"""Claim extraction scaffolding for document contradiction detection.

This module offers one primary function:
	extract_claims(text, ...)

Key features:
- Select model via parameter.
- Run against local or remote LLM backends.
- Toggle between Claimify and a direct prompt-based extraction flow.
- Include a smoke test that runs claim extraction twice on a long story.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple, cast

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
	from external.claimify.src.claimify import Claimify
except Exception:
	Claimify = None


BackendType = Literal["local", "remote"]


@dataclass
class ExtractionConfig:
	"""Configuration for one extraction run."""

	model_name: str
	backend: str = "local"
	temperature: float = 0.2
	max_new_tokens: int = 1024
	use_claimify: bool = False
	remote_url: Optional[str] = None
	remote_api_key: Optional[str] = None
	remote_headers: Optional[Dict[str, str]] = None
	question_for_claimify: str = "Extract all specific and verifiable claims from the answer."
	verbose: bool = True


_LOCAL_MODEL_CACHE: Dict[str, Tuple[Any, Any, torch.device]] = {}


def _log(message: str, verbose: bool) -> None:
	if verbose:
		print(message)


def _normalize_claims(claims: List[str]) -> List[str]:
	"""Normalize and de-duplicate claims while preserving order."""
	seen = set()
	normalized: List[str] = []

	for claim in claims:
		text = str(claim).strip()
		if not text:
			continue

		# Remove bullet/numbering prefixes often produced by LLMs.
		text = re.sub(r"^[-*]\s+", "", text)
		text = re.sub(r"^\d+[.)]\s+", "", text)
		text = text.strip()

		if text and text not in seen:
			seen.add(text)
			normalized.append(text)

	return normalized


def _claims_from_response(response: str) -> List[str]:
	"""Convert raw LLM output into a list of claim lines."""
	lines = response.splitlines()
	return _normalize_claims(lines)


def _direct_claim_prompt(text: str) -> str:
	"""Build a strict prompt for direct claim extraction without Claimify."""
	return (
		"You are an expert claim extractor. Extract claims from the text below.\n\n"
		"Rules:\n"
		"1) Every claim must be atomic: one checkable statement per line.\n"
		"2) Every claim must be self-contained and explicit. Resolve pronouns and avoid vague references.\n"
		"3) Prefer maximum uniqueness of entities: include identifying context in the claim itself.\n"
		"   Example: 'The mouse Tim has a red nose' (not: 'Tim has a red nose').\n"
		"4) Claims may be facts or conditional if-then statements.\n"
		"   Example fact: 'There exists a mouse named Tim'.\n"
		"   Example conditional: 'If the sky falls down, then every human wears a hat'.\n"
		"5) No summaries, no explanations, no metadata, no numbering.\n"
		"6) Output only claims, exactly one claim per line.\n\n"
		"Text:\n"
		f"{text}"
	)


def _load_local_model(model_name: str) -> Tuple[Any, Any, torch.device]:
	"""Load and cache a local HF model and tokenizer."""
	if model_name in _LOCAL_MODEL_CACHE:
		return _LOCAL_MODEL_CACHE[model_name]

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	dtype = torch.float16 if device.type == "cuda" else torch.float32

	tokenizer = AutoTokenizer.from_pretrained(model_name)
	model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
	model = cast(Any, model).to(device)
	_LOCAL_MODEL_CACHE[model_name] = (tokenizer, model, device)
	return tokenizer, model, device


def _call_local_llm(prompt: str, config: ExtractionConfig) -> str:
	"""Run local generation with transformers using chat template when available."""
	tokenizer, model, device = _load_local_model(config.model_name)

	messages = [{"role": "user", "content": prompt}]

	if hasattr(tokenizer, "apply_chat_template"):
		rendered = tokenizer.apply_chat_template(
			messages,
			tokenize=False,
			add_generation_prompt=True,
			enable_thinking=False,
		)
	else:
		rendered = prompt

	model_inputs = tokenizer([rendered], return_tensors="pt").to(device)
	do_sample = config.temperature > 0
	generate_kwargs = {
		"max_new_tokens": config.max_new_tokens,
		"do_sample": do_sample,
	}
	if do_sample:
		generate_kwargs["temperature"] = config.temperature

	generated = model.generate(**model_inputs, **generate_kwargs)
	output_ids = generated[0][len(model_inputs.input_ids[0]) :].tolist()

	# Qwen-style marker for optional reasoning blocks; keep best-effort stripping.
	try:
		index = len(output_ids) - output_ids[::-1].index(151668)
	except ValueError:
		index = 0

	return tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip()


def _call_remote_llm(prompt: str, config: ExtractionConfig) -> str:
	"""Call a remote OpenAI-compatible Chat Completions endpoint."""
	api_key = config.remote_api_key or os.getenv("OPENAI_API_KEY")
	if not api_key:
		raise RuntimeError("Remote backend requested but OPENAI_API_KEY is missing.")

	remote_url = config.remote_url or os.getenv("OPENAI_CHAT_COMPLETIONS_URL")
	if not remote_url:
		remote_url = "https://api.openai.com/v1/chat/completions"

	payload = {
		"model": config.model_name,
		"messages": [{"role": "user", "content": prompt}],
		"temperature": config.temperature,
		"max_tokens": config.max_new_tokens,
	}

	headers = {
		"Content-Type": "application/json",
		"Authorization": f"Bearer {api_key}",
	}
	if config.remote_headers:
		headers.update(config.remote_headers)

	request = urllib.request.Request(
		remote_url,
		data=json.dumps(payload).encode("utf-8"),
		headers=headers,
		method="POST",
	)

	try:
		with urllib.request.urlopen(request) as response:
			body = response.read().decode("utf-8")
	except urllib.error.HTTPError as exc:
		details = exc.read().decode("utf-8", errors="ignore")
		raise RuntimeError(f"Remote LLM call failed with HTTP {exc.code}: {details}") from exc
	except urllib.error.URLError as exc:
		raise RuntimeError(f"Remote LLM call failed: {exc.reason}") from exc

	data = json.loads(body)
	choices = data.get("choices", [])
	if not choices:
		raise RuntimeError("Remote LLM call returned no choices.")

	message = choices[0].get("message", {})
	content = message.get("content", "")
	if not content:
		raise RuntimeError("Remote LLM returned an empty content field.")

	return content.strip()


def _call_llm(prompt: str, config: ExtractionConfig) -> str:
	"""Dispatch to the configured LLM backend."""
	if config.backend == "local":
		return _call_local_llm(prompt, config)
	if config.backend == "remote":
		return _call_remote_llm(prompt, config)
	raise ValueError(f"Unsupported backend '{config.backend}'. Use 'local' or 'remote'.")


def extract_claims(
	text: str,
	model_name: str,
	backend: str = "local",
	use_claimify: bool = False,
	temperature: float = 0.2,
	max_new_tokens: int = 1024,
	remote_url: Optional[str] = None,
	remote_api_key: Optional[str] = None,
	remote_headers: Optional[Dict[str, str]] = None,
	question_for_claimify: str = "Extract all specific and verifiable claims from the answer.",
	verbose: bool = True,
) -> List[str]:
	"""Extract a list of claims from one text string.

	Args:
		text: Input document/story as one string.
		model_name: LLM model identifier.
		backend: 'local' or 'remote'.
		use_claimify: If True, run Claimify pipeline. If False, use direct prompt-based extraction.
		temperature: Decoding temperature for LLM generation.
		max_new_tokens: Generation token limit.
		remote_url: Optional remote endpoint URL for chat completions.
		remote_api_key: Optional API key override for remote calls.
		remote_headers: Optional extra headers for remote calls.
		question_for_claimify: Question passed into Claimify with answer=text.
		verbose: Print debug output.

	Returns:
		A normalized list of extracted claims.
	"""
	if not isinstance(text, str) or not text.strip():
		raise ValueError("'text' must be a non-empty string.")
	if not isinstance(model_name, str) or not model_name.strip():
		raise ValueError("'model_name' must be a non-empty string.")

	config = ExtractionConfig(
		model_name=model_name,
		backend=backend,
		temperature=temperature,
		max_new_tokens=max_new_tokens,
		use_claimify=use_claimify,
		remote_url=remote_url,
		remote_api_key=remote_api_key,
		remote_headers=remote_headers,
		question_for_claimify=question_for_claimify,
		verbose=verbose,
	)

	_log(
		(
			"Starting claim extraction | "
			f"backend={config.backend} | model={config.model_name} | use_claimify={config.use_claimify}"
		),
		config.verbose,
	)

	if config.use_claimify:
		if Claimify is None:
			raise RuntimeError(
				"Claimify is not available. Set use_claimify=False or ensure external.claimify is importable."
			)

		def llm_adapter(prompt: str, temp: float) -> str:
			adapter_config = ExtractionConfig(
				model_name=config.model_name,
				backend=config.backend,
				temperature=temp,
				max_new_tokens=config.max_new_tokens,
				remote_url=config.remote_url,
				remote_api_key=config.remote_api_key,
				remote_headers=config.remote_headers,
				verbose=config.verbose,
			)
			return _call_llm(prompt, adapter_config)

		claimify = Claimify(llm_function=llm_adapter)
		claims = claimify.extract_claims(question=config.question_for_claimify, answer=text)
		claims = _normalize_claims(claims)
	else:
		prompt = _direct_claim_prompt(text)
		response = _call_llm(prompt, config)
		claims = _claims_from_response(response)

	_log(f"Extracted {len(claims)} claims.", config.verbose)
	return claims


def smoke_test_claim_extractor() -> None:
	"""Smoke test: run extraction twice on one long and complex story."""
	story = (
		"In the valley of Meridia, a mouse named Tim wore a red brass-button coat and carried a map "
		"that his grandmother had drawn in 1984. Tim claimed the map showed three bridges over the "
		"river, but the mayor had announced last winter that only two bridges remained after flooding. "
		"At dawn, engineer Nora inspected the eastern bridge and told the council that the structure was "
		"safe for carts lighter than 500 kilograms. During the same meeting, historian Elias argued that "
		"the eastern bridge had already collapsed in 1999. Tim then said that if the eastern bridge was "
		"really safe, then every farmer from Oak District would deliver grain by sunset. By noon, two "
		"farmers from Oak District delivered grain, while four others reported blocked roads. In the "
		"afternoon, meteorologist Lina predicted heavy rain by evening, and she added that if rainfall "
		"exceeded 30 millimeters, then the western bridge would close automatically. The rain gauge later "
		"measured 34 millimeters. Nevertheless, the western bridge remained open until midnight according "
		"to the city logbook. Separately, archivist Omar wrote that there exists a lighthouse keeper named "
		"Rae who had never visited Meridia, even though a travel diary signed by Rae described a market "
		"visit in Meridia in 2022. At the festival, the announcer said the town had exactly 120 lanterns, "
		"while the inventory sheet listed 118 lanterns and two repaired frames. Finally, council minutes "
		"stated that if the budget was approved on Tuesday, then the school roof would be repaired before "
		"October, but the budget vote was postponed to Thursday."
	)

	has_remote_key = bool(os.getenv("OPENAI_API_KEY"))

	run_configs = [
		{
			"label": "Run 1 - Local model",
			"model_name": os.getenv("CLAIM_MODEL_1", "Qwen/Qwen3.5-0.8B"),
			"backend": "local",
		},
		{
			"label": "Run 2 - Remote model" if has_remote_key else "Run 2 - Local fallback model",
			"model_name": (
				os.getenv("CLAIM_MODEL_2", "gpt-4o-mini")
				if has_remote_key
				else os.getenv("CLAIM_MODEL_2", "Qwen/Qwen2.5-0.5B-Instruct")
			),
			"backend": "remote" if has_remote_key else "local",
		},
	]

	for cfg in run_configs:
		print("\n" + "=" * 80)
		print(cfg["label"])
		print(f"backend={cfg['backend']} | model={cfg['model_name']}")
		print("=" * 80)

		try:
			claims = extract_claims(
				text=story,
				model_name=cfg["model_name"],
				backend=cfg["backend"],
				use_claimify=False,
				temperature=0.1,
				max_new_tokens=1536,
				verbose=True,
			)
			for idx, claim in enumerate(claims, start=1):
				print(f"{idx:02d}. {claim}")
		except Exception as exc:
			print(f"Extraction failed for {cfg['label']}: {exc}")


if __name__ == "__main__":
	smoke_test_claim_extractor()
