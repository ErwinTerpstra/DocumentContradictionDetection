from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional

try:
    from external.claimify.src.claimify import Claimify
except Exception:
    Claimify = None

from src.claim_extraction.backends.local import call_local_llm
from src.claim_extraction.backends.remote import call_remote_llm
from src.claim_extraction.config import ExtractionConfig, LOCAL_DEFAULT_MAX_NEW_TOKENS, REMOTE_DEFAULT_MAX_NEW_TOKENS
from src.claim_extraction.prompts import (
    DIRECT_CLAIM_PROMPT_TEMPLATE,
    REPAIR_ONE_LINER_PROMPT_TEMPLATE,
    TEMPORAL_NORMALIZATION_ONLY_PROMPT_TEMPLATE,
)


def _log(message: str, verbose: bool) -> None:
    if verbose:
        # Reuse notebook run logger when available so extractor details land in the same file.
        run_logger = logging.getLogger("extraction_txt")
        if run_logger.handlers:
            run_logger.info(f"[extractor] {message}")
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


def _normalize_claims_keep_order(claims: List[str]) -> List[str]:
    """Normalize claims while preserving order and duplicates."""
    normalized: List[str] = []

    for claim in claims:
        text = str(claim).strip()
        if not text:
            continue

        # Remove bullet/numbering prefixes often produced by LLMs.
        text = re.sub(r"^[-*]\s+", "", text)
        text = re.sub(r"^\d+[.)]\s+", "", text)
        text = text.strip()

        if text:
            normalized.append(text)

    return normalized


def _claims_from_response(response: str) -> List[str]:
    """Convert raw LLM output into a list of claim lines."""
    lines = response.splitlines()
    return _normalize_claims(lines)


def _claims_from_response_keep_order(response: str) -> List[str]:
    """Convert raw LLM output into claim lines while preserving duplicates."""
    lines = response.splitlines()
    return _normalize_claims_keep_order(lines)


def _call_llm(prompt: str, config: ExtractionConfig) -> str:
    """Dispatch to the configured LLM backend."""
    if config.backend == "local":
        return call_local_llm(prompt, config)
    if config.backend == "remote":
        return call_remote_llm(prompt, config)
    raise ValueError(f"Unsupported backend '{config.backend}'. Use 'local' or 'remote'.")


def extract_claims(
    text: str,
    model_name: str,
    backend: str = "local",
    use_claimify: bool = False,
    temperature: float = 0.2,
    max_new_tokens: Optional[int] = None,
    remote_url: Optional[str] = None,
    remote_api_key: Optional[str] = None,
    remote_headers: Optional[Dict[str, str]] = None,
    remote_timeout: float = 120.0,
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
        max_new_tokens: Generation token limit. If omitted, uses backend defaults:
            local=1024 and remote=3072.
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
    if backend not in ("local", "remote"):
        raise ValueError(f"'backend' must be 'local' or 'remote', got '{backend}'.")

    resolved_max_new_tokens = max_new_tokens
    if resolved_max_new_tokens is None:
        resolved_max_new_tokens = (
            REMOTE_DEFAULT_MAX_NEW_TOKENS
            if backend == "remote"
            else LOCAL_DEFAULT_MAX_NEW_TOKENS
        )

    config = ExtractionConfig(
        model_name=model_name,
        backend=backend,
        temperature=temperature,
        max_new_tokens=resolved_max_new_tokens,
        use_claimify=use_claimify,
        remote_url=remote_url,
        remote_api_key=remote_api_key,
        remote_headers=remote_headers,
        remote_timeout=remote_timeout,
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
        prompt = DIRECT_CLAIM_PROMPT_TEMPLATE.format(text=text)
        response = _call_llm(prompt, config)
        claims = _claims_from_response(response)
   
        if len(claims) == 1:
            _log("One-line claim output detected after first extraction attempt.", config.verbose)
            _log(f"First attempt normalized one-line output: {claims[0]}", config.verbose)
            _log("Running one-line repair prompt (single attempt).", config.verbose)

            repair_prompt = REPAIR_ONE_LINER_PROMPT_TEMPLATE.format(text=claims[0])
            repair_config = ExtractionConfig(
                model_name=config.model_name,
                backend=config.backend,
                temperature=0.0,
                max_new_tokens=config.max_new_tokens,
                remote_url=config.remote_url,
                remote_api_key=config.remote_api_key,
                remote_headers=config.remote_headers,
                remote_timeout=config.remote_timeout,
                verbose=config.verbose,
            )
            _log("Repair call uses temperature=0.0.", config.verbose)
            repair_response = _call_llm(repair_prompt, repair_config)
            repaired_claims = _claims_from_response(repair_response)

            _log(f"Repair prompt produced {len(repaired_claims)} claim lines.", config.verbose)
            if len(repaired_claims) > 1:
                claims = repaired_claims
                _log("Repair successful: one-line output was split into multiple claims.", config.verbose)
            else:
                _log(f"Repair raw output: {repair_response}", config.verbose)
                if repaired_claims:
                    _log(f"Repair normalized one-line output: {repaired_claims[0]}", config.verbose)
                else:
                    _log("Repair normalized output is empty.", config.verbose)
                _log("Repair result is still one line (or empty); caller may skip this record.", config.verbose)

    _log(f"Extracted {len(claims)} claims.", config.verbose)
    return claims


def apply_temporal_corrections(
    claims: List[str],
    model_name: str,
    backend: str = "local",
    temperature: float = 0.0,
    max_new_tokens: Optional[int] = None,
    remote_url: Optional[str] = None,
    remote_api_key: Optional[str] = None,
    remote_headers: Optional[Dict[str, str]] = None,
    remote_timeout: float = 120.0,
    verbose: bool = True,
) -> List[str]:
    """Apply temporal normalization rules to already extracted claims.

    This function preserves claim order where possible and only applies the
    temporal correction pass.
    """
    if not isinstance(claims, list):
        raise ValueError("'claims' must be a list of strings.")
    if not claims:
        return []
    if any(not isinstance(claim, str) for claim in claims):
        raise ValueError("All items in 'claims' must be strings.")
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("'model_name' must be a non-empty string.")
    if backend not in ("local", "remote"):
        raise ValueError(f"'backend' must be 'local' or 'remote', got '{backend}'.")

    input_claims = _normalize_claims_keep_order(claims)
    if not input_claims:
        return []

    resolved_max_new_tokens = max_new_tokens
    if resolved_max_new_tokens is None:
        resolved_max_new_tokens = (
            REMOTE_DEFAULT_MAX_NEW_TOKENS
            if backend == "remote"
            else LOCAL_DEFAULT_MAX_NEW_TOKENS
        )

    config = ExtractionConfig(
        model_name=model_name,
        backend=backend,
        temperature=temperature,
        max_new_tokens=resolved_max_new_tokens,
        use_claimify=False,
        remote_url=remote_url,
        remote_api_key=remote_api_key,
        remote_headers=remote_headers,
        remote_timeout=remote_timeout,
        verbose=verbose,
    )

    _log(
        (
            "Starting temporal correction pass | "
            f"backend={config.backend} | model={config.model_name} | claims={len(input_claims)}"
        ),
        config.verbose,
    )

    prompt = TEMPORAL_NORMALIZATION_ONLY_PROMPT_TEMPLATE.format(text="\n".join(input_claims))
    response = _call_llm(prompt, config)
    corrected_claims = _claims_from_response_keep_order(response)

    if len(corrected_claims) != len(input_claims):
        _log(
            (
                "Temporal correction returned a different claim count "
                f"(in={len(input_claims)}, out={len(corrected_claims)}); returning corrected claims."
            ),
            config.verbose,
        )

    _log(f"Temporal correction completed for {len(corrected_claims)} claims.", config.verbose)
    return corrected_claims
