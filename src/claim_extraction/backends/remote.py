from __future__ import annotations

import json
import urllib.error
import urllib.request

from dotenv import dotenv_values

from src.claim_extraction.config import DOTENV_PATH, ExtractionConfig


def call_remote_llm(prompt: str, config: ExtractionConfig) -> str:
    """Call a remote OpenAI-compatible Chat Completions endpoint."""
    api_key = config.remote_api_key or dotenv_values(DOTENV_PATH).get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Remote backend requested but OPENAI_API_KEY is missing.")

    remote_url = config.remote_url or dotenv_values(DOTENV_PATH).get("OPENAI_CHAT_COMPLETIONS_URL")
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
