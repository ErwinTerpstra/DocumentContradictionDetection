from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Literal, Optional

BackendType = Literal["local", "remote"]

DOTENV_PATH = os.path.join(os.path.dirname(__file__), "..", "..", ".env")

LOCAL_DEFAULT_MAX_NEW_TOKENS = 1024
REMOTE_DEFAULT_MAX_NEW_TOKENS = 3072


@dataclass
class ExtractionConfig:
    """Configuration for one extraction run."""

    model_name: str
    backend: BackendType = "local"
    temperature: float = 0.2
    max_new_tokens: int = LOCAL_DEFAULT_MAX_NEW_TOKENS
    use_claimify: bool = False
    remote_url: Optional[str] = None
    remote_api_key: Optional[str] = None
    remote_headers: Optional[Dict[str, str]] = None
    remote_timeout: float = 120.0
    question_for_claimify: str = "Extract all specific and verifiable claims from the answer."
    verbose: bool = True
