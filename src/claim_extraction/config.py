from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Literal, Optional

BackendType = Literal["local", "remote"]

DOTENV_PATH = os.path.join(os.path.dirname(__file__), "..", "..", ".env")


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
