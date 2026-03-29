from __future__ import annotations

"""Smoke test: run claim extraction on a ContraDoc story with all configured models.

Run from the repo root:
    python scripts/smoke_test_claim_extractor.py
"""

import json
import os
import random
import sys

# Ensure repo root is on the path when running as a script.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import dotenv_values

from src.claim_extraction.config import DOTENV_PATH
from src.claim_extraction.extractor import extract_claims

CONTRADOC_STORY_ID_OVERRIDE = "3489825766_2"  # Set to a specific ContraDoc story ID, or "" for random.

FALLBACK_STORY = (
    "In the valley of Meridia, a mouse named Tim wore a red brass-button coat and carried a map that his"
    " grandmother had drawn in 1984. Tim claimed the map showed three bridges over the river, but the mayor had"
    " announced last winter that only two bridges remained after flooding. At dawn, engineer Nora inspected the"
    " eastern bridge and told the council that the structure was safe for carts lighter than 500 kilograms."
    " During the same meeting, historian Elias argued that the eastern bridge had already collapsed in 1999."
    " Tim then said that if the eastern bridge was really safe, then every farmer from Oak District would deliver"
    " grain by sunset. By noon, two farmers from Oak District delivered grain, while four others reported blocked"
    " roads. In the afternoon, meteorologist Lina predicted heavy rain by evening, and she added that if rainfall"
    " exceeded 30 millimeters, then the western bridge would close automatically. The rain gauge later measured"
    " 34 millimeters. Nevertheless, the western bridge remained open until midnight according to the city logbook."
    " Separately, archivist Omar wrote that there exists a lighthouse keeper named Rae who had never visited"
    " Meridia, even though a travel diary signed by Rae described a market visit in Meridia in 2022. At the"
    " festival, the announcer said the town had exactly 120 lanterns, while the inventory sheet listed 118"
    " lanterns and two repaired frames. Finally, council minutes stated that if the budget was approved on"
    " Tuesday, then the school roof would be repaired before October, but the budget vote was postponed to"
    " Thursday."
)


def _load_story() -> str:
    dataset_path = os.path.join(os.path.dirname(__file__), "..", "datasets", "ContraDoc", "ContraDoc.json")
    try:
        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        all_examples = []
        for split_name in ("pos", "neg"):
            split = dataset.get(split_name, {})
            if isinstance(split, dict):
                all_examples.extend(split.values())

        if CONTRADOC_STORY_ID_OVERRIDE:
            example = dataset.get("pos", {}).get(CONTRADOC_STORY_ID_OVERRIDE)
            if example is None:
                example = dataset.get("neg", {}).get(CONTRADOC_STORY_ID_OVERRIDE)
            if isinstance(example, dict) and isinstance(example.get("text"), str) and example["text"].strip():
                print(f"Loaded ContraDoc story override: {CONTRADOC_STORY_ID_OVERRIDE}")
                return example["text"].strip()

        candidates = [e for e in all_examples if isinstance(e, dict) and isinstance(e.get("text"), str) and e["text"].strip()]
        if candidates:
            example = random.choice(candidates)
            print(f"Loaded random ContraDoc story: {example.get('unique id', 'unknown-id')}")
            return example["text"].strip()

    except Exception as exc:
        print(f"Could not load ContraDoc story. Using fallback story. Reason: {exc}")

    return FALLBACK_STORY


def main() -> None:
    story = _load_story()

    print("\n" + "=" * 80)
    print("Smoke Test Story")
    print("=" * 80)
    print(story)
    print("=" * 80 + "\n")

    env = dotenv_values(DOTENV_PATH)
    has_remote_key = bool(env.get("OPENAI_API_KEY"))

    run_configs = [
        {"label": "Run 1 - Local model", "model_name": env.get("CLAIM_MODEL_1"), "backend": "local"},
        {"label": "Run 2 - Local model", "model_name": env.get("CLAIM_MODEL_2"), "backend": "local"},
    ]

    if has_remote_key:
        remote_model = env.get("CLAIM_MODEL_REMOTE")
        if not remote_model:
            print("OPENAI_API_KEY is set but CLAIM_MODEL_REMOTE is missing; skipping remote run.")
        else:
            run_configs.append({"label": "Run 3 - Remote model", "model_name": remote_model, "backend": "remote"})

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
    main()
