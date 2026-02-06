"""Export full prompts (system + user) seen by the model in each COAT (exp_vifd) stage.

Purpose:
- Reproduce/check prompt correctness (especially after switching dataset or templates)
- Avoid calling online LLMs (uses FakeClient by default) to save cost

Run:
- conda run -n coat python export_all_stage_prompts.py

Output:
- exp_vifd/outputs/prompts_audit/<timestamp>/*

Notes:
- “Stages” here refers to the LLM-calling stages (step1 factor discovery, step2 annotation, factor selection).
- step1.5/step3 do not call LLMs, so there are no prompts to export.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List

from config import Config
from prompts import (
    build_annotation_prompt,
    build_annotation_prompt_for_new_factors,
    build_factor_discovery_prompt,
    build_factor_selection_prompt,
    build_batch_samples_text,
)
from utils import load_dataset, load_data_distribution, save_prompt, format_sample


def _write_messages(out_dir: str, name: str, system: str | None, user: str) -> None:
    """Save both plain text and messages JSON for review."""
    system = system or ""
    txt = (
        "# SYSTEM\n" + system + "\n\n" +
        "# USER\n" + user + "\n"
    )
    save_prompt(txt, os.path.join(out_dir, f"{name}.txt"))
    save_prompt(
        json.dumps(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            ensure_ascii=False,
            indent=2,
        ),
        os.path.join(out_dir, f"{name}.messages.json"),
    )


class _FakeMessage:
    def __init__(self, content: str):
        self.content = content


class _FakeChoice:
    def __init__(self, content: str):
        self.message = _FakeMessage(content)


class _FakeCompletion:
    def __init__(self, content: str):
        self.choices = [_FakeChoice(content)]


class FakeClient:
    """Minimal simulation of OpenAI client.chat.completions.create."""

    def __init__(self):
        self.model_name = "fake-model"

        class _Completions:
            @staticmethod
            def create(model: str, messages: List[Dict[str, str]], temperature: float = 0.7):
                # Return a stable factor definition for downstream prompt generation
                content = json.dumps(
                    {
                        "factors": {
                            "job_post_completeness": {
                                "description": "Job posting text completeness level (description + requirements + benefits completeness and structure)",
                                "possible_values": [0, 1, 2, 3, -1],
                                "annotation_criteria": "Read description/requirements/benefits: if all three are empty or extremely short=0; only one is relatively detailed=1; at least two are detailed and clear=2; all three are detailed with bullets/sections=3; unknown=-1.",
                                "column_based": ["description", "requirements", "benefits"],
                            },
                            "role_family": {
                                "description": "Job function category (summarized from title/description/function)",
                                "possible_values": [0, 1, 2, 3, 4, 5, -1],
                                "annotation_criteria": "Use keywords in title/function/description: 0=tech/IT/engineering; 1=sales/business; 2=marketing/operations; 3=customer service/support; 4=management/leadership; 5=other; insufficient info=-1.",
                                "column_based": ["title", "description", "function"],
                            },
                        },
                        "recommended_focus_factors": [
                            {
                                "factor_name": "job_post_completeness",
                                "reasoning": "Text columns provide strong signals, multi-level values, and relate to multiple information dimensions",
                                "expected_value_diversity": "Expected 3-4 distinct values",
                            }
                        ],
                    },
                    ensure_ascii=False,
                )
                return _FakeCompletion(content)

        class _Chat:
            completions = _Completions

        self.chat = _Chat()


def main() -> None:
    cfg = Config()
    cfg.create_directories()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(cfg.OUTPUTS_DIR, "prompts_audit", ts)
    os.makedirs(out_dir, exist_ok=True)

    meta = {
        "dataset_path": cfg.DATASET_PATH,
        "prompts_library_dir": cfg.PROMPTS_LIBRARY_DIR,
        "data_distribution_path": cfg.DATA_DISTRIBUTION_PATH,
        "generated_at": ts,
    }
    save_prompt(json.dumps(meta, ensure_ascii=False, indent=2), os.path.join(out_dir, "_meta.json"))

    # 1) step1: factor discovery prompt
    df = load_dataset(cfg.DATASET_PATH)
    data_distribution = load_data_distribution(cfg.DATA_DISTRIBUTION_PATH)
    samples = df.sample(min(cfg.SAMPLES_PER_ITERATION, len(df)), random_state=42)
    samples_text = build_batch_samples_text(samples, cfg.SAMPLES_PER_ITERATION)

    user_prompt = build_factor_discovery_prompt(
        samples_text=samples_text,
        data_distribution=data_distribution,
        domain_context=cfg.DOMAIN_CONTEXT,
        iteration_num=1,
        existing_factors=None,
        deleted_factors=None,
    )
    _write_messages(out_dir, "01_factor_discovery", cfg.SYSTEM_INSTRUCTION_FACTOR_DISCOVERY, user_prompt)

    # 2) To generate later-stage prompts, use FakeClient to create example factors (no cost)
    fake = FakeClient()
    factors_obj = json.loads(fake.chat.completions.create(model=fake.model_name, messages=[{"role": "user", "content": user_prompt}], temperature=0.0).choices[0].message.content)
    factors = factors_obj.get("factors", {}) if isinstance(factors_obj, dict) else {}

    factors_json = json.dumps(factors, ensure_ascii=False, indent=2)

    # 3) Factor selection (if you use this step later)
    user_prompt_sel = build_factor_selection_prompt(factors_json=factors_json, domain_context=cfg.DOMAIN_CONTEXT)
    _write_messages(out_dir, "02_factor_selection", "", user_prompt_sel)

    # 4) step2: full annotation prompts (first 3 samples)
    n_show = min(3, len(df))
    for i in range(n_show):
        row = df.iloc[i]
        sample_text = format_sample(row)
        user_prompt_ann = build_annotation_prompt(sample_text=sample_text, factors_json=factors_json, domain_context=cfg.DOMAIN_CONTEXT)
        _write_messages(out_dir, f"03_annotation_full_sample{i+1}", cfg.SYSTEM_INSTRUCTION_ANNOTATION, user_prompt_ann)

    # 5) step2: incremental annotation prompts (use subset of factors as new_factors)
    new_factors = {}
    for k in list(factors.keys())[:1]:
        new_factors[k] = factors[k]
    new_factors_json = json.dumps(new_factors, ensure_ascii=False, indent=2)

    for i in range(n_show):
        row = df.iloc[i]
        sample_text = format_sample(row)
        user_prompt_inc = build_annotation_prompt_for_new_factors(
            sample_text=sample_text,
            new_factors_json=new_factors_json,
            domain_context=cfg.DOMAIN_CONTEXT,
        )
        _write_messages(out_dir, f"04_annotation_incremental_sample{i+1}", cfg.SYSTEM_INSTRUCTION_ANNOTATION, user_prompt_inc)

    print(f"✓ Exported full prompts for all stages to: {out_dir}")


if __name__ == "__main__":
    main()



