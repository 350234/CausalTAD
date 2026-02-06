#!/usr/bin/env python3
"""Aggregate multiple per-column weight JSONs into a single weighted total.

Input JSON format (each file):
{
  "weights": {
    "job_id": 0,
    "title": 8,
    "location": 3
  }
}

Processing:
1) Ensure all files have identical column-name sets (order does not matter).
2) Normalize each weight set by ai / sum(ai).
3) Compute total weight per column: sum_i (alpha_i * normalized_i[col]).

Output JSON (default: data/fakejob/fakejob_all_weights.json):
{
  "input_files": [...],
  "alphas": [...],
  "weights": {"col": 0.123, ...}
}
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping


@dataclass(frozen=True)
class WeightsPayload:
	path: str
	weights: Dict[str, float]


def _read_json(path: str) -> Any:
	if not os.path.exists(path):
		raise FileNotFoundError(f"Weights JSON not found: {path}")
	with open(path, "r", encoding="utf-8") as f:
		return json.load(f)


def _parse_weights_payload(path: str) -> WeightsPayload:
	data = _read_json(path)
	if not isinstance(data, dict):
		raise ValueError(f"Invalid weights JSON (expected object): {path}")
	weights = data.get("weights")
	if not isinstance(weights, dict):
		raise ValueError(f"Invalid weights JSON: key 'weights' must be an object: {path}")

	parsed: Dict[str, float] = {}
	for k, v in weights.items():
		col = str(k).strip()
		if not col:
			raise ValueError(f"Invalid column name in weights JSON: {path}")
		if not isinstance(v, (int, float)):
			raise ValueError(
				f"Invalid weight for column '{col}' in {path}: expected number, got {type(v).__name__}"
			)
		parsed[col] = float(v)

	if not parsed:
		raise ValueError(f"Empty weights in JSON: {path}")
	return WeightsPayload(path=path, weights=parsed)


def _validate_same_columns(payloads: List[WeightsPayload]) -> List[str]:
	"""Return stable-sorted column list after verifying column sets match."""
	base_cols = set(payloads[0].weights.keys())
	for p in payloads[1:]:
		cols = set(p.weights.keys())
		if cols != base_cols:
			missing = sorted(base_cols - cols)
			extra = sorted(cols - base_cols)
			msg_parts = [f"Column set mismatch between files:", f"- base: {payloads[0].path}", f"- this: {p.path}"]
			if missing:
				msg_parts.append(f"- missing in {p.path}: {missing}")
			if extra:
				msg_parts.append(f"- extra in {p.path}: {extra}")
			raise ValueError("\n".join(msg_parts))
	return sorted(base_cols)


def _normalize(weights: Mapping[str, float]) -> Dict[str, float]:
	total = sum(float(v) for v in weights.values())
	if total == 0:
		raise ValueError("Cannot normalize weights: sum(weights)=0")
	return {k: float(v) / total for k, v in weights.items()}


def aggregate_weights(
	weights_payloads: List[WeightsPayload], alphas: List[float]
) -> Dict[str, float]:
	if not weights_payloads:
		raise ValueError("No weights files provided")
	if len(weights_payloads) != len(alphas):
		raise ValueError(
			f"Number of weights files must equal number of alphas (files={len(weights_payloads)}, alphas={len(alphas)})"
		)

	columns = _validate_same_columns(weights_payloads)
	normalized_list = [_normalize(p.weights) for p in weights_payloads]

	agg: Dict[str, float] = {c: 0.0 for c in columns}
	for alpha, norm in zip(alphas, normalized_list, strict=True):
		for col in columns:
			agg[col] += float(alpha) * norm[col]
	return agg


def _parse_csv_list(value: str) -> List[str]:
	# Accept either: "a,b,c" or "a b c" style when passed as a single string
	if value is None:
		return []
	s = value.strip()
	if not s:
		return []
	if "," in s:
		return [x.strip() for x in s.split(",") if x.strip()]
	return [x.strip() for x in s.split() if x.strip()]


def main() -> None:
	parser = argparse.ArgumentParser(
		description=(
			"Read multiple per-column weight JSON files, normalize each, then combine them with given coefficients "
			"to produce a single aggregated weights JSON."
		)
	)

	parser.add_argument(
		"--weights-files",
		nargs="+",
		default=["data/fakejob/fakejob_weights_12_30.json"],
		help=(
			"One or more input weight JSON files. Each must contain a top-level 'weights' object. "
			"(default: data/fakejob/fakejob_weights_12_30.json)"
		),
	)
	parser.add_argument(
		"--alphas",
		nargs="+",
		type=float,
		default=[1.0],
		help=(
			"One or more coefficients, same count as --weights-files. "
			"Total weight per column = sum_i alpha_i * normalized_i[col]. (default: 1.0)"
		),
	)
	parser.add_argument(
		"--num-files",
		type=int,
		default=None,
		help="Optional sanity-check: expected number of weight files.",
	)
	parser.add_argument(
		"--num-weights",
		type=int,
		default=None,
		help="Optional sanity-check: expected number of coefficients.",
	)
	parser.add_argument(
		"--out",
		default="data/fakejob/fakejob_all_weights.json",
		help="Output path for aggregated weights JSON (default: data/fakejob/fakejob_all_weights.json)",
	)
	args = parser.parse_args()

	weights_files: List[str] = list(args.weights_files)
	alphas: List[float] = list(args.alphas)

	if args.num_files is not None and args.num_files != len(weights_files):
		raise ValueError(
			f"--num-files mismatch: expected {args.num_files}, got {len(weights_files)} from --weights-files"
		)
	if args.num_weights is not None and args.num_weights != len(alphas):
		raise ValueError(f"--num-weights mismatch: expected {args.num_weights}, got {len(alphas)} from --alphas")

	payloads = [_parse_weights_payload(p) for p in weights_files]
	agg = aggregate_weights(payloads, alphas)

	out_dir = os.path.dirname(args.out)
	if out_dir:
		os.makedirs(out_dir, exist_ok=True)

	out_payload = {
		"input_files": weights_files,
		"alphas": alphas,
		"weights": agg,
	}
	with open(args.out, "w", encoding="utf-8") as f:
		json.dump(out_payload, f, ensure_ascii=False, indent=2)
		f.write("\n")


if __name__ == "__main__":
	main()




