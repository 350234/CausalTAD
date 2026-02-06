#!/usr/bin/env python3
'''

python3 ./get_weights.py --csv data/fakejob/fake_job_postings.csv --factors data/fakejob/fakejob_factors.json --out data/fakejob/fakejob_weights.json --skip-columns "fraudulent"

'''

import argparse
import csv
import json
import os
from typing import Any, Dict, List

def _read_csv_header(csv_path: str) -> List[str]:
	if not os.path.exists(csv_path):
		raise FileNotFoundError(f"CSV not found: {csv_path}")

	with open(csv_path, "r", encoding="utf-8", newline="") as f:
		reader = csv.reader(f)
		try:
			header = next(reader)
		except StopIteration as exc:
			raise ValueError(f"CSV is empty: {csv_path}") from exc

	header = [h.strip() for h in header if h is not None]
	if not header:
		raise ValueError(f"CSV header is empty: {csv_path}")
	return header


def _read_factors_json(json_path: str) -> Dict[str, Any]:
	if not os.path.exists(json_path):
		raise FileNotFoundError(f"Factors JSON not found: {json_path}")

	with open(json_path, "r", encoding="utf-8") as f:
		data = json.load(f)
	if not isinstance(data, dict):
		raise ValueError(f"Invalid factors JSON (expected object): {json_path}")
	return data


def compute_column_weights(
	columns: List[str], factors_payload: Dict[str, Any], skip_columns: List[str] | None = None
) -> Dict[str, int]:
	"""Compute weights for columns, optionally skipping some columns.

	Args:
		columns: list of column names from CSV header
		factors_payload: loaded factors JSON payload
		skip_columns: list of column names to skip (these will not appear in result)

	Returns:
		dict mapping column -> weight for non-skipped columns
	"""
	skip_set = {s.strip() for s in (skip_columns or []) if s is not None}
	weights: Dict[str, int] = {c: 0 for c in columns if c not in skip_set}
	factors = factors_payload.get("factors", {})
	if not isinstance(factors, dict):
		raise ValueError('Invalid factors JSON: key "factors" must be an object')

	for _, factor in factors.items():
		if not isinstance(factor, dict):
			continue
		column_based = factor.get("column_based", [])
		if column_based is None:
			continue
		if not isinstance(column_based, list):
			raise ValueError('Invalid factors JSON: factor."column_based" must be a list when present')

		column_set = {str(x).strip() for x in column_based}
		for col in columns:
			if col in column_set and col in weights:
				weights[col] += 1

	return weights


def main() -> None:
	parser = argparse.ArgumentParser(
		description=(
			"Compute per-column weights from a CSV header and a factors JSON. "
			"Weight increases by 1 for each factor whose column_based contains the column name."
		)
	)
	parser.add_argument(
		"--csv",
		default="data/vifd/carclaims_train.csv",
		help="Path to CSV whose header contains column names (default: data/vifd/carclaims_train.csv)",
	)
	parser.add_argument(
		"--factors",
		default="data/vifd/vifd_factors.json",
		help="Path to factors JSON (default: data/vifd/vifd_factors.json)",
	)
	parser.add_argument(
		"--out",
		default="data/vifd/vifd_weights.json",
		help="Path to output weights JSON (default: data/vifd/vifd_weights.json)",
	)
	parser.add_argument(
		"--skip-columns",
		default="fraud found",
		help=(
			"Comma-separated list of column names to skip when saving weights "
			"(default: 'fraud found')"
		),
	)
	args = parser.parse_args()

	columns = _read_csv_header(args.csv)
	factors_payload = _read_factors_json(args.factors)
	skip_columns = [s.strip() for s in args.skip_columns.split(",") if s.strip()]
	weights = compute_column_weights(columns, factors_payload, skip_columns=skip_columns)

	out_dir = os.path.dirname(args.out)
	if out_dir:
		os.makedirs(out_dir, exist_ok=True)

	payload = {
		"csv": args.csv,
		"factors": args.factors,
		"num_columns": len(columns),
		"weights": weights,
		"skipped_columns": skip_columns,
	}
	with open(args.out, "w", encoding="utf-8") as f:
		json.dump(payload, f, ensure_ascii=False, indent=2)
		f.write("\n")


if __name__ == "__main__":
	main()



