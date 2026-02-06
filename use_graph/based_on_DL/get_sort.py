#!/usr/bin/env python3

"""Generate column ordering based on a factor causal order.

Inputs (defaults are for fakejob dataset):
- Factor ordering JSON: contains a 1D list at key "causal_order".
- Factor-to-columns mapping JSON: factors.<factor>.column_based is a list of column names.
- CSV with column-name set: first row header is used to detect missing columns.

Logic:
1) Iterate factors in causal_order, append their mapped columns (shuffled per factor), skipping duplicates.
2) Load CSV header, find any columns not in the ordering, and randomly insert them into the ordering.
3) Save result as JSON 2D list: [ [col1, col2, ...] ].
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


def _load_json(path: Path):
	with path.open("r", encoding="utf-8") as f:
		return json.load(f)


def _dump_json(path: Path, obj) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8") as f:
		json.dump(obj, f, ensure_ascii=False, indent=2)


def _build_factor_to_columns(factors_json: dict) -> Dict[str, List[str]]:
	factors = factors_json.get("factors")
	if not isinstance(factors, dict):
		raise ValueError("Invalid factors json: missing object key 'factors'.")

	factor_to_cols: Dict[str, List[str]] = {}
	for factor_name, meta in factors.items():
		if not isinstance(meta, dict):
			continue
		cols = meta.get("column_based", [])
		if cols is None:
			cols = []
		if not isinstance(cols, list) or not all(isinstance(c, str) for c in cols):
			raise ValueError(
				f"Invalid factors json: factors.{factor_name}.column_based must be list[str]."
			)
		factor_to_cols[str(factor_name)] = list(cols)

	return factor_to_cols


def _extract_causal_order(graph_json: dict, graph_path: Path) -> List[str]:
	order = graph_json.get("causal_order")
	if not isinstance(order, list) or not all(isinstance(x, str) for x in order):
		raise ValueError(
			f"Invalid causal graph json at {graph_path}: expected key 'causal_order' as list[str]."
		)
	return list(order)


def _load_csv_header(csv_path: Path) -> List[str]:
	with csv_path.open("r", encoding="utf-8", newline="") as f:
		reader = csv.reader(f)
		header = next(reader, None)
	if header is None:
		raise ValueError(f"CSV file appears empty: {csv_path}")
	# Keep raw header strings; strip only surrounding whitespace.
	return [h.strip() for h in header]


def factor_order_to_column_order(
	factor_order: Sequence[str],
	factor_to_columns: Dict[str, List[str]],
	all_columns: Sequence[str],
	rng: random.Random,
) -> List[str]:
	seen: set[str] = set()
	col_order: List[str] = []

	for factor in factor_order:
		cols = list(factor_to_columns.get(factor, []))
		# Shuffle column order within a factor
		rng.shuffle(cols)
		for col in cols:
			if col not in seen:
				seen.add(col)
				col_order.append(col)

	# Check CSV header for missing columns and insert randomly
	missing = [c for c in all_columns if c not in seen]
	for col in missing:
		pos = rng.randint(0, len(col_order))
		col_order.insert(pos, col)
		seen.add(col)

	return col_order


def _default_output_path(graph_path: Path) -> Path:
	return graph_path.parent / "sort_graph.json"


def main(argv: Sequence[str] | None = None) -> int:
	parser = argparse.ArgumentParser(
		description="Sort column names based on factor causal order and save as JSON."
	)
	parser.add_argument(
		"--graph",
		default="data/fakejob/causal_graph_DL.json",
		help="Factor ordering (causal graph) JSON. Default: data/fakejob/causal_graph_DL.json",
	)
	parser.add_argument(
		"--factors",
		default="data/fakejob/fakejob_factors.json",
		help="Factors mapping JSON. Default: data/fakejob/fakejob_factors.json",
	)
	parser.add_argument(
		"--columns_csv",
		default="data/fakejob/fake_job_postings.csv",
		help="CSV file whose header is the full column set. Default: data/fakejob/fake_job_postings.csv",
	)
	parser.add_argument(
		"--out",
		default=None,
		help="Output JSON path. Default: <graph_dir>/sort_graph.json",
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=None,
		help="Random seed for shuffling/inserting columns.",
	)
	args = parser.parse_args(list(argv) if argv is not None else None)

	graph_path = Path(args.graph)
	factors_path = Path(args.factors)
	columns_csv_path = Path(args.columns_csv)
	out_path = Path(args.out) if args.out else _default_output_path(graph_path)

	graph_json = _load_json(graph_path)
	factor_order = _extract_causal_order(graph_json, graph_path)

	factors_json = _load_json(factors_path)
	factor_to_columns = _build_factor_to_columns(factors_json)

	all_columns = _load_csv_header(columns_csv_path)
	rng = random.Random(args.seed)

	col_order = factor_order_to_column_order(
		factor_order=factor_order,
		factor_to_columns=factor_to_columns,
		all_columns=all_columns,
		rng=rng,
	)

	_dump_json(out_path, [col_order])
	print(f"Wrote column ordering (2D list) to: {out_path}")

	return 0


if __name__ == "__main__":
	raise SystemExit(main())





