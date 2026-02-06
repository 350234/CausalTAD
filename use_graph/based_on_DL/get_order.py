#!/usr/bin/env python3

'''

python3 ./get_sort.py --orders data/fakejob/fakejob_full_orders_random.json --factors data/fakejob/fakejob_factors.json

'''

from __future__ import annotations

import argparse
import json
import os
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


def _validate_orders(orders: object, orders_path: Path) -> List[List[str]]:
	if not isinstance(orders, list) or not all(isinstance(x, list) for x in orders):
		raise ValueError(
			f"Invalid orders json at {orders_path}: expected a 2D list (list[list[str]])."
		)
	for i, row in enumerate(orders):
		if not all(isinstance(node, str) for node in row):
			bad = [type(node).__name__ for node in row if not isinstance(node, str)][:5]
			raise ValueError(
				f"Invalid orders json at {orders_path}: orders[{i}] contains non-string nodes: {bad}"
			)
	return orders  # type: ignore[return-value]


def factor_orders_to_column_orders(
	factor_orders: Sequence[Sequence[str]],
	factor_to_columns: Dict[str, List[str]],
	rng: random.Random,
) -> List[List[str]]:
	out: List[List[str]] = []

	for ordering in factor_orders:
		seen: set[str] = set()
		col_order: List[str] = []

		for node in ordering:
			if node in factor_to_columns:
				cols = list(factor_to_columns.get(node, []))
				rng.shuffle(cols)
				for col in cols:
					if col not in seen:
						seen.add(col)
						col_order.append(col)
			else:
				# Node not in factors mapping: treat as a column name directly.
				if node not in seen:
					seen.add(node)
					col_order.append(node)

		out.append(col_order)

	return out


def _default_output_path(orders_path: Path) -> Path:
	# Default name for the full expanded 2D column orders.
	return orders_path.parent / "sort_graph.json"


def _default_random_output_path(orders_path: Path) -> Path:
	# Default name for a single randomly selected ordering from the 2D list.
	return orders_path.parent / "sort_graph_random.json"


def main(argv: Sequence[str] | None = None) -> int:
	parser = argparse.ArgumentParser(
		description="Expand factor orderings into column-name orderings and save as JSON."
	)
	parser.add_argument(
		"--orders",
		nargs="+",
		default=["data/vifd/vifd_full_orders_random.json"],
		help=(
			"Factor ordering JSON file(s). Default: data/vifd/vifd_full_orders_random.json"
		),
	)
	parser.add_argument(
		"--factors",
		default="data/vifd/vifd_factors.json",
		help="Factors mapping JSON. Default: data/vifd/vifd_factors.json",
	)
	parser.add_argument(
		"--out",
		default=None,
		help=(
			"Output JSON path. Only valid when a single --orders file is provided; "
			"otherwise outputs are written next to each orders file with suffix *_colorders.json"
		),
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=None,
		help="Random seed for shuffling columns within each factor node.",
	)
	args = parser.parse_args(list(argv) if argv is not None else None)

	orders_paths = [Path(p) for p in args.orders]
	factors_path = Path(args.factors)
	out_path = Path(args.out) if args.out else None

	if out_path is not None and len(orders_paths) != 1:
		raise SystemExit("--out can only be used when exactly one --orders file is provided.")

	factors_json = _load_json(factors_path)
	factor_to_columns = _build_factor_to_columns(factors_json)
	rng = random.Random(args.seed)

	for orders_path in orders_paths:
		orders_json = _load_json(orders_path)
		factor_orders = _validate_orders(orders_json, orders_path)
		col_orders = factor_orders_to_column_orders(factor_orders, factor_to_columns, rng)

		# Full 2D list output
		target_full = out_path if out_path is not None else _default_output_path(orders_path)
		_dump_json(target_full, col_orders)
		print(f"Wrote column orders: {target_full}")

		# Randomly pick one ordering from the 2D list and save it separately.
		if col_orders:
			single = rng.choice(col_orders)
		else:
			single = []

		if out_path is not None:
			# If user provided an explicit out path, place the random file next to it.
			target_random = out_path.with_name(out_path.stem + "_random.json")
		else:
			target_random = _default_random_output_path(orders_path)

		_dump_json(target_random, single)
		print(f"Wrote random single ordering: {target_random}")

	return 0


if __name__ == "__main__":
	raise SystemExit(main())




