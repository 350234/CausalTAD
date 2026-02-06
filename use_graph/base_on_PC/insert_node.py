
#!/usr/bin/env python3

'''

python3 ./insert_node.py --topo  data/fakejob/fakejob_topo_orders.json --excluded data/fakejob/fakejob_topo_orders_excluded.json

'''

import argparse
import json
import random
from pathlib import Path
from typing import List, Optional, Sequence


def _default_topo_path() -> Path:
	return Path(__file__).resolve().parent / "data" / "vifd" / "vifd_topo_orders.json"


def _default_excluded_path() -> Path:
	return (
		Path(__file__).resolve().parent
		/ "data"
		/ "vifd"
		/ "vifd_topo_orders_excluded.json"
	)


def _default_output_path(topo_path: Path, strategy: str) -> Path:
	# e.g. vifd_topo_orders -> vifd_full_orders_random
	stem = topo_path.stem
	if "topo_orders" in stem:
		stem = stem.replace("topo_orders", f"full_orders_{strategy}")
	else:
		stem = f"{stem}_full_orders_{strategy}"
	return topo_path.with_name(stem + ".json")


def _validate_topo_orders(topo_orders: object) -> List[List[str]]:
	if not isinstance(topo_orders, list):
		raise ValueError("Invalid topo orders json: expected a 2D list")
	parsed: List[List[str]] = []
	for i, row in enumerate(topo_orders):
		if not isinstance(row, list):
			raise ValueError(f"Invalid topo orders json: row {i} is not a list")
		# Be tolerant: cast scalars to strings to guarantee output nodes are strings.
		parsed.append([str(x) for x in row])
	return parsed


def _validate_excluded_nodes(excluded_nodes: object) -> List[str]:
	if not isinstance(excluded_nodes, list):
		raise ValueError("Invalid excluded nodes json: expected a 1D list")
	return [str(x) for x in excluded_nodes]


def _remaining_nodes(excluded_nodes: Sequence[str], partial_order: Sequence[str]) -> List[str]:
	# Remaining nodes are explicitly provided by excluded_nodes.
	# Still guard against overlap with partial_order to avoid duplicates.
	seen = set(partial_order)
	return [n for n in excluded_nodes if n not in seen]



def insert_front(excluded_nodes: Sequence[str], partial_order: Sequence[str]) -> List[str]:
	rem = _remaining_nodes(excluded_nodes, partial_order)
	return list(rem) + list(partial_order)



def insert_back(excluded_nodes: Sequence[str], partial_order: Sequence[str]) -> List[str]:
	rem = _remaining_nodes(excluded_nodes, partial_order)
	return list(partial_order) + list(rem)


def insert_random(
	excluded_nodes: Sequence[str],
	partial_order: Sequence[str],
	*,
	rng: random.Random,
) -> List[str]:
	rem = _remaining_nodes(excluded_nodes, partial_order)
	order = list(partial_order)

	# Randomize insertion order, then insert each node at a random position
	rng.shuffle(rem)
	for n in rem:
		pos = rng.randrange(0, len(order) + 1)
		order.insert(pos, n)
	return order


def main() -> int:
	parser = argparse.ArgumentParser(
		description=(
			"Given partial topological orders (2D list) and excluded nodes (1D list), "
			"insert the excluded nodes into each order using different strategies."
		)
	)
	parser.add_argument(
		"--topo",
		type=str,
		default=str(_default_topo_path()),
		help="Path to partial topo orders json (default: data/vifd/vifd_topo_orders.json)",
	)
	parser.add_argument(
		"--excluded",
		type=str,
		default=str(_default_excluded_path()),
		help=(
			"Path to excluded nodes json (default: data/vifd/vifd_topo_orders_excluded.json)"
		),
	)
	parser.add_argument(
		"--strategy",
		type=str,
		default="all",
		choices=["random", "front", "back", "all"],
		help="Insertion strategy: random/front/back/all (default: all)",
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=None,
		help="Optional random seed (only affects random strategy)",
	)
	parser.add_argument(
		"--output",
		type=str,
		default=None,
		help=(
			"Output json path. If strategy=all, this is treated as a directory; "
			"otherwise it's a file path. Default: alongside topo json."
		),
	)
	args = parser.parse_args()

	topo_path = Path(args.topo)
	excluded_path = Path(args.excluded)
	if not topo_path.is_file():
		raise FileNotFoundError(f"Topo orders json not found: {topo_path}")
	if not excluded_path.is_file():
		raise FileNotFoundError(f"Excluded nodes json not found: {excluded_path}")

	with topo_path.open("r", encoding="utf-8") as f:
		topo_orders_raw = json.load(f)
	topo_orders = _validate_topo_orders(topo_orders_raw)

	with excluded_path.open("r", encoding="utf-8") as f:
		excluded_raw = json.load(f)
	excluded_nodes = _validate_excluded_nodes(excluded_raw)

	# De-duplicate excluded nodes while preserving order.
	seen_excluded = set()
	excluded_nodes_unique: List[str] = []
	for n in excluded_nodes:
		if n not in seen_excluded:
			seen_excluded.add(n)
			excluded_nodes_unique.append(n)
	excluded_nodes = excluded_nodes_unique

	for i, order in enumerate(topo_orders):
		if len(order) != len(set(order)):
			raise ValueError(f"Topo order row {i} has duplicated nodes")
		# Note: we do not require a full graph here; excluded nodes are provided explicitly.

	strategies = [args.strategy] if args.strategy != "all" else ["random", "front", "back"]
	rng = random.Random(args.seed)

	for strategy in strategies:
		if args.output:
			if args.strategy == "all":
				out_dir = Path(args.output)
				out_dir.mkdir(parents=True, exist_ok=True)
				out_path = out_dir / _default_output_path(topo_path, strategy).name
			else:
				out_path = Path(args.output)
				out_path.parent.mkdir(parents=True, exist_ok=True)
		else:
			out_path = _default_output_path(topo_path, strategy)
			out_path.parent.mkdir(parents=True, exist_ok=True)

		full_orders: List[List[str]] = []
		for order in topo_orders:
			if strategy == "front":
				full = insert_front(excluded_nodes, order)
			elif strategy == "back":
				full = insert_back(excluded_nodes, order)
			elif strategy == "random":
				full = insert_random(excluded_nodes, order, rng=rng)
			else:
				raise ValueError(f"Unknown strategy: {strategy}")

			# Result should contain all nodes from partial order + (non-overlapping) excluded nodes.
			expected_len = len(order) + len(_remaining_nodes(excluded_nodes, order))
			if len(full) != expected_len:
				raise ValueError(
					f"Strategy {strategy} produced invalid length: {len(full)} vs expected {expected_len}"
				)
			if len(full) != len(set(full)):
				raise ValueError(f"Strategy {strategy} produced duplicated nodes")
			full_orders.append(full)

		with out_path.open("w", encoding="utf-8") as f:
			json.dump(full_orders, f, ensure_ascii=False, indent=2)

		print(
			f"strategy={strategy}: topo_in={len(topo_orders)}; "
			f"excluded={len(excluded_nodes)}; saved={len(full_orders)} -> {out_path}"
		)

	return 0


if __name__ == "__main__":
	raise SystemExit(main())





