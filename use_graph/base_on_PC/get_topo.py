#!/usr/bin/env python3

'''

python3 ./get_topo.py --input data/fakejob/fakejob_graph.json --weights data/fakejob/fakejob_weights.json --output data/fakejob/fakejob_topo_orders.json --max-orders 200

'''


import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Optional, Sequence, Set, Tuple


def _is_close(a: float, b: float, eps: float = 1e-9) -> bool:
	return abs(a - b) <= eps


def infer_directed_edges(edges: Sequence[dict]) -> Set[Tuple[str, str]]:
	"""Infer directed edges from paired relations.

	Rule:
	  If value(A->B) == -1.0 and value(B->A) == 1.0, infer directed edge A -> B.
	  (Symmetric case inferred accordingly.)
	"""

	value_map: Dict[Tuple[str, str], float] = {}
	for e in edges:
		src = e.get("from")
		dst = e.get("to")
		val = e.get("value")
		if not isinstance(src, str) or not isinstance(dst, str):
			continue
		if not isinstance(val, (int, float)):
			continue
		value_map[(src, dst)] = float(val)

	directed: Set[Tuple[str, str]] = set()
	seen_pairs: Set[Tuple[str, str]] = set()
	for (a, b), ab in value_map.items():
		# Only decide once per unordered pair.
		key = (a, b) if a <= b else (b, a)
		if key in seen_pairs:
			continue
		seen_pairs.add(key)

		ba = value_map.get((b, a))
		if ba is None:
			continue
		if _is_close(ab, -1.0) and _is_close(ba, 1.0):
			directed.add((a, b))
		elif _is_close(ab, 1.0) and _is_close(ba, -1.0):
			directed.add((b, a))

	return directed


def topo_sorts_all(
	nodes: Sequence[str],
	directed_edges: Iterable[Tuple[str, str]],
	*,
	max_orders: int = 0,
) -> List[List[str]]:
	"""Enumerate topological orders (deterministic).

	This uses backtracking with a sorted available-node list, producing stable
	orders across runs.

	If max_orders > 0, stop after generating that many orders.
	"""

	node_set = set(nodes)
	adj: DefaultDict[str, Set[str]] = defaultdict(set)
	indeg: Dict[str, int] = {n: 0 for n in nodes}

	for u, v in directed_edges:
		if u not in node_set or v not in node_set:
			continue
		if v in adj[u]:
			continue
		adj[u].add(v)
		indeg[v] += 1

	available: List[str] = sorted([n for n in nodes if indeg[n] == 0])
	order: List[str] = []
	results: List[List[str]] = []

	def backtrack(avail: List[str]) -> None:
		if max_orders > 0 and len(results) >= max_orders:
			return
		if len(order) == len(nodes):
			results.append(order.copy())
			return

		if not avail:
			return

		for idx, n in enumerate(list(avail)):
			order.append(n)

			changed: List[Tuple[str, int]] = []
			newly_zero: List[str] = []
			for m in adj.get(n, ()):  # neighbors
				indeg[m] -= 1
				changed.append((m, 1))
				if indeg[m] == 0:
					newly_zero.append(m)

			next_avail = avail[:idx] + avail[idx + 1 :]
			if newly_zero:
				next_avail = sorted(next_avail + newly_zero)

			backtrack(next_avail)

			for m, delta in changed:
				indeg[m] += delta
			order.pop()

			if max_orders > 0 and len(results) >= max_orders:
				return

	backtrack(available)
	return results


def topo_sort_one(
	nodes: Sequence[str],
	directed_edges: Iterable[Tuple[str, str]],
) -> List[str]:
	"""Return one topological order if possible (deterministic Kahn).

	If no order exists (cycle), return [].
	"""
	node_set = set(nodes)
	adj: DefaultDict[str, Set[str]] = defaultdict(set)
	indeg: Dict[str, int] = {n: 0 for n in nodes}

	for u, v in directed_edges:
		if u not in node_set or v not in node_set:
			continue
		if v in adj[u]:
			continue
		adj[u].add(v)
		indeg[v] += 1

	avail: List[str] = sorted([n for n in nodes if indeg[n] == 0])
	order: List[str] = []
	while avail:
		n = avail.pop(0)
		order.append(n)
		for m in sorted(adj.get(n, ())):
			indeg[m] -= 1
			if indeg[m] == 0:
				# keep deterministic ordering
				avail.append(m)
				avail.sort()

	return order if len(order) == len(nodes) else []


def find_cycle(nodes: Sequence[str], directed_edges: Iterable[Tuple[str, str]]) -> List[str]:
	"""Return a list of nodes forming a cycle if any, else empty list.

	Uses DFS with recursion stack to find one directed cycle and reconstructs
	the cycle path.
	"""
	node_set = set(nodes)
	adj: DefaultDict[str, List[str]] = defaultdict(list)
	for u, v in directed_edges:
		if u in node_set and v in node_set:
			if v not in adj[u]:
				adj[u].append(v)

	visited: Set[str] = set()
	onstack: Set[str] = set()
	parent: Dict[str, Optional[str]] = {}

	def dfs(u: str) -> Optional[List[str]]:
		visited.add(u)
		onstack.add(u)
		for v in adj.get(u, ()):  # neighbors
			if v not in visited:
				parent[v] = u
				res = dfs(v)
				if res:
					return res
			elif v in onstack:
				# reconstruct cycle from v..u..v
				path = [v]
				cur = u
				while cur is not None and cur != v:
					path.append(cur)
					cur = parent.get(cur)
				path.append(v)
				path.reverse()
				return path
		onstack.remove(u)
		return None

	for n in nodes:
		if n not in visited:
			parent[n] = None
			res = dfs(n)
			if res:
				return res
	return []


def _default_weights_path() -> Path:
	return Path(__file__).resolve().parent / "data" / "vifd" / "vifd_weights.json"


def _as_2d_list(values: Sequence[str]) -> List[List[str]]:
	# Always emit 2D list format as required.
	return [list(values)]


def _load_zero_weight_nodes(weights_path: Path) -> List[str]:
	"""Load nodes with weight==0 from weights json.

	Expected format:
	{
	  "weights": {
	    "feature name": 0,
	    ...
	  }
	}
	"""
	if not weights_path.is_file():
		raise FileNotFoundError(f"Weights json not found: {weights_path}")
	with weights_path.open("r", encoding="utf-8") as f:
		obj = json.load(f)
	weights = obj.get("weights")
	if not isinstance(weights, dict):
		raise ValueError("Invalid weights json: missing dict field 'weights'")
	zero_nodes: List[str] = []
	for name, w in weights.items():
		if not isinstance(name, str):
			continue
		if not isinstance(w, (int, float)):
			continue
		if _is_close(float(w), 0.0):
			zero_nodes.append(name)
	return sorted(set(zero_nodes))

def sample_topo_orders_randomized(
	nodes: Sequence[str],
	directed_edges: Iterable[Tuple[str, str]],
	max_orders: int = 200,
	attempts_multiplier: int = 10,
	seed: Optional[int] = None,
) -> List[List[str]]:
	"""Sample up to `max_orders` diverse topological orders using randomized Kahn.

	The function runs multiple randomized Kahn runs (random tie-breaking) and collects
	unique orders until `max_orders` unique orders are found or attempts exhausted.
	"""
	if max_orders <= 0:
		return []

	rnd = random.Random(seed)
	node_set = set(nodes)
	adj: DefaultDict[str, Set[str]] = defaultdict(set)
	base_indeg: Dict[str, int] = {n: 0 for n in nodes}

	for u, v in directed_edges:
		if u not in node_set or v not in node_set:
			continue
		if v in adj[u]:
			continue
		adj[u].add(v)
		base_indeg[v] += 1

	unique_orders: Set[Tuple[str, ...]] = set()
	attempts = 0
	max_attempts = max(1000, max_orders * attempts_multiplier)

	while len(unique_orders) < max_orders and attempts < max_attempts:
		attempts += 1
		indeg = dict(base_indeg)
		avail = [n for n in nodes if indeg[n] == 0]
		order: List[str] = []

		while avail:
			n = rnd.choice(avail)
			order.append(n)
			avail.remove(n)
			for m in adj.get(n, ()):  # neighbors
				indeg[m] -= 1
				if indeg[m] == 0:
					avail.append(m)

		if len(order) == len(nodes):
			unique_orders.add(tuple(order))

	# return as list of lists
	return [list(t) for t in sorted(unique_orders)]


def sample_topo_orders(
	nodes: Sequence[str],
	directed_edges: Iterable[Tuple[str, str]],
	*,
	num: int = 200,
	attempts_limit: Optional[int] = None,
	seed: Optional[int] = None,
) -> List[List[str]]:
	"""Sample up to `num` diverse topological orders using randomized Kahn's algorithm.

	This repeatedly performs Kahn's algorithm but when multiple zero-indegree nodes
	are available it picks one uniformly at random. The procedure repeats until
	`num` unique orders are found or `attempts_limit` attempts are exhausted.
	"""

	if num <= 0:
		return []

	node_set = set(nodes)
	adj: DefaultDict[str, Set[str]] = defaultdict(set)
	indeg_base: Dict[str, int] = {n: 0 for n in nodes}

	for u, v in directed_edges:
		if u not in node_set or v not in node_set:
			continue
		if v in adj[u]:
			continue
		adj[u].add(v)
		indeg_base[v] += 1

	rng = random.Random(seed)
	attempts_limit = attempts_limit or max(1000, num * 100)

	results: List[List[str]] = []
	seen: Set[Tuple[str, ...]] = set()

	for _ in range(attempts_limit):
		indeg = indeg_base.copy()
		avail = [n for n in nodes if indeg[n] == 0]
		order: List[str] = []

		while avail:
			choice = rng.choice(avail)
			avail.remove(choice)
			order.append(choice)
			for nb in adj.get(choice, ()):  # type: ignore[arg-type]
				indeg[nb] -= 1
				if indeg[nb] == 0:
					avail.append(nb)

		if len(order) != len(nodes):
			# invalid (cycle), skip
			continue

		key = tuple(order)
		if key not in seen:
			seen.add(key)
			results.append(list(key))
			if len(results) >= num:
				break

	return results


def _footrule_distance(order_a: Sequence[str], order_b: Sequence[str]) -> float:
	"""Spearman footrule distance between two permutations.

	Lower => more similar. Higher => less similar.
	"""
	if len(order_a) != len(order_b):
		return 0.0
	pos_a = {n: i for i, n in enumerate(order_a)}
	pos_b = {n: i for i, n in enumerate(order_b)}
	# If node sets differ, distance is not meaningful.
	if pos_a.keys() != pos_b.keys():
		return 0.0
	return float(sum(abs(pos_a[n] - pos_b[n]) for n in pos_a.keys()))


def select_diverse_orders(
	orders: List[List[str]],
	*,
	max_orders: int,
	seed: Optional[int] = 0,
) -> List[List[str]]:
	"""Select up to max_orders orders with low similarity.

	Greedy farthest-point selection using Spearman footrule distance.
	"""
	if max_orders <= 0:
		return []
	if len(orders) <= max_orders:
		return orders

	rnd = random.Random(seed)
	# If there are many candidate orders, downsample deterministically to
	# a manageable working set to avoid O(N^2) blowup in distance computations.
	max_candidates = min(1000, max_orders * 5)
	working: List[List[str]]
	if len(orders) > max_candidates:
		idxs = rnd.sample(range(len(orders)), max_candidates)
		working = [orders[i] for i in idxs]
	else:
		working = list(orders)

	# Precompute position maps for each order for fast footrule distance.
	pos_maps: List[Dict[str, int]] = []
	common_nodes: Optional[Set[str]] = None
	for o in working:
		pos = {n: i for i, n in enumerate(o)}
		pos_maps.append(pos)
		if common_nodes is None:
			common_nodes = set(pos.keys())
		elif common_nodes != set(pos.keys()):
			# If node sets differ across orders, fallback to simpler behavior.
			return working[:max_orders]

	selected: List[List[str]] = []
	selected_pos: List[Dict[str, int]] = []

	# Start from a random seed order (deterministic with seed).
	start_idx = rnd.randrange(len(working))
	selected.append(working.pop(start_idx))
	selected_pos.append(pos_maps.pop(start_idx))

	# Efficient distance using precomputed pos maps.
	def footrule_between(pos_a: Dict[str, int], pos_b: Dict[str, int]) -> float:
		return float(sum(abs(pos_a[n] - pos_b[n]) for n in pos_a.keys()))

	while working and len(selected) < max_orders:
		best_idx = 0
		best_score = -1.0
		for i, pos in enumerate(pos_maps):
			min_d = min(footrule_between(pos, sp) for sp in selected_pos)
			if min_d > best_score:
				best_score = min_d
				best_idx = i
		selected.append(working.pop(best_idx))
		selected_pos.append(pos_maps.pop(best_idx))

	return selected


def _default_input_path() -> Path:
	return Path(__file__).resolve().parent / "data" / "vifd" / "vifd_graph.json"


def _default_output_path(input_path: Path) -> Path:
	stem = input_path.stem
	if "graph" in stem:
		stem = stem.replace("graph", "topo_orders")
	else:
		stem = f"{stem}_topo_orders"
	return input_path.with_name(stem + ".json")


def main() -> int:
	parser = argparse.ArgumentParser(
		description=(
			"Get topological orders among nodes connected by inferred directed edges "
			"from a DAG json graph."
		)
	)
	parser.add_argument(
		"--input",
		type=str,
		default=str(_default_input_path()),
		help="Path to graph json (default: data/vifd/vifd_graph.json)",
	)
	parser.add_argument(
		"--weights",
		type=str,
		default=str(_default_weights_path()),
		help="Path to weights json (default: data/vifd/vifd_weights.json)",
	)
	parser.add_argument(
		"--output",
		type=str,
		default=None,
		help="Path to write topo orders json (default: alongside input)",
	)
	parser.add_argument(
		"--max-orders",
		type=int,
		default=200,
		help=(
			"Max number of topo orders to save (default: 200). "
			"Use 0 to try enumerating all (may be very large)."
		),
	)
	args = parser.parse_args()

	input_path = Path(args.input)
	if not input_path.is_file():
		raise FileNotFoundError(f"Input json not found: {input_path}")
	weights_path = Path(args.weights)

	output_path = Path(args.output) if args.output else _default_output_path(input_path)
	output_path.parent.mkdir(parents=True, exist_ok=True)

	with input_path.open("r", encoding="utf-8") as f:
		graph = json.load(f)

	edges = graph.get("edges")
	if not isinstance(edges, list):
		raise ValueError("Invalid graph json: missing list field 'edges'")

	directed_edges = infer_directed_edges(edges)
	involved_nodes: Set[str] = set()
	for u, v in directed_edges:
		involved_nodes.add(u)
		involved_nodes.add(v)

	nodes_for_sort = sorted(involved_nodes)
	if not nodes_for_sort:
		topo_orders = []
	else:
		# NOTE: Do NOT enumerate all topo orders (can be enormous).
		# Instead: sample many candidates via randomized Kahn, then select a diverse subset.
		num = int(args.max_orders)
		# First, quickly check whether the inferred directed graph is a DAG.
		one_order = topo_sort_one(nodes_for_sort, directed_edges)
		if not one_order:
				# Find a concrete cycle and report it in the error message.
				cycle = find_cycle(nodes_for_sort, directed_edges)
				if cycle:
					raise ValueError(
						"No topological order found for inferred directed edges. "
						f"Graph contains a directed cycle: {' -> '.join(cycle)}"
					)
				else:
					raise ValueError(
						"No topological order found for inferred directed edges. "
						"Graph may contain a cycle under inferred directions (could not reconstruct the cycle)."
					)

		candidates = sample_topo_orders_randomized(
			nodes_for_sort,
			directed_edges,
			max_orders=max(200, num * 5),
			attempts_multiplier=50,
			seed=0,
		)
		# Ensure we always return at least one valid order for a DAG.
		if not candidates:
			topo_orders = [one_order]
		else:
			selected = select_diverse_orders(candidates, max_orders=num, seed=0)
			if selected:
				topo_orders = selected
			else:
				topo_orders = [one_order]

	# If the graph is a DAG, topo_orders must be non-empty now.
	assert (not nodes_for_sort) or topo_orders

	with output_path.open("w", encoding="utf-8") as f:
		json.dump(topo_orders, f, ensure_ascii=False, indent=2)

	# Also save nodes that did not participate in sorting (present in graph['nodes'] but not in involved_nodes)
	all_nodes_raw = graph.get("nodes") or []
	if not isinstance(all_nodes_raw, list):
		raise ValueError("Invalid graph json: field 'nodes' must be a list")
	all_nodes = [n for n in all_nodes_raw if isinstance(n, str)]
	uninvolved = sorted([n for n in all_nodes if n not in involved_nodes])

	# Save nodes that did not participate in sorting OR are not in the graph (weights==0)
	zero_weight_nodes = _load_zero_weight_nodes(weights_path)
	excluded_nodes = sorted(set(uninvolved).union(zero_weight_nodes))
	excluded_path = output_path.with_name(output_path.stem + "_excluded.json")
	with excluded_path.open("w", encoding="utf-8") as f:
		json.dump(excluded_nodes, f, ensure_ascii=False, indent=2)

	print(
		f"Inferred directed edges: {len(directed_edges)}; "
		f"nodes involved: {len(nodes_for_sort)}; "
		f"topo orders saved: {len(topo_orders)} -> {output_path}; "
		f"excluded nodes saved: {len(excluded_nodes)} -> {excluded_path}"
	)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())




