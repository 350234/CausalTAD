'''

Code: obtain column weights based on a causal graph.


python use_graph/based_on_DL/get_weights_based_on_graph.py \
    --graph data/fakejob/causal_graph_DL.json \
    --factor_to_columns data/fakejob/fakejob_factors.json \
    --output_weights data/fakejob/fakejob_weights_based_on_graph_1-10.json \
    --random_seed 42

        

Inputs: default paths as follows
1. data/fakejob/causal_graph_DL.json, which contains the graph structure output produced by the DirectLiNGAM
function described at:
https://causal-learn.dowhy.cn/en/latest/search_methods_index/Causal%20discovery%20methods%20based%20on%20constrained%20functional%20causal%20models/lingam.html#directlingam
The code must strictly read edges according to the file format, and after reading it should print a few nodes' edge
information concisely in the terminal so I can verify the behavior.

2. data/fakejob/fakejob_factors.json, which contains factor definitions. The column_based field indicates
which columns map to each factor.

Code behavior:
1. Parse edge information associated with each factor according to the input format, including edge direction and
edge weight, i.e., the from/to and weight fields in "edges" (weight does not affect direction; direction is determined
by from/to).

2. Count which factors are mapped from each column and save it.

3. For each column, compute as follows:

For each factor mapped from the current column:
    For each edge from this factor to another factor not mapped from the current column:
        Accumulate the absolute value of the edge weight.

4. After computation, save the final value for each factor to file as weights;
Saving format reference: data/fakejob/fakejob_weights_12_30.json, the output format must be identical.



Added feature:
In the original step 3, now additionally accumulate for each factor: the absolute value of edge weights from factors
not mapped from the current column to the current factor.

After completion, please use a debug script in the terminal to show in detail the computation process for one column,
and first perform a manual verification of the code.


'''

import argparse
import json
import os
import random
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser(description="Calculate column weights based on causal graph.")
    parser.add_argument("--graph", type=str, default="../../data/fakejob/causal_graph_DL.json", help="Path to causal graph JSON")
    parser.add_argument("--factor_to_columns", type=str, default="../../data/fakejob/fakejob_factors.json", help="Path to factors JSON")
    parser.add_argument("--output_weights", type=str, default="output_weights.json", help="Path to output weights JSON")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed")
    return parser.parse_args()

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def main():
    args = parse_args()
    
    # Check if files exist
    if not os.path.exists(args.graph):
        print(f"Error: Graph file not found at {args.graph}")
        return
    if not os.path.exists(args.factor_to_columns):
        print(f"Error: Factors file not found at {args.factor_to_columns}")
        return

    random.seed(args.random_seed)

    # 1. Load Data
    print(f"Loading graph from {args.graph}")
    graph_data = load_json(args.graph)
    
    print(f"Loading factors from {args.factor_to_columns}")
    factors_data = load_json(args.factor_to_columns)
    
    # Print first 5 edges for verification
    edges = graph_data.get("edges", [])
    print("\n--- First 5 Edges Check ---")
    for i, edge in enumerate(edges[:5]):
        print(f"Edge {i}: {edge.get('from')} -> {edge.get('to')}, weight: {edge.get('weight')}")
    print("---------------------------\n")

    # 2. Build Mappings
    # col_to_factors: column_name -> set of factor_names associated with it
    col_to_factors = defaultdict(set)
    # factor_to_cols: factor_name -> set of column_names that generate it
    factor_to_cols = defaultdict(set)

    factors_dict = factors_data.get("factors", {})
    
    for factor_name, factor_info in factors_dict.items():
        columns = factor_info.get("column_based", [])
        for col in columns:
            col_to_factors[col].add(factor_name)
            factor_to_cols[factor_name].add(col)

    print(f"Identified {len(col_to_factors)} columns and {len(factors_dict)} factors.")

    # Pre-process edges for faster access
    edge_outgoing = defaultdict(list)
    edge_incoming = defaultdict(list)
    
    for edge in edges:
        source = edge.get("from")
        target = edge.get("to")
        weight = edge.get("weight", 0.0)
        edge_outgoing[source].append((target, weight))
        edge_incoming[target].append((source, weight))

    # 3. Calculate Weights
    final_column_weights = {}
    
    # Iterate over all columns found in the factors file
    all_columns = list(col_to_factors.keys())
    
    for col_name in all_columns:
        current_col_weight = 0.0
        
        # Factors derived from this column
        my_factors = col_to_factors[col_name]
        
        for factor in my_factors:
            # 1. Outgoing edges: Factor -> External
            for target, weight in edge_outgoing[factor]:
                is_internal = (target in my_factors)
                if not is_internal:
                    current_col_weight += abs(weight)
            
            # 2. Incoming edges: External -> Factor (New Requirement)
            for source, weight in edge_incoming[factor]:
                is_internal = (source in my_factors)
                if not is_internal:
                    current_col_weight += abs(weight)
        
        final_column_weights[col_name] = current_col_weight

    # 4. Save Output
    output_data = {
        "csv": "data/fakejob/fake_job_postings.csv", 
        "factors": args.factor_to_columns,
        "num_columns": len(final_column_weights),
        "weights": final_column_weights
    }
    
    output_dir = os.path.dirname(args.output_weights)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    with open(args.output_weights, 'w') as f:
        json.dump(output_data, f, indent=2)
        
    print(f"\nWeights saved to {args.output_weights}")
    
    # Briefly print results for verification
    print("\n--- Calculated Weights (First 5 by Name) ---")
    sorted_weights = sorted(final_column_weights.items(), key=lambda x: x[0])
    for k, v in sorted_weights[:5]:
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    main()



