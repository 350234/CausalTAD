"""
Step 3: Causal learning module.
Learn causal structure using causallearn PC/FCI/GES algorithms.
Can run independently or be called by main.py.
"""
import os
import json
import pandas as pd
import numpy as np
from typing import List, Dict
from config import Config
from utils import log_message
from datetime import datetime

# causallearn imports
try:
    from causallearn.search.ConstraintBased.FCI import fci
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.search.ScoreBased.GES import ges
    from causallearn.utils.GraphUtils import GraphUtils
    CAUSALLEARN_AVAILABLE = True
except ImportError:
    CAUSALLEARN_AVAILABLE = False
    print("⚠ causallearn is not installed. Please run: pip install causallearn")


def prepare_factor_data(df_annotated, factors):
    """
    Prepare factor data for causal discovery algorithms.

    Parameters:
    -----------
    df_annotated : DataFrame
        Annotated dataset
    factors : dict
        Factor definitions

    Returns:
    --------
    df_factors : DataFrame
        DataFrame with factor columns only (numeric)
    factor_names : list
        Factor name list
    encoding_maps : dict
        Mapping from factor values to numeric codes
    """
    factor_names = list(factors.keys())
    df_factors = df_annotated[factor_names].copy()
    
    encoding_maps = {}
    
    # Encode each factor to numeric
    for factor_name in factor_names:
        if df_factors[factor_name].dtype == 'object':
            unique_values = df_factors[factor_name].unique()
            encoding_map = {val: idx for idx, val in enumerate(unique_values)}
            encoding_maps[factor_name] = encoding_map
            df_factors[factor_name] = df_factors[factor_name].map(encoding_map)
    
    print(f"✓ Factor data prepared: {df_factors.shape}")
    print(f"  Factor list: {factor_names}")
    
    return df_factors, factor_names, encoding_maps


def run_pc_algorithm(df_factors, alpha, independence_test):
    """
    Run the PC algorithm.

    Parameters:
    -----------
    df_factors : DataFrame
        Factor data (numeric)
    alpha : float
        Significance level
    independence_test : str
        Independence test method

    Returns:
    --------
    G : Graph
        Causal graph
    edges : list
        Edge list
    """
    if not CAUSALLEARN_AVAILABLE:
        raise ImportError("causallearn is not installed")
    
    print(f"\nRunning PC algorithm (alpha={alpha}, test={independence_test})...")
    
    # Convert to numpy array
    data = df_factors.values.astype(float)
    
    # Run PC
    cg = pc(
        data,
        alpha=alpha,
        indep_test=independence_test,
        stable=True,
        verbose=False
    )
    
    G = cg.G
    edges = G.get_graph_edges() if hasattr(G, 'get_graph_edges') else []
    
    print(f"✓ PC completed, detected {len(edges)} edges")
    
    return G, edges


def run_fci_algorithm(df_factors, alpha, independence_test):
    """
    Run the FCI algorithm.

    Parameters:
    -----------
    df_factors : DataFrame
        Factor data (numeric)
    alpha : float
        Significance level
    independence_test : str
        Independence test method

    Returns:
    --------
    G : PAG
        Partial ancestral graph
    edges : list
        Edge list
    """
    if not CAUSALLEARN_AVAILABLE:
        raise ImportError("causallearn is not installed")
    
    print(f"\nRunning FCI algorithm (alpha={alpha}, test={independence_test})...")
    
    # Convert to numpy array
    data = df_factors.values.astype(float)
    
    # Run FCI
    G, edges = fci(
        data,
        independence_test_method=independence_test,
        alpha=alpha,
        verbose=False
    )
    
    print(f"✓ FCI completed, detected {len(edges)} edges")
    
    return G, edges
    '''"""
    Legacy function flow:
    Run the GES algorithm

    Parameters:
    -----------
    df_factors : DataFrame
        Factor data (numeric)

    Returns:
    --------
    G : PAG
        Partial ancestral graph
    edges : list
        Edge list
    """
    if not CAUSALLEARN_AVAILABLE:
        raise ImportError("causallearn is not installed")
    
    print(f"\nRunning GES algorithm: {ges_algorithm}...")

    # Convert to numpy array
    data = df_factors.values.astype(float)

    # ges
    if ges_algorithm.lower() == "bic":
        graph = ges(X = data, score_func = "local_score_BIC")
    elif ges_algorithm.lower() == "bdeu":
        graph = ges(X = data, score_func = "local_score_BDeu")
    elif ges_algorithm.lower() == "bic_from_cov":
        graph = ges(X = data, score_func = "local_score_BIC_from_cov")
    elif ges_algorithm.lower() == "marginal_multi":
        graph = ges(X = data, score_func = "local_score_marginal_multi")
    elif ges_algorithm.lower() == "cv_multi":
        graph = ges(X = data, score_func = "local_score_CV_multi")
    elif ges_algorithm.lower() == "marginal_general":
        graph = ges(X = data, score_func = "local_score_marginal_general")
    elif ges_algorithm.lower() == "cv_general":
        graph = ges(X = data, score_func = "local_score_CV_general")
    else:
        raise ValueError(f"Unknown GES scoring function: {ges_algorithm}")
    G = graph['G']

    if G is None:
        raise RuntimeError("Failed to call causallearn.ges")

    edges = get_ges_graph_edges(graph)

    try:
        print(f"✓ GES completed, detected {len(edges)} edges")
    except Exception:
        print("✓ GES completed; edge parsing may require further inspection")

    return G, edges'''


def extract_markov_blanket(G, edges, factor_names, focus_factor):
    """
    Extract the Markov blanket for a target factor.

    Parameters:
    -----------
    G : PAG
        Causal graph
    edges : list
        Edge list
    factor_names : list
        Factor name list
    focus_factor : str
        Target factor name

    Returns:
    --------
    mb_factors : list
        Factors in the Markov blanket
    """
    if focus_factor not in factor_names:
        print(f"⚠ Warning: focus_factor '{focus_factor}' is not in factor list")
        return []
    
    focus_idx = factor_names.index(focus_factor)
    
    # Collect nodes directly connected to focus_factor
    mb_indices = set()
    
    for edge in edges:
        i, j, endpoint_i, endpoint_j = edge
        if i == focus_idx:
            mb_indices.add(j)
        elif j == focus_idx:
            mb_indices.add(i)
    
    mb_factors = [factor_names[idx] for idx in mb_indices]
    
    print(f"\n✓ Markov blanket extraction completed:")
    print(f"  Target factor: {focus_factor}")
    print(f"  Markov blanket factor count: {len(mb_factors)}")
    for factor in mb_factors:
        print(f"    - {factor}")
    
    return mb_factors


def visualize_graph(G, factor_names, output_path):
    """
    Visualize the causal graph (optional).

    Parameters:
    -----------
    G : PAG
        Causal graph
    factor_names : list
        Factor names
    output_path : str
        Output path
    """
    try:
        from causallearn.utils.GraphUtils import GraphUtils
        import matplotlib.pyplot as plt
        
        pyd = GraphUtils.to_pydot(G, labels=factor_names)
        pyd.write_png(output_path)
        print(f"✓ Causal graph saved: {output_path}")
        
    except ImportError:
        print("⚠ Graph visualization requires pydot and graphviz")
    except Exception as e:
        print(f"⚠ Visualization failed: {str(e)}")


def normalize_edges(edges, factor_names, algorithm):
    """
    Normalize edge formats across algorithms.

    Parameters:
    -----------
    edges : list
        Raw edge list
    factor_names : list
        Factor name list
    algorithm : str
        Algorithm name

    Returns:
    --------
    edges_normalized : list
        Normalized edge list [{from, to, type}]
    """
    edges_normalized = []
    
    if algorithm == 'pc' or algorithm == 'fci':
        # Edge format returned by PC/FCI
        for edge in edges:
            try:
                # Extract node and endpoint info
                node1 = edge.get_node1() if hasattr(edge, 'get_node1') else edge.node1
                node2 = edge.get_node2() if hasattr(edge, 'get_node2') else edge.node2
                
                # Get node names
                name1 = str(node1.get_name() if hasattr(node1, 'get_name') else node1)
                name2 = str(node2.get_name() if hasattr(node2, 'get_name') else node2)
                
                # If names are numeric indices, convert to factor names
                try:
                    idx1 = int(name1) if name1.isdigit() else factor_names.index(name1)
                    idx2 = int(name2) if name2.isdigit() else factor_names.index(name2)
                    name1 = factor_names[idx1]
                    name2 = factor_names[idx2]
                except (ValueError, IndexError):
                    pass
                
                edges_normalized.append({
                    "from": name1,
                    "to": name2,
                    "type": "undetermined"  # PC/FCI may have undetermined edges
                })
            except Exception as e:
                print(f"⚠️ Edge parsing failed: {str(e)}")
                continue
    
    elif algorithm == 'ges':
        # Edge format returned by GES (already simple)
        for edge in edges:
            edges_normalized.append({
                "from": factor_names[edge.node1],
                "to": factor_names[edge.node2],
                "type": "directed" if edge.edge == 1 else "undirected"
            })
    
    return edges_normalized


def extract_markov_blanket_for_focus(G, factor_names, focus_factor):
    """
    Extract Markov blanket for focus_factor from the full causal graph.

    Parameters:
    -----------
    G : Graph
        Causal graph object
    factor_names : list
        Factor name list
    focus_factor : str
        Target factor name

    Returns:
    --------
    mb_factors : list
        Factor names in the Markov blanket
    """
    if focus_factor not in factor_names:
        print(f"⚠️ focus_factor '{focus_factor}' is not in the factor list")
        return []
    
    focus_idx = factor_names.index(focus_factor)
    
    # Extract neighbor nodes
    try:
        # Try using graph adjacency matrix
        if hasattr(G, 'graph'):
            adj_matrix = G.graph
            n = len(factor_names)
            neighbors = set()
            
            for i in range(n):
                if i == focus_idx:
                    continue
                # Check if there is an edge (either direction)
                if adj_matrix[focus_idx, i] != 0 or adj_matrix[i, focus_idx] != 0:
                    neighbors.add(i)
            
            mb_factors = [factor_names[i] for i in neighbors]
            
        else:
            # Fallback: return empty list
            print("⚠️ Unable to extract Markov Blanket (graph structure not supported)")
            mb_factors = []
    
    except Exception as e:
        print(f"⚠️ Markov Blanket extraction failed: {str(e)}")
        mb_factors = []
    
    return mb_factors


def learn_causal_structure(config, df_annotated, factors, focus_factors=None):
    """
    Learn causal structure (use all factors; no filtering by focus_factor).

    Parameters:
    -----------
    config : Config
        Config object
    df_annotated : DataFrame
        Annotated full dataset
    factors : dict
        Confirmed factor definitions
    focus_factors : list, optional
        focus_factor list (only for MB extraction, does not affect causal learning)

    Returns:
    --------
    results : dict
        Contains causal graph, edges, markov_blankets, etc.
    """
    print("\n" + "="*60)
    print("🔗 Start causal learning")
    print("="*60)
    
    # 1. Prepare factor data (all confirmed factors)
    df_factors, factor_names, encoding_maps = prepare_factor_data(
        df_annotated, factors
    )
    
    # 2. Run causal discovery algorithm
    algorithm = config.CAUSAL_ALGORITHM
    alpha = config.CAUSAL_ALPHA
    independence_test = config.INDEPENDENCE_TEST
    
    if algorithm == 'pc':
        G, edges = run_pc_algorithm(df_factors, alpha, independence_test)
    elif algorithm == 'fci':
        G, edges = run_fci_algorithm(df_factors, alpha, independence_test)
    elif algorithm == 'ges':
        G, edges = run_ges_algorithm(df_factors, alpha)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    # 3. Normalize edge format
    edges_normalized = normalize_edges(edges, factor_names, algorithm)
    
    print(f"\n✓ Causal learning completed:")
    print(f"  Algorithm: {algorithm.upper()}")
    print(f"  Factor count: {len(factor_names)}")
    print(f"  Edge count: {len(edges_normalized)}")
    
    # 4. Extract Markov Blanket (optional)
    markov_blankets = {}
    if config.EXTRACT_MARKOV_BLANKET and focus_factors:
        print(f"\nExtracting Markov Blanket...")
        for focus in focus_factors:
            focus_name = focus.get('factor_name') if isinstance(focus, dict) else focus
            mb = extract_markov_blanket_for_focus(G, factor_names, focus_name)
            if mb:
                markov_blankets[focus_name] = mb
                print(f"  {focus_name}: {len(mb)} factors")
    
    # 5. Save graph (optional)
    try:
        pdy = GraphUtils.to_pydot(G, labels=factor_names)
        graph_str = pdy.to_string()
    except Exception as e:
        print(f"⚠️ Graph generation failed: {str(e)}")
        graph_str = None
    
    # 6. Return results
    results = {
        "algorithm": algorithm,
        "factor_names": factor_names,
        "edges": edges_normalized,
        "markov_blankets": markov_blankets,
        "graph_string": graph_str,
        "encoding_maps": encoding_maps
    }
    
    log_message(
        f"Causal learning completed: {algorithm}, {len(factor_names)} factors, {len(edges_normalized)} edges",
        config.LOG_FILE
    )
    
    return results


def run_ges_algorithm(df_factors, alpha):
    """Run the GES algorithm (simplified)."""
    print(f"\nRunning GES algorithm (alpha={alpha})...")
    data = df_factors.values.astype(float)
    
    result_dict = ges(X=data, score_func="local_score_BIC")
    G = result_dict['G']
    
    # Parse edges
    edges = []
    for i in range(G.num_vars):
        for j in range(i + 1, G.num_vars):
            a = G.graph[j, i]
            b = G.graph[i, j]
            if a == 1 and b == -1:
                edges.append(type('Edge', (), {'node1': i, 'node2': j, 'edge': 1})())
            elif a == -1 and b == 1:
                edges.append(type('Edge', (), {'node1': j, 'node2': i, 'edge': 1})())
            elif a == -1 and b == -1:
                edges.append(type('Edge', (), {'node1': i, 'node2': j, 'edge': 0})())
    
    print(f"✓ GES completed, detected {len(edges)} edges")
    return G, edges


def save_causal_results(config, results, iteration, timestamp):
    """
    Save causal learning results.

    Parameters:
    -----------
    config : Config
        Config object
    results : dict
        Causal learning results
    iteration : int
        Iteration number
    timestamp : str
        Timestamp
    """
    # Create output directory
    save_dir = os.path.join(
        config.RESULTS_DIR,
        f"iter{iteration}_{timestamp}"
    )
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Save causal structure JSON
    structure_path = os.path.join(save_dir, "causal_structure.json")
    with open(structure_path, 'w', encoding='utf-8') as f:
        json.dump({
            'algorithm': results['algorithm'],
            'iteration': iteration,
            'timestamp': timestamp,
            'factor_names': results['factor_names'],
            'edges': results['edges'],
            'edge_count': len(results['edges'])
        }, f, indent=2, ensure_ascii=False)
    
    # 2. Save Markov Blankets (if any)
    if results.get('markov_blankets'):
        for focus_name, mb in results['markov_blankets'].items():
            mb_path = os.path.join(save_dir, f"mb_{focus_name}.json")
            with open(mb_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'focus_factor': focus_name,
                    'markov_blanket': mb,
                    'size': len(mb)
                }, f, indent=2, ensure_ascii=False)
    
    # 3. Save causal graph image (if any)
    if results.get('graph_string'):
        graph_path = os.path.join(save_dir, "causal_graph.png")
        try:
            import pydot
            graphs = pydot.graph_from_dot_data(results['graph_string'])
            if graphs:
                graphs[0].write_png(graph_path)
                print(f"✓ Causal graph saved: {graph_path}")
        except Exception as e:
            print(f"⚠️ Failed to save causal graph: {str(e)}")
    
    # 4. Save summary
    summary_path = os.path.join(save_dir, "summary.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("Causal learning result summary\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Algorithm: {results['algorithm'].upper()}\n")
        f.write(f"Iteration: {iteration}\n")
        f.write(f"Time: {timestamp}\n\n")
        f.write(f"Factor count: {len(results['factor_names'])}\n")
        f.write(f"Edge count: {len(results['edges'])}\n\n")
        
        if results.get('markov_blankets'):
            f.write("Markov Blankets:\n")
            for focus_name, mb in results['markov_blankets'].items():
                f.write(f"  {focus_name}: {len(mb)} factors\n")
                f.write(f"    {', '.join(mb)}\n")
    
    print(f"\n✓ Results saved to: {save_dir}")
    log_message(f"Causal learning results saved to: {save_dir}", config.LOG_FILE)
    
    return save_dir


def main():
    """
    Standalone entry point.
    """
    print("="*60)
    print("Step 3: Causal learning")
    print("="*60)
    
    # Load config
    config = Config()
    
    # Check if annotated data exists
    annotated_path = os.path.join(config.RESULTS_DIR, "annotated_data.csv")
    if not os.path.exists(annotated_path):
        print(f"❌ Error: annotated data not found: {annotated_path}")
        print("Please run step2_annotation.py first")
        return
    
    # Ask for focus_factor
    print("\nPlease specify target factor (focus_factor), or press Enter to skip Markov Blanket extraction:")
    focus_factor = input("focus_factor: ").strip()
    
    if not focus_factor:
        focus_factor = None
        print("Skipping Markov Blanket extraction")

    # Ask for causal learning algorithm
    print("\nPlease specify causal learning algorithm (default fci):")
    algorithm = input("algorithm: ").strip()
    
    if not algorithm:
        algorithm = "fci"
        print("Using default algorithm: fci")
    
    if algorithm == "ges":
        print("\nPlease choose GES scoring function (default BIC): bic, bdeu, bic_from_cov, marginal_multi, cv_multi, marginal_general, cv_general")
        ges_algorithm = input("ges_algorithm: ").strip()
        if not ges_algorithm:
            ges_algorithm = "bic"
        print(f"Using GES algorithm: {ges_algorithm}")
    
    # Run causal learning
    results = learn_causal_structure(config, focus_factor=focus_factor, algorithm=algorithm, ges_algorithm = ges_algorithm if algorithm=="ges" else "bic")
    
    print("\n" + "="*60)
    print("✓ Causal learning completed!")
    print(f"  Edge count detected: {len(results['edges'])}")
    if results['markov_blanket']:
        print(f"  Markov blanket factor count: {len(results['markov_blanket'])}")
    print("="*60)


if __name__ == "__main__":
    main()




