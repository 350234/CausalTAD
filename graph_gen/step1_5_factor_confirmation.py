"""
Step 1.5: Factor confirmation.
Use pairwise independence tests to remove fully isolated factors.
"""
import numpy as np
import pandas as pd
from causallearn.utils.cit import CIT
from utils import log_message


def confirm_factors_pairwise(config, df_annotated, factors, focus_factors):
    """
    Factor confirmation: pairwise independence testing.

    Logic:
    1. For each factor, test independence with every other factor
    2. If a factor is independent of all others → delete (isolated factor)
    3. If a focus_factor is deleted → remove it from focus_factor list

    Parameters:
    -----------
    config : Config
        Config object
    df_annotated : DataFrame
        Annotated data (all factor columns)
    factors : dict
        Factor definitions
    focus_factors : list
        Recommended focus_factor list

    Returns:
    --------
    confirmed_factors : dict
        Confirmed factors
    deleted_factors : dict
        Deleted factors
    updated_focus_factors : list
        Updated focus_factor list
    """
    print("\n" + "="*60)
    print("Factor confirmation: pairwise independence test")
    print("="*60)
    
    factor_names = list(factors.keys())
    
    if len(factor_names) < 2:
        print("⚠️ Factor count < 2, skipping independence test")
        return factors, {}, focus_factors
    
    # Prepare data matrix (numeric encoding)
    factor_data = df_annotated[factor_names].copy()
    
    # Encode (if still strings)
    encoding_maps = {}
    for fname in factor_names:
        if factor_data[fname].dtype == 'object':
            unique_vals = factor_data[fname].unique()
            encoding_maps[fname] = {val: idx for idx, val in enumerate(unique_vals)}
            factor_data[fname] = factor_data[fname].map(encoding_maps[fname])
    
    data_matrix = factor_data.values.astype(float)
    
    # Conditional independence test
    try:
        ci_test = CIT(data_matrix, config.INDEPENDENCE_TEST)
        alpha = config.CAUSAL_ALPHA
    except Exception as e:
        print(f"⚠️ Failed to create independence test: {str(e)}; keep all factors")
        return factors, {}, focus_factors
    
    # Test each factor
    isolated_factors = set()
    factor_relations = {}
    
    for i, factor_i in enumerate(factor_names):
        is_isolated = True
        related_factors = []
        
        for j, factor_j in enumerate(factor_names):
            if i == j:
                continue
            
            try:
                # Test factor_i ⊥ factor_j
                p_value = ci_test(i, j, [])
                
                if p_value < alpha:  # Not independent
                    is_isolated = False
                    related_factors.append((factor_j, p_value))
            except Exception as e:
                # If test fails, assume not independent (conservative)
                is_isolated = False
                continue
        
        if is_isolated:
            print(f"❌ {factor_i}: independent from all other factors (isolated factor)")
            isolated_factors.add(factor_i)
        else:
            print(f"✓ {factor_i}: related to {len(related_factors)} factors")
            factor_relations[factor_i] = related_factors
            # Print top related factors
            if related_factors:
                related_factors.sort(key=lambda x: x[1])
                top_3 = related_factors[:min(3, len(related_factors))]
                print(f"    Most related: {', '.join([f'{name}(p={pv:.4f})' for name, pv in top_3])}")
    
    # Delete isolated factors
    confirmed_factors = {k: v for k, v in factors.items() if k not in isolated_factors}
    deleted_factors = {k: v for k, v in factors.items() if k in isolated_factors}
    
    # Update focus_factors list
    updated_focus_factors = []
    removed_focus = []
    
    for focus in focus_factors:
        # Support dict or string format
        if isinstance(focus, dict):
            fname = focus.get('factor_name')
        else:
            fname = focus
            focus = {'factor_name': fname}
        
        if fname in isolated_factors:
            removed_focus.append(fname)
        else:
            updated_focus_factors.append(focus)
    
    # Report
    print("\n" + "-"*60)
    print("Confirmation results:")
    print(f"  Original factor count: {len(factors)}")
    print(f"  Isolated factor count: {len(isolated_factors)}")
    print(f"  Confirmed factor count: {len(confirmed_factors)}")
    
    if removed_focus:
        print("\n⚠️ The following focus_factors were removed (isolated):")
        for fname in removed_focus:
            print(f"    - {fname}")
    
    print(f"\n✓ Remaining valid focus_factors: {len(updated_focus_factors)}")
    for focus in updated_focus_factors:
        fname = focus.get('factor_name') if isinstance(focus, dict) else focus
        print(f"    - {fname}")
    print("-"*60)
    
    log_message(
        f"Factor confirmation completed: kept {len(confirmed_factors)}/{len(factors)}, "
        f"{len(updated_focus_factors)} valid focus_factors",
        config.LOG_FILE
    )
    
    return confirmed_factors, deleted_factors, updated_focus_factors


def main():
    """
    Standalone test entry point.
    """
    print("This module is intended to be called as a submodule; it does not run standalone")


if __name__ == "__main__":
    main()


