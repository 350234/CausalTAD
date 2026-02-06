"""
Main pipeline: complete iterative execution script that integrates all steps.
"""
import json
import os
from datetime import datetime
from config import Config
from step1_factor_discovery import discover_factors
from step2_annotation import annotate_dataset, annotate_new_factors_only
from step1_5_factor_confirmation import confirm_factors_pairwise
from step3_causal_learning import learn_causal_structure, save_causal_results
from utils import (
    log_message, load_dataset,
    validate_focus_factors
)


def main():
    """
    Run the full COAT pipeline (iterative).
    """
    print("="*70)
    print("🚀 COAT causal discovery pipeline - auto insurance fraud detection (iterative)")
    print("="*70)
    
    # ========== Initialize config ==========
    config = Config()
    config.create_directories()
    config.print_config()
    
    # ========== Initialize state ==========
    all_factors = {}              # Accumulated factors
    deleted_factors = {}          # Deleted factors (kept in memory)
    all_focus_factors = []        # Current valid focus_factors
    df_full = load_dataset(config.DATASET_PATH)  # Full dataset
    
    iteration = 0
    
    try:
        # ========== Iteration loop ==========
        while True:
            iteration += 1
            print("\n" + "="*70)
            print(f"Iteration {iteration}")
            print("="*70)
            
            if iteration > config.MAX_ITERATIONS:
                print(f"\nReached max iterations {config.MAX_ITERATIONS}, stopping")
                break
            
            # ========== Module 1: Factor discovery ==========
            print("\n" + "-"*60)
            print("Module 1: Factor discovery")
            print("-"*60)
            
            log_message(f"Start factor discovery for iteration {iteration}", config.LOG_FILE)
            
            result = discover_factors(
                config,
                iteration_num=iteration,
                df=df_full if iteration > 1 else None,  # Round 2+ uses df for cluster sampling
                existing_factors=all_factors if iteration > 1 else None,
                deleted_factors=deleted_factors,
                focus_factors=all_focus_factors if iteration > 1 else None,
                save_outputs=False
            )
            
            new_factors = result.get('factors', {})
            recommended_focus = result.get('recommended_focus_factors', [])
            
            print(f"\n✓ Discovered {len(new_factors)} new factors")
            
            # ========== Human-in-the-loop review ==========
            # 1. Prepare ordered factor list (new factors first)
            ordered_factors = {k: new_factors[k] for k in new_factors}
            ordered_factors.update(all_factors) # Append old factors
            
            # 2. Save to current round file
            current_factors_path = os.path.join(config.FACTORS_DIR, f"factors_iter{iteration}.json")
            os.makedirs(os.path.dirname(current_factors_path), exist_ok=True)
            # Review file uses a human-friendly structure consistent with LLM output
            # Users only need to edit recommended_focus_factors; no need to maintain focus_factor
            with open(current_factors_path, 'w', encoding='utf-8') as f:
                json.dump(
                    {
                        "factors": ordered_factors,
                        "recommended_focus_factors": recommended_focus,
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
            print(f"✓ Factor definitions saved: {current_factors_path}")
            
            # 3. Pause and wait for user edits
            from utils import wait_for_user_review
            wait_for_user_review(config, current_factors_path, new_factor_names=list(new_factors.keys()))
            
            # 4. Reload reviewed factors
            log_message(f"Reloading reviewed factors: {current_factors_path}", config.LOG_FILE)
            with open(current_factors_path, 'r', encoding='utf-8') as f:
                reviewed = json.load(f)

            if not isinstance(reviewed, dict) or 'factors' not in reviewed:
                raise ValueError(
                    "Review file format error: must be a JSON object with top-level field 'factors'."
                )

            loaded_factors = reviewed.get('factors')
            loaded_recommended_focus = reviewed.get('recommended_focus_factors')

            if not isinstance(loaded_factors, dict):
                raise ValueError("Review file field 'factors' must be an object (dict).")

            # 4.1 Read and validate user-edited recommended_focus_factors (missing/empty => error)
            if loaded_recommended_focus is None or loaded_recommended_focus == [] or loaded_recommended_focus == "":
                raise ValueError(
                    "Human-in-the-loop review file is missing 'recommended_focus_factors'."
                    "Please fill focus factors (list) in top-level field recommended_focus_factors of factors_iter*.json."
                    "Format can be [{\"factor_name\": ...}, ...] or [\"name1\", \"name2\"]."
                )

            # Extract focus names for validation/printing
            focus_names = []
            if isinstance(loaded_recommended_focus, list):
                for x in loaded_recommended_focus:
                    if isinstance(x, dict):
                        n = x.get('factor_name')
                    else:
                        n = x
                    if isinstance(n, str) and n.strip():
                        focus_names.append(n.strip())
            elif isinstance(loaded_recommended_focus, str):
                # Compatible with a single string input
                if loaded_recommended_focus.strip():
                    focus_names = [loaded_recommended_focus.strip()]
                    loaded_recommended_focus = focus_names
            else:
                raise ValueError(
                    f"Unsupported recommended_focus_factors type: {type(loaded_recommended_focus)}."
                    "Please use a list or string."
                )

            if not focus_names:
                raise ValueError("recommended_focus_factors is empty or cannot be parsed to factor_name.")

            # Ensure focus_names exist in factor table (otherwise a late warning occurs: 'not found in annotated data')
            missing_focus = [n for n in focus_names if n not in loaded_factors]
            if missing_focus:
                raise ValueError(
                    "recommended_focus_factors contains non-existent factor names (check typos or renames):"
                    + ", ".join(missing_focus)
                )

            print(f"\n✓ User-confirmed focus_factor ({len(focus_names)}):")
            for n in focus_names:
                print(f"  - {n}")

            # Use the user-confirmed focus list for this and subsequent rounds (keep structure for dict/str compatibility)
            if iteration == 1:
                recommended_focus = loaded_recommended_focus
            else:
                all_focus_factors = loaded_recommended_focus
            
            # 5. Compute actual new factors (user may rename/delete/add)
            # NOTE: all_factors is assumed to be the state after last round, excluding this round's LLM proposals that were deleted.
            actual_new_factors = {k: v for k, v in loaded_factors.items() if k not in all_factors}
            
            print(f"\n✓ User-confirmed new factors: {len(actual_new_factors)}")
            for k in actual_new_factors:
                print(f"  + {k}")
            
            # 6. Update in-memory state
            all_factors = loaded_factors
            new_factors = actual_new_factors # Update new_factors for annotation module
            
            # ========== Module 2: Factor annotation ==========
            print("\n" + "-"*60)
            print("Module 2: Factor annotation")
            print("-"*60)
            
            log_message(f"Start factor annotation for iteration {iteration}", config.LOG_FILE)
            
            if iteration == 1:
                # Round 1: annotate all factors
                df_full = annotate_dataset(config, df_full, all_factors)
            else:
                # Subsequent rounds: annotate new factors only (incremental)
                df_full = annotate_new_factors_only(config, df_full, new_factors)
            
            # ========== Module 3: Factor confirmation ==========
            print("\n" + "-"*60)
            print("Module 3: Factor confirmation (pairwise independence test)")
            print("-"*60)
            
            log_message(f"Start factor confirmation for iteration {iteration}", config.LOG_FILE)
            
            # Use recommended focus_factors in round 1; otherwise use existing ones
            focus_for_confirmation = recommended_focus if iteration == 1 else all_focus_factors
            
            confirmed_factors, this_round_deleted, updated_focus_factors = confirm_factors_pairwise(
                config,
                df_full,
                all_factors,
                focus_for_confirmation
            )
            
            # Update state
            all_factors = confirmed_factors
            all_focus_factors = updated_focus_factors
            
            # ⭐ Accumulate deleted factors
            deleted_factors.update(this_round_deleted)
            
            print(f"\n✓ Confirmed {len(confirmed_factors)} factors")
            print(f"✓ Valid focus_factor count: {len(all_focus_factors)}")
            
            # Validate focus_factors
            if all_focus_factors:
                valid_focus, warnings = validate_focus_factors(
                    df_full,
                    all_focus_factors,
                    min_unique_values=3
                )
                
                if warnings:
                    print("\n⚠️  focus_factor validation warnings:")
                    for w in warnings:
                        print(f"  - {w}")
                
                all_focus_factors = valid_focus
            
            # ========== Module 4: Causal learning (each round) ==========
            print("\n" + "-"*60)
            print("Module 4: Causal learning")
            print("-"*60)
            
            log_message(f"Start causal learning for iteration {iteration}", config.LOG_FILE)
            
            # Run causal discovery with all confirmed factors
            results = learn_causal_structure(
                config,
                df_full,
                all_factors,
                focus_factors=all_focus_factors
            )
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            save_dir = save_causal_results(config, results, iteration, timestamp)
            
            # ========== Module 5: Stopping criteria ==========
            print("\n" + "-"*60)
            print("Module 5: Stopping criteria")
            print("-"*60)
            
            if iteration > 1:
                new_factors_count = len(new_factors)
                print(f"  New factors this round: {new_factors_count}")
                
                if new_factors_count < 2:
                    print(f"\n✓ New factor count ({new_factors_count}) < 2, stopping condition met")
                    log_message(f"Stopped after iteration {iteration}: insufficient new factors", config.LOG_FILE)
                    break
            
            if len(all_focus_factors) == 0:
                print("\n⚠️  All focus_factors have been deleted; cannot continue")
                log_message("All focus_factors deleted, terminating early", config.LOG_FILE)
                break
            
            print(f"\nContinuing to next iteration...")
        
        # ========== Final summary ==========
        print("\n" + "="*70)
        print("🎉 Full pipeline completed!")
        print("="*70)
        print(f"\nFinal stats:")
        print(f"  Total iterations: {iteration}")
        print(f"  Confirmed factors: {len(all_factors)}")
        print(f"  Deleted factors: {len(deleted_factors)}")
        print(f"  Valid focus_factors: {len(all_focus_factors)}")
        
        if all_focus_factors:
            print(f"\nValid focus_factors:")
            for focus in all_focus_factors:
                fname = focus.get('factor_name') if isinstance(focus, dict) else focus
                print(f"  - {fname}")
        
        print(f"\nResult file locations:")
        print(f"  Last round results: {save_dir}")
        print(f"  Log file: {config.LOG_FILE}")
        print("="*70)
        
        log_message(f"Full pipeline completed, total iterations: {iteration}", config.LOG_FILE)
        
    except KeyboardInterrupt:
        print("\n\nExecution interrupted by user")
        log_message("Execution interrupted by user", config.LOG_FILE)
    except Exception as e:
        print(f"\n\n❌ Error during execution: {str(e)}")
        import traceback
        traceback.print_exc()
        log_message(f"Execution error: {str(e)}", config.LOG_FILE)
        raise


if __name__ == "__main__":
    main()



