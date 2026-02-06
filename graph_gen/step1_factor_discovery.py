"""
Step 1: Factor discovery module.
Can run independently or be called by main.py.
"""
import os
import json
from config import Config
from utils import (
    create_llm_client, load_dataset, 
    load_data_distribution, save_factors, save_prompt,
    display_prompt, log_message, parse_factor_discovery_response,
    call_llm_with_retry, cluster_entropy_sample
)
from prompts import build_factor_discovery_prompt, build_batch_samples_text


def discover_factors(config, iteration_num=1, df=None, existing_factors=None, 
                     deleted_factors=None, focus_factors=None, save_outputs=True):
    """
    Run factor discovery.

    Parameters:
    -----------
    config : Config
        Config object
    iteration_num : int
        Iteration number
    df : DataFrame, optional
        Dataset (if provided, used for cluster sampling)
    existing_factors : dict, optional
        Existing factors (avoid duplicates)
    deleted_factors : dict, optional
        Deleted factors (avoid re-proposing)
    focus_factors : list, optional
        focus_factor list (for cluster sampling)

    Returns:
    --------
    result : dict
        Contains factors and recommended_focus_factors
    """
    print("\n" + "="*60)
    print(f"🔍 Start factor discovery (iteration {iteration_num})")
    print("="*60)
    
    # 1. Load data
    if df is None:
        log_message("Loading dataset...", config.LOG_FILE)
        df = load_dataset(config.DATASET_PATH)
    
    data_distribution = load_data_distribution(config.DATA_DISTRIBUTION_PATH)
    
    # 2. Sampling
    log_message(f"Sampling {config.SAMPLES_PER_ITERATION} samples...", config.LOG_FILE)
    
    if iteration_num == 1:
        # Round 1: random sampling
        print("  Sampling strategy: random sampling")
        samples = df.sample(min(config.SAMPLES_PER_ITERATION, len(df)), random_state=42)
    else:
        # Subsequent rounds: clustering + entropy selection
        print("  Sampling strategy: factor clustering + entropy selection")
        if existing_factors and focus_factors:
            samples = cluster_entropy_sample(
                df_annotated=df,
                factor_names=list(existing_factors.keys()),
                focus_factors=focus_factors,
                n_samples=config.SAMPLES_PER_ITERATION,
                config=config
            )
        else:
            # Fallback: random sampling
            print("  ⚠️ Missing factors or focus_factors, falling back to random sampling")
            samples = df.sample(min(config.SAMPLES_PER_ITERATION, len(df)), random_state=42)
    
    samples_text = build_batch_samples_text(samples, config.SAMPLES_PER_ITERATION)
    
    # 3. Build prompt
    log_message("Building factor discovery prompt...", config.LOG_FILE)
    prompt = build_factor_discovery_prompt(
        samples_text=samples_text,
        data_distribution=data_distribution,
        domain_context=config.DOMAIN_CONTEXT,
        iteration_num=iteration_num,
        existing_factors=existing_factors,
        deleted_factors=deleted_factors
    )
    
    # Save prompt
    prompt_path = os.path.join(
        config.PROMPTS_DIR, 
        f"factor_discovery_iter{iteration_num}.txt"
    )
    save_prompt(prompt, prompt_path)
    display_prompt(prompt, f"Factor discovery prompt (iteration {iteration_num})")
    
    # 4. Call LLM (with retries)
    log_message("Calling LLM for factor discovery...", config.LOG_FILE)
    client = create_llm_client(config.API_BETTER)
    
    result = call_llm_with_retry(
        client=client,
        prompt=prompt,
        system_instruction=config.SYSTEM_INSTRUCTION_FACTOR_DISCOVERY,
        parser_func=parse_factor_discovery_response,
        max_retries=3,
        temperature=0.7
    )
    
    # Save raw response
    response_path = os.path.join(
        config.RESPONSES_DIR,
        f"factor_discovery_iter{iteration_num}_response.txt"
    )
    # Save result as JSON string
    save_prompt(json.dumps(result, indent=2, ensure_ascii=False), response_path)
    
    # 5. Extract results
    factors = result.get('factors', {})
    recommended_focus = result.get('recommended_focus_factors', [])
    
    print(f"\n✓ Extracted {len(factors)} factors:")
    for factor_name in factors.keys():
        print(f"  - {factor_name}")
    
    print(f"\n✓ Recommended focus_factor ({len(recommended_focus)}):")
    for focus in recommended_focus:
        fname = focus.get('factor_name', focus) if isinstance(focus, dict) else focus
        print(f"  - {fname}")
    
    # Save factor definitions (optional)
    # NOTE: In the integrated main.py pipeline, factors are saved/loaded during Human-in-the-loop;
    # exp_vifd/outputs/factors/factors_iter{n}.json, so we allow disabling here to avoid duplication.
    if save_outputs:
        factors_path = os.path.join(
            config.RESULTS_DIR,
            f"factors_iter{iteration_num}.json"
        )
        save_factors(factors, focus_factor=None, filepath=factors_path)
    
    log_message(f"Factor discovery completed: {len(factors)} factors", config.LOG_FILE)
    
    return result


def main():
    """
    Standalone entry point.
    """
    print("="*60)
    print("Step 1: Factor discovery")
    print("="*60)
    
    # Load config
    config = Config()
    config.create_directories()
    config.print_config()
    
    # Run factor discovery (multiple iterations)
    all_factors = {}
    
    for iteration in range(1, config.MAX_ITERATIONS + 1):
        factors = discover_factors(config, iteration_num=iteration)
        
        # Merge factors (later rounds may discover new or refine existing factors)
        all_factors.update(factors)
        
        print(f"\n✓ Iteration {iteration} completed, total factors: {len(all_factors)}")
    
    # Save final factors
    final_path = os.path.join(config.RESULTS_DIR, "factors_final.json")
    save_factors(all_factors, focus_factor=None, filepath=final_path)
    
    print("\n" + "="*60)
    print(f"✓ Factor discovery complete! Total factors discovered: {len(all_factors)}")
    print(f"✓ Results saved to: {final_path}")
    print("="*60)


if __name__ == "__main__":
    main()



