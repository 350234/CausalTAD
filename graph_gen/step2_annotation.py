"""
Step 2: Data annotation module.
Can run independently or be called by main.py.
"""
import os
import json
import pandas as pd
from tqdm import tqdm
from config import Config
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from utils import (
    create_llm_client, call_llm, load_dataset, 
    load_factors, save_annotations, format_sample,
    log_message
)
from prompts import build_annotation_prompt, build_annotation_prompt_for_new_factors


_thread_local = threading.local()


def _get_thread_client(config):
    """Get a per-thread LLM client (safer than sharing one client across threads)."""
    if not hasattr(_thread_local, "client"):
        _thread_local.client = create_llm_client(config.API_BASE)
    return _thread_local.client


def _coerce_int_or_none(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        # avoid NaN
        if v != v:
            return None
        return int(v)
    if isinstance(v, str):
        s = v.strip()
        if s == "":
            return None
        if s in {"unknown", "UNKNOWN", "Unknown", "None", "null", "NULL", "na", "NA"}:
            return None
        try:
            return int(s)
        except Exception:
            return None
    return None


def _missing_value_for_factor(factor_info):
    """Prefer -1 for missing/unknown if present in possible_values, else 0."""
    try:
        pv = factor_info.get("possible_values")
    except Exception:
        pv = None
    if isinstance(pv, list) and (-1 in pv or "-1" in pv):
        return -1
    return 0


def _normalize_annotations(raw_annotations, factors_dict):
    """Ensure every factor key exists and every value is an int (or missing code).

    This prevents NaN when building DataFrames and supports the requirement that
    missing/unknown must still be encoded as a numeric value.
    """
    if not isinstance(raw_annotations, dict):
        raw_annotations = {}
    out = {}
    for fname, finfo in factors_dict.items():
        missing = _missing_value_for_factor(finfo if isinstance(finfo, dict) else {})
        v = raw_annotations.get(fname, None)
        iv = _coerce_int_or_none(v)
        out[fname] = missing if iv is None else iv
    return out


def _iter_index_batches(index, batch_size):
    for start in range(0, len(index), batch_size):
        yield index[start:start + batch_size]


def annotate_single_sample(client, sample_row, factors, domain_context, system_instruction, log_file=None):
    """
    Annotate a single sample (all factors).

    Parameters:
    -----------
    client : OpenAI
        LLM client
    sample_row : Series
        Sample row
    factors : dict
        Factor definitions
    domain_context : str
        Domain context
    system_instruction : str
        System instruction

    Returns:
    --------
    annotations : dict
        Annotation results {factor_name: value}
    """
    # Format sample
    sample_text = format_sample(sample_row)
    
    # Serialize factor definitions to JSON
    factors_json = json.dumps(factors, indent=2, ensure_ascii=False)
    
    # Build prompt
    prompt = build_annotation_prompt(
        sample_text=sample_text,
        factors_json=factors_json,
        domain_context=domain_context
    )
    
    # Call LLM (up to 5 retries; log only on final failure)
    last_err = None
    for attempt in range(1, 6):
        try:
            response = call_llm(
                client=client,
                prompt=prompt,
                system_instruction=system_instruction + "\n\nIMPORTANT: Output ONLY a JSON object. Do NOT use markdown or code fences.",
                temperature=0.3  # Lower temperature for annotation
            )

            # Parse JSON (compat with historical output: extract from code block if present)
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            elif "```" in response:
                json_start = response.find("```") + 3
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            else:
                json_str = response.strip()

            raw = json.loads(json_str)
            normalized = _normalize_annotations(raw, factors)

            # If all missing, treat as failure and retry (improve completeness)
            if normalized and all(v == _missing_value_for_factor(factors.get(k, {})) for k, v in normalized.items()):
                raise ValueError("All factor values are missing/unknown")

            return normalized

        except Exception as e:
            last_err = e
            continue

    # Final failure: log only here
    if log_file:
        try:
            log_message(f"⚠ Annotation failed after final retry (5/5): {str(last_err)}", log_file)
        except Exception:
            pass
    return _normalize_annotations({}, factors)


def annotate_single_sample_new_factors(client, sample_row, new_factors, domain_context, system_instruction, log_file=None):
    """
    Annotate new factors for a single sample (incremental annotation).

    Parameters:
    -----------
    client : OpenAI
        LLM client
    sample_row : Series
        Sample row
    new_factors : dict
        New factor definitions (new factors only)
    domain_context : str
        Domain context
    system_instruction : str
        System instruction

    Returns:
    --------
    annotations : dict
        Annotation results {new_factor_name: value}
    """
    # Format sample
    sample_text = format_sample(sample_row)
    
    # Serialize new factor definitions to JSON
    new_factors_json = json.dumps(new_factors, indent=2, ensure_ascii=False)
    
    # Build prompt (use incremental annotation prompt)
    prompt = build_annotation_prompt_for_new_factors(
        sample_text=sample_text,
        new_factors_json=new_factors_json,
        domain_context=domain_context
    )
    
    # Call LLM (up to 5 retries; log only on final failure)
    last_err = None
    for attempt in range(1, 6):
        try:
            response = call_llm(
                client=client,
                prompt=prompt,
                system_instruction=(
                    system_instruction
                    + "\n\nIMPORTANT: Output ONLY a JSON object. Do NOT use markdown or code fences."
                    + "\n⚠️ Important: Only annotate the specified new factors; do not annotate other factors."
                ),
                temperature=0.3
            )

            # Parse JSON (compat with historical output: extract from code block if present)
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            elif "```" in response:
                json_start = response.find("```") + 3
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            else:
                json_str = response.strip()

            raw = json.loads(json_str)
            if not isinstance(raw, dict):
                raise ValueError("Annotation output is not a JSON object")

            # Keep only new factor annotations, then fill missing to avoid NaN
            filtered_raw = {k: raw.get(k) for k in new_factors.keys()}
            normalized = _normalize_annotations(filtered_raw, new_factors)

            if normalized and all(v == _missing_value_for_factor(new_factors.get(k, {})) for k, v in normalized.items()):
                raise ValueError("All new-factor values are missing/unknown")

            return normalized

        except Exception as e:
            last_err = e
            continue

    if log_file:
        try:
            log_message(f"⚠ New-factor annotation failed after final retry (5/5): {str(last_err)}", log_file)
        except Exception:
            pass
    return _normalize_annotations({}, new_factors)


def annotate_dataset(config, df=None, factors=None, factors_path=None):
    """
    Annotate the entire dataset.

    Parameters:
    -----------
    config : Config
        Config object
    df : DataFrame, optional
        Dataset in memory. If None, load from config.DATASET_PATH.
    factors : dict, optional
        Factor definitions in memory. If None, load from factors_path.
    factors_path : str, optional
        Factor definition file path, used only when factors is None.

    Returns:
    --------
    df_annotated : DataFrame
        Annotated dataset
    """
    print("\n" + "="*60)
    print("🏷️  Full dataset annotation")
    print("="*60)
    
    # 1. Prepare data
    if df is None:
        log_message("Loading dataset from disk...", config.LOG_FILE)
        df = load_dataset(config.DATASET_PATH)
    else:
        log_message("Using dataset provided in memory", config.LOG_FILE)
    
    # 2. Prepare factors
    if factors is None:
        if factors_path is None:
            factors_path = os.path.join(config.RESULTS_DIR, "factors_final.json")
        
        log_message(f"Loading factor definitions from disk: {factors_path}", config.LOG_FILE)
        factors, _ = load_factors(factors_path)
    else:
        log_message(f"Using factor definitions from memory ({len(factors)} factors)", config.LOG_FILE)
    
    print("✓ Factors to annotate:")
    for factor_name in factors.keys():
        print(f"  - {factor_name}")
    
    # 3. Parallel batch annotation (one parameter controls: concurrency + batch size)
    batch_size = int(getattr(config, "ANNOTATION_BATCH_SIZE", 30))
    if batch_size <= 0:
        batch_size = 30

    print(f"\nStart annotating {len(df)} samples... (batch_size={batch_size})")
    all_results = {}

    # tqdm progress bar: update per sample
    with tqdm(total=len(df), desc="Annotation progress") as pbar:
        for batch_idx in _iter_index_batches(list(df.index), batch_size):
            batch_df = df.loc[batch_idx]

            def _worker(idx, row):
                client = _get_thread_client(config)
                return annotate_single_sample(
                    client=client,
                    sample_row=row,
                    factors=factors,
                    domain_context=config.DOMAIN_CONTEXT,
                    system_instruction=config.SYSTEM_INSTRUCTION_ANNOTATION,
                    log_file=config.LOG_FILE,
                )

            future_to_idx = {}
            with ThreadPoolExecutor(max_workers=batch_size) as ex:
                for idx, row in batch_df.iterrows():
                    fut = ex.submit(_worker, idx, row)
                    future_to_idx[fut] = idx

                for fut in as_completed(future_to_idx):
                    idx = future_to_idx[fut]
                    all_results[idx] = fut.result()
                    pbar.update(1)
    
    # 5. Merge annotations into original data
    # Note: if df already contains previous annotations, handle carefully to avoid column conflicts
    # This implementation assumes full annotation is in the first round or uses overwrite update
    # Ensure alignment to df.index; ensure full columns to avoid NaN
    df_annotations = pd.DataFrame.from_dict(all_results, orient='index')
    df_annotations = df_annotations.reindex(df.index)

    for fname in factors.keys():
        if fname not in df_annotations.columns:
            missing = _missing_value_for_factor(factors.get(fname, {}))
            df_annotations[fname] = missing
    df_annotations = df_annotations[list(factors.keys())]
    
    # Remove existing columns with same names to avoid duplicate columns error
    cols_to_use = [c for c in df.columns if c not in df_annotations.columns]
    df_clean = df[cols_to_use]
    
    df_annotated = pd.concat([df_clean, df_annotations], axis=1)
    
    # 6. Save results (single final save)
    output_path = os.path.join(config.RESULTS_DIR, "annotated_data.csv")
    save_annotations(df_annotated, output_path)
    
    log_message(f"Data annotation completed: {len(df_annotated)} rows", config.LOG_FILE)
    
    print("\n" + "="*60)
    print("✓ Annotation completed!")
    print(f"✓ Results saved to: {output_path}")
    print("="*60)
    
    return df_annotated


def annotate_new_factors_only(config, df, new_factors):
    """
    Incrementally annotate only new factors (save time).

    Parameters:
    -----------
    config : Config
        Config object
    df : DataFrame
        Partially annotated dataset
    new_factors : dict
        New factor definitions

    Returns:
    --------
    df_annotated : DataFrame
        Dataset with new factor annotations added
    """
    print("\n" + "="*60)
    print(f"🏷️  Incremental annotation (only {len(new_factors)} new factors)")
    print("="*60)
    
    print("✓ New factors:")
    for factor_name in new_factors.keys():
        print(f"  - {factor_name}")
    
    # Parallel incremental annotation (one parameter controls: concurrency + batch size)
    batch_size = int(getattr(config, "ANNOTATION_BATCH_SIZE", 30))
    if batch_size <= 0:
        batch_size = 30

    print(f"\nStart annotating new factors for {len(df)} samples... (batch_size={batch_size})")
    all_results = {}

    with tqdm(total=len(df), desc="Incremental annotation progress") as pbar:
        for batch_idx in _iter_index_batches(list(df.index), batch_size):
            batch_df = df.loc[batch_idx]

            def _worker(idx, row):
                client = _get_thread_client(config)
                return annotate_single_sample_new_factors(
                    client=client,
                    sample_row=row,
                    new_factors=new_factors,
                    domain_context=config.DOMAIN_CONTEXT,
                    system_instruction=config.SYSTEM_INSTRUCTION_ANNOTATION,
                    log_file=config.LOG_FILE,
                )

            future_to_idx = {}
            with ThreadPoolExecutor(max_workers=batch_size) as ex:
                for idx, row in batch_df.iterrows():
                    fut = ex.submit(_worker, idx, row)
                    future_to_idx[fut] = idx

                for fut in as_completed(future_to_idx):
                    idx = future_to_idx[fut]
                    all_results[idx] = fut.result()
                    pbar.update(1)

    df_new_annotations = pd.DataFrame.from_dict(all_results, orient='index')
    df_new_annotations = df_new_annotations.reindex(df.index)

    for fname in new_factors.keys():
        if fname not in df_new_annotations.columns:
            missing = _missing_value_for_factor(new_factors.get(fname, {}))
            df_new_annotations[fname] = missing
    df_new_annotations = df_new_annotations[list(new_factors.keys())]

    df_annotated = pd.concat([df, df_new_annotations], axis=1)
    
    # Save results (single final save)
    output_path = os.path.join(config.RESULTS_DIR, "annotated_data.csv")
    save_annotations(df_annotated, output_path)
    
    log_message(f"New-factor annotation completed: {len(df_annotated)} rows", config.LOG_FILE)
    
    print("\n" + "="*60)
    print("✓ Incremental annotation completed!")
    print(f"✓ Results saved to: {output_path}")
    print("="*60)
    
    return df_annotated


def main():
    """
    Standalone entry point.
    """
    print("="*60)
    print("Step 2: Data annotation")
    print("="*60)
    
    # Load config
    config = Config()
    config.create_directories()
    
    # Check if factor definitions exist
    factors_path = os.path.join(config.RESULTS_DIR, "factors_final.json")
    if not os.path.exists(factors_path):
        print(f"❌ Error: factor definition file not found: {factors_path}")
        print("Please run step1_factor_discovery.py first")
        return
    
    # Run annotation
    df_annotated = annotate_dataset(config, factors_path=factors_path)
    
    print(f"\nAnnotated dataset shape: {df_annotated.shape}")
    print(f"Factor columns: {[col for col in df_annotated.columns if col not in load_dataset(config.DATASET_PATH).columns]}")


if __name__ == "__main__":
    main()



