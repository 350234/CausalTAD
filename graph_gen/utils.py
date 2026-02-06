"""
Utility functions module - basic functionality.
"""
import os
import json
import pandas as pd
import numpy as np
from openai import OpenAI
from datetime import datetime


# ==================== LLM client management ====================

def create_llm_client(api_config):
    """
    Create an OpenAI client.

    Parameters:
    -----------
    api_config : dict
        API config dict containing api_key, base_url, model_name

    Returns:
    --------
    client : OpenAI
        OpenAI client instance
    """
    client = OpenAI(
        api_key=api_config['api_key'],
        base_url=api_config['base_url']
    )
    client.model_name = api_config['model_name']  # Attach model name
    return client


def call_llm(client, prompt, system_instruction=None, temperature=0.7):
    """
    Call the LLM and return its response.

    Parameters:
    -----------
    client : OpenAI
        OpenAI client
    prompt : str
        User prompt
    system_instruction : str, optional
        System instruction
    temperature : float
        Temperature

    Returns:
    --------
    response : str
        LLM text response
    """
    messages = []
    
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    
    messages.append({"role": "user", "content": prompt})
    
    try:
        completion = client.chat.completions.create(
            model=client.model_name,
            messages=messages,
            temperature=temperature
        )
        response = completion.choices[0].message.content
        return response
    except Exception as e:
        print(f"❌ LLM call failed: {str(e)}")
        raise


# ==================== Data loading and processing ====================

def load_dataset(path):
    """
    Load dataset.

    Parameters:
    -----------
    path : str
        Dataset path

    Returns:
    --------
    df : DataFrame
        Dataset
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file does not exist: {path}")

    # NOTE:
    # Downstream logic formats samples as "col is value" for LLM reading.
    # For datasets with many text/categorical columns (e.g., fake_job_postings_*),
    # missing values hurt prompt readability, so we fill missing/blank with 'unknown'.
    df = pd.read_csv(path)

    # Treat whitespace-only strings as missing
    try:
        df = df.replace(r'^\s*$', np.nan, regex=True)
    except Exception:
        # If regex replace fails on specific pandas/data, ignore whitespace detection
        pass

    df = df.fillna('unknown')
    print(f"✓ Dataset loaded: {len(df)} rows, {len(df.columns)} columns")
    return df


def load_data_distribution(path):
    """
    Read data distribution prompt text.

    Parameters:
    -----------
    path : str
        Data distribution prompt file path

    Returns:
    --------
    content : str
        Data distribution description text
    """
    if not os.path.exists(path):
        print(f"⚠ Data distribution file does not exist: {path}")
        return ""
    
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print(f"✓ Data distribution prompt loaded ({len(content)} chars)")
    return content


def format_sample(row, columns=None):
    """
    Format a single sample as "column is value".

    Parameters:
    -----------
    row : Series
        Data row
    columns : list, optional
        Column names to display (default all)

    Returns:
    --------
    formatted : str
        Formatted text
    """
    if columns is None:
        columns = row.index
    
    lines = []
    for col in columns:
        value = row[col]
        # Handle different value types
        if pd.isna(value):
            value_str = "N/A"
        elif isinstance(value, (int, np.integer)):
            value_str = str(int(value))
        elif isinstance(value, (float, np.floating)):
            value_str = f"{value:.2f}"
        else:
            value_str = str(value)
        
        lines.append(f"{col} is {value_str}")
    
    return "\n".join(lines)


def random_sample_data(df, n_samples, random_state=None):
    """
    Randomly sample data.

    Parameters:
    -----------
    df : DataFrame
        Dataset
    n_samples : int
        Number of samples
    random_state : int, optional
        Random seed

    Returns:
    --------
    samples : DataFrame
        Sampled data
    """
    if n_samples >= len(df):
        return df
    
    return df.sample(n=n_samples, random_state=random_state)


# ==================== Prompt management ====================

def save_prompt(prompt, filepath):
    """
    Save a prompt to file.

    Parameters:
    -----------
    prompt : str
        Prompt content
    filepath : str
        Save path
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(prompt)
    
    print(f"✓ Prompt saved: {filepath}")


def load_prompt(filepath):
    """
    Load a prompt from file (if it exists).

    Parameters:
    -----------
    filepath : str
        File path

    Returns:
    --------
    prompt : str or None
        Prompt content, or None if file does not exist
    """
    if not os.path.exists(filepath):
        return None
    
    with open(filepath, 'r', encoding='utf-8') as f:
        prompt = f.read()
    
    print(f"✓ Prompt loaded: {filepath}")
    return prompt


def display_prompt(prompt, title="PROMPT"):
    """
    Display the prompt in the terminal.

    Parameters:
    -----------
    prompt : str
        Prompt content
    title : str
        Title
    """
    print("\n" + "="*60)
    print(f"{title}")
    print("="*60)
    print(prompt)
    print("="*60 + "\n")


# ==================== Result saving ====================

def save_factors(factors, filepath, focus_factor=None):
    """
    Save factor definitions to JSON.

    Parameters:
    -----------
    factors : dict
        Factor definition dict
    filepath : str
        Save path
    focus_factor : str, optional
        Selected focus_factor
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    output = {
        "factors": factors,
        "focus_factor": focus_factor,
        "timestamp": datetime.now().isoformat()
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Factor definitions saved: {filepath}")


def load_factors(filepath):
    """
    Load factor definitions from JSON.

    Parameters:
    -----------
    filepath : str
        File path

    Returns:
    --------
    factors : dict
        Factor definitions
    focus_factor : str
        focus_factor name
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Factor definition file does not exist: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"✓ Factor definitions loaded: {filepath}")
    return data['factors'], data.get('focus_factor', None)


def save_annotations(df, filepath):
    """
    Save annotations to CSV.

    Parameters:
    -----------
    df : DataFrame
        Dataset containing annotation results
    filepath : str
        Save path
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    df.to_csv(filepath, index=False)
    print(f"✓ Annotation results saved: {filepath} ({len(df)} rows)")


# ==================== Logging ====================

def log_message(message, log_file=None):
    """
    Record a log message.

    Parameters:
    -----------
    message : str
        Log message
    log_file : str, optional
        Log file path
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    
    print(log_entry)
    
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry + "\n")


# ==================== Cluster sampling ====================

def cluster_entropy_sample(df_annotated, factor_names, focus_factors, n_samples, config=None):
    """
    Cluster annotated factor values + entropy-based selection.

    Parameters:
    -----------
    df_annotated : DataFrame
        Full data with annotated factors
    factor_names : list
        Factor name list
    focus_factors : list
        focus_factor list (for entropy calculation)
    n_samples : int
        Number of samples
    config : Config, optional
        Config object (for robustness checks)

    Returns:
    --------
    samples : DataFrame
        Sampled data
    """
    from sklearn.cluster import KMeans
    from scipy.stats import entropy
    
    # Robustness checks
    if config:
        if len(factor_names) < config.CLUSTER_MIN_FACTORS:
            print(f"⚠️ Factor count ({len(factor_names)}) < {config.CLUSTER_MIN_FACTORS}, falling back to random sampling")
            return df_annotated.sample(min(n_samples, len(df_annotated)), random_state=42)
        
        if len(df_annotated) < config.CLUSTER_MIN_SAMPLES:
            print(f"⚠️ Sample count ({len(df_annotated)}) < {config.CLUSTER_MIN_SAMPLES}, falling back to random sampling")
            return df_annotated.sample(min(n_samples, len(df_annotated)), random_state=42)
    
    # Extract factor value matrix (encoded as numeric)
    factor_data = df_annotated[factor_names].copy()
    
    # Encode (if still strings)
    for fname in factor_names:
        if factor_data[fname].dtype == 'object':
            unique_vals = factor_data[fname].unique()
            encoding_map = {val: idx for idx, val in enumerate(unique_vals)}
            factor_data[fname] = factor_data[fname].map(encoding_map)
    
    data_matrix = factor_data.values.astype(float)
    
    # Number of clusters
    n_clusters = min(len(factor_names) + 1, len(df_annotated) // 20, 5)
    
    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(data_matrix)
    except Exception as e:
        print(f"⚠️ Clustering failed: {str(e)}, falling back to random sampling")
        return df_annotated.sample(min(n_samples, len(df_annotated)), random_state=42)
    
    # Compute overall entropy for each cluster
    cluster_entropies = []
    for c in range(n_clusters):
        cluster_mask = (clusters == c)
        cluster_df = df_annotated[cluster_mask]
        
        # Compute entropy for each focus_factor
        entropies_for_this_cluster = []
        for focus in focus_factors:
            focus_name = focus.get('factor_name', focus) if isinstance(focus, dict) else focus
            if focus_name in df_annotated.columns:
                value_counts = cluster_df[focus_name].value_counts().values
                if len(value_counts) > 1:
                    ent = entropy(value_counts / value_counts.sum())
                else:
                    ent = 0
                entropies_for_this_cluster.append(ent)
        
        # Aggregate entropy: average
        avg_entropy = np.mean(entropies_for_this_cluster) if entropies_for_this_cluster else 0
        cluster_entropies.append(avg_entropy)
    
    # Select the cluster with highest entropy
    best_cluster_idx = np.argmax(cluster_entropies)
    print(f"  Clustering result: {n_clusters} clusters")
    print(f"  Selected cluster #{best_cluster_idx}, avg entropy={cluster_entropies[best_cluster_idx]:.3f}")
    
    # Sample from that cluster
    best_cluster_samples = df_annotated[clusters == best_cluster_idx]
    return best_cluster_samples.sample(min(n_samples, len(best_cluster_samples)), random_state=42)


# ==================== LLM response parsing ====================

def parse_factor_discovery_response(response):
    """
    Parse factor discovery response.

    Validation:
    1. Valid JSON
    2. Contains factors and recommended_focus_factors
    3. possible_values are numeric
    """
    import json
    
    # Extract JSON
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
    
    result = json.loads(json_str)
    
    # Validate
    if 'factors' not in result:
        raise ValueError("Missing 'factors' field")
    
    if 'recommended_focus_factors' not in result:
        raise ValueError("Missing 'recommended_focus_factors' field")
    
    # Ensure possible_values are numeric
    for fname, finfo in result['factors'].items():
        values = finfo.get('possible_values', [])
        if not all(isinstance(v, (int, float)) for v in values):
            raise ValueError(f"{fname} possible_values contains non-numeric values: {values}")
    
    return result


def call_llm_with_retry(client, prompt, system_instruction, 
                        parser_func=None, max_retries=3, temperature=0.7):
    """
    Call the LLM and parse; retry on failure.
    """
    import json
    
    for attempt in range(max_retries):
        try:
            response = call_llm(client, prompt, system_instruction, temperature)
            
            if parser_func:
                parsed = parser_func(response)
                return parsed
            else:
                return response
                
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            print(f"⚠️ Parse failed {attempt+1}/{max_retries}: {str(e)}")
            
            if attempt < max_retries - 1:
                print("Retrying...")
                # Lower temperature for more deterministic output
                temperature = max(0.3, temperature - 0.2)
            else:
                print("❌ Reached max retry attempts")
                raise
    
    return None


def validate_focus_factors(df_annotated, candidates, min_unique_values=3):
    """
    Validate focus_factors.

    Parameters:
    -----------
    df_annotated : DataFrame
        Annotated data
    candidates : list
        Candidate focus_factor list
    min_unique_values : int
        Minimum unique values

    Returns:
    --------
    valid_factors : list
        Valid focus_factors
    warnings : list
        Warning messages
    """
    valid = []
    warnings = []
    
    for candidate in candidates:
        # Support dict or string format
        if isinstance(candidate, dict):
            factor_name = candidate.get('factor_name')
        else:
            factor_name = candidate
            candidate = {'factor_name': factor_name}
        
        if factor_name not in df_annotated.columns:
            warnings.append(f"{factor_name} not found in annotated data")
            continue
        
        unique_count = df_annotated[factor_name].nunique()
        if unique_count < min_unique_values:
            warnings.append(
                f"{factor_name} has only {unique_count} unique values,"
                f" cannot serve as focus_factor (need at least {min_unique_values})"
            )
        else:
            valid.append(candidate)
    
    return valid, warnings


def wait_for_user_review(config, file_path, new_factor_names=None):
    """
    Pause program, wait for user to review and edit file.

    Parameters:
    -----------
    config : Config
        Config object
    file_path : str
        File path to review
    new_factor_names : list, optional
        New factor names for this round (for logging)
    """
    import time
    signal_file = os.path.join(config.OUTPUTS_DIR, "PAUSE_FOR_REVIEW")
    
    # 1. Construct prompt message
    msg = [
        "\n" + "="*60,
        "⏸️  Program paused (Human-in-the-loop)",
        "="*60,
        f"File to review: {file_path}",
        "Instructions:",
        "1. Open the file above in your editor",
        "2. Check and correct LLM-generated factors (especially new ones)",
        "3. Save your changes",
        f"4. Delete the signal file to continue: {signal_file}",
        "="*60
    ]
    
    if new_factor_names:
        msg.append(f"New factors this round ({len(new_factor_names)}):")
        for name in new_factor_names:
            msg.append(f"  + {name}")
        msg.append("="*60)
            
    log_message("\n".join(msg), config.LOG_FILE)
    
    # 2. Create signal file
    with open(signal_file, 'w') as f:
        f.write(f"Please review: {file_path}\n")
        f.write("Delete this file to continue execution.\n")
        if new_factor_names:
            f.write(f"\nNew factors to check: {', '.join(new_factor_names)}")
    
    # 3. Wait loop
    print(f"\n[PAUSED] Waiting for review. Delete '{signal_file}' to continue...")
    while os.path.exists(signal_file):
        time.sleep(5)
        
    log_message("▶️ Signal file deleted, reloading data and continuing...", config.LOG_FILE)





