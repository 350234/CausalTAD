"""Configuration file - VIFD (Vehicle Insurance Fraud Detection) experiment."""

import os
from datetime import datetime


def _read_required_text_file(path: str, desc: str) -> str:
    """Read a required text file.

    Convention: all prompts (data distribution / task background / system prompts / module user prompt templates)
    are stored under the directory pointed to by Config.PROMPTS_LIBRARY_DIR.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing required prompt file: {desc}\n"
            f"Expected path: {path}\n"
            f"Please add this file under the directory pointed to by Config.PROMPTS_LIBRARY_DIR."
        )
    with open(path, 'r', encoding='utf-8') as f:
        # Keep file content as-is (do not strip) to preserve leading/trailing newlines
        return f.read()

class Config:
    """Experiment configuration."""
    
    # ==================== API configuration ====================
    # IMPORTANT: Before running, please set your API credentials in these dictionaries.
    # Required keys: 'api_key', 'base_url', 'model_name'
    # Example:
    # API_BETTER = {
    #     'api_key': '<your-api-key-here>',
    #     'base_url': 'https://api.openai.com/v1',
    #     'model_name': 'gpt-4'
    # }
    
    # API-Better: powerful but expensive, used for harder tasks like factor discovery
    # NOTE: For anonymity and security, API credentials are NOT included in this repository.
    # Users must configure their own API keys before running the experiments.
    # Example configuration (replace with your actual values):
    # API_BETTER = {
    #     'api_key': 'your-api-key-here',
    #     'base_url': 'https://api.openai.com/v1',  # or your custom endpoint
    #     'model_name': 'gpt-4'  # or your preferred model
    # }
    API_BETTER = {
        'api_key': 'PLACEHOLDER_API_KEY',  # ⚠️ REPLACE WITH YOUR ACTUAL API KEY
        'base_url': 'PLACEHOLDER_BASE_URL',  # ⚠️ REPLACE WITH YOUR ACTUAL BASE URL
        'model_name': 'PLACEHOLDER_MODEL_NAME'  # ⚠️ REPLACE WITH YOUR ACTUAL MODEL NAME
    }
    
    # API-Base: cheaper, used for batch tasks like factor annotation
    # NOTE: For anonymity and security, API credentials are NOT included in this repository.
    # Users must configure their own API keys before running the experiments.
    API_BASE = {
        'api_key': 'PLACEHOLDER_API_KEY',  # ⚠️ REPLACE WITH YOUR ACTUAL API KEY
        'base_url': 'PLACEHOLDER_BASE_URL',  # ⚠️ REPLACE WITH YOUR ACTUAL BASE URL
        'model_name': 'PLACEHOLDER_MODEL_NAME'  # ⚠️ REPLACE WITH YOUR ACTUAL MODEL NAME
    }
    
    # ==================== Path configuration ====================
    # Base path
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # ==================== Prompt directory configuration (core) ====================
    # You can change this parameter to any directory (absolute or relative to BASE_DIR), as long as it contains
    # all required prompt files for this experiment:
    # - data_distribution_prompt.md
    # - task_background_prompt.md
    # - system_prompt_factor_discovery.txt
    # - system_prompt_annotation.txt
    # - factor_discovery_prompt.md / factor_full_annotation_prompt.md / factor_incremental_annotation_prompt.md / factor_selection_prompt.md
    PROMPTS_LIBRARY_DIR = os.path.join(BASE_DIR, "myprompts", "vifd")
    
    # NOTE: change the prompt package path above when switching datasets

    # Dataset path NOTE: update here when switching datasets
    DATASET_PATH = os.path.join(BASE_DIR, "datasets", "carclaims_test.csv")

    # Data distribution prompt: read from PROMPTS_LIBRARY_DIR
    DATA_DISTRIBUTION_PATH = os.path.join(PROMPTS_LIBRARY_DIR, "data_distribution_prompt.md")
    
    # Output paths
    # Note: PROMPTS_DIR here refers to runtime-generated prompt text (for reproducibility),
    # which is different from PROMPTS_LIBRARY_DIR (external template library).
    # All outputs are written to a timestamped directory to avoid overwriting history.
    # Format: exp_vifd/outputs-YYYYMMDD_HHMMSS (seconds precision)
    RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUTS_DIR = os.path.join(BASE_DIR, f"outputs-{RUN_TIMESTAMP}")

    # Runtime-generated prompt text (for reproducibility)
    PROMPTS_DIR = os.path.join(OUTPUTS_DIR, "prompts")
    FACTORS_DIR = os.path.join(OUTPUTS_DIR, "factors")
    ANNOTATIONS_DIR = os.path.join(OUTPUTS_DIR, "annotations")
    GRAPHS_DIR = os.path.join(OUTPUTS_DIR, "causal_graphs")
    LOGS_DIR = os.path.join(OUTPUTS_DIR, "logs")
    LOG_FILE = os.path.join(LOGS_DIR, "run.log")
    RESULTS_DIR = OUTPUTS_DIR
    RESPONSES_DIR = os.path.join(OUTPUTS_DIR, "responses")
    
    # ==================== Experiment parameters ====================
    # Iteration settings
    MAX_ITERATIONS = 5
    
    # Sampling settings
    SAMPLES_PER_ITERATION = 8  # Samples shown to LLM per iteration (fixed, not cumulative)

    # Annotation concurrency/batching (one parameter controls: thread count + batch size)
    ANNOTATION_BATCH_SIZE = 1000
    
    # Causal discovery algorithm parameters
    CAUSAL_ALGORITHM = 'pc'  # Options: 'pc', 'fci', 'ges'
    INDEPENDENCE_TEST = 'chisq'  # Use chisq for discrete data, fisherz for continuous
    CAUSAL_ALPHA = 0.05  # Significance level
    CAUSAL_DEPTH = 3  # Search depth
    FCI_ALPHA = CAUSAL_ALPHA  # Provide fci alpha value
    
    # Markov Blanket extraction
    EXTRACT_MARKOV_BLANKET = False  # Off by default
    
    # Clustering robustness parameters
    CLUSTER_MIN_SAMPLES = 50  # Minimum samples for clustering
    CLUSTER_MIN_FACTORS = 2  # Minimum factors for clustering
    
    # ==================== Domain background (external file) ====================
    DOMAIN_CONTEXT_PATH = os.path.join(PROMPTS_LIBRARY_DIR, "task_background_prompt.md")
    DOMAIN_CONTEXT = _read_required_text_file(DOMAIN_CONTEXT_PATH, "task_background_prompt.md")

    # ==================== System prompts (external files) ====================
    SYSTEM_INSTRUCTION_FACTOR_DISCOVERY_PATH = os.path.join(PROMPTS_LIBRARY_DIR, "system_prompt_factor_discovery.txt")
    SYSTEM_INSTRUCTION_FACTOR_DISCOVERY = _read_required_text_file(
        SYSTEM_INSTRUCTION_FACTOR_DISCOVERY_PATH,
        "system_prompt_factor_discovery.txt",
    )

    SYSTEM_INSTRUCTION_ANNOTATION_PATH = os.path.join(PROMPTS_LIBRARY_DIR, "system_prompt_annotation.txt")
    SYSTEM_INSTRUCTION_ANNOTATION = _read_required_text_file(
        SYSTEM_INSTRUCTION_ANNOTATION_PATH,
        "system_prompt_annotation.txt",
    )
    
    @classmethod
    def create_directories(cls):
        """Create all required directories."""
        dirs = [
            cls.PROMPTS_DIR,
            cls.OUTPUTS_DIR,
            cls.FACTORS_DIR,
            cls.ANNOTATIONS_DIR,
            cls.GRAPHS_DIR,
            cls.LOGS_DIR,
            cls.RESPONSES_DIR,   # Added
            cls.RESULTS_DIR      # In case of differences
        ]
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
        print("✓ All directories created/verified")
    
    @classmethod
    def print_config(cls):
        """Print current configuration."""
        print("\n" + "="*60)
        print("EXPERIMENT CONFIGURATION")
        print("="*60)
        print(f"Dataset: {cls.DATASET_PATH}")
        print(f"Max Iterations: {cls.MAX_ITERATIONS}")
        print(f"Samples per Iteration: {cls.SAMPLES_PER_ITERATION}")
        print(f"FCI Alpha: {cls.FCI_ALPHA}")
        print(f"API Better Model: {cls.API_BETTER['model_name']}")
        print(f"API Base Model: {cls.API_BASE['model_name']}")
        print("="*60 + "\n")



