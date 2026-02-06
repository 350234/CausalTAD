"""
Prompt construction module.
"""

from __future__ import annotations

import os
from string import Template
from typing import Dict

from config import Config


_TEMPLATE_CACHE: Dict[str, str] = {}


def _template_path(filename: str) -> str:
    return os.path.join(Config.PROMPTS_LIBRARY_DIR, filename)


def _load_template(filename: str) -> str:
    """Load a template from Config.PROMPTS_LIBRARY_DIR with a simple cache."""
    path = _template_path(filename)
    if path in _TEMPLATE_CACHE:
        return _TEMPLATE_CACHE[path]
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing prompt template file: {filename}\n"
            f"Expected path: {path}\n"
            f"Please place the template under the directory pointed to by Config.PROMPTS_LIBRARY_DIR."
        )
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    _TEMPLATE_CACHE[path] = content
    return content


def _render_template(filename: str, **kwargs) -> str:
    """Render template using $name placeholders."""
    raw = _load_template(filename)
    return Template(raw).substitute(**kwargs)


def build_factor_discovery_prompt(samples_text, data_distribution, domain_context,
                                   iteration_num, existing_factors=None, deleted_factors=None):
    """
    Build the factor discovery prompt.

    Parameters:
    -----------
    samples_text : str
        Formatted sample text
    data_distribution : str
        Data distribution description
    domain_context : str
        Domain background knowledge
    iteration_num : int
        Current iteration number
    existing_factors : dict, optional
        Existing factors (avoid duplicates)
    deleted_factors : dict, optional
        Deleted factors (avoid re-proposing)

    Returns:
    --------
    prompt : str
        Full factor discovery prompt
    """
    
    # Existing factors hint
    existing_factors_hint = ""
    if existing_factors and iteration_num > 1:
        factor_list = "\n".join([f"  - {name}" for name in existing_factors.keys()])
        existing_factors_hint = f"""
    ## Already discovered factors (avoid duplicates)
    The following factors were found in previous rounds. Please do NOT extract them again:
    {factor_list}

    """
    
    # Deleted factors hint
    deleted_factors_hint = ""
    if deleted_factors and len(deleted_factors) > 0:
        deleted_list = "\n".join([f"  - {name}: verified as an isolated factor unrelated to others"
                                   for name in deleted_factors.keys()])
        deleted_factors_hint = f"""
    ## ⚠️ Deleted useless factors (do NOT propose again)
    The following factors were found in previous iterations but have been statistically verified as **isolated factors** (unrelated to others) and were removed:
    {deleted_list}

    **Please do NOT propose these factors or similar ones again!**

    """

    prompt = _render_template(
        "factor_discovery_prompt.md",
        iteration_num=str(iteration_num),
        domain_context=domain_context,
        data_distribution=data_distribution,
        existing_factors_hint=existing_factors_hint,
        deleted_factors_hint=deleted_factors_hint,
        samples_text=samples_text,
    )
    
    return prompt


def build_annotation_prompt(sample_text, factors_json, domain_context):
    """
    Build the annotation prompt for a single sample.

    Parameters:
    -----------
    sample_text : str
        Formatted text of a single sample
    factors_json : str
        JSON string of factor definitions
    domain_context : str
        Domain background knowledge

    Returns:
    --------
    prompt : str
        Annotation prompt
    """
    
    prompt = _render_template(
        "factor_full_annotation_prompt.md",
        domain_context=domain_context,
        factors_json=factors_json,
        sample_text=sample_text,
    )
    
    return prompt


def build_annotation_prompt_for_new_factors(sample_text, new_factors_json, domain_context):
    """
    Build the prompt to annotate only new factors (incremental annotation).

    Parameters:
    -----------
    sample_text : str
        Formatted text of a single sample
    new_factors_json : str
        JSON string of new factor definitions (new factors only)
    domain_context : str
        Domain background knowledge

    Returns:
    --------
    prompt : str
        Annotation prompt
    """
    
    prompt = _render_template(
        "factor_incremental_annotation_prompt.md",
        domain_context=domain_context,
        new_factors_json=new_factors_json,
        sample_text=sample_text,
    )
    
    return prompt


def build_factor_selection_prompt(factors_json, domain_context):
    """
    Build the factor selection prompt (for choosing focus_factor).

    Parameters:
    -----------
    factors_json : str
        JSON string of factor definitions
    domain_context : str
        Domain background knowledge

    Returns:
    --------
    prompt : str
        Factor selection prompt
    """

    prompt = _render_template(
        "factor_selection_prompt.md",
        domain_context=domain_context,
        factors_json=factors_json,
    )
    
    return prompt


def build_factor_summary_prompt(factors):
    """
    Build a factor summary prompt (for human review).

    Parameters:
    -----------
    factors : dict
        Factor definition dict

    Returns:
    --------
    summary : str
        Formatted factor summary text
    """
    
    summary = "# Discovered Factors Summary\n\n"
    
    for idx, (factor_name, factor_info) in enumerate(factors.items(), 1):
        summary += f"## {idx}. {factor_name}\n"
        summary += f"**Description**: {factor_info['description']}\n\n"
        # possible_values is often a list of int; direct join will raise TypeError
        summary += f"**Possible Values**: {', '.join(map(str, factor_info['possible_values']))}\n\n"
        summary += f"**Annotation Criteria**:\n{factor_info['annotation_criteria']}\n\n"
        summary += "-" * 60 + "\n\n"
    
    return summary


def build_batch_samples_text(df, n_samples, columns=None):
    """
    Format samples in batch into prompt text.

    Parameters:
    -----------
    df : DataFrame
        Dataset
    n_samples : int
        Number of samples
    columns : list, optional
        Column names to display

    Returns:
    --------
    samples_text : str
        Formatted sample text
    """
    from utils import random_sample_data, format_sample
    
    samples = random_sample_data(df, n_samples)
    
    samples_text = ""
    for idx, (_, row) in enumerate(samples.iterrows(), 1):
        samples_text += f"### Sample {idx}\n"
        samples_text += format_sample(row, columns)
        samples_text += "\n\n"
    
    return samples_text



