"""
LaTeX table and report generation for hair transfer evaluation results.

This module provides functions to generate publication-quality LaTeX tables
and documents for comparing hair transfer methods across multiple metrics.
"""

from __future__ import annotations

import os
import textwrap
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd


# -----------------------------
# Global Metrics (computed once per dataset, not per sample)
# -----------------------------

GLOBAL_METRICS = ['clip_i', 'fid', 'fid_clip']


# -----------------------------
# Metric Categories and Metadata
# -----------------------------

METRIC_CATEGORIES = {
    'hair_transfer': {
        'display_name': 'Hair Transfer Quality',
        'description': 'Metrics measuring how well the generated hair matches the reference hair',
        # Note: clip_i is a global metric, so we only include clip_i_per_sample here
        'metrics': ['dinov3_hair_similarity', 'pixio_hair_similarity', 'clip_i_per_sample', 'hair_prompt_similarity'],
    },
    'identity_preservation': {
        'display_name': 'Identity Preservation',
        'description': 'Metrics measuring preservation of original face identity',
        'metrics': ['ids'],
    },
    'distribution_quality': {
        'display_name': 'Distribution Quality',
        'description': 'Metrics measuring distribution similarity between source and generated images (full images)',
        'metrics': ['fid', 'fid_clip'],
    },
    'nonhair_region': {
        'display_name': 'Non-Hair Region Quality',
        'description': 'Metrics measuring quality of non-hair intersected regions (face + background)',
        'metrics': ['dreamsim', 'ssim_nonhair_intersection', 'psnr_nonhair_intersection', 'lpips'],
    },
}

# Human-readable display names for metrics
METRIC_DISPLAY_NAMES = {
    'clip_i': 'CLIP-I',
    'clip_i_per_sample': 'CLIP-I (per-sample)',
    'fid': 'FID',
    'fid_clip': 'FID-CLIP',
    'ssim_nonhair_intersection': 'SSIM (non-hair)',
    'psnr_nonhair_intersection': 'PSNR (non-hair)',
    'lpips': 'LPIPS (non-hair)',
    'dreamsim': 'DreamSim (hair)',
    'ids': 'IDS',
    'dinov3_hair_similarity': 'DINO Hair Sim',
    'pixio_hair_similarity': 'Pixio Hair Sim',
    'hair_prompt_similarity': 'Hair Prompt Sim',
}

# Metric short names for compact LaTeX tables
METRIC_LATEX_NAMES = {
    'dinov3_hair_similarity': r'DINO$_\text{hair}$',
    'pixio_hair_similarity': r'Pixio$_\text{hair}$',
    'ids': 'IDS',
    'clip_i': 'CLIP-I',
    'clip_i_per_sample': 'CLIP-I',
    'fid': 'FID',
    'fid_clip': 'FID-CLIP',
    'dreamsim': r'DreamSim$_\text{hair}$',
    'ssim_nonhair_intersection': r'SSIM$_\text{nh}$',
    'psnr_nonhair_intersection': r'PSNR$_\text{nh}$',
    'lpips': r'LPIPS$_\text{nh}$',
    'hair_prompt_similarity': r'Prompt$_\text{hair}$',
}

# HairPort method prefix - all methods starting with this are considered "ours"
HAIRPORT_METHOD_PREFIX = 'hairport'


def is_hairport_method(method: str) -> bool:
    """
    Check if a method is a HairPort variant.
    
    All methods starting with 'hairport' are considered HairPort variants:
    - hairport
    - hairport_hi3dgen
    - hairport_direct3d_s2
    - etc.
    """
    return method.startswith(HAIRPORT_METHOD_PREFIX)


# Canonical method ordering (baselines first, then HairPort variants)
# HairPort variants are sorted alphabetically and placed at the end
METHOD_ORDER = [
    'barbershop',
    'styleyourhair',
    'hairclip',
    'hairclipv2',
    'hairfastgan',
    'stablehair',
    'hairfusion',
    # HairPort variants (will be sorted and added dynamically)
]

# Method short names for LaTeX
METHOD_LATEX_NAMES = {
    'hairfastgan': 'HairFastGAN',
    'hairport': r'\textbf{HairPort (Ours)}',
    'hairport_hi3dgen': r'\textbf{HairPort-Hi3DGen (Ours)}',
    'hairport_direct3d_s2': r'\textbf{HairPort-Direct3D (Ours)}',
    'hairport_insertanything': r'\textbf{HairPort-IA (Ours)}',
    'stablehair': 'StableHair',
    'hairclip': 'HairCLIP',
    'hairclipv2': 'HairCLIPv2',
    'hairfusion': 'HairFusion',
    'barbershop': 'Barbershop',
    'styleyourhair': 'StyleYourHair',
}

# Method display names
METHOD_DISPLAY_NAMES = {
    'hairfastgan': 'HairFastGAN',
    'hairport': 'HairPort (Ours)',
    'hairport_hi3dgen': 'HairPort-Hi3DGen (Ours)',
    'hairport_direct3d_s2': 'HairPort-Direct3D (Ours)',
    'hairport_insertanything': 'HairPort-IA (Ours)',
    'stablehair': 'StableHair',
    'hairclip': 'HairCLIP',
    'hairclipv2': 'HairCLIPv2',
    'hairfusion': 'HairFusion',
    'barbershop': 'Barbershop',
    'styleyourhair': 'StyleYourHair',
}

# Metrics where higher is better (True) or lower is better (False)
METRIC_HIGHER_IS_BETTER = {
    'clip_i': True,
    'clip_i_per_sample': True,
    'fid': False,
    'fid_clip': False,
    'ssim_nonhair_intersection': True,
    'psnr_nonhair_intersection': True,
    'lpips': False,
    'dreamsim': False,
    'ids': True,
    'dinov3_hair_similarity': True,
    'pixio_hair_similarity': True,
    'hair_prompt_similarity': True,
}


# -----------------------------
# Helper Functions
# -----------------------------

def get_metric_display_name(metric: str, include_direction: bool = False) -> str:
    """Get human-readable display name for a metric."""
    name = METRIC_DISPLAY_NAMES.get(metric, metric.replace('_', ' ').title())
    if include_direction:
        direction = '↑' if is_higher_better(metric) else '↓'
        name = f"{name} {direction}"
    return name


def get_metric_latex_name(metric: str) -> str:
    """Get LaTeX-formatted name for a metric."""
    return METRIC_LATEX_NAMES.get(metric, metric.replace('_', r'\_'))


def get_method_display_name(method: str) -> str:
    """Get human-readable display name for a method."""
    if method in METHOD_DISPLAY_NAMES:
        return METHOD_DISPLAY_NAMES[method]
    # Handle unknown HairPort variants dynamically
    if is_hairport_method(method):
        # Convert hairport_some_variant to HairPort-SomeVariant (Ours)
        suffix = method[len(HAIRPORT_METHOD_PREFIX):].lstrip('_')
        if suffix:
            # Convert snake_case to Title Case
            suffix_display = suffix.replace('_', ' ').title().replace(' ', '-')
            return f'HairPort-{suffix_display} (Ours)'
        return 'HairPort (Ours)'
    return method.replace('_', ' ').title()


def get_method_latex_name(method: str) -> str:
    """Get LaTeX-formatted name for a method."""
    if method in METHOD_LATEX_NAMES:
        return METHOD_LATEX_NAMES[method]
    # Handle unknown HairPort variants dynamically
    if is_hairport_method(method):
        # Convert hairport_some_variant to \textbf{HairPort-SomeVariant (Ours)}
        suffix = method[len(HAIRPORT_METHOD_PREFIX):].lstrip('_')
        if suffix:
            # Convert snake_case to Title Case
            suffix_display = suffix.replace('_', ' ').title().replace(' ', '-')
            return r'\textbf{HairPort-' + suffix_display + r' (Ours)}'
        return r'\textbf{HairPort (Ours)}'
    return method.replace('_', r'\_')


def is_higher_better(metric: str) -> bool:
    """Return True if higher values are better for this metric."""
    return METRIC_HIGHER_IS_BETTER.get(metric, True)


def get_metric_category(metric: str) -> str:
    """Get the category a metric belongs to."""
    for category, info in METRIC_CATEGORIES.items():
        if metric in info['metrics']:
            return category
    return 'other'


def get_ordered_methods(methods: List[str], our_method: str = "hairport") -> List[str]:
    """
    Sort methods according to METHOD_ORDER, with all HairPort variants at the end.
    
    Baselines are sorted according to METHOD_ORDER, then alphabetically for unknown ones.
    All HairPort variants (methods starting with 'hairport') are placed at the end,
    sorted alphabetically among themselves.
    
    Args:
        methods: List of method names to sort
        our_method: Deprecated parameter, kept for backward compatibility.
            All hairport variants are now automatically placed at the end.
    
    Returns:
        Sorted list of methods with HairPort variants at the end
    """
    # Create ordering dict from METHOD_ORDER
    order_dict = {m: i for i, m in enumerate(METHOD_ORDER)}
    
    # Separate HairPort variants from baselines
    hairport_methods = [m for m in methods if is_hairport_method(m)]
    baseline_methods = [m for m in methods if not is_hairport_method(m)]
    
    # Sort baseline methods: first by METHOD_ORDER position, then alphabetically for unknown ones
    def baseline_sort_key(m):
        if m in order_dict:
            return (0, order_dict[m], m)  # Known methods: sort by position
        return (1, 0, m)  # Unknown methods: alphabetically after known ones
    
    sorted_baselines = sorted(baseline_methods, key=baseline_sort_key)
    
    # Sort HairPort variants alphabetically
    sorted_hairport = sorted(hairport_methods)
    
    # Combine: baselines first, then HairPort variants
    return sorted_baselines + sorted_hairport


# -----------------------------
# Best Value Determination
# -----------------------------

def _determine_best_values(
    results_dict: Dict[str, pd.DataFrame],
    global_metrics_dict: Dict[str, Dict[str, float]],
    metrics: List[str],
) -> Dict[str, Tuple[str, float]]:
    """
    Determine the best method and value for each metric.
    
    Returns:
        Dictionary mapping metric name to (best_method, best_value)
    """
    best_values = {}
    methods = list(results_dict.keys())
    
    for metric in metrics:
        is_global = metric in GLOBAL_METRICS
        
        if is_global:
            values = {}
            for method in methods:
                if method in global_metrics_dict and metric in global_metrics_dict[method]:
                    values[method] = global_metrics_dict[method][metric]
        else:
            values = {}
            for method in methods:
                if metric in results_dict[method].columns:
                    val = results_dict[method][metric].mean(skipna=True)
                    if not np.isnan(val):
                        values[method] = val
        
        if values:
            if is_higher_better(metric):
                best_method = max(values, key=values.get)
            else:
                best_method = min(values, key=values.get)
            best_values[metric] = (best_method, values[best_method])
    
    return best_values


def _determine_second_best_values(
    results_dict: Dict[str, pd.DataFrame],
    global_metrics_dict: Dict[str, Dict[str, float]],
    metrics: List[str],
    best_values: Dict[str, Tuple[str, float]],
) -> Dict[str, Tuple[str, float]]:
    """
    Determine the second-best method and value for each metric.
    
    Returns:
        Dictionary mapping metric name to (second_best_method, second_best_value)
    """
    second_best_values = {}
    methods = list(results_dict.keys())
    
    for metric in metrics:
        if metric not in best_values:
            continue
            
        best_method = best_values[metric][0]
        is_global = metric in GLOBAL_METRICS
        
        if is_global:
            values = {}
            for method in methods:
                if method != best_method and method in global_metrics_dict and metric in global_metrics_dict[method]:
                    values[method] = global_metrics_dict[method][metric]
        else:
            values = {}
            for method in methods:
                if method != best_method and metric in results_dict[method].columns:
                    val = results_dict[method][metric].mean(skipna=True)
                    if not np.isnan(val):
                        values[method] = val
        
        if values:
            if is_higher_better(metric):
                second_method = max(values, key=values.get)
            else:
                second_method = min(values, key=values.get)
            second_best_values[metric] = (second_method, values[second_method])
    
    return second_best_values


def _format_metric_value(
    value: float,
    std: Optional[float],
    metric: str,
    is_best: bool = False,
    is_second_best: bool = False,
    include_std: bool = False,
    precision: int = 2,
) -> str:
    """
    Format a metric value for LaTeX with optional highlighting.
    """
    if metric in ['fid', 'fid_clip']:
        precision = 2
    elif metric in ['psnr_full', 'psnr_nonhair_intersection']:
        precision = 2
    
    if std is not None and include_std:
        val_str = f"{value:.{precision}f}"
        std_str = f"{std:.{precision}f}"
        formatted = f"{val_str}{{\\scriptsize$\\pm${std_str}}}"
    else:
        formatted = f"{value:.{precision}f}"
    
    if is_best:
        formatted = r"\textbf{" + formatted + "}"
    elif is_second_best:
        formatted = r"\underline{" + formatted + "}"
    
    return formatted


# -----------------------------
# Main Table Generation
# -----------------------------

def generate_latex_main_table(
    results_dict: Dict[str, pd.DataFrame],
    global_metrics_dict: Dict[str, Dict[str, float]],
    output_path: str,
    caption: str = "Quantitative comparison of hair transfer methods.",
    label: str = "tab:main_comparison",
    metrics_to_include: Optional[List[str]] = None,
    highlight_best: bool = True,
    highlight_second_best: bool = True,
    include_std: bool = False,
    our_method: str = "hairport",
) -> str:
    """
    Generate a publication-ready LaTeX table for the main comparison.
    
    Args:
        results_dict: Dictionary mapping method names to DataFrames
        global_metrics_dict: Dictionary mapping method names to global metrics
        output_path: Path to save the .tex file
        caption: Table caption
        label: LaTeX label for referencing
        metrics_to_include: List of metrics to include (None = all)
        highlight_best: Bold the best values
        highlight_second_best: Underline second-best values
        include_std: Include standard deviation
        our_method: Method name to highlight as "ours"
    
    Returns:
        LaTeX table string
    """
    methods = get_ordered_methods(list(results_dict.keys()), our_method)
    
    metadata_cols = ['target_id', 'source_id', 'pair_name']
    sample_df = list(results_dict.values())[0]
    all_metrics = [col for col in sample_df.columns if col not in metadata_cols]
    
    if metrics_to_include:
        all_metrics = [m for m in metrics_to_include if m in all_metrics or m in GLOBAL_METRICS]
    
    global_metrics = [m for m in GLOBAL_METRICS if m in all_metrics or metrics_to_include is None]
    per_pair_metrics = [m for m in all_metrics if m not in global_metrics]
    
    # Order metrics by category
    ordered_metrics = []
    for category in ['hair_transfer', 'identity_preservation', 'nonhair_region']:
        cat_metrics = METRIC_CATEGORIES.get(category, {}).get('metrics', [])
        for m in cat_metrics:
            if m in per_pair_metrics and m not in ordered_metrics:
                ordered_metrics.append(m)
    for m in per_pair_metrics:
        if m not in ordered_metrics:
            ordered_metrics.append(m)
    
    ordered_metrics = ordered_metrics + global_metrics
    
    best_values = _determine_best_values(results_dict, global_metrics_dict, ordered_metrics)
    second_best_values = _determine_second_best_values(results_dict, global_metrics_dict, ordered_metrics, best_values)
    
    n_metrics = len(ordered_metrics)
    col_spec = "l" + "c" * n_metrics
    
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{" + caption + "}")
    lines.append(r"\label{" + label + "}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")
    
    header_parts = ["Method"]
    for metric in ordered_metrics:
        latex_name = get_metric_latex_name(metric)
        direction = r"$\uparrow$" if is_higher_better(metric) else r"$\downarrow$"
        header_parts.append(f"{latex_name} {direction}")
    
    lines.append(" & ".join(header_parts) + r" \\")
    lines.append(r"\midrule")
    
    for method in methods:
        df = results_dict[method]
        method_name = get_method_latex_name(method)
        
        row_parts = [method_name]
        
        for metric in ordered_metrics:
            is_global = metric in global_metrics
            
            if is_global:
                if method in global_metrics_dict and metric in global_metrics_dict[method]:
                    value = global_metrics_dict[method][metric]
                    std = None
                else:
                    row_parts.append("-")
                    continue
            else:
                if metric in df.columns:
                    value = df[metric].mean(skipna=True)
                    std = df[metric].std(skipna=True) if include_std else None
                else:
                    row_parts.append("-")
                    continue
            
            if np.isnan(value):
                row_parts.append("-")
                continue
            
            is_best = highlight_best and metric in best_values and best_values[metric][0] == method
            is_second = highlight_second_best and metric in second_best_values and second_best_values[metric][0] == method
            
            formatted = _format_metric_value(value, std, metric, is_best, is_second, include_std)
            row_parts.append(formatted)
        
        lines.append(" & ".join(row_parts) + r" \\")
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\vspace{-2mm}")
    lines.append(r"\end{table*}")
    
    latex_str = "\n".join(lines)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(latex_str)
    
    return latex_str


def generate_latex_category_tables(
    results_dict: Dict[str, pd.DataFrame],
    global_metrics_dict: Dict[str, Dict[str, float]],
    output_dir: str,
    highlight_best: bool = True,
    highlight_second_best: bool = True,
    our_method: str = "hairport",
    include_std: bool = False,
) -> Dict[str, str]:
    """
    Generate separate LaTeX tables for each metric category.
    
    Each table includes:
    - Appropriate title and caption based on the category
    - All metrics in that category with direction indicators (↑/↓)
    - Best values in bold, second-best underlined
    - Mean ± std for per-sample metrics
    
    Args:
        results_dict: Dictionary mapping method names to DataFrames
        global_metrics_dict: Dictionary mapping method names to global metrics
        output_dir: Directory to save the .tex files
        highlight_best: Bold the best values
        highlight_second_best: Underline second-best values
        our_method: Method name to highlight as "ours"
        include_std: Include standard deviation in results
    
    Returns:
        Dictionary mapping category names to LaTeX strings
    """
    os.makedirs(output_dir, exist_ok=True)
    
    methods = get_ordered_methods(list(results_dict.keys()), our_method)
    latex_tables = {}
    
    # Enhanced category information with detailed captions
    category_info = {
        'hair_transfer': {
            'title': 'Hair Transfer Quality',
            'caption': (
                r'\textbf{Hair Transfer Quality.} '
                r'Metrics measuring similarity between the reference hair and generated hair regions. '
                r'DINO$_\text{hair}$ uses DINOv2 vision foundation model embeddings for robust hair similarity measurement; '
                r'CLIP-I measures CLIP image similarity between reference (hair donor) and generated images; '
                r'Prompt$_\text{hair}$ compares hair description embeddings using Qwen3-Embedding-8B. '
                r'Higher values indicate better hair transfer fidelity. '
                r'Best results in \textbf{bold}, second-best \underline{underlined}.'
            ),
            'label': 'tab:hair_transfer',
        },
        'identity_preservation': {
            'title': 'Identity Preservation',
            'caption': (
                r'\textbf{Identity Preservation.} '
                r'Metrics measuring how well the original face identity is preserved after hair transfer. '
                r'IDS uses InsightFace ArcFace embeddings to measure cosine similarity between source and generated faces. '
                r'Higher values indicate better identity preservation. '
                r'Best results in \textbf{bold}, second-best \underline{underlined}.'
            ),
            'label': 'tab:identity_preservation',
        },
        'nonhair_region': {
            'title': 'Non-Hair Region Quality',
            'caption': (
                r'\textbf{Non-Hair Region Quality.} '
                r'Metrics measuring quality preservation in non-hair intersected regions (face + background) between source and generated images. '
                r'All metrics are computed on regions where neither source nor generated image has hair: W = (1-hair\_src)*(1-hair\_gen). '
                r'FID$_\text{nh}$ and FID-CLIP$_\text{nh}$ measure distributional similarity; '
                r'DreamSim$_\text{nh}$ measures perceptual distance; '
                r'SSIM$_\text{nh}$ and PSNR$_\text{nh}$ measure structural/pixel similarity; '
                r'LPIPS$_\text{nh}$ measures perceptual similarity. '
                r'Best results in \textbf{bold}, second-best \underline{underlined}.'
            ),
            'label': 'tab:nonhair_region',
        },
    }
    
    for category, cat_info in METRIC_CATEGORIES.items():
        metrics = cat_info['metrics']
        
        # Get available metrics from the data
        sample_df = list(results_dict.values())[0]
        # Check which global metrics are actually present in global_metrics_dict
        available_global = set()
        for method_globals in global_metrics_dict.values():
            available_global.update(method_globals.keys())
        # A metric is available if it's in the DataFrame OR if it's a global metric that's actually present
        available_metrics = [m for m in metrics if m in sample_df.columns or m in available_global]
        
        if not available_metrics:
            continue
        
        # Determine best and second-best values for each metric
        best_values = _determine_best_values(results_dict, global_metrics_dict, available_metrics)
        second_best_values = _determine_second_best_values(results_dict, global_metrics_dict, available_metrics, best_values)
        
        n_metrics = len(available_metrics)
        col_spec = "l" + "c" * n_metrics
        
        # Get category-specific info
        info = category_info.get(category, {})
        title = info.get('title', cat_info.get('display_name', category.replace("_", " ").title()))
        caption = info.get('caption', f'{title} metrics. Best in bold, second-best underlined.')
        label = info.get('label', f'tab:{category}')
        
        # Build LaTeX table
        lines = []
        lines.append(r"\begin{table}[t]")
        lines.append(r"\centering")
        lines.append(r"\caption{" + caption + "}")
        lines.append(r"\label{" + label + "}")
        lines.append(r"\small")
        lines.append(r"\begin{tabular}{" + col_spec + "}")
        lines.append(r"\toprule")
        
        # Header row with metric names and direction indicators
        header_parts = ["Method"]
        for metric in available_metrics:
            latex_name = get_metric_latex_name(metric)
            direction = r"$\uparrow$" if is_higher_better(metric) else r"$\downarrow$"
            header_parts.append(f"{latex_name} {direction}")
        
        lines.append(" & ".join(header_parts) + r" \\")
        lines.append(r"\midrule")
        
        # Data rows for each method
        for method in methods:
            df = results_dict[method]
            method_name = get_method_latex_name(method)
            row_parts = [method_name]
            
            for metric in available_metrics:
                is_global = metric in GLOBAL_METRICS
                
                if is_global:
                    # Global metrics come from global_metrics_dict
                    if method in global_metrics_dict and metric in global_metrics_dict[method]:
                        value = global_metrics_dict[method][metric]
                        std = None  # Global metrics don't have per-sample std
                    else:
                        row_parts.append("-")
                        continue
                else:
                    # Per-sample metrics: compute mean and std
                    if metric in df.columns:
                        value = df[metric].mean(skipna=True)
                        std = df[metric].std(skipna=True) if include_std else None
                    else:
                        row_parts.append("-")
                        continue
                
                if np.isnan(value):
                    row_parts.append("-")
                    continue
                
                # Determine if this is best or second-best
                is_best = highlight_best and metric in best_values and best_values[metric][0] == method
                is_second = highlight_second_best and metric in second_best_values and second_best_values[metric][0] == method
                
                # Format the value with highlighting
                formatted = _format_metric_value(value, std, metric, is_best, is_second, include_std)
                row_parts.append(formatted)
            
            lines.append(" & ".join(row_parts) + r" \\")
        
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")
        
        latex_str = "\n".join(lines)
        latex_tables[category] = latex_str
        
        # Save to file
        output_path = os.path.join(output_dir, f"table_{category}.tex")
        with open(output_path, 'w') as f:
            f.write(latex_str)
        print(f"  Saved: {output_path}")
    
    return latex_tables


def generate_latex_compact_table(
    results_dict: Dict[str, pd.DataFrame],
    global_metrics_dict: Dict[str, Dict[str, float]],
    output_path: str,
    caption: str = "Quantitative comparison of hair transfer methods. Best results in \\textbf{bold}, second-best \\underline{underlined}.",
    label: str = "tab:comparison",
    our_method: str = "hairport",
) -> str:
    """
    Generate a compact, publication-ready LaTeX table suitable for conference papers.
    Groups metrics by category with multi-column headers.
    """
    methods = get_ordered_methods(list(results_dict.keys()), our_method)
    
    # Get the set of all available metrics from the data
    sample_df = list(results_dict.values())[0]
    available_in_df = set(sample_df.columns)
    available_global = set()
    for method_globals in global_metrics_dict.values():
        available_global.update(method_globals.keys())
    all_available = available_in_df | available_global
    
    metric_groups = [
        {
            'name': 'Hair Transfer',
            'metrics': ['dinov3_hair_similarity', 'pixio_hair_similarity', 'clip_i'],
        },
        {
            'name': 'Identity',
            'metrics': ['ids'],
        },
        {
            'name': 'Non-Hair Region',
            'metrics': ['fid', 'fid_clip', 'lpips', 'dreamsim'],
        },
    ]
    
    # Filter each group to only include available metrics
    filtered_groups = []
    for group in metric_groups:
        available_metrics = [m for m in group['metrics'] if m in all_available]
        if available_metrics:
            filtered_groups.append({
                'name': group['name'],
                'metrics': available_metrics,
            })
    metric_groups = filtered_groups
    
    if not metric_groups:
        print("  Warning: No metrics available for compact table")
        return ""
    
    all_metrics = []
    for group in metric_groups:
        all_metrics.extend(group['metrics'])
    
    best_values = _determine_best_values(results_dict, global_metrics_dict, all_metrics)
    second_best_values = _determine_second_best_values(results_dict, global_metrics_dict, all_metrics, best_values)
    
    total_cols = sum(len(g['metrics']) for g in metric_groups)
    col_spec = "l" + "c" * total_cols
    
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{" + caption + "}")
    lines.append(r"\label{" + label + "}")
    lines.append(r"\resizebox{0.9\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")
    
    header1_parts = [""]
    for group in metric_groups:
        n_cols = len(group['metrics'])
        header1_parts.append(r"\multicolumn{" + str(n_cols) + r"}{c}{" + group['name'] + "}")
    lines.append(" & ".join(header1_parts) + r" \\")
    
    col_idx = 2
    cmidrules = []
    for group in metric_groups:
        n_cols = len(group['metrics'])
        cmidrules.append(f"\\cmidrule(lr){{{col_idx}-{col_idx + n_cols - 1}}}")
        col_idx += n_cols
    lines.append(" ".join(cmidrules))
    
    header2_parts = ["Method"]
    for group in metric_groups:
        for metric in group['metrics']:
            latex_name = get_metric_latex_name(metric)
            direction = r"$\uparrow$" if is_higher_better(metric) else r"$\downarrow$"
            header2_parts.append(f"{latex_name}{direction}")
    lines.append(" & ".join(header2_parts) + r" \\")
    lines.append(r"\midrule")
    
    for method in methods:
        df = results_dict[method]
        method_name = get_method_latex_name(method)
        row_parts = [method_name]
        
        for group in metric_groups:
            for metric in group['metrics']:
                is_global = metric in GLOBAL_METRICS
                
                if is_global:
                    if method in global_metrics_dict and metric in global_metrics_dict[method]:
                        value = global_metrics_dict[method][metric]
                        std = None
                    else:
                        row_parts.append("-")
                        continue
                else:
                    if metric in df.columns:
                        value = df[metric].mean(skipna=True)
                        std = df[metric].std(skipna=True)
                    else:
                        row_parts.append("-")
                        continue
                
                if np.isnan(value):
                    row_parts.append("-")
                    continue
                
                is_best = metric in best_values and best_values[metric][0] == method
                is_second = metric in second_best_values and second_best_values[metric][0] == method
                
                formatted = _format_metric_value(value, None, metric, is_best, is_second, include_std=False)
                row_parts.append(formatted)
        
        lines.append(" & ".join(row_parts) + r" \\")
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\end{table}")
    
    latex_str = "\n".join(lines)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(latex_str)
    
    return latex_str


def generate_latex_figure_commands(
    output_dir: str,
    figure_files: List[str],
    output_path: str,
) -> str:
    """
    Generate LaTeX commands for including figures with proper formatting.
    """
    lines = []
    lines.append("% Auto-generated figure inclusion commands")
    lines.append(f"% Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    for fig_file in figure_files:
        fig_name = os.path.splitext(os.path.basename(fig_file))[0]
        cmd_name = fig_name.replace("_", "").replace("-", "")
        
        lines.append(f"% Figure: {fig_name}")
        lines.append(f"\\newcommand{{\\fig{cmd_name}}}{{%")
        lines.append(f"  \\includegraphics[width=0.8\\textwidth]{{{fig_file}}}%")
        lines.append("}")
        lines.append("")
    
    latex_str = "\n".join(lines)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(latex_str)
    
    return latex_str


def generate_latex_template(
    output_dir: str,
    results_dict: Dict[str, pd.DataFrame],
    global_metrics_dict: Dict[str, Dict[str, float]],
    paper_title: str = "Hair Transfer Evaluation Results",
    our_method: str = "hairport",
) -> str:
    """
    Generate a complete, compilable LaTeX document template.
    """
    methods = get_ordered_methods(list(results_dict.keys()), our_method)
    
    template = textwrap.dedent(r'''
    % ============================================================================
    % Hair Transfer Evaluation Results - LaTeX Template
    % Generated automatically by latex_builder.py
    % ============================================================================
    
    \documentclass[11pt]{article}
    
    % ============================================================================
    % Packages
    % ============================================================================
    \usepackage[utf8]{inputenc}
    \usepackage[T1]{fontenc}
    \usepackage{booktabs}       % Professional tables
    \usepackage{multirow}       % Multi-row cells
    \usepackage{multicol}       % Multi-column layout
    \usepackage{graphicx}       % Images
    \usepackage{xcolor}         % Colors
    \usepackage{amsmath}        % Math
    \usepackage{amssymb}        % Math symbols
    \usepackage{siunitx}        % Number formatting
    \usepackage{subcaption}     % Subfigures
    \usepackage{hyperref}       % Hyperlinks
    \usepackage[margin=1in]{geometry}
    
    % ============================================================================
    % Custom Commands
    % ============================================================================
    \newcommand{\best}[1]{\textbf{#1}}
    \newcommand{\secondbest}[1]{\underline{#1}}
    \newcommand{\methodname}{HairPort}
    
    % Metric arrows
    \newcommand{\up}{$\uparrow$}
    \newcommand{\down}{$\downarrow$}
    
    % ============================================================================
    % Document
    % ============================================================================
    \begin{document}
    
    \title{''' + paper_title + r'''}
    \author{Auto-generated Report}
    \date{\today}
    \maketitle
    
    % ============================================================================
    % Abstract
    % ============================================================================
    \begin{abstract}
    This document presents quantitative evaluation results for hair transfer methods.
    All metrics are computed on a standardized benchmark dataset.
    Best results are shown in \textbf{bold}, second-best are \underline{underlined}.
    Arrows indicate metric direction: \up{} = higher is better, \down{} = lower is better.
    \end{abstract}
    
    % ============================================================================
    % Main Results
    % ============================================================================
    \section{Quantitative Results}
    
    Table~\ref{tab:main_comparison} presents a comprehensive comparison of all methods across all metrics.
    
    \input{tables/main_comparison.tex}
    
    % ============================================================================
    % Results by Category
    % ============================================================================
    \section{Results by Category}
    
    We organize our evaluation metrics into three categories, each measuring different aspects of hair transfer quality.
    
    \subsection{Hair Transfer Quality}
    
    Table~\ref{tab:hair_transfer} evaluates how well each method transfers the reference hair style to the target image.
    We use DINOv2 and CLIP embeddings to measure semantic similarity between the reference and generated hair regions,
    as well as text-based hair description similarity.
    
    \input{tables/table_hair_transfer.tex}
    
    \subsection{Identity Preservation}
    
    Table~\ref{tab:identity_preservation} measures how well each method preserves the original face identity after hair transfer.
    Identity preservation is crucial for realistic hair transfer applications.
    
    \input{tables/table_identity_preservation.tex}
    
    \subsection{Non-Hair Region Quality}
    
    Table~\ref{tab:nonhair_region} evaluates the quality of non-hair intersected regions (face + background).
    All metrics in this category are computed on regions where neither source nor generated image has hair,
    ensuring we measure how well the method preserves areas that should remain unchanged during hair transfer.
    
    \input{tables/table_nonhair_region.tex}
    
    % ============================================================================
    % Compact Summary
    % ============================================================================
    \section{Compact Summary}
    
    Table~\ref{tab:comparison} provides a compact summary of key metrics for quick comparison.
    
    \input{tables/compact_comparison.tex}
    
    % ============================================================================
    % Figures
    % ============================================================================
    \section{Visualizations}
    
    \begin{figure}[t]
        \centering
        \includegraphics[width=0.8\textwidth]{figures/comparison_all_metrics.png}
        \caption{Comparison of all metrics across methods.}
        \label{fig:comparison_all}
    \end{figure}
    
    % ============================================================================
    % Metric Definitions
    % ============================================================================
    \section{Metric Definitions}
    
    \paragraph{Hair Transfer Quality}
    \begin{itemize}
        \item \textbf{DINO$_\text{hair}$}: DINOv2 cosine similarity between reference and generated hair regions. DINOv2 is a vision foundation model that produces high-quality dense features. Higher is better.
    \end{itemize}
    
    \paragraph{Identity Preservation}
    \begin{itemize}
        \item \textbf{IDS}: Identity similarity using InsightFace embeddings between source and generated faces. Higher is better.
        \item \textbf{CLIP-I}: CLIP image similarity between source and generated images. Higher is better.
    \end{itemize}
    
    \paragraph{Non-Hair Region Preservation}
    \begin{itemize}
        \item \textbf{SSIM$_\text{bg}$}: Structural Similarity Index on non-hair regions. Higher is better.
        \item \textbf{PSNR$_\text{bg}$}: Peak Signal-to-Noise Ratio on non-hair regions. Higher is better.
        \item \textbf{LPIPS$_\text{bg}$}: Learned Perceptual Image Patch Similarity on non-hair regions. Lower is better.
    \end{itemize}
    
    \paragraph{Full Image Quality}
    \begin{itemize}
        \item \textbf{FID}: Fr\'echet Inception Distance measuring distribution similarity. Lower is better.
        \item \textbf{SSIM}: Structural Similarity Index on full image. Higher is better.
        \item \textbf{PSNR}: Peak Signal-to-Noise Ratio on full image. Higher is better.
        \item \textbf{LPIPS}: Learned Perceptual Image Patch Similarity on full image. Lower is better.
        \item \textbf{DreamSim}: Perceptual similarity using DreamSim model. Lower is better.
    \end{itemize}
    
    \end{document}
    ''').strip()
    
    os.makedirs(output_dir, exist_ok=True)
    template_path = os.path.join(output_dir, "main.tex")
    with open(template_path, 'w') as f:
        f.write(template)
    
    return template


def generate_all_latex_outputs(
    results_dict: Dict[str, pd.DataFrame],
    global_metrics_dict: Dict[str, Dict[str, float]],
    output_dir: str,
    paper_title: str = "Hair Transfer Evaluation Results",
    our_method: str = "hairport",
):
    """
    Generate all LaTeX outputs: tables, figure commands, and template.
    
    Args:
        results_dict: Dictionary mapping method names to DataFrames
        global_metrics_dict: Dictionary mapping method names to global metrics
        output_dir: Base output directory for LaTeX files
        paper_title: Title for the paper template
        our_method: Method name to highlight as "ours"
    """
    import shutil
    
    print("\n" + "="*60)
    print("GENERATING LATEX OUTPUTS")
    print("="*60)
    
    # Create directory structure
    latex_dir = os.path.join(output_dir, "latex_report")
    tables_dir = os.path.join(latex_dir, "tables")
    figures_dir = os.path.join(latex_dir, "figures")
    
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    # 1. Generate main comparison table
    print("\nGenerating main comparison table...")
    main_table_path = os.path.join(tables_dir, "main_comparison.tex")
    generate_latex_main_table(
        results_dict, global_metrics_dict,
        main_table_path,
        caption="Quantitative comparison of hair transfer methods. Best results in \\textbf{bold}, second-best \\underline{underlined}. $\\uparrow$ indicates higher is better, $\\downarrow$ indicates lower is better.",
        label="tab:main_comparison",
        our_method=our_method,
    )
    print(f"  Saved: {main_table_path}")
    
    # 2. Generate compact table
    print("\nGenerating compact comparison table...")
    compact_table_path = os.path.join(tables_dir, "compact_comparison.tex")
    generate_latex_compact_table(
        results_dict, global_metrics_dict,
        compact_table_path,
        our_method=our_method,
    )
    print(f"  Saved: {compact_table_path}")
    
    # 3. Generate category-specific tables
    print("\nGenerating category-specific tables...")
    generate_latex_category_tables(
        results_dict, global_metrics_dict,
        tables_dir,
        our_method=our_method,
    )
    
    # 4. Copy/link figures to latex figures directory
    print("\nSetting up figures directory...")
    comparison_dir = os.path.join(output_dir, "comparison")
    if os.path.exists(comparison_dir):
        for item in os.listdir(comparison_dir):
            src = os.path.join(comparison_dir, item)
            dst = os.path.join(figures_dir, item)
            if os.path.isfile(src) and src.endswith('.png'):
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
            elif os.path.isdir(src):
                dst_dir = os.path.join(figures_dir, item)
                if not os.path.exists(dst_dir):
                    shutil.copytree(src, dst_dir)
        print(f"  Figures copied to: {figures_dir}")
    
    # 5. Generate LaTeX template
    print("\nGenerating LaTeX template...")
    generate_latex_template(
        latex_dir, results_dict, global_metrics_dict,
        paper_title=paper_title,
    )
    print(f"  Template saved to: {os.path.join(latex_dir, 'main.tex')}")
    
    # 6. Generate Makefile
    makefile_content = textwrap.dedent('''
    # Makefile for LaTeX compilation
    
    MAIN = main
    LATEX = pdflatex
    
    all: $(MAIN).pdf
    
    $(MAIN).pdf: $(MAIN).tex tables/*.tex
    \t$(LATEX) $(MAIN)
    \t$(LATEX) $(MAIN)
    
    clean:
    \trm -f *.aux *.log *.out *.bbl *.blg *.toc *.lof *.lot
    
    cleanall: clean
    \trm -f $(MAIN).pdf
    
    .PHONY: all clean cleanall
    ''').strip()
    
    makefile_path = os.path.join(latex_dir, "Makefile")
    with open(makefile_path, 'w') as f:
        f.write(makefile_content)
    print(f"  Makefile saved to: {makefile_path}")
    
    # 7. Generate README
    readme_content = textwrap.dedent(f'''
    # LaTeX Outputs for Hair Transfer Evaluation
    
    Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    
    ## Directory Structure
    
    ```
    latex_report/
    ├── main.tex           # Main LaTeX document
    ├── Makefile           # For easy compilation
    ├── tables/
    │   ├── main_comparison.tex
    │   ├── compact_comparison.tex
    │   └── table_*.tex
    └── figures/
        └── *.png
    ```
    
    ## Compilation
    
    ```bash
    cd latex_report
    make
    ```
    
    ## Methods Evaluated
    
    {chr(10).join([f"- {m}" for m in results_dict.keys()])}
    ''').strip()
    
    readme_path = os.path.join(latex_dir, "README.md")
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    print(f"  README saved to: {readme_path}")
    
    print("\n" + "="*60)
    print("LATEX GENERATION COMPLETE")
    print("="*60)
    print(f"\nAll LaTeX files saved to: {latex_dir}")
