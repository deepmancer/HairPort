"""
Visualization utilities for hair transfer evaluation results.

This module provides functions to create publication-quality visualizations
for comparing hair transfer methods across multiple metrics.
"""

from __future__ import annotations

import os
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from .latex_builder import (
    METRIC_CATEGORIES,
    GLOBAL_METRICS,
    get_metric_display_name,
    get_method_display_name,
    is_higher_better,
    get_metric_category,
    get_ordered_methods,
    is_hairport_method,
)


# -----------------------------
# Font Configuration
# -----------------------------

def _configure_professional_fonts():
    """Configure professional fonts for visualizations."""
    try:
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['font.size'] = 11
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['axes.labelsize'] = 11
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 10
    except Exception:
        pass  # Fall back to defaults


# -----------------------------
# Category-Specific Visualizations
# -----------------------------

def _create_category_visualizations(
    df: pd.DataFrame, 
    output_dir: str, 
    method: str,
    category: str,
    category_info: Dict[str, Any]
):
    """
    Create visualizations for a specific metric category.
    """
    category_metrics = [m for m in category_info['metrics'] if m in df.columns]
    if not category_metrics:
        return
    
    method_display = get_method_display_name(method)
    category_dir = os.path.join(output_dir, category)
    os.makedirs(category_dir, exist_ok=True)
    
    sns.set_style("whitegrid")
    _configure_professional_fonts()
    
    n_metrics = len(category_metrics)
    
    # 1. Box plots for category metrics
    fig, axes = plt.subplots(1, n_metrics, figsize=(4*n_metrics, 5))
    axes = np.atleast_1d(axes).flatten()
    
    for idx, metric in enumerate(category_metrics):
        ax = axes[idx]
        sns.boxplot(y=df[metric], ax=ax, color='skyblue')
        display_name = get_metric_display_name(metric, include_direction=True)
        ax.set_ylabel(display_name)
        ax.set_title(display_name)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'{category_info["display_name"]} - {method_display}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(category_dir, "boxplots.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Distribution plots
    n_rows = (n_metrics + 1) // 2
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 4*n_rows))
    axes = np.atleast_1d(axes).flatten()
    
    for idx, metric in enumerate(category_metrics):
        ax = axes[idx]
        sns.histplot(df[metric].dropna(), kde=True, ax=ax, color='steelblue', bins=20)
        display_name = get_metric_display_name(metric, include_direction=True)
        ax.set_xlabel(display_name)
        ax.set_ylabel('Frequency')
        ax.set_title(f'{display_name} Distribution')
        ax.grid(True, alpha=0.3)
        
        mean_val = df[metric].mean()
        median_val = df[metric].median()
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
        ax.axvline(median_val, color='green', linestyle='-.', linewidth=2, label=f'Median: {median_val:.3f}')
        ax.legend(fontsize=8)
    
    for idx in range(n_metrics, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'{category_info["display_name"]} - {method_display}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(category_dir, "distributions.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Summary bar plot
    fig, ax = plt.subplots(figsize=(max(8, n_metrics * 2), 5))
    means = [df[m].mean() for m in category_metrics]
    stds = [df[m].std() for m in category_metrics]
    x_pos = np.arange(len(category_metrics))
    display_names = [get_metric_display_name(m, include_direction=True) for m in category_metrics]
    
    bars = ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, 
                  color='steelblue', edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Value')
    ax.set_title(f'{category_info["display_name"]} - {method_display}')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(display_names, rotation=30, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(category_dir, "summary_bars.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  {category_info['display_name']} visualizations saved to {category_dir}")


# -----------------------------
# Single Method Visualizations
# -----------------------------

def create_method_visualizations(df: pd.DataFrame, output_dir: str, method: str):
    """
    Create comprehensive visualizations for a single method's results.
    
    Args:
        df: DataFrame with metric values for each sample
        output_dir: Directory to save visualizations
        method: Method name
    """
    method_display = get_method_display_name(method)
    
    sns.set_style("whitegrid")
    _configure_professional_fonts()
    
    metrics = df.columns.tolist()
    n_metrics = len(metrics)
    
    print(f"\nCreating visualizations for {method_display}...")
    
    # Create category-specific visualizations
    for category, info in METRIC_CATEGORIES.items():
        _create_category_visualizations(df, output_dir, method, category, info)
    
    # Create overall visualizations in main directory
    
    # 1. Box plots for all metrics
    fig, axes = plt.subplots(1, n_metrics, figsize=(3*n_metrics, 5))
    axes = np.atleast_1d(axes).flatten()
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        sns.boxplot(y=df[metric], ax=ax, color='skyblue')
        display_name = get_metric_display_name(metric, include_direction=True)
        ax.set_ylabel(display_name)
        ax.set_title(display_name, fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'All Metrics - {method_display}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    box_plot_path = os.path.join(output_dir, "boxplots.png")
    plt.savefig(box_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Box plots saved to {box_plot_path}")
    
    # 2. Distribution plots (histograms with KDE)
    n_cols = min(4, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    axes = np.atleast_1d(axes).flatten()
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        sns.histplot(df[metric].dropna(), kde=True, ax=ax, color='steelblue', bins=20)
        display_name = get_metric_display_name(metric, include_direction=True)
        ax.set_xlabel(display_name)
        ax.set_ylabel('Frequency')
        ax.set_title(f'{display_name} Distribution', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        mean_val = df[metric].mean()
        median_val = df[metric].median()
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
        ax.axvline(median_val, color='green', linestyle='-.', linewidth=2, label=f'Median: {median_val:.3f}')
        ax.legend(fontsize=7)
    
    for idx in range(n_metrics, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'Metric Distributions - {method_display}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    dist_plot_path = os.path.join(output_dir, "distributions.png")
    plt.savefig(dist_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Distribution plots saved to {dist_plot_path}")
    
    # 3. Correlation heatmap with proper display names
    if n_metrics > 1:
        fig, ax = plt.subplots(figsize=(max(10, n_metrics * 0.8), max(8, n_metrics * 0.6)))
        corr_matrix = df.corr()
        
        display_labels = [get_metric_display_name(m, include_direction=True) for m in corr_matrix.columns]
        
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                    square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax,
                    xticklabels=display_labels, yticklabels=display_labels,
                    annot_kws={'fontsize': 9})
        ax.set_title(f'Metric Correlation Matrix - {method_display}')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        corr_plot_path = os.path.join(output_dir, "correlation_matrix.png")
        plt.savefig(corr_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Correlation matrix saved to {corr_plot_path}")
    
    # 4. Summary bar plot with display names
    fig, ax = plt.subplots(figsize=(max(10, n_metrics * 1.2), 6))
    means = df.mean()
    stds = df.std()
    x_pos = np.arange(len(metrics))
    display_names = [get_metric_display_name(m, include_direction=True) for m in metrics]
    
    bars = ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, 
                  color='steelblue', edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Value')
    ax.set_title(f'Mean Metric Values (± Std) - {method_display}')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(display_names, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std,
                f'{mean:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    summary_bar_path = os.path.join(output_dir, "summary_bars.png")
    plt.savefig(summary_bar_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Summary bar plot saved to {summary_bar_path}")
    
    # 5. Per-pair metric visualization (if reasonable number of pairs)
    if len(df) <= 50:
        fig, ax = plt.subplots(figsize=(max(12, len(df) * 0.3), 8))
        df_plot = df.copy()
        df_plot.columns = [get_metric_display_name(m, include_direction=True) for m in df_plot.columns]
        df_plot.index = [f"Pair {i+1}" for i in range(len(df))]
        df_plot.plot(kind='bar', ax=ax, width=0.8)
        ax.set_xlabel('Sample Pairs')
        ax.set_ylabel('Metric Value')
        ax.set_title(f'Per-Pair Metric Values - {method_display}')
        ax.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=90, fontsize=8)
        plt.tight_layout()
        per_pair_path = os.path.join(output_dir, "per_pair_metrics.png")
        plt.savefig(per_pair_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Per-pair metrics saved to {per_pair_path}")


# -----------------------------
# Comparison Visualizations
# -----------------------------

def create_comparison_visualizations(
    results_dict: Dict[str, pd.DataFrame],
    global_metrics_dict: Dict[str, Dict[str, float]],
    output_dir: str,
    our_method: str = "hairport",
):
    """
    Create comparison visualizations between multiple methods.
    
    Args:
        results_dict: Dictionary mapping method names to their per-pair results DataFrames
        global_metrics_dict: Dictionary mapping method names to their global metrics
        output_dir: Directory to save comparison visualizations
        our_method: Method name to highlight and place last in ordering
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create category subdirectories (include 'other' for uncategorized metrics)
    category_dirs = {}
    for category in list(METRIC_CATEGORIES.keys()) + ['other']:
        cat_dir = os.path.join(output_dir, category)
        os.makedirs(cat_dir, exist_ok=True)
        category_dirs[category] = cat_dir
    
    sns.set_style("whitegrid")
    _configure_professional_fonts()
    
    methods = get_ordered_methods(list(results_dict.keys()), our_method)
    colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))
    method_colors = {m: colors[i] for i, m in enumerate(methods)}
    
    method_display_names = {m: get_method_display_name(m) for m in methods}
    
    # Get all metric columns (exclude metadata)
    metadata_cols = ['target_id', 'source_id', 'pair_name']
    sample_df = list(results_dict.values())[0]
    all_metric_cols = [col for col in sample_df.columns if col not in metadata_cols]
    
    # Separate global and per-pair metrics
    global_metric_names = GLOBAL_METRICS
    per_pair_metric_cols = [col for col in all_metric_cols if col not in global_metric_names]
    
    # Group metrics by category (include 'other' for uncategorized metrics)
    metrics_by_category = {cat: [] for cat in METRIC_CATEGORIES.keys()}
    metrics_by_category['other'] = []  # For uncategorized metrics
    for metric in per_pair_metric_cols:
        category = get_metric_category(metric)
        if category and category in metrics_by_category:
            metrics_by_category[category].append(metric)
    
    # 1. Create category-specific bar charts
    for category, category_metrics in metrics_by_category.items():
        if not category_metrics:
            continue
            
        cat_dir = category_dirs[category]
        cat_display = category.replace('_', ' ').title()
        
        fig, ax = plt.subplots(figsize=(max(10, len(category_metrics) * 2), 7))
        x = np.arange(len(category_metrics))
        width = 0.8 / len(methods)
        
        for i, method in enumerate(methods):
            df = results_dict[method]
            means = [df[col].mean() if col in df.columns else np.nan for col in category_metrics]
            stds = [df[col].std() if col in df.columns else np.nan for col in category_metrics]
            offset = (i - len(methods)/2 + 0.5) * width
            ax.bar(x + offset, means, width, yerr=stds, label=method_display_names[method],
                   color=method_colors[method], capsize=3, alpha=0.8)
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Value')
        ax.set_title(f'{cat_display} Metrics Comparison (Mean ± Std)')
        ax.set_xticks(x)
        ax.set_xticklabels([get_metric_display_name(m, include_direction=True) for m in category_metrics], 
                          rotation=45, ha='right')
        ax.legend(title='Method')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(cat_dir, f'{category}_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {os.path.join(cat_dir, f'{category}_comparison.png')}")
    
    # 2. Overall per-pair metrics comparison
    fig, ax = plt.subplots(figsize=(max(14, len(per_pair_metric_cols) * 1.5), 7))
    x = np.arange(len(per_pair_metric_cols))
    width = 0.8 / len(methods)
    
    for i, method in enumerate(methods):
        df = results_dict[method]
        means = [df[col].mean() if col in df.columns else np.nan for col in per_pair_metric_cols]
        stds = [df[col].std() if col in df.columns else np.nan for col in per_pair_metric_cols]
        offset = (i - len(methods)/2 + 0.5) * width
        ax.bar(x + offset, means, width, yerr=stds, label=method_display_names[method],
               color=method_colors[method], capsize=3, alpha=0.8)
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Value')
    ax.set_title('All Per-Pair Metrics Comparison (Mean ± Std)')
    ax.set_xticks(x)
    ax.set_xticklabels([get_metric_display_name(m, include_direction=True) for m in per_pair_metric_cols], 
                      rotation=45, ha='right')
    ax.legend(title='Method')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_all_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.join(output_dir, 'comparison_all_metrics.png')}")
    
    # 3. Global metrics comparison bar plot
    if global_metrics_dict:
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(global_metric_names))
        width = 0.8 / len(methods)
        
        for i, method in enumerate(methods):
            if method in global_metrics_dict:
                values = [global_metrics_dict[method].get(m, 0) for m in global_metric_names]
                offset = (i - len(methods)/2 + 0.5) * width
                ax.bar(x + offset, values, width, label=method_display_names[method], 
                       color=method_colors[method], alpha=0.8)
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Value')
        ax.set_title('Global Metrics Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels([get_metric_display_name(m, include_direction=True) for m in global_metric_names])
        ax.legend(title='Method')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'comparison_global_metrics.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {os.path.join(output_dir, 'comparison_global_metrics.png')}")
    
    # 4. Box plots comparison for each metric (organized by category)
    for category, category_metrics in metrics_by_category.items():
        cat_dir = category_dirs[category]
        for metric in category_metrics:
            fig, ax = plt.subplots(figsize=(8, 6))
            data_to_plot = []
            labels = []
            methods_with_data = []  # Track methods that have this metric
            for method in methods:
                if metric in results_dict[method].columns:
                    data_to_plot.append(results_dict[method][metric].dropna().values)
                    labels.append(method_display_names[method])
                    methods_with_data.append(method)
            
            if not data_to_plot:  # Skip if no data for this metric
                plt.close()
                continue
                
            bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
            for patch, method in zip(bp['boxes'], methods_with_data):
                patch.set_facecolor(method_colors[method])
                patch.set_alpha(0.7)
            
            metric_display = get_metric_display_name(metric, include_direction=True)
            ax.set_ylabel(metric_display)
            ax.set_title(f'{metric_display} - Method Comparison')
            ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig(os.path.join(cat_dir, f'boxplot_{metric}.png'), dpi=300, bbox_inches='tight')
            plt.close()
    print(f"Saved box plots for all metrics (organized by category)")
    
    # 5. Radar/Spider chart for normalized metrics
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Filter out metrics with NaN values for radar chart
    valid_radar_metrics = []
    for metric in per_pair_metric_cols:
        has_valid_data = True
        for method in methods:
            # Check if column exists and has valid data
            if metric not in results_dict[method].columns:
                has_valid_data = False
                break
            if results_dict[method][metric].isna().all():
                has_valid_data = False
                break
        if has_valid_data:
            valid_radar_metrics.append(metric)
    
    if len(valid_radar_metrics) > 2:
        radar_metrics = valid_radar_metrics
        angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()
        angles += angles[:1]
        
        for method in methods:
            df = results_dict[method]
            values = []
            for metric in radar_metrics:
                if metric not in df.columns:
                    values.append(0.5)  # Default value for missing metrics
                    continue
                val = df[metric].mean(skipna=True)
                all_vals = []
                for m in methods:
                    if metric in results_dict[m].columns:
                        v = results_dict[m][metric].mean(skipna=True)
                        if not np.isnan(v):
                            all_vals.append(v)
                if len(all_vals) == 0:
                    norm_val = 0.5
                else:
                    min_val, max_val = min(all_vals), max(all_vals)
                    if max_val - min_val > 0:
                        norm_val = (val - min_val) / (max_val - min_val)
                    else:
                        norm_val = 0.5
                # Invert for "lower is better" metrics
                if not is_higher_better(metric):
                    norm_val = 1 - norm_val
                values.append(norm_val if not np.isnan(norm_val) else 0.5)
            values += values[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, label=method_display_names[method], 
                   color=method_colors[method])
            ax.fill(angles, values, alpha=0.25, color=method_colors[method])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([get_metric_display_name(m, include_direction=True) for m in radar_metrics])
        ax.set_title('Normalized Metrics Radar Chart\n(Higher = Better)', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'comparison_radar_chart.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {os.path.join(output_dir, 'comparison_radar_chart.png')}")
    else:
        plt.close()
        print("Skipping radar chart (not enough valid metrics)")
    
    # 6. Create comparison tables
    _create_comparison_tables(results_dict, global_metrics_dict, output_dir, methods, 
                             method_display_names, global_metric_names, per_pair_metric_cols,
                             metrics_by_category, category_dirs)


def _create_comparison_tables(
    results_dict: Dict[str, pd.DataFrame],
    global_metrics_dict: Dict[str, Dict[str, float]],
    output_dir: str,
    methods: List[str],
    method_display_names: Dict[str, str],
    global_metric_names: List[str],
    per_pair_metric_cols: List[str],
    metrics_by_category: Dict[str, List[str]],
    category_dirs: Dict[str, str],
):
    """Create comparison tables and summary files."""
    
    # Overall comparison table
    comparison_data = []
    for method in methods:
        df = results_dict[method]
        row = {'Method': method_display_names[method]}
        if method in global_metrics_dict:
            for gm in global_metric_names:
                display_name = get_metric_display_name(gm, include_direction=True)
                row[display_name] = f"{global_metrics_dict[method].get(gm, 0):.4f}"
        for metric in per_pair_metric_cols:
            display_name = get_metric_display_name(metric, include_direction=True)
            if metric in df.columns:
                mean_val = df[metric].mean()
                std_val = df[metric].std()
                row[display_name] = f"{mean_val:.4f} ± {std_val:.4f}"
            else:
                row[display_name] = "N/A"
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(os.path.join(output_dir, 'comparison_table.csv'), index=False)
    print(f"Saved: {os.path.join(output_dir, 'comparison_table.csv')}")
    
    latex_str = comparison_df.to_latex(index=False, escape=False)
    with open(os.path.join(output_dir, 'comparison_table.tex'), 'w') as f:
        f.write(latex_str)
    print(f"Saved: {os.path.join(output_dir, 'comparison_table.tex')}")
    
    # Category-specific tables
    for category, category_metrics in metrics_by_category.items():
        if not category_metrics:
            continue
        cat_dir = category_dirs[category]
        cat_data = []
        for method in methods:
            df = results_dict[method]
            row = {'Method': method_display_names[method]}
            for metric in category_metrics:
                display_name = get_metric_display_name(metric, include_direction=True)
                if metric in df.columns:
                    mean_val = df[metric].mean()
                    std_val = df[metric].std()
                    row[display_name] = f"{mean_val:.4f} ± {std_val:.4f}"
                else:
                    row[display_name] = "N/A"
            cat_data.append(row)
        cat_df = pd.DataFrame(cat_data)
        cat_df.to_csv(os.path.join(cat_dir, f'{category}_table.csv'), index=False)
        print(f"Saved: {os.path.join(cat_dir, f'{category}_table.csv')}")
    
    # Summary text file
    with open(os.path.join(output_dir, 'comparison_summary.txt'), 'w') as f:
        f.write("="*80 + "\n")
        f.write("METHODS COMPARISON SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        f.write("Methods Evaluated:\n")
        f.write("-"*40 + "\n")
        for method in methods:
            f.write(f"  • {method} → {method_display_names[method]}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("Global Metrics:\n")
        f.write("-"*40 + "\n")
        for gm in global_metric_names:
            display_name = get_metric_display_name(gm, include_direction=True)
            f.write(f"\n{display_name}:\n")
            for method in methods:
                if method in global_metrics_dict:
                    val = global_metrics_dict[method].get(gm, 0)
                    f.write(f"  {method_display_names[method]:20s}: {val:.4f}\n")
        
        # Per-category results
        for category, category_metrics in metrics_by_category.items():
            if not category_metrics:
                continue
            cat_display = category.replace('_', ' ').upper()
            f.write("\n" + "="*80 + "\n")
            f.write(f"{cat_display} METRICS (Mean ± Std):\n")
            f.write("-"*40 + "\n")
            for metric in category_metrics:
                display_name = get_metric_display_name(metric, include_direction=True)
                f.write(f"\n{display_name}:\n")
                for method in methods:
                    df = results_dict[method]
                    if metric in df.columns:
                        mean_val = df[metric].mean()
                        std_val = df[metric].std()
                        f.write(f"  {method_display_names[method]:20s}: {mean_val:.4f} ± {std_val:.4f}\n")
                    else:
                        f.write(f"  {method_display_names[method]:20s}: N/A\n")
        
        # Determine best method for each metric
        f.write("\n" + "="*80 + "\n")
        f.write("BEST METHOD PER METRIC:\n")
        f.write("-"*40 + "\n")
        for category, category_metrics in metrics_by_category.items():
            if not category_metrics:
                continue
            cat_display = category.replace('_', ' ').title()
            f.write(f"\n[{cat_display}]\n")
            for metric in category_metrics:
                display_name = get_metric_display_name(metric, include_direction=True)
                means = {}
                for m in methods:
                    if metric in results_dict[m].columns:
                        val = results_dict[m][metric].mean(skipna=True)
                        if not np.isnan(val):
                            means[m] = val
                if not means:
                    f.write(f"  {display_name:35s}: No valid data\n")
                    continue
                if is_higher_better(metric):
                    best = max(means, key=means.get)
                    f.write(f"  {display_name:35s}: {method_display_names[best]} (best = {means[best]:.4f})\n")
                else:
                    best = min(means, key=means.get)
                    f.write(f"  {display_name:35s}: {method_display_names[best]} (best = {means[best]:.4f})\n")
    
    print(f"Saved: {os.path.join(output_dir, 'comparison_summary.txt')}")


# -----------------------------
# Summary Statistics
# -----------------------------

def save_summary_statistics(df: pd.DataFrame, output_dir: str, method: str = None):
    """
    Save summary statistics to CSV and text files with proper display names.
    """
    summary = df.describe()
    
    summary_csv_path = os.path.join(output_dir, "summary_statistics.csv")
    summary.to_csv(summary_csv_path)
    
    method_display = get_method_display_name(method) if method else "Unknown"
    
    summary_txt_path = os.path.join(output_dir, "summary_statistics.txt")
    with open(summary_txt_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"SUMMARY STATISTICS - {method_display}\n")
        f.write("="*80 + "\n\n")
        
        for category, info in METRIC_CATEGORIES.items():
            category_metrics = [m for m in df.columns if m in info['metrics']]
            if category_metrics:
                f.write(f"\n{info['display_name']}\n")
                f.write("-"*40 + "\n")
                f.write(f"{info['description']}\n\n")
                for metric in category_metrics:
                    if metric in df.columns:
                        mean_val = df[metric].mean()
                        std_val = df[metric].std()
                        display_name = get_metric_display_name(metric)
                        direction = "↑" if is_higher_better(metric) else "↓"
                        f.write(f"  {display_name:30s}: {mean_val:8.4f} ± {std_val:8.4f} {direction}\n")
                f.write("\n")
        
        uncategorized = [m for m in df.columns if get_metric_category(m) is None]
        if uncategorized:
            f.write("\nOther Metrics\n")
            f.write("-"*40 + "\n")
            for metric in uncategorized:
                mean_val = df[metric].mean()
                std_val = df[metric].std()
                display_name = get_metric_display_name(metric)
                f.write(f"  {display_name:30s}: {mean_val:8.4f} ± {std_val:8.4f}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("Legend: ↑ = higher is better, ↓ = lower is better\n")
    
    print(f"Summary statistics saved to {summary_csv_path} and {summary_txt_path}")
