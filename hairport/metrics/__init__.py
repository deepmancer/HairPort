"""
Hair Transfer Evaluation Metrics Package.

This package provides a modular evaluation pipeline for hair transfer methods:

Modules:
- metrics: Individual metric implementations (CLIP-I, FID, SSIM, PSNR, LPIPS, etc.)
- visualizers: Visualization utilities for single methods and comparisons
- latex_builder: Publication-quality LaTeX table and report generation
- metric_manager: Main evaluation pipeline orchestrating all components

Usage:
    from hairport.metrics import evaluate_and_compare, Sample
    
    # Run evaluation on multiple methods
    results, global_metrics = evaluate_and_compare(
        output_dir="/path/to/evaluation",
        methods=['hairfastgan', 'stablehair', 'hairport'],
        device="cuda",
    )

Output Directory Structure:
    evaluation/
    ├── hairport/              # All HairPort variant results
    │   ├── hairport_hi3dgen/
    │   ├── hairport_direct3d_s2/
    │   └── ...
    ├── baselines/             # Baseline method results
    │   ├── hairfastgan/
    │   ├── stablehair/
    │   └── ...
    ├── comparison/            # Cross-method comparison visualizations
    └── latex_report/          # Publication-ready LaTeX tables and figures
        ├── main.tex
        ├── tables/
        └── figures/
"""

# Import main data structures
from .metrics import (
    Sample,
    Metric,
    MetricSuite,
    CLIPIMetric,
    FIDMetric,
    SSIMMetric,
    PSNRMetric,
    LPIPSMetric,
    DreamSimMetric,
    IDSMetric,
    DINOv3HairSimilarityMetric,
    PixioHairSimilarityMetric,
)

# Import visualization functions
from .visualizers import (
    create_method_visualizations,
    create_comparison_visualizations,
    save_summary_statistics,
)

# Import LaTeX generation functions
from .latex_builder import (
    generate_latex_main_table,
    generate_latex_category_tables,
    generate_latex_compact_table,
    generate_all_latex_outputs,
    get_metric_display_name,
    get_method_display_name,
    is_higher_better,
    METRIC_CATEGORIES,
)

# Import main evaluation functions
from .metric_manager import (
    evaluate_method,
    evaluate_and_compare,
    SAMMaskExtractor,
    PATHS,
    BASE_DIR,
    get_all_available_methods,
    is_hairport_method,
)

__all__ = [
    # Data structures
    'Sample',
    'Metric',
    'MetricSuite',
    
    # Metric classes
    'CLIPIMetric',
    'FIDMetric',
    'SSIMMetric',
    'PSNRMetric',
    'LPIPSMetric',
    'DreamSimMetric',
    'IDSMetric',
    'DINOv3HairSimilarityMetric',
    'PixioHairSimilarityMetric',
    
    # Visualization functions
    'create_method_visualizations',
    'create_comparison_visualizations',
    'save_summary_statistics',
    
    # LaTeX generation
    'generate_latex_main_table',
    'generate_latex_category_tables',
    'generate_latex_compact_table',
    'generate_all_latex_outputs',
    'get_metric_display_name',
    'get_method_display_name',
    'is_higher_better',
    'METRIC_CATEGORIES',
    
    # Evaluation pipeline
    'evaluate_method',
    'evaluate_and_compare',
    'SAMMaskExtractor',
    'PATHS',
    'BASE_DIR',
    'get_all_available_methods',
    'is_hairport_method',
]
