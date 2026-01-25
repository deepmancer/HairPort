import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from scipy.ndimage import map_coordinates

import warnings

def create_color_palette(n_colors, colormap='tab10'):
    """
    Generate a palette of distinct colors with improved diversity and distinguishability.
    
    Args:
        n_colors: Number of colors to generate
        colormap: Base colormap to use ('tab10', 'tab20', 'Set1', 'Set2', 'Set3', 'Paired', 
                 'Dark2', 'Accent', 'plasma', 'viridis', 'hsv', 'rainbow')
    
    Returns:
        List of color tuples (R, G, B, A) or color strings
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    
    # For small numbers, use high-quality qualitative palettes
    if n_colors <= 10:
        if colormap == 'tab10' or colormap == 'default':
            cmap = plt.cm.get_cmap('tab10')
            return [cmap(i) for i in range(n_colors)]
        elif colormap in ['Set1', 'Set2', 'Set3', 'Dark2', 'Accent', 'Paired']:
            cmap = plt.cm.get_cmap(colormap)
            return [cmap(i / max(1, n_colors - 1)) for i in range(n_colors)]
    
    # For medium numbers, use tab20 or extended qualitative palettes
    elif n_colors <= 20:
        if colormap == 'tab20':
            cmap = plt.cm.get_cmap('tab20')
            return [cmap(i) for i in range(n_colors)]
        elif colormap == 'Paired':
            cmap = plt.cm.get_cmap('Paired')
            return [cmap(i) for i in range(min(n_colors, 12))]
    
    # For large numbers, use perceptually uniform or HSV-based generation
    if n_colors > 20:
        if colormap == 'hsv' or colormap == 'rainbow':
            # HSV color space for maximum distinguishability
            hues = np.linspace(0, 1, n_colors, endpoint=False)
            # Add slight randomization to avoid too-regular patterns
            np.random.shuffle(hues)
            colors = []
            for h in hues:
                # Vary saturation and value slightly for more diversity
                s = 0.7 + 0.3 * np.random.random()  # 0.7-1.0
                v = 0.6 + 0.4 * np.random.random()  # 0.6-1.0
                rgb = mcolors.hsv_to_rgb([h, s, v])
                colors.append((*rgb, 1.0))
            return colors
        
        elif colormap in ['plasma', 'viridis', 'inferno', 'magma', 'cividis']:
            # Perceptually uniform colormaps
            cmap = plt.cm.get_cmap(colormap)
            # Skip very dark/light ends for better distinguishability
            indices = np.linspace(0.1, 0.9, n_colors)
            return [cmap(i) for i in indices]
        
        elif colormap == 'golden_ratio':
            # Golden ratio based hue distribution for optimal spacing
            golden_ratio = (1 + 5**0.5) / 2
            colors = []
            for i in range(n_colors):
                hue = (i / golden_ratio) % 1
                # Alternate between high and medium saturation/value
                sat = 0.8 if i % 2 == 0 else 0.6
                val = 0.9 if i % 3 != 0 else 0.7
                rgb = mcolors.hsv_to_rgb([hue, sat, val])
                colors.append((*rgb, 1.0))
            return colors
    
    # Fallback: use the specified colormap with even distribution
    try:
        cmap = plt.cm.get_cmap(colormap)
        if n_colors == 1:
            return [cmap(0.5)]
        return [cmap(i / (n_colors - 1)) for i in range(n_colors)]
    except:
        # Ultimate fallback: simple rainbow
        return [plt.cm.hsv(i / max(1, n_colors - 1)) for i in range(n_colors)]

def visualize_head_with_strands_3d(head_mesh, strands, title="Head Mesh with Hair Strands",
                                    head_color='lightblue', head_opacity=0.95,
                                    hue=None, hue_legend=None, strand_opacity=0.8,
                                    n_strands=200, n_points_per_strand=50, figsize=(14, 10),
                                    show_head_wireframe=False, wireframe_color='gray'):
    """
    Visualize a head mesh with randomly sampled hair strands in an interactive 3D plot.
    
    Args:
        head_mesh: trimesh.Trimesh object representing the head
        strands: list of numpy arrays, each array is shape (N, 3) representing a strand
        title: plot title
        head_color: color of the head mesh
        head_opacity: opacity of the head mesh (0-1)
        hue: numpy array or torch tensor of shape (N,) with integer cluster IDs for each strand
        hue_legend: list of strings with length equal to unique count of hue values, for custom legend labels
        strand_opacity: opacity of the strands (0-1)
        n_strands: number of strands to randomly sample and display
        n_points_per_strand: number of points to sample along each strand (not used in this function, but can be useful for future extensions)
        figsize: figure size tuple
        show_head_wireframe: whether to show head mesh wireframe
        wireframe_color: color of wireframe edges
    """
    import plotly.graph_objects as go
    import re
    
    # Randomly sample strands
    if hasattr(strands, 'cpu'):
        strands = strands.cpu().numpy()

    if isinstance(strands, np.ndarray) and len(strands.shape) == 3:
        strands = [strands[i] for i in range(strands.shape[0])]

    n_available = len(strands)
    n_to_sample = min(n_strands, n_available)
    sampled_indices = np.random.choice(n_available, n_to_sample, replace=False)
    sampled_strands = [strands[i] for i in sampled_indices]
    
    # Process hue parameter for color assignment
    if hue is not None:
        # Convert to numpy if needed
        if hasattr(hue, 'cpu'):
            hue = hue.cpu().numpy()
        
        # Sample the hue values corresponding to sampled strands
        sampled_hue = hue[sampled_indices]
        
        # Get unique cluster IDs and create color palette
        unique_clusters = np.unique(sampled_hue)
        n_clusters = len(unique_clusters)
        cluster_colors = create_color_palette(n_clusters, 'tab10')
        
        # Create cluster label mapping and analyze hierarchy
        hierarchical_legend = False
        cluster_hierarchy = {}
        
        if hue_legend is not None:
            if len(hue_legend) != n_clusters:
                raise ValueError(f"hue_legend length ({len(hue_legend)}) must match number of unique clusters ({n_clusters})")
            
            # Check if legend contains hierarchical cluster pattern
            cluster_pattern = re.compile(r'(.+) - Group (.+)$')
            
            for i, cluster_id in enumerate(unique_clusters):
                legend_title = hue_legend[i]
                match = cluster_pattern.match(legend_title)
                
                if match:
                    hierarchical_legend = True
                    prefix = match.group(1)
                    cluster_num = match.group(2)
                    
                    if cluster_num not in cluster_hierarchy:
                        cluster_hierarchy[cluster_num] = {}
                    
                    cluster_hierarchy[cluster_num][cluster_id] = {
                        'prefix': prefix,
                        'full_title': legend_title,
                        'color_idx': i
                    }
            
            # Create cluster label mapping
            if hierarchical_legend:
                cluster_label_map = {cluster_id: hue_legend[i] for i, cluster_id in enumerate(unique_clusters)}
            else:
                cluster_label_map = {cluster_id: hue_legend[i] for i, cluster_id in enumerate(unique_clusters)}
        else:
            cluster_label_map = {cluster_id: f"Cluster {cluster_id}" for cluster_id in unique_clusters}
        
        # Create color mapping from cluster ID to color
        color_map = {cluster_id: cluster_colors[i] for i, cluster_id in enumerate(unique_clusters)}
        strand_colors = [color_map[cluster_id] for cluster_id in sampled_hue]
    else:
        # Use default random colors if no hue provided
        strand_colors = create_color_palette(n_to_sample, 'plasma')
        hierarchical_legend = False
    
    # Extract head mesh vertices and faces
    head_vertices = head_mesh.vertices
    head_faces = head_mesh.faces
    
    # Create head mesh trace
    head_trace = go.Mesh3d(
        x=head_vertices[:, 0],
        y=head_vertices[:, 1],
        z=head_vertices[:, 2],
        i=head_faces[:, 0],
        j=head_faces[:, 1],
        k=head_faces[:, 2],
        color=head_color,
        opacity=head_opacity,
        name='Head Mesh',
        hovertemplate='<b>Head</b><br>' +
                        'X: %{x:.3f}<br>' +
                        'Y: %{y:.3f}<br>' +
                        'Z: %{z:.3f}<extra></extra>'
    )
    
    # Create figure starting with head mesh
    fig = go.Figure(data=[head_trace])
    
    # Add head wireframe if requested
    if show_head_wireframe:
        edge_trace = []
        for face in head_faces:
            for i in range(3):
                v1 = head_vertices[face[i]]
                v2 = head_vertices[face[(i + 1) % 3]]
                edge_trace.extend([v1, v2, [None, None, None]])
        
        edge_array = np.array(edge_trace)
        
        wireframe_trace = go.Scatter3d(
            x=edge_array[:, 0],
            y=edge_array[:, 1],
            z=edge_array[:, 2],
            mode='lines',
            line=dict(color=wireframe_color, width=1),
            name='Head Wireframe',
            hoverinfo='skip',
            showlegend=False
        )
        fig.add_trace(wireframe_trace)
    
    # Add group header traces for hierarchical legend
    group_headers_added = set()
    if hierarchical_legend:
        for cluster_num in sorted(cluster_hierarchy.keys()):
            main_group = f"Cluster {cluster_num}"
            if main_group not in group_headers_added:
                # Add invisible trace to serve as group header
                header_trace = go.Scatter3d(
                    x=[None], y=[None], z=[None],
                    mode='markers',
                    marker=dict(size=0, opacity=0),
                    name=f"<b>{main_group}</b>",
                    legendgroup=main_group,
                    showlegend=True,
                    hoverinfo='skip'
                )
                fig.add_trace(header_trace)
                group_headers_added.add(main_group)
    
    # Track which clusters have been added to legend to avoid duplicates
    if hierarchical_legend:
        # For hierarchical legend, track by (cluster_num, prefix) combinations
        legend_groups_added = set()
    else:
        # For regular legend, track by cluster_id
        clusters_in_legend = set()
    
    # Add each sampled strand
    for i, strand in enumerate(sampled_strands):
        color = strand_colors[i % len(strand_colors)]
        if isinstance(color, tuple) and len(color) >= 3:
            color_str = f'rgba({int(color[0]*255)},{int(color[1]*255)},{int(color[2]*255)},{strand_opacity})'
        else:
            color_str = color
        
        
        # Downsample strand points if needed, preserving geometry
        if len(strand) > n_points_per_strand and n_points_per_strand > 2:
            # Always keep root (first point) and tip (last point)
            root_point = strand[0:1]  # Keep as 2D array
            tip_point = strand[-1:]   # Keep as 2D array
            
            if n_points_per_strand == 2:
                # Only root and tip
                strand = np.vstack([root_point, tip_point])
            else:
                # Geometry-aware downsampling for intermediate points
                middle_points_needed = n_points_per_strand - 2
                
                # Calculate cumulative arc length along the strand
                segments = np.diff(strand, axis=0)
                segment_lengths = np.linalg.norm(segments, axis=1)
                cumulative_lengths = np.concatenate([[0], np.cumsum(segment_lengths)])
                total_length = cumulative_lengths[-1]
                
                if total_length > 0:
                    # Sample points at uniform arc length intervals
                    target_lengths = np.linspace(0, total_length, n_points_per_strand)
                    
                    # Find indices and interpolation weights for target lengths
                    sampled_points = []
                    
                    for target_length in target_lengths:
                        if target_length <= 0:
                            # Root point
                            sampled_points.append(strand[0])
                        elif target_length >= total_length:
                            # Tip point
                            sampled_points.append(strand[-1])
                        else:
                            # Interpolate along the curve
                            idx = np.searchsorted(cumulative_lengths, target_length) - 1
                            idx = max(0, min(idx, len(strand) - 2))
                            
                            # Linear interpolation between adjacent points
                            t = (target_length - cumulative_lengths[idx]) / (cumulative_lengths[idx + 1] - cumulative_lengths[idx])
                            t = np.clip(t, 0, 1)
                            
                            interpolated_point = (1 - t) * strand[idx] + t * strand[idx + 1]
                            sampled_points.append(interpolated_point)
                    
                    strand = np.array(sampled_points)
                else:
                    # Degenerate case: all points are the same
                    strand = np.vstack([root_point, tip_point])
        # Determine legend settings
        if hue is not None:
            cluster_id = sampled_hue[i]
            cluster_label = cluster_label_map[cluster_id]
            
            if hierarchical_legend:
                # Find the cluster number and prefix for this cluster_id
                cluster_num = None
                prefix = None
                for cnum, cluster_data in cluster_hierarchy.items():
                    if cluster_id in cluster_data:
                        cluster_num = cnum
                        prefix = cluster_data[cluster_id]['prefix']
                        break
                
                if cluster_num is not None and prefix is not None:
                    # Create hierarchical legend group names
                    main_group = f"Cluster {cluster_num}"
                    sub_group = prefix
                    legend_group_key = (cluster_num, prefix)
                    
                    # Only show legend for first strand of each (cluster_num, prefix) combination
                    show_in_legend = legend_group_key not in legend_groups_added
                    if show_in_legend:
                        legend_groups_added.add(legend_group_key)
                    
                    # Use sub_group as the legend name, but group by main_group
                    legend_name = f"&nbsp;&nbsp;&nbsp;&nbsp;{sub_group}"  # Indent sub-groups
                    legend_group = main_group
                else:
                    # Fallback if pattern doesn't match
                    show_in_legend = cluster_id not in clusters_in_legend if 'clusters_in_legend' in locals() else True
                    legend_name = cluster_label
                    legend_group = cluster_label
                    if 'clusters_in_legend' not in locals():
                        clusters_in_legend = set()
                    if show_in_legend:
                        clusters_in_legend.add(cluster_id)
            else:
                # Regular legend behavior
                show_in_legend = cluster_id not in clusters_in_legend
                if show_in_legend:
                    clusters_in_legend.add(cluster_id)
                    legend_name = cluster_label
                    legend_group = cluster_label
                else:
                    legend_name = cluster_label
                    legend_group = cluster_label
            
            # Hover text includes cluster info
            cluster_info = f" ({cluster_label})"
        else:
            # No hue provided - don't show legend for individual strands
            show_in_legend = False
            legend_name = f'Strand {sampled_indices[i]}'
            legend_group = None
            cluster_info = ""
        
        strand_trace = go.Scatter3d(
            x=strand[:, 0],
            y=strand[:, 1],
            z=strand[:, 2],
            mode='lines+markers',
            line=dict(
                color=color_str,
                width=4
            ),
            marker=dict(
                size=2,
                color=color_str,
                opacity=strand_opacity
            ),
            name=legend_name,
            legendgroup=legend_group if hue is not None else None,
            showlegend=show_in_legend,
            hovertemplate=f'<b>Strand {sampled_indices[i]}{cluster_info}</b><br>' +
                            'Point: %{pointNumber}<br>' +
                            'X: %{x:.3f}<br>' +
                            'Y: %{y:.3f}<br>' +
                            'Z: %{z:.3f}<extra></extra>'
        )
        fig.add_trace(strand_trace)
    
    # Compute combined bounding box
    all_points = [head_vertices]
    all_points.extend(sampled_strands)
    all_vertices = np.vstack(all_points)
    bounds = [all_vertices.min(axis=0), all_vertices.max(axis=0)]
    
    # Calculate strand statistics
    strand_lengths = [len(strand) for strand in sampled_strands] if n_points_per_strand > 0 else [n_points_per_strand] * n_to_sample
    avg_strand_length = np.mean(strand_lengths)
    total_points = sum(strand_lengths)
    
    # Add cluster statistics to title if hue is provided
    cluster_stats = ""
    if hue is not None:
        if hierarchical_legend:
            n_clusters_sampled = len(cluster_hierarchy)
            n_groups_sampled = len(legend_groups_added)
            cluster_stats = f" | Hierarchical Clusters: {n_clusters_sampled}, Groups: {n_groups_sampled}"
        else:
            n_clusters_sampled = len(np.unique(sampled_hue))
            cluster_stats = f" | Clusters: {n_clusters_sampled}"
    
    # Determine if legend should be shown
    show_legend = hue is not None
    
    # Update layout
    fig.update_layout(
        title=f'{title}<br><sub>Head: {len(head_vertices):,}v, {len(head_faces):,}f | Strands: {n_to_sample}/{n_available} | Avg Length: {avg_strand_length:.1f}{cluster_stats}</sub>',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.2, y=1.2, z=1.2)
            ),
            bgcolor='white'
        ),
        width=figsize[0]*100,
        height=figsize[1]*100,
        showlegend=show_legend,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='rgba(0,0,0,0.2)',
            borderwidth=1,
            groupclick="togglegroup" if hierarchical_legend else "togglegroup"
        ) if show_legend else None
    )
    
    fig.show()
    
    return sampled_indices

def plot_histogram(array, title=None, bins=100, figsize=(10, 6), color='skyblue', alpha=0.7, log_scale=False):
    # Flatten array and get non-zero values
    flat_array = array.flatten()
    non_zero_values = flat_array[flat_array != 0]
    
    if len(non_zero_values) == 0:
        print("No non-zero values found in the array.")
        return
    
    plt.figure(figsize=figsize)
    
    # Create histogram
    n, bins_edges, patches = plt.hist(non_zero_values, bins=bins, color=color, alpha=alpha, 
                                     edgecolor='black', linewidth=0.5, log=log_scale)
    
    # Add title
    if title:
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
    else:
        plt.title('Histogram of Non-Zero Values', fontsize=16, fontweight='bold', pad=20)
    
    # Labels and formatting
    plt.xlabel('Value', fontsize=12, fontweight='bold')
    ylabel_text = 'Frequency (log scale)' if log_scale else 'Frequency'
    plt.ylabel(ylabel_text, fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Statistics
    mean_val = np.mean(non_zero_values)
    median_val = np.median(non_zero_values)
    std_val = np.std(non_zero_values)
    min_val = np.min(non_zero_values)
    max_val = np.max(non_zero_values)
    
    # Add vertical lines for mean and median
    plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.4f}')
    plt.axvline(median_val, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_val:.4f}')
    
    # Statistics text box
    stats_text = f'Count: {len(non_zero_values):,}\nMin: {min_val:.4f}\nMax: {max_val:.4f}\nStd: {std_val:.4f}'
    plt.text(0.98, 0.98, stats_text, transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    # Legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.98, 0.7))
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print(f"Non-zero values statistics:")
    print(f"Total count: {len(non_zero_values):,}")
    print(f"Percentage of non-zero values: {100 * len(non_zero_values) / len(flat_array):.2f}%")
    print(f"Range: [{min_val:.4f}, {max_val:.4f}]")
    if log_scale:
        print(f"Note: Histogram displayed with logarithmic y-axis")

def visualize_tsne_result(tsne_embeddings, hue=None, hue_legend=None, title="t-SNE Embeddings Visualization",
                         point_size=6, alpha=0.8, show_density=True, show_stats=True,
                         colormap='tab10', figsize=(12, 10), show_legend=True, existing_fig=None,
                         density_contours=True, marginal_plots=False, n_points=10000):
    """
    Create an interactive visualization of t-SNE embeddings with optional cluster coloring.
    
    Args:
        tsne_embeddings (np.ndarray or torch.Tensor): Shape (N, 2) representing t-SNE coordinates
        hue (np.ndarray or torch.Tensor, optional): Shape (N,) cluster IDs for color coding
        hue_legend (list, optional): Custom titles for each unique cluster in hue. 
                                    Length should match number of unique clusters.
        title (str): Plot title
        point_size (int): Size of scatter plot points
        alpha (float): Point opacity (0-1)
        show_density (bool): Whether to show density estimation overlay
        show_stats (bool): Whether to display statistics annotation
        colormap (str): Plotly colormap name for clusters
        figsize (tuple): Figure size (width, height)
        show_legend (bool): Whether to show legend
        existing_fig (plotly.graph_objects.Figure, optional): Existing figure to add traces to
        density_contours (bool): Whether to show density contours
        marginal_plots (bool): Whether to show marginal distribution plots
        n_points (int): Number of points to sample for downsampling t-SNE embeddings

    Returns:
        dict: Contains figure object and computed statistics
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    from scipy import stats
    from sklearn.neighbors import KernelDensity
    import re
    
    # Convert inputs to numpy arrays
    if hasattr(tsne_embeddings, 'cpu'):
        tsne_embeddings = tsne_embeddings.cpu().numpy()
    tsne_embeddings = np.asarray(tsne_embeddings)

    if tsne_embeddings.shape[1] != 2:
        raise ValueError(f"Expected tsne_embeddings shape (N, 2), got {tsne_embeddings.shape}")
    
    # Process hue parameter
    cluster_ids = None
    if hue is not None:
        if hasattr(hue, 'cpu'):
            hue = hue.cpu().numpy()
        cluster_ids = np.asarray(hue)
        if len(cluster_ids) != len(tsne_embeddings):
            raise ValueError(f"Length mismatch: tsne_embeddings {len(tsne_embeddings)} vs hue {len(cluster_ids)}")
    
    if n_points is not None and len(tsne_embeddings) > n_points:
        # Randomly sample n_points from tsne_embeddings
        indices = np.random.choice(len(tsne_embeddings), n_points, replace=False)
        tsne_embeddings = tsne_embeddings[indices]
        if hue is not None:
            cluster_ids = cluster_ids[indices]
    # Extract coordinates
    x_coords = tsne_embeddings[:, 0]
    y_coords = tsne_embeddings[:, 1]
    
    # Create subplot structure based on marginal_plots option
    if existing_fig is None:
        if marginal_plots:
            fig = make_subplots(
                rows=2, cols=2,
                column_widths=[0.8, 0.2],
                row_heights=[0.2, 0.8],
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                    [{"secondary_y": False}, {"secondary_y": False}]],
                subplot_titles=('X Distribution', 'Y Distribution', '', ''),
                horizontal_spacing=0.02,
                vertical_spacing=0.02
            )
            main_row, main_col = 2, 1
            x_hist_row, x_hist_col = 1, 1
            y_hist_row, y_hist_col = 2, 2
        else:
            fig = go.Figure()
            main_row, main_col = None, None
    else:
        fig = existing_fig
        if marginal_plots:
            raise ValueError("Cannot add marginal plots to an existing figure. Please create a new figure.")
        main_row, main_col = None, None
    
    # Compute statistics
    stats_dict = {
        'n_points': len(tsne_embeddings),
        'x_range': (float(np.min(x_coords)), float(np.max(x_coords))),
        'y_range': (float(np.min(y_coords)), float(np.max(y_coords))),
        'x_std': float(np.std(x_coords)),
        'y_std': float(np.std(y_coords)),
    }
    
    if cluster_ids is not None:
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)
        stats_dict['n_clusters'] = n_clusters
        stats_dict['cluster_distribution'] = {
            int(cluster): int(np.sum(cluster_ids == cluster)) 
            for cluster in unique_clusters
        }
        
        # Create cluster label mapping and analyze hierarchy
        hierarchical_legend = False
        cluster_hierarchy = {}
        
        # Validate hue_legend if provided
        if hue_legend is not None:
            if len(hue_legend) != n_clusters:
                raise ValueError(f"hue_legend length ({len(hue_legend)}) must match number of unique clusters ({n_clusters})")
            
            # Check if legend contains hierarchical cluster pattern
            cluster_pattern = re.compile(r'(.+) - Group (.+)$')
            
            for i, cluster_id in enumerate(unique_clusters):
                legend_title = hue_legend[i]
                match = cluster_pattern.match(legend_title)
                
                if match:
                    hierarchical_legend = True
                    prefix = match.group(1)
                    cluster_num = match.group(2)
                    
                    if cluster_num not in cluster_hierarchy:
                        cluster_hierarchy[cluster_num] = {}
                    
                    cluster_hierarchy[cluster_num][cluster_id] = {
                        'prefix': prefix,
                        'full_title': legend_title,
                        'color_idx': i
                    }
            
            # Create cluster label mapping
            cluster_title_map = {cluster_id: hue_legend[i] for i, cluster_id in enumerate(unique_clusters)}
        else:
            cluster_title_map = {cluster: f"Cluster {cluster}" for cluster in unique_clusters}
        
        # Create color mapping
        colors = px.colors.qualitative.Set1 if n_clusters <= 9 else px.colors.qualitative.Plotly
        if colormap in px.colors.qualitative.__dict__:
            colors = getattr(px.colors.qualitative, colormap)
        
        color_map = {cluster: colors[i % len(colors)] for i, cluster in enumerate(unique_clusters)}
        
        # Add group header traces for hierarchical legend
        group_headers_added = set()
        if hierarchical_legend:
            for cluster_num in sorted(cluster_hierarchy.keys()):
                main_group = f"Cluster {cluster_num}"
                if main_group not in group_headers_added:
                    # Add invisible trace to serve as group header
                    header_trace = go.Scatter(
                        x=[None], y=[None],
                        mode='markers',
                        marker=dict(size=0, opacity=0),
                        name=f"<b>{main_group}</b>",
                        legendgroup=main_group,
                        showlegend=True,
                        hoverinfo='skip'
                    )
                    
                    if marginal_plots:
                        fig.add_trace(header_trace, row=main_row, col=main_col)
                    else:
                        fig.add_trace(header_trace)
                    group_headers_added.add(main_group)
        
        # Track which clusters have been added to legend to avoid duplicates
        if hierarchical_legend:
            # For hierarchical legend, track by (cluster_num, prefix) combinations
            legend_groups_added = set()
        else:
            # For regular legend, track by cluster_id
            clusters_in_legend = set()
        
        # Create main scatter plot with clusters
        for i, cluster in enumerate(unique_clusters):
            mask = cluster_ids == cluster
            cluster_x = x_coords[mask]
            cluster_y = y_coords[mask]
            cluster_count = len(cluster_x)
            cluster_title = cluster_title_map[cluster]
            
            # Determine legend settings for hierarchical clustering
            if hierarchical_legend:
                # Find the cluster number and prefix for this cluster_id
                cluster_num = None
                prefix = None
                for cnum, cluster_data in cluster_hierarchy.items():
                    if cluster in cluster_data:
                        cluster_num = cnum
                        prefix = cluster_data[cluster]['prefix']
                        break
                
                if cluster_num is not None and prefix is not None:
                    # Create hierarchical legend group names
                    main_group = f"Cluster {cluster_num}"
                    sub_group = prefix
                    legend_group_key = (cluster_num, prefix)
                    
                    # Only show legend for first occurrence of each (cluster_num, prefix) combination
                    show_in_legend = legend_group_key not in legend_groups_added
                    if show_in_legend:
                        legend_groups_added.add(legend_group_key)
                    
                    # Use sub_group as the legend name, but group by main_group
                    legend_name = f"&nbsp;&nbsp;&nbsp;&nbsp;{sub_group}"  # Indent sub-groups
                    legend_group = main_group
                else:
                    # Fallback if pattern doesn't match
                    show_in_legend = cluster not in clusters_in_legend if 'clusters_in_legend' in locals() else True
                    legend_name = f'{cluster_title} (n={cluster_count})'
                    legend_group = cluster_title
                    if 'clusters_in_legend' not in locals():
                        clusters_in_legend = set()
                    if show_in_legend:
                        clusters_in_legend.add(cluster)
            else:
                # Regular legend behavior
                show_in_legend = cluster not in clusters_in_legend
                if show_in_legend:
                    clusters_in_legend.add(cluster)
                legend_name = f'{cluster_title} (n={cluster_count})'
                legend_group = cluster_title
            
            # Create hover text with detailed information
            hover_text = [
                f'{cluster_title}<br>'
                f'Point: {j}<br>'
                f'X: {x:.3f}<br>'
                f'Y: {y:.3f}<br>'
                f'Cluster Size: {cluster_count}'
                for j, (x, y) in enumerate(zip(cluster_x, cluster_y))
            ]
            
            scatter_trace = go.Scatter(
                x=cluster_x,
                y=cluster_y,
                mode='markers',
                marker=dict(
                    size=point_size,
                    color=color_map[cluster],
                    opacity=alpha,
                    line=dict(width=0.5, color='white'),
                ),
                name=legend_name,
                legendgroup=legend_group if hierarchical_legend else cluster_title,
                showlegend=show_in_legend and show_legend,
                hovertemplate='%{text}<extra></extra>',
                text=hover_text
            )
            
            if marginal_plots:
                fig.add_trace(scatter_trace, row=main_row, col=main_col)
            else:
                fig.add_trace(scatter_trace)
                
    else:
        # Single color scatter plot
        hover_text = [
            f'Point: {i}<br>X: {x:.3f}<br>Y: {y:.3f}'
            for i, (x, y) in enumerate(zip(x_coords, y_coords))
        ]
        
        scatter_trace = go.Scatter(
            x=x_coords,
            y=y_coords,
            mode='markers',
            marker=dict(
                size=point_size,
                color='blue',
                opacity=alpha,
                line=dict(width=0.5, color='white'),
            ),
            name=f'Embeddings (n={len(tsne_embeddings)})',
            hovertemplate='%{text}<extra></extra>',
            text=hover_text,
            showlegend=show_legend
        )
        
        if marginal_plots:
            fig.add_trace(scatter_trace, row=main_row, col=main_col)
        else:
            fig.add_trace(scatter_trace)
    
    # Add density contours if requested
    if density_contours and show_density:
        try:
            # Create density estimation
            kde = KernelDensity(kernel='gaussian', bandwidth=0.5)
            kde.fit(tsne_embeddings)
            
            # Create grid for contour plotting
            x_min, x_max = np.min(x_coords), np.max(x_coords)
            y_min, y_max = np.min(y_coords), np.max(y_coords)
            x_margin = (x_max - x_min) * 0.1
            y_margin = (y_max - y_min) * 0.1
            
            xx, yy = np.meshgrid(
                np.linspace(x_min - x_margin, x_max + x_margin, 50),
                np.linspace(y_min - y_margin, y_max + y_margin, 50)
            )
            grid_points = np.c_[xx.ravel(), yy.ravel()]
            density = np.exp(kde.score_samples(grid_points)).reshape(xx.shape)
            
            contour_trace = go.Contour(
                x=xx[0],
                y=yy[:, 0],
                z=density,
                showscale=False,
                opacity=0.3,
                colorscale='Viridis',
                name='Density',
                hoverinfo='skip',
                showlegend=False,
                contours=dict(
                    showlabels=False,
                    start=np.min(density),
                    end=np.max(density),
                    size=(np.max(density) - np.min(density)) / 8
                )
            )
            
            if marginal_plots:
                fig.add_trace(contour_trace, row=main_row, col=main_col)
            else:
                fig.add_trace(contour_trace)
                
        except Exception as e:
            print(f"Warning: Could not generate density contours: {e}")
    
    # Add marginal plots if requested
    if marginal_plots and cluster_ids is not None:
        # X-axis marginal histogram
        for cluster in unique_clusters:
            mask = cluster_ids == cluster
            cluster_title = cluster_title_map[cluster]
            fig.add_trace(
                go.Histogram(
                    x=x_coords[mask],
                    name=cluster_title,
                    marker_color=color_map[cluster],
                    opacity=0.7,
                    showlegend=False,
                    nbinsx=30
                ),
                row=x_hist_row, col=x_hist_col
            )
        
        # Y-axis marginal histogram
        for cluster in unique_clusters:
            mask = cluster_ids == cluster
            cluster_title = cluster_title_map[cluster]
            fig.add_trace(
                go.Histogram(
                    y=y_coords[mask],
                    name=cluster_title,
                    marker_color=color_map[cluster],
                    opacity=0.7,
                    showlegend=False,
                    nbinsy=30
                ),
                row=y_hist_row, col=y_hist_col
            )
    elif marginal_plots:
        # Single color marginal plots
        fig.add_trace(
            go.Histogram(
                x=x_coords,
                marker_color='blue',
                opacity=0.7,
                showlegend=False,
                nbinsx=30
            ),
            row=x_hist_row, col=x_hist_col
        )
        
        fig.add_trace(
            go.Histogram(
                y=y_coords,
                marker_color='blue',
                opacity=0.7,
                showlegend=False,
                nbinsy=30
            ),
            row=y_hist_row, col=y_hist_col
        )
    
    # Create statistics annotation text
    if show_stats:
        stats_text = f"<b>Statistics</b><br>"
        stats_text += f"Points: {stats_dict['n_points']:,}<br>"
        stats_text += f"X range: [{stats_dict['x_range'][0]:.2f}, {stats_dict['x_range'][1]:.2f}]<br>"
        stats_text += f"Y range: [{stats_dict['y_range'][0]:.2f}, {stats_dict['y_range'][1]:.2f}]<br>"
        stats_text += f"X std: {stats_dict['x_std']:.3f}<br>"
        stats_text += f"Y std: {stats_dict['y_std']:.3f}"
        
        if cluster_ids is not None:
            if hierarchical_legend:
                n_clusters_sampled = len(cluster_hierarchy)
                n_groups_sampled = len(legend_groups_added) if 'legend_groups_added' in locals() else 0
                stats_text += f"<br><b>Hierarchical Clusters: {n_clusters_sampled}, Groups: {n_groups_sampled}</b><br>"
            else:
                stats_text += f"<br><b>Clusters: {stats_dict['n_clusters']}</b><br>"
                
            for cluster, count in stats_dict['cluster_distribution'].items():
                percentage = 100 * count / stats_dict['n_points']
                cluster_title = cluster_title_map[cluster]
                stats_text += f"{cluster_title}: {count} ({percentage:.1f}%)<br>"
        
        # Position annotation based on subplot layout
        annotation_x = 0.02 if not marginal_plots else 0.75
        annotation_y = 0.98 if not marginal_plots else 0.45
        
        annotation = dict(
            x=annotation_x, y=annotation_y,
            xref="paper", yref="paper",
            text=stats_text,
            showarrow=False,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="gray",
            borderwidth=1,
            font=dict(size=10),
            align="left"
        )
    
    # Update layout
    layout_updates = dict(
        title=dict(
            text=title,
            x=0.5,
            font=dict(size=16, color='black')
        ),
        width=figsize[0] * 80,
        height=figsize[1] * 80,
        template='plotly_white',
        hovermode='closest'
    )
    
    if show_stats:
        layout_updates['annotations'] = [annotation]
    
    if marginal_plots:
        # Update axes for subplots
        fig.update_xaxes(title_text="t-SNE 1", row=main_row, col=main_col)
        fig.update_yaxes(title_text="t-SNE 2", row=main_row, col=main_col)
        fig.update_xaxes(title_text="", showticklabels=False, row=x_hist_row, col=x_hist_col)
        fig.update_yaxes(title_text="Count", row=x_hist_row, col=x_hist_col)
        fig.update_xaxes(title_text="Count", row=y_hist_row, col=y_hist_col)
        fig.update_yaxes(title_text="", showticklabels=False, row=y_hist_row, col=y_hist_col)
    else:
        layout_updates.update({
            'xaxis_title': "t-SNE 1",
            'yaxis_title': "t-SNE 2"
        })
    
    if show_legend:
        layout_updates['legend'] = dict(
            x=1.02,
            y=1,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='rgba(0,0,0,0.2)',
            borderwidth=1,
            groupclick="togglegroup" if hierarchical_legend else "togglegroup"
        )
    else:
        layout_updates['showlegend'] = False
    
    fig.update_layout(**layout_updates)
    
    # Show the figure
    fig.show()
    
    # Print summary to console
    print(f"\n{title}")
    print("=" * len(title))
    print(f"Total points: {stats_dict['n_points']:,}")
    if cluster_ids is not None:
        if hierarchical_legend:
            n_clusters_sampled = len(cluster_hierarchy)
            n_groups_sampled = len(legend_groups_added) if 'legend_groups_added' in locals() else 0
            print(f"Hierarchical Clusters: {n_clusters_sampled}, Groups: {n_groups_sampled}")
        else:
            print(f"Number of clusters: {stats_dict['n_clusters']}")
        print("Cluster distribution:")
        for cluster, count in stats_dict['cluster_distribution'].items():
            percentage = 100 * count / stats_dict['n_points']
            cluster_title = cluster_title_map[cluster]
            print(f"  {cluster_title}: {count:4d} points ({percentage:5.1f}%)")
    
    return {
        'figure': fig,
        'statistics': stats_dict,
        'embeddings': tsne_embeddings,
        'cluster_ids': cluster_ids
    }

def show_density_map(density_map, title=None, figsize=(8, 6), relative_log_scale=False):
    # Convert to numpy if needed
    if hasattr(density_map, 'cpu'):
        density_map = density_map.cpu().numpy()
    if len(density_map.shape) > 2:
        density_map = density_map.squeeze()
    
    # Apply relative log scaling if requested
    if relative_log_scale:
        # Add small epsilon to avoid log(0) issues
        epsilon = np.finfo(density_map.dtype).eps
        display_data = np.log(density_map + epsilon)
        colorbar_title = 'Log Density Value'
    else:
        display_data = density_map
        colorbar_title = 'Density Value'
    
    # Get dimensions for UV coordinate calculation
    height, width = density_map.shape
    
    # Create UV coordinate arrays
    u_coords = np.linspace(0, 1, width)
    v_coords = np.linspace(0, 1, height)
    u_grid, v_grid = np.meshgrid(u_coords, v_coords)
    
    # Create hover text with UV coordinates and density values
    hover_text = []
    for i in range(height):
        hover_row = []
        for j in range(width):
            u_val = u_grid[i, j]
            v_val = 1 - v_grid[i, j]  # Flip V coordinate to match image coordinates
            density_val = density_map[i, j]
            hover_row.append(f'U: {u_val:.3f}<br>V: {v_val:.3f}<br>Density: {density_val:.3f}')
        hover_text.append(hover_row)
    
    # Create interactive plot with Plotly
    fig = go.Figure(data=go.Heatmap(
        z=display_data,
        colorscale='viridis',
        text=hover_text,
        hovertemplate='%{text}<extra></extra>',
        colorbar=dict(title=colorbar_title, titleside="right")
    ))
    
    # Calculate statistics from original data
    min_val = np.min(density_map)
    max_val = np.max(density_map)
    mean_val = np.mean(density_map)
    std_val = np.std(density_map)
    
    # Set title
    plot_title = title if title else "Density Map"
    scale_note = " (Log Scale)" if relative_log_scale else ""
    full_title = f"{plot_title}{scale_note}"
    
    # Update layout
    fig.update_layout(
        title=dict(text=full_title, x=0.5, font=dict(size=14, color='black')),
        width=figsize[0] * 100,
        height=figsize[1] * 120,
        xaxis_title="Width (pixels)",
        yaxis_title="Height (pixels)",
        annotations=[
            dict(
                x=0.02, y=0.98,
                xref="paper", yref="paper",
                text=f"Min: {min_val:.3f}<br>Max: {max_val:.3f}<br>Mean: {mean_val:.3f}<br>Std: {std_val:.3f}",
                showarrow=False,
                bgcolor="white",
                bordercolor="gray",
                borderwidth=1,
                font=dict(size=10),
                align="left"
            )
        ]
    )
    
    fig.show()
    


def visualize_root_uvs_on_density_map(density_map, strand_root_uv_01,
                                      title="Strand Root UV Coordinates on Density Map",
                                      point_size=2, alpha=0.8,
                                      colormap_density='viridis', colormap_points='Reds', n_points=100):
    
    # Convert to numpy if needed
    if hasattr(density_map, 'cpu'):
        density_map = density_map.cpu().numpy()
    if hasattr(strand_root_uv_01, 'cpu'):
        strand_root_uv_01 = strand_root_uv_01.cpu().numpy()
    if len(density_map.shape) > 2:
        density_map = density_map.squeeze()

    height, width = density_map.shape
    
    # Randomly sample n_points from strand_root_uv_01
    n_available = len(strand_root_uv_01)
    n_to_sample = min(n_points, n_available)
    sampled_indices = np.random.choice(n_available, n_to_sample, replace=False)
    strand_root_uv_01 = strand_root_uv_01[sampled_indices]
    # Convert UV coordinates (0-1) to pixel coordinates for sampling density
    # UV coordinates are already in 0-1 range, so we just scale to image dimensions
    pixel_coords = strand_root_uv_01 * np.array([width-1, height-1])
    
    # For sampling from the density map, we need to flip V coordinate since:
    # - UV space: V=0 is bottom, V=1 is top
    # - Image space: row=0 is top, row=height-1 is bottom
    pixel_coords_for_sampling = pixel_coords.copy()
    pixel_coords_for_sampling[:, 1] = height - 1 - pixel_coords_for_sampling[:, 1]
    
    # Sample density values at strand root locations
    strand_densities = map_coordinates(
        density_map,
        [pixel_coords_for_sampling[:, 1], pixel_coords_for_sampling[:, 0]],  # [row, col]
        order=1, mode='nearest'
    )
    
    # Create UV coordinate arrays for the heatmap (0 to 1 range)
    u_coords = np.linspace(0, 1, width)
    v_coords = np.linspace(0, 1, height)
    
    # Create hover text with UV coordinates and density values
    hover_text = []
    for i in range(len(strand_root_uv_01)):
        u_val = strand_root_uv_01[i, 0]
        v_val = strand_root_uv_01[i, 1]
        density_val = strand_densities[i]
        hover_text.append(f'Strand {i}<br>U: {u_val:.3f}<br>V: {v_val:.3f}<br>Density: {density_val:.3f}')
    
    fig = go.Figure()
    
    # Add density map heatmap with UV coordinates
    # Do NOT flip the density map - keep it as-is to match the original orientation
    fig.add_trace(go.Heatmap(
        z=density_map,
        x=u_coords,
        y=v_coords,
        colorscale=colormap_density,
        showscale=True,
        colorbar=dict(title="Density", x=1.05, len=0.4),
        name="Density Map",
        hovertemplate='U: %{x:.3f}<br>V: %{y:.3f}<br>Density: %{z:.3f}<extra></extra>'
    ))
    
    # Add strand root points with UV coordinates (already in 0-1 range)
    fig.add_trace(go.Scatter(
        x=strand_root_uv_01[:, 0],  # U coordinates (0-1)
        y=strand_root_uv_01[:, 1],  # V coordinates (0-1)
        mode='markers',
        marker=dict(
            size=point_size,
            color=strand_densities,
            colorscale=colormap_points,
            opacity=alpha,
            line=dict(width=0.5, color='white'),
            showscale=True,
            colorbar=dict(title="Point Density", x=1.15, len=0.4)
        ),
        text=hover_text,
        hovertemplate='%{text}<extra></extra>',
        name='Strand Roots'
    ))
    
    fig.update_layout(
        title=dict(text=title, x=0.5),
        height=800,
        width=900,  # Slightly wider to accommodate two colorbars
        showlegend=False
    )
    
    # Update axes to show UV coordinates (0 to 1 range)
    # Flip the y-axis by reversing the range to match UV coordinate system
    fig.update_xaxes(title_text="U Coordinate", range=[0, 1])
    fig.update_yaxes(title_text="V Coordinate", range=[1, 0])  # Flipped: top=1, bottom=0
    
    fig.show()
    
    return {
        'uv_coords': strand_root_uv_01, 
        'strand_densities': strand_densities,
        'pixel_coords': pixel_coords
    }
