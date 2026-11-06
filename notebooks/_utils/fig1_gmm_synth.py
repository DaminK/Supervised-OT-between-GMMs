"""Helpers for the Figure 1 synthetic GMM notebook."""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, proj3d
from matplotlib.patches import Ellipse, FancyArrowPatch
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerPatch
from scipy.stats import ortho_group
import matplotlib.colors as mcolors
import matplotlib.patheffects as path_effects

# ==========================================
# 1. DATA GENERATION
# ==========================================
def generate_fig1_data(config):
    np.random.seed(config['random_seed'])
    
    # --- Signal Archetypes (Latent Space) ---
    # Increased base_offset to 12.0
    base = config['base_offset']
    shift = config['mean_shift']
    v_maj = config['var_major']
    v_min = config['var_minor']
    
    # Process A (Horizontal)
    cov_X_i = [[v_maj * 0.9, 0], [0, v_min * 4.0]] 
    mu_X_i =  [base - shift, 0] 
    cov_Y_i = [[v_maj * 1.8, 0], [0, v_min * 0.4]]
    mu_Y_i =  [base + shift, 0] 
    
    # Process B (Vertical)
    cov_X_j = [[v_min * 1.5, 0], [0, v_maj * 1.6]] 
    mu_X_j = [0, base + shift]
    cov_Y_j = [[v_min * 2.5, 0], [0, v_maj * 0.7]] 
    mu_Y_j = [0, base - shift]
    
    signal_means = np.array([mu_X_i, mu_Y_i, mu_X_j, mu_Y_j])
    signal_covs = np.array([cov_X_i, cov_Y_i, cov_X_j, cov_Y_j])
    
    rotations = [20, -5, np.random.uniform(-10, 10), np.random.uniform(-10, 10)]
    for k in range(4):
        angle = rotations[k]
        theta = np.radians(angle)
        c, s = np.cos(theta), np.sin(theta)
        R_local = np.array([[c, -s], [s, c]])
        signal_covs[k] = R_local @ signal_covs[k] @ R_local.T

    # --- Noise Modes (Latent) ---
    n_noise = config['n_noise_modes']
    noise_means = np.random.uniform(-1.0, 1.0, size=(n_noise, 2))
    
    noise_covs = []
    for _ in range(n_noise):
        v1 = np.random.uniform(0.1, 0.5)
        v2 = np.random.uniform(0.1, 0.5)
        cov = np.diag([v1, v2])
        ang = np.random.uniform(0, 180)
        theta = np.radians(ang); c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        noise_covs.append(R @ cov @ R.T)
    noise_covs = np.array(noise_covs)
    
    # --- Noise Modes (High-Dim Structure) ---
    n_dim = config['n_dim']
    n_hd = n_dim - 2
    noise_hd_centers = []
    noise_hd_covs = []
    for _ in range(n_noise):
        center = np.random.randn(n_hd)
        center = center / (np.linalg.norm(center) + 1e-9) * np.random.uniform(12.0, 16.0)
        noise_hd_centers.append(center)
        A = np.random.randn(n_hd, n_hd)
        cov_hd = (A @ A.T) + np.eye(n_hd) * 0.1
        cov_hd = cov_hd / np.max(cov_hd) * 2.0 
        noise_hd_covs.append(cov_hd)
        
    # --- Signal Modes (High-Dim Structure) ---
    signal_hd_covs = []
    for _ in range(4):
        A = np.random.randn(n_hd, n_hd)
        cov_hd = (A @ A.T) * 0.02 + np.eye(n_hd) * 0.05 
        signal_hd_covs.append(cov_hd)
    
    # --- Generate Patients ---
    n_patients_per_class = 3
    if 'mixing_first3_rows' in config:
        desired_rows = np.array(config['mixing_first3_rows'], dtype=float)
        if desired_rows.shape[1] < n_dim:
            desired_rows = np.pad(desired_rows, ((0, 0), (0, n_dim - desired_rows.shape[1])))
        elif desired_rows.shape[1] > n_dim:
            desired_rows = desired_rows[:, :n_dim]
        rng_mix = np.random.default_rng(config['random_seed'] + 17)
        random_rows = rng_mix.normal(size=(n_dim - desired_rows.shape[0], n_dim))
        seed_rows = np.vstack([desired_rows, random_rows])
        q, _ = np.linalg.qr(seed_rows.T)
        Q_high = q.T
    else:
        Q_high = ortho_group.rvs(dim=n_dim, random_state=config['random_seed'] + 1)
    
    theta_b = np.radians(config['rotation_degrees'])
    cb, sb = np.cos(theta_b), np.sin(theta_b)
    R_b = np.array([[cb, -sb], [sb, cb]])

    all_samples = {'high': [], 'rot': [], 'clean': [], 'labels': [], 'mode_names': [], 'patient_ids': []}
    
    agg_means_2d = np.vstack([signal_means, noise_means])
    agg_covs_2d = np.concatenate([signal_covs, noise_covs])
    agg_means_rot = agg_means_2d @ R_b.T
    agg_covs_rot = np.array([R_b @ c @ R_b.T for c in agg_covs_2d])

    patient_id_counter = 0
    for label in [0, 1]: 
        for p_idx in range(n_patients_per_class):
            pid = patient_id_counter
            patient_id_counter += 1
            
            pat_means, pat_covs = [], []
            sig_indices = [0, 2] if label == 0 else [1, 3]
            for s_idx in sig_indices:
                m = signal_means[s_idx] + np.random.randn(2) * 0.15
                c = signal_covs[s_idx] * np.random.uniform(0.9, 1.1)
                pat_means.append(m)
                pat_covs.append(c)
            
            n_sig = int(config['n_samples_signal_per_mode'] / n_patients_per_class) 
            
            # Signal
            for i, s_idx in enumerate(sig_indices):
                mode_n = 'A' if s_idx in [0, 1] else 'B'
                X_clean = np.random.multivariate_normal(pat_means[i], pat_covs[i], n_sig)
                X_rot = X_clean @ R_b.T
                
                X_pad = np.zeros((n_sig, n_dim))
                X_pad[:, :2] = X_clean
                hd_noise = np.random.multivariate_normal(np.zeros(n_hd), signal_hd_covs[s_idx], n_sig)
                X_pad[:, 2:] = hd_noise
                X_high = X_pad @ Q_high.T
                
                all_samples['clean'].append(X_clean)
                all_samples['rot'].append(X_rot)
                all_samples['high'].append(X_high)
                all_samples['labels'].extend([label]*n_sig)
                all_samples['mode_names'].extend([mode_n]*n_sig)
                all_samples['patient_ids'].extend([pid]*n_sig)

            # Noise
            total_sig_p = 2 * n_sig
            total_mass_p = total_sig_p / config['mass_ratio_signal']
            n_noi_p = int((total_mass_p - total_sig_p) / n_noise)
            
            for n_idx in range(n_noise):
                X_clean = np.random.multivariate_normal(noise_means[n_idx], noise_covs[n_idx], n_noi_p)
                X_rot = X_clean @ R_b.T
                
                X_pad = np.zeros((n_noi_p, n_dim))
                X_pad[:, :2] = X_clean 
                noise_hd_samples = np.random.multivariate_normal(
                    mean=np.zeros(n_hd), 
                    cov=noise_hd_covs[n_idx], 
                    size=n_noi_p
                )
                X_pad[:, 2:] = noise_hd_centers[n_idx] + noise_hd_samples
                X_high = X_pad @ Q_high.T
                
                all_samples['clean'].append(X_clean)
                all_samples['rot'].append(X_rot)
                all_samples['high'].append(X_high)
                all_samples['labels'].extend([label]*n_noi_p)
                all_samples['mode_names'].extend(['Noise']*n_noi_p)
                all_samples['patient_ids'].extend([pid]*n_noi_p)

    return {
        'agg_means_2d': agg_means_2d, 'agg_covs_2d': agg_covs_2d,
        'agg_means_rot': agg_means_rot, 'agg_covs_rot': agg_covs_rot,
        'samples': all_samples,
        'X_high_all': np.vstack(all_samples['high']),
        'n_noise': n_noise,
        'noise_means': noise_means,
        'noise_hd_centers': np.array(noise_hd_centers),
        'Q_high': Q_high,
        'R_b': R_b,
        'disc_pairs': [(0, 1), (2, 3)]
    }

# ==========================================
# 2. HELPERS
# ==========================================

def get_tight_bounds(means, covs, padding_sigma=3.5):
    x_min, x_max = np.inf, -np.inf
    y_min, y_max = np.inf, -np.inf
    
    # Iterate ALL modes (Signal + Noise) to catch everything
    for i in range(len(means)):
        sig_x = np.sqrt(covs[i][0,0])
        sig_y = np.sqrt(covs[i][1,1])
        x_min = min(x_min, means[i][0] - padding_sigma * sig_x)
        x_max = max(x_max, means[i][0] + padding_sigma * sig_x)
        y_min = min(y_min, means[i][1] - padding_sigma * sig_y)
        y_max = max(y_max, means[i][1] + padding_sigma * sig_y)
        
    return x_min, x_max, y_min, y_max

def plot_ellipse_custom(mean, cov, ax, color, linestyle='-', alpha=1.0, is_diag_approx=False, lw=1.5, fill_alpha=0.2):
    if is_diag_approx:
        cov_draw = np.diag(np.diag(cov))
        fill = False
        z_ord = 20
    else:
        cov_draw = cov
        fill = True
        z_ord = 10

    lambda_, v = np.linalg.eig(cov_draw)
    lambda_ = np.sqrt(np.real(lambda_))
    v = np.real(v)
    angle = np.rad2deg(np.arctan2(v[1, 0], v[0, 0]))
    
    ell = Ellipse(xy=mean, width=lambda_[0]*4, height=lambda_[1]*4,
                  angle=angle, linestyle=linestyle, linewidth=lw, zorder=z_ord)
    
    edge_rgba = mcolors.to_rgba(color, alpha)
    ell.set_edgecolor(edge_rgba)
    
    if fill:
        face_rgba = mcolors.to_rgba(color, fill_alpha)
        ell.set_facecolor(face_rgba)
    else:
        ell.set_facecolor('none')

    ax.add_patch(ell)

def get_discriminative_axis_segments_2d(means, pairs, axis_length):
    segments = []
    origin = np.zeros(2)
    for pair in pairs:
        pair_means = means[list(pair)]
        center = pair_means.mean(axis=0)
        direction = center.copy()
        norm = np.linalg.norm(direction)
        if norm < 1e-9:
            direction = pair_means[1] - pair_means[0]
            norm = np.linalg.norm(direction)
        if norm < 1e-9:
            direction = np.array([1.0, 0.0])
        else:
            direction = direction / norm
        segments.append((origin, axis_length * direction))
    return segments

def get_discriminative_axis_segments_high(data, config):
    segments = []
    n_dim = config['n_dim']
    for start_2d, end_2d in get_discriminative_axis_segments_2d(
        data['agg_means_2d'], data['disc_pairs'], config['axis_length_3d']
    ):
        direction = end_2d - start_2d
        norm = np.linalg.norm(direction)
        if norm > 1e-9:
            start_2d = start_2d - config['axis_back_extension_3d'] * direction / norm
        endpoints = []
        for point_2d in (start_2d, end_2d):
            point_high = np.zeros(n_dim)
            point_high[:2] = point_2d
            endpoints.append(point_high @ data['Q_high'].T)
        segments.append(np.array(endpoints)[:, :3])
    return segments

def draw_discriminative_axes_2d(ax, means, data, config):
    for start, end in get_discriminative_axis_segments_2d(
        means, data['disc_pairs'], config['axis_length_2d']
    ):
        ann = ax.annotate(
            '', xy=end, xytext=start,
            arrowprops=dict(
                arrowstyle='-|>', color=config['axis_color'],
                lw=config['axis_width'], alpha=config['axis_alpha'],
                shrinkA=0, shrinkB=0, mutation_scale=config['axis_arrow_scale'],
            ),
            zorder=config['axis_zorder'],
        )
        ann.set_clip_on(False)

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)

def draw_arrow_3d(ax, start, end, color, lw, alpha, zorder, mutation_scale):
    arrow = Arrow3D(
        [start[0], end[0]], [start[1], end[1]], [start[2], end[2]],
        mutation_scale=mutation_scale, lw=lw, arrowstyle='-|>',
        color=color, alpha=alpha, zorder=zorder,
    )
    arrow.set_clip_on(False)
    ax.add_artist(arrow)

def draw_discriminative_axes_3d(ax, data, config):
    for segment in get_discriminative_axis_segments_high(data, config):
        start, end = segment
        draw_arrow_3d(
            ax, start, end, color=config['axis_color'], lw=config.get('axis_width_3d', config['axis_width']),
            alpha=config['axis_alpha'], zorder=config['axis_zorder'],
            mutation_scale=config['axis_arrow_scale_3d'],
        )

def get_noise_axis_segments_high(data, config):
    segments = []
    origin = np.zeros(3)
    for dim_idx in range(2, config['n_dim']):
        point_high = np.zeros(config['n_dim'])
        point_high[dim_idx] = config['noise_axis_length_3d']
        end = (point_high @ data['Q_high'].T)[:3]
        segments.append(np.array([origin, end]))
    return segments

def draw_noise_axes_3d(ax, data, config):
    if not config.get('show_noise_axes_3d', False):
        return
    for segment in get_noise_axis_segments_high(data, config):
        start, end = segment
        vec = end - start
        ax.quiver(
            start[0], start[1], start[2], vec[0], vec[1], vec[2],
            color=config['noise_axis_color'], linewidth=config['noise_axis_width'],
            alpha=config['noise_axis_alpha'], arrow_length_ratio=config['axis_arrow_ratio'],
            normalize=False, zorder=config['axis_zorder']
        )

def high_point_from_latent(point_2d, data, config):
    point_high = np.zeros(config['n_dim'])
    point_high[:2] = point_2d
    return (point_high @ data['Q_high'].T)[:3]

def signal_mode_name(k):
    return [r'$\mathcal{X}_l$', r'$\mathcal{Y}_j$', r'$\mathcal{X}_k$', r'$\mathcal{Y}_i$'][k]

def label_position_2d(k, means, data, config, offset=None):
    if offset is None:
        offset = config['label_offset_2d']
    pair = next(pair for pair in data['disc_pairs'] if k in pair)
    direction = means[pair[1]] - means[pair[0]]
    norm = np.linalg.norm(direction)
    if norm < 1e-9:
        return means[k]
    sign = -1 if k == pair[0] else 1
    return means[k] + sign * offset * direction / norm

def add_3d_signal_labels(ax, data, config):
    colors = config['colors']
    for k, mean in enumerate(data['agg_means_2d'][:4]):
        point = high_point_from_latent(label_position_2d(k, data['agg_means_2d'], data, config, offset=config['label_offset_3d']), data, config)
        label = 0 if k in [0, 2] else 1
        txt = ax.text(point[0], point[1], point[2], signal_mode_name(k),
                      color=colors[label], ha='center', va='center',
                      fontweight='bold', fontsize=config['label_fontsize_3d'], zorder=80)
        txt.set_path_effects([path_effects.withStroke(linewidth=2.6, foreground='white')])

def set_3d_bounds(ax, points, config):
    q_low, q_high = config['bounds_quantile_3d']
    low = np.percentile(points, q_low, axis=0)
    high = np.percentile(points, q_high, axis=0)
    pad = (high - low) * config['axis_bound_pad_3d']
    ax.set_xlim(low[0] - pad[0], high[0] + pad[0])
    ax.set_ylim(low[1] - pad[1], high[1] + pad[1])
    ax.set_zlim(low[2] - pad[2], high[2] + pad[2])

def style_3d_panel(ax, config):
    ax.set_xlabel('Dim 1', labelpad=-13, fontsize=14)
    ax.set_ylabel('Dim 2', labelpad=-13, fontsize=14)
    ax.set_zlabel('Dim ...', labelpad=-13, fontsize=14)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    try:
        ax.set_box_aspect((1, 1, 1), zoom=config['box_zoom_3d'])
    except TypeError:
        ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=config['elev_3d'], azim=config['azim_3d'])

def make_legend_ellipse(legend, orig_handle, xdescent, ydescent, width, height, fontsize):
    return Ellipse((xdescent + width / 2, ydescent + height / 2),
                   width=width, height=height * 0.65)

def ellipse_legend_handle(color, label):
    return Ellipse((0, 0), width=1.8, height=0.8, angle=0,
                   facecolor=mcolors.to_rgba(color, 0.22),
                   edgecolor=color, linewidth=1.8, label=label)

def covariance_legend_handle(linestyle, label):
    return Ellipse((0, 0), width=1.8, height=0.8, angle=0,
                   facecolor='none', edgecolor='0.25', linewidth=1.9,
                   linestyle=linestyle, label=label)

def gmm_marker(pid):
    # Hardcoded so patient/GMM symbols stay consistent between panels A and B.
    # The class-1 patient shown in panel B is pid 3, so it uses the inverted triangle.
    return ['o', '^', 's', 'v', 'D', 'P'][pid % 6]

# ==========================================
# 3. DRAWING FUNCTIONS
# ==========================================

def draw_panel_A(ax, data, config):
    X_high = data['X_high_all']
    samples = data['samples']
    labels = np.array(samples['labels'])
    pids = np.array(samples['patient_ids'])
    colors = config['colors']
    
    unique_pids = np.unique(pids)
    for pid in unique_pids:
        mask = (pids == pid)
        lbl = labels[mask][0]
        m = gmm_marker(pid)
        pts = X_high[mask]
        ax.scatter(pts[:,0], pts[:,1], pts[:,2], c=colors[lbl], marker=m,
                   s=config['point_size_3d'], alpha=config['point_alpha_3d'],
                   edgecolors='none', depthshade=False)

    set_3d_bounds(ax, X_high[:, :3], config)
    style_3d_panel(ax, config)
    
    # Hardcoded compact legend: show representative GMM symbols while the full
    # pid-specific marker mapping above keeps panels A and B visually consistent.
    distribution_handles = [
        Line2D([], [], linestyle='none', marker='o', markersize=7.0,
               markerfacecolor='0.35', markeredgecolor='0.35', label=r'GMM $\mathcal{X}$'),
        Line2D([], [], linestyle='none', marker='v', markersize=7.0,
               markerfacecolor='0.35', markeredgecolor='0.35', label=r'GMM $\mathcal{Y}$'),
        Line2D([], [], linestyle='none', marker='s', markersize=7.0,
               markerfacecolor='0.35', markeredgecolor='0.35', label=r'GMM $\ldots$'),
    ]
    label_handles = [
        Line2D([], [], linestyle='none', marker='o', markersize=7.0,
               markerfacecolor=colors[0], markeredgecolor='none', label='Class 0'),
        Line2D([], [], linestyle='none', marker='o', markersize=7.0,
               markerfacecolor=colors[1], markeredgecolor='none', label='Class 1'),
    ]
    leg_dist = ax.legend(handles=distribution_handles, loc='upper right',
                         fontsize=config['legend_fontsize'], title='Distribution',
                         title_fontsize=config['legend_title_fontsize'],
                         ncol=1, framealpha=config['legend_framealpha'], facecolor='white',
                         borderpad=0.45, labelspacing=0.35, handletextpad=0.5,
                         handlelength=1.0, bbox_to_anchor=(0.98, 0.98), borderaxespad=0.0)
    leg_dist._legend_box.align = 'left'
    ax.add_artist(leg_dist)
    leg_label = ax.legend(handles=label_handles, loc='upper right',
                          fontsize=config['legend_fontsize'], title='Label',
                          title_fontsize=config['legend_title_fontsize'],
                          ncol=1, framealpha=config['legend_framealpha'], facecolor='white',
                          borderpad=0.45, labelspacing=0.35, handletextpad=0.5,
                          handlelength=1.0, bbox_to_anchor=(0.60, 0.98), borderaxespad=0.0)
    leg_label._legend_box.align = 'left'

def draw_panel_B(ax, data, config):
    X_high = data['X_high_all']
    samples = data['samples']
    labels = np.array(samples['labels'])
    pids = np.array(samples['patient_ids'])
    mode_names = np.array(samples['mode_names'])
    colors = config['colors']
    background_color = config['background_color']
    
    # Hardcoded minimal subset for panel B: one patient/GMM per class, with
    # markers inherited from gmm_marker() to match the same points in panel A.
    selected_pids = []
    for lbl in [0, 1]:
        class_pids = np.unique(pids[labels == lbl])
        if len(class_pids) > 0:
            selected_pids.append(class_pids[0])
    unique_pids = np.array(selected_pids)

    draw_discriminative_axes_3d(ax, data, config)

    for pid in unique_pids:
        mask = (pids == pid) & (mode_names == 'Noise')
        if not np.any(mask):
            continue
        lbl = labels[mask][0]
        m = gmm_marker(pid)
        pts = X_high[mask]
        ax.scatter(pts[:,0], pts[:,1], pts[:,2], c=background_color, marker=m,
                   s=config['background_point_size_3d'], alpha=config['background_alpha_3d'],
                   edgecolors='none', depthshade=False)

    for pid in unique_pids:
        mask = (pids == pid) & (mode_names != 'Noise')
        if not np.any(mask):
            continue
        lbl = labels[mask][0]
        m = gmm_marker(pid)
        pts = X_high[mask]
        ax.scatter(pts[:,0], pts[:,1], pts[:,2], c=colors[lbl], marker=m,
                   s=config['signal_point_size_3d'], alpha=config['signal_alpha_3d'],
                   edgecolors='white', linewidths=0.25, depthshade=False)

    add_3d_signal_labels(ax, data, config)

    set_3d_bounds(ax, X_high[:, :3], config)
    style_3d_panel(ax, config)

    leg_B = [
        Line2D([0], [0], color='w', marker='o', markerfacecolor=colors[0],
               markeredgecolor='white', label='Discriminative Class 0'),
        Line2D([0], [0], color='w', marker='o', markerfacecolor=colors[1],
               markeredgecolor='white', label='Discriminative Class 1'),
        Line2D([0], [0], color='w', marker='o', markerfacecolor=background_color,
               markeredgecolor='none', label='Non-Discriminative'),
    ]
    legend = ax.legend(handles=leg_B, loc='upper left', fontsize=config['legend_fontsize'],
                       title='Gaussian Components', title_fontsize=config['legend_title_fontsize'],
                       framealpha=config['legend_framealpha'], facecolor='white')
    legend._legend_box.align = 'left'

def set_2d_bounds(ax, means, covs, data, config):
    x1, x2, y1, y2 = get_tight_bounds(means, covs, padding_sigma=3.5)
    axis_segments = get_discriminative_axis_segments_2d(means, data['disc_pairs'], config['axis_length_2d'])
    axis_points = np.vstack(axis_segments)
    x1 = min(x1, axis_points[:, 0].min()); x2 = max(x2, axis_points[:, 0].max())
    y1 = min(y1, axis_points[:, 1].min()); y2 = max(y2, axis_points[:, 1].max())
    x_pad = (x2 - x1) * config['axis_bound_pad_2d']
    y_pad = (y2 - y1) * config['axis_bound_pad_2d']
    ax.set_xlim(x1 - x_pad, x2 + x_pad); ax.set_ylim(y1 - y_pad, y2 + y_pad)

def draw_panel_C(ax, data, config):
    samples = data['samples']
    colors = config['colors']
    labels = np.array(samples['labels'])
    mode_names = np.array(samples['mode_names'])
    
    flat_rot = np.vstack(samples['rot'])
    noise_mask = mode_names == 'Noise'
    signal_mask = ~noise_mask
    ax.scatter(flat_rot[noise_mask,0], flat_rot[noise_mask,1], c=config['background_color'],
               s=config['noise_point_size_2d'], alpha=config['noise_point_alpha_2d'],
               zorder=1, edgecolors='none')
    signal_cols = [colors[l] for l in labels[signal_mask]]
    ax.scatter(flat_rot[signal_mask,0], flat_rot[signal_mask,1], c=signal_cols,
               s=config['point_size_2d'], alpha=config['point_alpha_2d'],
               zorder=2, edgecolors='none')

    means = data['agg_means_rot']
    covs = data['agg_covs_rot']
    n_total = len(means)
    
    fw = config['focused_width']; nfw = config['nonfocused_width']
    fa = config['focused_alpha']; nfa = config['nonfocused_alpha']
    
    draw_discriminative_axes_2d(ax, means, data, config)

    for k in range(n_total):
        if k >= 4:
            plot_ellipse_custom(means[k], covs[k], ax, 'gray', '-', config['noise_ellipse_alpha'], False, lw=nfw, fill_alpha=0.05)
        else:
            label = 0 if k in [0, 2] else 1
            mode_name = signal_mode_name(k)
            plot_ellipse_custom(means[k], covs[k], ax, colors[label], '-', fa, False, lw=fw, fill_alpha=config['fill_alpha'])
            plot_ellipse_custom(means[k], covs[k], ax, colors[label], '--', nfa, True, lw=nfw)
            label_pos = label_position_2d(k, means, data, config)
            txt = ax.text(label_pos[0], label_pos[1], mode_name,
                          color=colors[label], ha='center', va='center', fontweight='bold', fontsize=config['label_fontsize_2d'], zorder=30)
            txt.set_path_effects([path_effects.withStroke(linewidth=3.0, foreground='white')])

    set_2d_bounds(ax, means, covs, data, config)
    ax.axhline(0, color='k', linestyle=':', alpha=0.3)
    ax.axvline(0, color='k', linestyle=':', alpha=0.3)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel('Latent Dim 1', fontsize=14); ax.set_ylabel('Latent Dim 2', fontsize=14)
    
    leg_CD = [
        ellipse_legend_handle(colors[0], 'Discriminative Class 0'),
        ellipse_legend_handle(colors[1], 'Discriminative Class 1'),
        ellipse_legend_handle('gray', 'Non-Discriminative'),
    ]
    legend = ax.legend(handles=leg_CD, loc='upper right', fontsize=config['legend_fontsize'],
                       title='Gaussian Components', title_fontsize=config['legend_title_fontsize'],
                       framealpha=config['legend_framealpha'], facecolor='white',
                       handler_map={Ellipse: HandlerPatch(patch_func=make_legend_ellipse)})
    legend._legend_box.align = 'left'

def draw_panel_D(ax, data, config):
    samples = data['samples']
    colors = config['colors']
    labels = np.array(samples['labels'])
    mode_names = np.array(samples['mode_names'])
    
    flat_clean = np.vstack(samples['clean'])
    noise_mask = mode_names == 'Noise'
    signal_mask = ~noise_mask
    ax.scatter(flat_clean[noise_mask,0], flat_clean[noise_mask,1], c=config['background_color'],
               s=config['noise_point_size_2d'], alpha=config['noise_point_alpha_2d'],
               zorder=1, edgecolors='none')
    signal_cols = [colors[l] for l in labels[signal_mask]]
    ax.scatter(flat_clean[signal_mask,0], flat_clean[signal_mask,1], c=signal_cols,
               s=config['point_size_2d'], alpha=config['point_alpha_2d'],
               zorder=2, edgecolors='none')

    means = data['agg_means_2d']
    covs = data['agg_covs_2d']
    n_total = len(means)
    
    fw = config['focused_width']; nfw = config['nonfocused_width']
    fa = config['focused_alpha']; nfa = config['nonfocused_alpha']

    draw_discriminative_axes_2d(ax, means, data, config)

    for k in range(n_total):
        if k >= 4:
            plot_ellipse_custom(means[k], covs[k], ax, 'gray', '-', config['noise_ellipse_alpha'], False, lw=nfw, fill_alpha=0.05)
        else:
            label = 0 if k in [0, 2] else 1
            mode_name = signal_mode_name(k)
            plot_ellipse_custom(means[k], covs[k], ax, colors[label], '-', nfa, False, lw=nfw, fill_alpha=config['fill_alpha'])
            plot_ellipse_custom(means[k], covs[k], ax, colors[label], '--', fa, True, lw=fw)
            label_pos = label_position_2d(k, means, data, config)
            txt = ax.text(label_pos[0], label_pos[1], mode_name,
                          color=colors[label], ha='center', va='center', fontweight='bold', fontsize=config['label_fontsize_2d'], zorder=30)
            txt.set_path_effects([path_effects.withStroke(linewidth=3.0, foreground='white')])

    set_2d_bounds(ax, means, covs, data, config)
    ax.axhline(0, color='k', linestyle=':', alpha=0.3)
    ax.axvline(0, color='k', linestyle=':', alpha=0.3)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel('Latent Dim 1', fontsize=14); ax.set_ylabel('Latent Dim 2', fontsize=14)
    
    leg_D = [
        covariance_legend_handle('-', 'Full Covariance'),
        covariance_legend_handle('--', 'Diagonal Approx.'),
    ]
    legend = ax.legend(handles=leg_D, loc='upper right', fontsize=config['legend_fontsize'],
                       title='Gaussian Components', title_fontsize=config['legend_title_fontsize'],
                       framealpha=config['legend_framealpha'], facecolor='white',
                       handler_map={Ellipse: HandlerPatch(patch_func=make_legend_ellipse)})
    legend._legend_box.align = 'left'

