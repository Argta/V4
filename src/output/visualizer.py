"""Matplotlib-based visualization replacing the old MATLAB scripts.

Generates multi-panel figures showing:
- 3D room scene with source trajectory
- Left/right ear waveforms
- Spectrograms
- ITD/ILD analysis
- DOA trajectory comparison (v3.0)
- Error histogram (v3.0)
- Confusion matrix (v3.0)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def draw_room_3d(ax, room_dims):
    """Draw a wireframe room box."""
    x, y, z = room_dims
    vertices = np.array([
        [0, 0, 0], [x, 0, 0], [x, y, 0], [0, y, 0],
        [0, 0, z], [x, 0, z], [x, y, z], [0, y, z],
    ])
    faces = [
        [0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
        [2, 3, 7, 6], [0, 3, 7, 4], [1, 2, 6, 5],
    ]
    for face in faces:
        poly = Poly3DCollection(
            [vertices[face]], alpha=0.1,
            facecolor="lightblue", edgecolor="gray", linewidth=0.5
        )
        ax.add_collection3d(poly)


def visualize(scene_cfg, stereo_signal, trajectory, fs, save_path=None):
    """Generate a comprehensive visualization figure.

    Args:
        scene_cfg: SceneConfig
        stereo_signal: (N, 2) stereo numpy array
        trajectory: (N, 3) trajectory array
        fs: sample rate
        save_path: optional path to save the figure
    """
    fig = plt.figure(figsize=(16, 12))

    # --- 1. 3D Room Scene (top-left, spans two rows) ---
    ax3d = fig.add_subplot(2, 3, 1, projection="3d")
    room = scene_cfg.room.dimensions
    draw_room_3d(ax3d, room)

    # Plot trajectory
    if trajectory is not None and len(trajectory) > 0:
        stride = max(1, len(trajectory) // 500)
        ax3d.plot(trajectory[::stride, 0],
                  trajectory[::stride, 1],
                  trajectory[::stride, 2],
                  "b-", linewidth=1, alpha=0.7, label="Trajectory")
        if len(trajectory) > 1:
            ax3d.scatter(*trajectory[0], c="green", s=50, marker="o", label="Start")
            ax3d.scatter(*trajectory[-1], c="red", s=50, marker="s", label="End")

    # Plot head center and ears
    mic = scene_cfg.microphone
    ax3d.scatter(*mic.head_center, c="black", s=80, marker="o", label="Head")
    ax3d.scatter(*mic.left_ear, c="blue", s=60, marker="^", label="Left Ear")
    ax3d.scatter(*mic.right_ear, c="red", s=60, marker="v", label="Right Ear")

    ax3d.set_xlabel("X (m)")
    ax3d.set_ylabel("Y (m)")
    ax3d.set_zlabel("Z (m)")
    ax3d.set_title(f"Room Scene: {scene_cfg.name}")
    ax3d.legend(loc="upper right", fontsize=7)
    ax3d.set_xlim(0, room[0])
    ax3d.set_ylim(0, room[1])
    ax3d.set_zlim(0, room[2])

    # --- 2. Waveforms ---
    t = np.arange(len(stereo_signal)) / fs
    ax_wf = fig.add_subplot(2, 3, 2)
    ax_wf.plot(t, stereo_signal[:, 0], "b-", linewidth=0.5, alpha=0.7, label="Left")
    ax_wf.plot(t, stereo_signal[:, 1], "r-", linewidth=0.5, alpha=0.7, label="Right")
    ax_wf.set_xlabel("Time (s)")
    ax_wf.set_ylabel("Amplitude")
    ax_wf.set_title("Binaural Waveforms")
    ax_wf.legend(fontsize=7)
    ax_wf.set_xlim(0, t[-1])
    ax_wf.grid(True, alpha=0.3)

    # --- 3. Spectrogram (Left) ---
    ax_spec_l = fig.add_subplot(2, 3, 3)
    ax_spec_l.specgram(stereo_signal[:, 0], Fs=fs, NFFT=1024,
                       noverlap=512, cmap="inferno")
    ax_spec_l.set_xlabel("Time (s)")
    ax_spec_l.set_ylabel("Frequency (Hz)")
    ax_spec_l.set_title("Spectrogram (Left Ear)")
    ax_spec_l.set_ylim(0, 8000)

    # --- 4. Spectrogram (Right) ---
    ax_spec_r = fig.add_subplot(2, 3, 4)
    ax_spec_r.specgram(stereo_signal[:, 1], Fs=fs, NFFT=1024,
                       noverlap=512, cmap="inferno")
    ax_spec_r.set_xlabel("Time (s)")
    ax_spec_r.set_ylabel("Frequency (Hz)")
    ax_spec_r.set_title("Spectrogram (Right Ear)")
    ax_spec_r.set_ylim(0, 8000)

    # --- 5. ITD: Cross-correlation ---
    ax_cc = fig.add_subplot(2, 3, 5)
    max_lag = int(fs * 0.001)  # 1ms max lag
    lags = np.arange(-max_lag, max_lag + 1) / fs * 1000  # ms

    window_s = 0.05  # 50ms windows for time-varying ITD
    window_len = int(window_s * fs)
    n_windows = max(1, len(stereo_signal) // window_len)

    itd_values = []
    itd_times = []
    for i in range(0, n_windows, max(1, n_windows // 50)):  # Downsample for display
        w0 = i * window_len
        w1 = min(w0 + window_len, len(stereo_signal))
        if w1 - w0 < window_len // 2:
            break
        xcorr = np.correlate(stereo_signal[w0:w1, 0],
                             stereo_signal[w0:w1, 1], mode="same")
        mid = len(xcorr) // 2
        segment = xcorr[mid - max_lag:mid + max_lag + 1]
        if len(segment) > 0:
            peak_idx = np.argmax(np.abs(segment))
            itd_values.append(lags[peak_idx])
            itd_times.append((w0 + w1) / 2 / fs)

    if itd_values:
        ax_cc.plot(itd_times, itd_values, "g.-", linewidth=0.8, markersize=3)
        ax_cc.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
        ax_cc.set_xlabel("Time (s)")
        ax_cc.set_ylabel("ITD (ms)")
        ax_cc.set_title("Time-varying ITD (cross-correlation)")
        ax_cc.grid(True, alpha=0.3)

    # --- 6. ILD: Energy difference ---
    ax_ild = fig.add_subplot(2, 3, 6)
    ild_db = []
    ild_times = []
    for i in range(0, n_windows, max(1, n_windows // 50)):
        w0 = i * window_len
        w1 = min(w0 + window_len, len(stereo_signal))
        if w1 - w0 < window_len // 2:
            break
        energy_l = np.sum(stereo_signal[w0:w1, 0] ** 2)
        energy_r = np.sum(stereo_signal[w0:w1, 1] ** 2)
        ild = 10 * np.log10((energy_l + 1e-10) / (energy_r + 1e-10))
        ild_db.append(ild)
        ild_times.append((w0 + w1) / 2 / fs)

    if ild_db:
        ax_ild.plot(ild_times, ild_db, "m.-", linewidth=0.8, markersize=3)
        ax_ild.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
        ax_ild.set_xlabel("Time (s)")
        ax_ild.set_ylabel("ILD (dB)")
        ax_ild.set_title("Time-varying ILD (L-R energy ratio)")
        ax_ild.grid(True, alpha=0.3)

    plt.suptitle(f"Binaural Simulation: {scene_cfg.name}",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Figure saved: {save_path}")
    else:
        plt.show()

    plt.close(fig)


def visualize_evaluation(scene_name, timestamps, doa_estimated, doa_truth,
                         errors, confusion_matrix, metrics,
                         save_path=None):
    """Generate DOA evaluation figure with 4 subplots.

    Args:
        scene_name: scene identifier
        timestamps: (M,) time in seconds
        doa_estimated: (M,) estimated azimuth
        doa_truth: (M,) true azimuth
        errors: (M,) angular errors
        confusion_matrix: 4x4 ndarray
        metrics: dict from compute_all_metrics
        save_path: optional save path
    """
    fig = plt.figure(figsize=(14, 10))

    # --- 1. DOA Trajectory ---
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(timestamps, doa_truth, "k-", linewidth=1.5, alpha=0.8, label="True DOA")
    ax1.plot(timestamps, doa_estimated, "r--", linewidth=1.0, alpha=0.7, label="Estimated DOA")
    ax1.fill_between(timestamps, doa_truth, doa_estimated, alpha=0.12, color="red")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Azimuth (deg)")
    ax1.set_title(f"DOA Estimation: {scene_name}")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-180, 180)

    # --- 2. Error Histogram ---
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.hist(errors, bins=36, range=(-180, 180), color="steelblue",
             edgecolor="white", alpha=0.85)
    ax2.axvline(x=0, color="black", linewidth=1)
    ax2.axvline(x=np.mean(errors), color="red", linestyle="--",
                linewidth=1, label=f"Mean={np.mean(errors):.1f} deg")
    ax2.set_xlabel("Angular Error (deg)")
    ax2.set_ylabel("Frame Count")
    ax2.set_title("Error Distribution")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # --- 3. Confusion Matrix ---
    ax3 = fig.add_subplot(2, 2, 3)
    labels = ["Front-R", "Front-L", "Back-R", "Back-L"]
    cm = confusion_matrix
    im = ax3.imshow(cm, cmap="YlOrRd", origin="upper")
    for i in range(4):
        for j in range(4):
            ax3.text(j, i, str(cm[i, j]), ha="center", va="center",
                     fontsize=10, fontweight="bold",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax3.set_xticks(range(4))
    ax3.set_xticklabels(labels, fontsize=8)
    ax3.set_yticks(range(4))
    ax3.set_yticklabels(labels, fontsize=8)
    ax3.set_xlabel("Estimated")
    ax3.set_ylabel("Truth")
    ax3.set_title("DOA Confusion Matrix")
    plt.colorbar(im, ax=ax3, shrink=0.8)

    # --- 4. Metrics Summary ---
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis("off")
    metric_lines = [
        f"Evaluation: {scene_name}",
        "",
        f"Frames:     {metrics.get('n_frames', 0)}",
        f"RMSE:       {metrics.get('rmse', 0):.2f} deg",
        f"MAE:        {metrics.get('mae', 0):.2f} deg",
        f"Acc (5 deg):  {metrics.get('accuracy_5deg', 0)*100:.1f}%",
        f"Acc (10 deg): {metrics.get('accuracy_10deg', 0)*100:.1f}%",
        f"Acc (30 deg): {metrics.get('accuracy_30deg', 0)*100:.1f}%",
        f"F/B Confusion: {metrics.get('front_back_confusion', 0)*100:.1f}%",
    ]
    ax4.text(0.05, 0.95, "\n".join(metric_lines),
             transform=ax4.transAxes, fontsize=11, fontfamily="monospace",
             verticalalignment="top",
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    plt.suptitle(f"Localization Evaluation: {scene_name}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Eval figure saved: {save_path}")
    else:
        plt.show()

    plt.close(fig)
