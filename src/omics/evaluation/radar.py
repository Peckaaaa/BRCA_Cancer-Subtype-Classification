
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  

from .metrics import METRIC_KEYS 

_LABELS = ["Accuracy", "F1", "Precision", "Recall", "AUC"]


def plot_radar(mean_by_method: dict[str, dict], title: str, out_path: str | Path) -> None:
    """
    mean_by_method : {method_name: {metric_key: value, ...}, ...}
                     metric keys are those in evaluation.metrics.METRIC_KEYS.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(_LABELS)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 8), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0.5, 1.0)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(_LABELS)
    rings = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    ax.set_yticks(rings)
    ax.set_yticklabels([f"{r:.1f}" for r in rings], color="grey", size=8)

    colors = plt.cm.tab10.colors
    for idx, (method, scores) in enumerate(mean_by_method.items()):
        vals = [float(np.clip(scores.get(k, 0.5), 0.5, 1.0)) for k in METRIC_KEYS]
        vals += vals[:1]
        color = colors[idx % len(colors)]
        ax.plot(angles, vals, color=color, linewidth=2, label=method)
        ax.fill(angles, vals, color=color, alpha=0.15)

    ax.set_title(title, fontsize=13, fontweight="bold", y=1.10)
    ax.legend(loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Radar chart saved -> {out_path}")
