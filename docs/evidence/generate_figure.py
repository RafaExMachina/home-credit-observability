"""Gera a figura do README a partir das métricas agregadas verificadas."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parents[2]
results = json.loads((root / "docs/evidence/results.json").read_text())
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
fig, axes = plt.subplots(1, 2, figsize=(11, 5.6))
fig.patch.set_facecolor("#f8fafc")
for ax, scenario, title, keys, subtitle in [
    (
        axes[0],
        results["data_drift"],
        "Data Drift · coorte independente",
        ["accuracy", "recall"],
        "3 features com drift · mesmos 8.000 registros antes/depois",
    ),
    (
        axes[1],
        results["concept_drift"]["performance"],
        "Concept Drift · cenário sintético",
        ["roc_auc", "recall"],
        "Entradas idênticas · zero testes de Data Drift positivos",
    ),
]:
    positions = np.arange(2)
    for label, dataset, offset, color, hatch in [
        ("Antes", "reference", -0.18, "#1e3a5f", ""),
        ("Depois", "production", 0.18, "#c05c22", "//"),
    ]:
        values = [scenario[dataset][key] for key in keys]
        bars = ax.bar(
            positions + offset,
            values,
            width=0.34,
            color=color,
            hatch=hatch,
            edgecolor="white",
            label=label,
            zorder=3,
        )
        ax.bar_label(
            bars,
            labels=[f"{v:.4f}".replace(".", ",") for v in values],
            padding=5,
            fontsize=11,
        )
    ax.set_xticks(
        positions,
        [
            "Acurácia" if k == "accuracy" else "ROC AUC" if k == "roc_auc" else "Recall"
            for k in keys
        ],
    )
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Valor da métrica (0 a 1)")
    ax.set_title(title, loc="left", fontweight="bold", fontsize=12, pad=18)
    ax.grid(axis="y", color="#dbe3ec", alpha=0.8, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#cbd5e1")
    ax.tick_params(axis="y", length=0)
    ax.text(0, -0.17, subtitle, transform=ax.transAxes, fontsize=9, color="#475569")
    ax.legend(loc="upper right", frameon=False, fontsize=9)
fig.suptitle(
    "A degradação exige monitorar distribuição e desempenho",
    x=0.07,
    ha="left",
    fontweight="bold",
    fontsize=16,
    color="#172033",
)
fig.text(
    0.07,
    0.025,
    "Fonte: execução verificada em 12/10/2026 · 8.000 registros por cenário · semente 42",
    fontsize=9,
    color="#475569",
)
fig.subplots_adjust(left=0.07, right=0.97, bottom=0.23, top=0.82, wspace=0.30)
fig.savefig(
    root / "docs/images/drift-comparison.png", dpi=180, facecolor=fig.get_facecolor()
)
plt.close(fig)
