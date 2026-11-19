import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ratio = 2


def cell_edges(values):
    if len(values) == 1:
        return np.array([values[0] - 0.5, values[0] + 0.5])

    midpoints = (values[:-1] + values[1:]) / 2
    return np.concatenate(
        ([values[0] - (midpoints[0] - values[0])],
         midpoints,
         [values[-1] + (values[-1] - midpoints[-1])])
    )


def main():
    data_path = Path(__file__).resolve().parent.parent / "osz.json"
    with data_path.open(encoding="utf-8") as json_file:
        data = json.load(json_file)

    records = [
        entry for entry in data.values()
        if isinstance(entry, dict) and entry.get("ratio") == ratio
    ]
    if not records:
        raise ValueError(f"No OSZ records found for ratio {ratio:g}")

    lambdas = np.array(sorted({float(entry["Lambda"]) for entry in records}))
    kv_values = np.array(sorted({float(entry["KV"]) for entry in records}))
    lambda_indices = {value: index for index, value in enumerate(lambdas)}
    kv_indices = {value: index for index, value in enumerate(kv_values)}
    scores = np.full((len(lambdas), len(kv_values)), np.nan)

    for entry in records:
        lambda_value = float(entry["Lambda"])
        kv_value = float(entry["KV"])
        scores[lambda_indices[lambda_value], kv_indices[kv_value]] = float(
            entry["score"]
        )

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("lightgray")
    fig, ax = plt.subplots(figsize=(8, 6))
    heatmap = ax.pcolormesh(
        cell_edges(kv_values),
        cell_edges(lambdas),
        np.ma.masked_invalid(scores),
        shading="flat",
        cmap=cmap,
        vmin=0,
        vmax=1,
    )
    ax.set_xticks(kv_values)
    ax.set_yticks(lambdas)
    ax.set_xlabel("KV")
    ax.set_ylabel("Lambda")
    ax.set_title(f"OSZ score (ratio = {ratio:g})")
    fig.colorbar(heatmap, ax=ax, label="Score")
    fig.tight_layout()

    output_path = data_path.with_name(f"osz_heatmap_ratio_{ratio:g}.png")
    fig.savefig(output_path, dpi=200)
    print(f"Saved heatmap to {output_path}")


if __name__ == "__main__":
    main()