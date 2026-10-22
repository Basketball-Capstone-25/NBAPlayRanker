"""Draw the comparison from the saved, unchanged evaluation report."""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

folder = Path(__file__).resolve().parent
source = folder / "report.json"
report = json.loads(source.read_text())
final = report["final_test"]
models = ["ridge", "prior_league", "prior_team"]
names = ["Ridge", "Previous league PPP", "Previous team PPP"]
colors = ["#365fa8", "#93a7bd", "#93a7bd"]
fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), dpi=160)
fig.patch.set_facecolor("white")
fig.suptitle("Next-season PPP: held-out 2024-25 results", x=0.03, ha="left",
             fontsize=17, fontweight="bold", color="#172642")
fig.text(0.03, 0.88, "2023-24 inputs only | 330 team/play-type rows | Lower error is better",
         fontsize=10, color="#4f5e70")
for axis, key, title in zip(axes, ["mae_ppp", "rmse_ppp"],
                          ["Mean absolute error (MAE)", "Root mean squared error (RMSE)"]):
    values = [final["metrics"][model][key] for model in models]
    bars = axis.barh(names, values, color=colors, height=0.56)
    axis.invert_yaxis()
    axis.set_xlim(0, 0.095)
    axis.set_xticks([0, 0.02, 0.04, 0.06, 0.08])
    axis.set_xlabel("Points per possession (PPP)", fontsize=9)
    axis.set_title(title, loc="left", fontsize=11, pad=12)
    axis.xaxis.grid(True, color="#e4e9ef", linewidth=0.7)
    axis.set_axisbelow(True)
    axis.tick_params(axis="both", length=0, labelsize=9, pad=7)
    for spine in axis.spines.values():
        spine.set_visible(False)
    for bar, value in zip(bars, values):
        axis.text(value + 0.0014, bar.get_y() + bar.get_height() / 2,
                  f"{value:.4f}", va="center", fontsize=10, color="#172642")
fig.subplots_adjust(left=0.18, right=0.96, top=0.75, bottom=0.28, wspace=1.05)
fig.text(0.03, 0.12,
         "Ridge has a small advantage over the league baseline and an average overprediction of 0.0223 PPP.\n"
         "This is one historical season-ahead test, not proof of future or pre-game accuracy.",
         fontsize=10, color="#4f5e70", linespacing=1.5)
output = folder / "results.png"
fig.savefig(output, facecolor="white")
plt.close(fig)
(folder / "chart-provenance.json").write_text(json.dumps({
    "source": "report.json",
    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "output": "results.png",
    "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    "metrics": {model: {key: final["metrics"][model][key]
                        for key in ("mae_ppp", "rmse_ppp")} for model in models},
    "note": "Chart only; the evaluation was not rerun or modified."
}, indent=2) + "\n")
