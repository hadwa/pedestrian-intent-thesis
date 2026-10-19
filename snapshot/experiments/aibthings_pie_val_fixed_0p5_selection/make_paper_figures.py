"""Paper figures: (1) RF feature-importance groups, (2) pipeline overview.

Inputs: rf_biu_bbox_group_importances.csv (this folder).
Outputs: fig_biu_bbox_importance.pdf/.png, fig_pipeline.pdf/.png (this folder).
Print-friendly, grayscale-safe, single-column IEEE size.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OUT = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({
    "font.size": 7.5,
    "font.family": "DejaVu Sans",
    "axes.linewidth": 0.6,
    "pdf.fonttype": 42,
})

INK = "#1f2430"
MUTED = "#5b6472"
BBOX_C = "#8a93a3"   # light neutral: BBox aggregate
BIU_C = "#2f6db3"    # single blue hue: BIU cues (darker in grayscale than BBOX_C)

grp = pd.read_csv(os.path.join(OUT, "rf_biu_bbox_group_importances.csv"))
grp = grp[grp["group"] != "BIU: other"].copy()
label_map = {
    "BIU: lead_foot_elevation": "lead_foot_elevation",
    "BIU: stride_length": "stride_length",
    "BIU: norm_shoulder_width": "norm_shoulder_width",
    "BIU: head_yaw_angle": "head_yaw_angle",
    "BIU: face_yaw_angle": "face_yaw_angle",
    "BIU: gaze_to_car_angle": "head_direction_proxy",
    "BIU: body_yaw_angle": "body_yaw_angle",
    "BIU: arm_swing_asymmetry": "arm_swing_asymmetry",
    "BIU: torso_lean_angle": "torso_lean_angle",
    "BIU: lead_foot_lifted": "lead_foot_lifted",
}
biu = grp[grp["group"] != "BBox"].copy()
biu["label"] = biu["group"].map(label_map)
biu = biu.sort_values("importance", ascending=True)
bbox_total = float(grp.loc[grp["group"] == "BBox", "importance"].iloc[0])
biu_total = float(biu["importance"].sum())

fig, (ax_top, ax) = plt.subplots(
    2, 1, figsize=(3.45, 2.55), gridspec_kw={"height_ratios": [1, 9], "hspace": 0.30})

# context strip: BBox vs BIU share of total importance
ax_top.barh([0], [bbox_total], color=BBOX_C, height=0.9, zorder=3)
ax_top.barh([0], [biu_total], left=[bbox_total], color=BIU_C, height=0.9, zorder=3)
ax_top.text(bbox_total / 2, 0, f"BBox {bbox_total:.0%}", ha="center", va="center",
            fontsize=6.2, color="#20242c")
ax_top.text(bbox_total + biu_total / 2, 0, f"BIU {biu_total + 0.001:.0%}", ha="center",
            va="center", fontsize=6.2, color="white", fontweight="bold")
ax_top.set_xlim(0, 1)
ax_top.axis("off")
ax_top.set_title("Share of total importance", fontsize=6.4, color=MUTED, pad=2)

# main panel: ranking of the ten named BIU cues
bars = ax.barh(biu["label"], biu["importance"], color=BIU_C, height=0.62, zorder=3)
for b, v in zip(bars, biu["importance"]):
    ax.text(v + 0.0008, b.get_y() + b.get_height() / 2, f"{v:.3f}",
            va="center", ha="left", fontsize=6.5, color=MUTED)
ax.set_xlabel("Gini importance of each BIU cue (share of total)", color=INK, fontsize=6.8)
ax.set_xlim(0, 0.055)
ax.tick_params(colors=INK, length=0, labelsize=6.8)
for s in ["top", "right", "left"]:
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(MUTED)
ax.xaxis.grid(True, color="#d7dbe2", linewidth=0.5, zorder=0)
ax.set_axisbelow(True)
fig.savefig(os.path.join(OUT, "fig_biu_bbox_importance.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "fig_biu_bbox_importance.png"), dpi=300, bbox_inches="tight")
plt.close(fig)

# ---- pipeline figure ----
fig, ax = plt.subplots(figsize=(3.45, 1.6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 3.6)
ax.axis("off")

def box(x, y, w, h, text, fc="#eef1f5", ec=MUTED, bold=False, fs=6.0):
    ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec,
                               linewidth=0.7, zorder=2, joinstyle="round"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=INK, zorder=3, fontweight="bold" if bold else "normal")

def arrow(x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.8))

# upstream perception (given)
box(0.15, 1.95, 2.15, 1.0, "Pedestrian\nboxes (given)")
box(0.15, 0.35, 2.15, 1.0, "Keypoints\n(ViTPose-S)")
# features
box(3.05, 1.95, 2.6, 1.0, "BBox features\n53 window stats")
box(3.05, 0.35, 2.6, 1.0, "10 BIU cues\n72 window stats")
# fusion + classifier
box(6.4, 1.15, 1.55, 1.0, "Fusion\n125-d")
box(8.5, 1.15, 1.35, 1.0, "RF:\ncross?", fc="#dbe7f6", ec=BIU_C, bold=True)
arrow(2.3, 2.45, 3.05, 2.45)
arrow(2.3, 0.85, 3.05, 0.85)
arrow(5.65, 2.45, 6.4, 1.85)
arrow(5.65, 0.85, 6.4, 1.45)
arrow(7.95, 1.65, 8.5, 1.65)
# dashed separation between upstream and downstream
ax.plot([2.72, 2.72], [0.15, 3.15], ls=(0, (3, 3)), lw=0.6, color="#b9c0cb")
ax.text(1.22, 3.32, "Upstream (given)", fontsize=6.0, color=MUTED, ha="center")
ax.text(6.4, 3.32, "Interpretable downstream stage (this work, CPU)",
        fontsize=6.0, color=MUTED, ha="center")
fig.tight_layout(pad=0.2)
fig.savefig(os.path.join(OUT, "fig_pipeline.pdf"))
fig.savefig(os.path.join(OUT, "fig_pipeline.png"), dpi=300)
print("figures written")
