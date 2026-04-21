import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import transforms

# ==========================
# 配置
# ==========================
EGO_DIM = 6
PARTNER_NUM = 63
PARTNER_DIM = 6
ROAD_NUM = 200
ROAD_DIM = 13

TOTAL_DIM = EGO_DIM + PARTNER_NUM * PARTNER_DIM + ROAD_NUM * ROAD_DIM


ROAD_TYPE_NAMES = [
    "none", "RoadLine", "RoadEdge", "RoadLane",
    "CrossWalk", "SpeedBump", "StopSign"
]
ROAD_TYPE_COLORS = {
    "none": "#999999",
    "RoadLine": "#ffffff",
    "RoadEdge": "#ff8800",
    "RoadLane": "#00bfff",
    "CrossWalk": "#00ff7f",
    "SpeedBump": "#ff00ff",
    "StopSign": "#ff3333",
}

def draw_vehicle(ax, cx, cy, length, width, yaw, color="cyan", alpha=0.8, lw=1.0):
    rect = Rectangle((-length/2, -width/2), length, width,
                     linewidth=lw, edgecolor=color, facecolor="none", alpha=alpha)
    t = transforms.Affine2D().rotate_around(0, 0, yaw).translate(cx, cy) + ax.transData
    rect.set_transform(t)
    ax.add_patch(rect)
    
    head_x = cx + (length / 2) * np.cos(yaw)
    head_y = cy + (length / 2) * np.sin(yaw)
    ax.plot([cx, head_x], [cy, head_y], color=color, linewidth=lw+0.5, alpha=alpha)

def visualize_single_obs_and_save(ego, partners, roads, save_path, dpi=160):
    """
    Visualize extracted features.
    :param ego: (6,) array
    :param partners: (M, 6) array
    :param roads: (N, 13) array — variable length
    """
    _, ego_len, ego_wid, rel_goal_x, rel_goal_y, is_collided = ego
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_facecolor("#1e1e1e")
    
    # ego
    ego_color = "red" if is_collided > 0.5 else "lime"
    draw_vehicle(ax, 0, 0, max(ego_len, 0.1), max(ego_wid, 0.1), 0.0, color=ego_color, lw=2.0)
    ax.scatter([0], [0], c=ego_color, s=60)
    
    # goal
    ax.scatter([rel_goal_x], [rel_goal_y], c="yellow", marker="*", s=180)
    ax.plot([0, rel_goal_x], [0, rel_goal_y], "--", color="yellow", alpha=0.6)
    
    # partner
    for p in partners:
        if np.allclose(p, 0.0, atol=1e-6):
            continue
        _, px, py, yaw, plen, pwid = p
        draw_vehicle(ax, px, py, max(plen, 0.1), max(pwid, 0.1), yaw, color="#00d4ff", alpha=0.75, lw=1.0)
        ax.scatter([px], [py], c="#00d4ff", s=10, alpha=0.9)
        
    # road
    for r in roads:
        if np.allclose(r, 0.0, atol=1e-6):
            continue
        x, y, seg_len, _, _, ori = r[:6]
        onehot = r[6:13]
        t_idx = int(np.argmax(onehot)) if np.sum(onehot) > 0 else 0
        c = ROAD_TYPE_COLORS[ROAD_TYPE_NAMES[t_idx]]
        ax.scatter([x], [y], c=c, s=8, alpha=0.9)
        dx = float(seg_len) * np.cos(float(ori))
        dy = float(seg_len) * np.sin(float(ori))
        ax.plot([x - dx, x + dx], [y - dy, y + dy], color=c, alpha=0.4, linewidth=1.0)
        
    ax.set_title("JSON Observation Visualization (single D=2984)")
    ax.set_xlabel("x (ego frame)")
    ax.set_ylabel("y (ego frame)")
    ax.axis("equal")
    ax.set_xlim(-80, 80)
    ax.set_ylim(-80, 80)
    ax.grid(True, alpha=0.2)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {save_path}")
