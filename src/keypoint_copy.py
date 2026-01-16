import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.patches as mpatches

# 1. MediaPipe 33 點連接關係
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), # Head
    (9, 10), # Mouth
    (11, 12), # Shoulders
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19), # Left Arm
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20), # Right Arm
    (11, 23), (12, 24), (23, 24), # Trunk
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31), # Left Leg
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32)  # Right Leg
]

# ====== Shape-difference demo (Option A) settings ======
SHOW_TRUNK_DIFF = True
DIFF_SHIFT = np.array([0.22, -0.02])   # (dx, dy) in your 2D coordinates; adjust to taste
DIFF_ALPHA = 0.18
DIFF_LINE_ALPHA = 0.5
DIFF_LW = 2.0

# ====== Option B: Shape Subspace Demo (Trunk) ======
SHOW_SUBSPACE = False

SUBSPACE_ORIGIN = np.array([1.10, 0.95])   # 右側位置（依你畫布微調）
SUBSPACE_W = 0.55
SUBSPACE_H = 0.40



# 2. 論文定義顏色 (Paper Colors)
PARTS = {
    "Head":      (list(range(0, 11)), "#E74C3C"),       # Red
    "Trunk":     ([11, 12, 23, 24],   "#2ECC71"),       # Green
    "L.Arm":     ([11, 13, 15, 17, 19, 21], "#3498DB"), # Blue (Left)
    "R.Arm":     ([12, 14, 16, 18, 20, 22], "#1A253A"), # Navy (Right)
    "L.Leg":     ([23, 25, 27, 29, 31],     "#9B59B6"), # Purple (Left)
    "R.Leg":     ([24, 26, 28, 30, 32],     "#FD79A8"), # Pink (Right)
}

# ====== 3D -> 2D 產生「側面走路 + 45度朝向螢幕」 ======
def rot_y(points_xyz: np.ndarray, yaw_deg: float) -> np.ndarray:
    """Rotate around vertical axis (Y)."""
    yaw = np.deg2rad(yaw_deg)
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[ c, 0, s],
                  [ 0, 1, 0],
                  [-s, 0, c]], dtype=float)
    return points_xyz @ R.T

def project_to_2d(points_xyz: np.ndarray,
                  proj_k: float = 0.90,
                  persp_k: float = 0.12) -> np.ndarray:
    """
    Simple perspective-ish projection:
      screen_x = (-X + proj_k * Z) / (1 + persp_k * Z)
      screen_y = ( Y ) / (1 + persp_k * Z)
    """
    X, Y, Z = points_xyz[:, 0], points_xyz[:, 1], points_xyz[:, 2]
    denom = 1.0 + persp_k * Z
    denom = np.clip(denom, 0.5, 2.0)
    x2 = (-X + proj_k * Z) / denom
    y2 = ( Y ) / denom
    return np.stack([x2, y2], axis=1)

def build_sidewalk_pose_3d(
    yaw_deg: float = 38,        # 20~35 通常好看；越大越朝向鏡頭
    stride: float = 0.60,         # 步伐前後距離
    arm_swing: float = 0.38,      # 手擺幅度
    shoulder_w: float = 0.26,     # 肩寬(用Z表示左右深度差)
    hip_w: float = 0.20,          # 髖寬(用Z表示左右深度差)
    torso_lean: float = 0.06,     # 身體微前傾
) -> np.ndarray:
    p = np.zeros((33, 3), dtype=float)

    # heights
    y_head = 1.78
    y_sh  = 1.52
    y_hip = 0.90
    y_knee_front = 0.48
    y_knee_back  = 0.56
    y_ankle_front = 0.12
    y_ankle_back  = 0.22

    # torso centers (side view)
    cx_hip = 0.00
    cx_sh  = cx_hip + torso_lean

    # near/far separation (Z)
    zL_sh, zR_sh = +shoulder_w/2, -shoulder_w/2
    zL_hip, zR_hip = +hip_w/2, -hip_w/2

    # Head (0-10): nose points forward (X+), tiny Z variations
    p[0]  = [cx_sh + 0.10, y_head, 0.00]
    p[1]  = [cx_sh + 0.07, y_head+0.03, +0.03]; p[4]  = [cx_sh + 0.07, y_head+0.03, -0.03]
    p[2]  = [cx_sh + 0.06, y_head+0.05, +0.05]; p[5]  = [cx_sh + 0.06, y_head+0.05, -0.05]
    p[3]  = [cx_sh + 0.05, y_head+0.07, +0.07]; p[6]  = [cx_sh + 0.05, y_head+0.07, -0.07]
    p[7]  = [cx_sh + 0.00, y_head+0.00, +0.12]; p[8]  = [cx_sh + 0.00, y_head+0.00, -0.12]
    p[9]  = [cx_sh + 0.10, y_head-0.08, +0.02]; p[10] = [cx_sh + 0.10, y_head-0.08, -0.02]

    # Trunk
    p[11] = [cx_sh,  y_sh,  zL_sh]
    p[12] = [cx_sh,  y_sh,  zR_sh]
    p[23] = [cx_hip, y_hip, zL_hip]
    p[24] = [cx_hip, y_hip, zR_hip]

    # ====== Legs (ergonomic knee bend, same bend direction) ======
    # bend_dir: +1 或 -1，用來決定「彎曲外凸」的方向；左右腿要一致就用同一個
    bend_dir = +1

    # 彎曲幅度：右腿小一點
    knee_bend_L = 0.18   # 左腿（前擺）彎較明顯
    knee_bend_R = 0.22   # 右腿（後擺）彎較小

    # 先定義 ankle 位置（跟你原本一致）
    ankle_L = np.array([cx_hip + stride*0.95, 0.12, zL_hip])
    ankle_R = np.array([cx_hip - stride*0.75, 0.22, zR_hip])

    hip_L = np.array([cx_hip, y_hip, zL_hip])
    hip_R = np.array([cx_hip, y_hip, zR_hip])

    # 讓 knee 先落在 hip->ankle 的線段上（t 控制膝蓋在腿上的比例）
    tL = 0.52
    tR = 0.50
    knee_L = (1 - tL) * hip_L + tL * ankle_L
    knee_R = (1 - tR) * hip_R + tR * ankle_R

    # 再把 knee 往「同一個方向」偏移，形成彎曲
    # 這裡偏移用 X(前後) + Y(高度) 的組合，能產生自然的屈膝外凸
    knee_L += np.array([+bend_dir * stride * knee_bend_L, 0.12, 0.0])
    knee_R += np.array([+bend_dir * stride * knee_bend_R, 0.08, 0.0])

    # 寫回到 mediapipe index
    p[25] = knee_L  # L.Knee
    p[27] = ankle_L # L.Ankle
    p[29] = [cx_hip + stride*1.05, 0.06, zL_hip + 0.03]
    p[31] = [cx_hip + stride*1.15, 0.05, zL_hip - 0.03]

    p[26] = knee_R  # R.Knee
    p[28] = ankle_R # R.Ankle
    p[30] = [cx_hip - stride*0.85, 0.07, zR_hip + 0.03]
    p[32] = [cx_hip - stride*0.70, 0.05, zR_hip - 0.03]



    # ====== Arms (ergonomic elbow bend, same bend direction) ======
    # 與腿一致：左右手彎曲方向相同，但幅度不同
    arm_bend_dir = -1   # 若方向顛倒，改成 -1

    elbow_bend_L = 0.10   # 左手（後擺）彎較明顯
    elbow_bend_R = 0.15   # 右手（前擺）彎較小

    # Wrist positions（沿用你原本的設定）
    wrist_L = np.array([cx_sh - arm_swing*0.75, 1.02, zL_sh])
    wrist_R = np.array([cx_sh + arm_swing*0.85, 1.05, zR_sh])

    shoulder_L = np.array([cx_sh, y_sh, zL_sh])
    shoulder_R = np.array([cx_sh, y_sh, zR_sh])

    # Elbow 初始：落在 shoulder -> wrist 線段上
    tL = 0.52
    tR = 0.50
    elbow_L = (1 - tL) * shoulder_L + tL * wrist_L
    elbow_R = (1 - tR) * shoulder_R + tR * wrist_R

    # 往同一方向偏移，形成自然彎曲
    elbow_L += np.array([arm_bend_dir * arm_swing * elbow_bend_L, -0.10, 0.0])
    elbow_R += np.array([arm_bend_dir * arm_swing * elbow_bend_R, -0.08, 0.0])
    # 寫回 MediaPipe index
    p[13] = elbow_L   # L.Elbow
    p[15] = wrist_L   # L.Wrist
    p[17] = [wrist_L[0] - 0.08, wrist_L[1] - 0.04, zL_sh + 0.03]
    p[19] = [wrist_L[0] - 0.10, wrist_L[1] - 0.06, zL_sh]
    p[21] = [wrist_L[0] - 0.08, wrist_L[1] - 0.06, zL_sh - 0.03]

    p[14] = elbow_R   # R.Elbow
    p[16] = wrist_R   # R.Wrist
    p[18] = [wrist_R[0] + 0.08, wrist_R[1] - 0.04, zR_sh + 0.03]
    p[20] = [wrist_R[0] + 0.10, wrist_R[1] - 0.06, zR_sh]
    p[22] = [wrist_R[0] + 0.08, wrist_R[1] - 0.06, zR_sh - 0.03]

    # yaw rotate to make 3/4
    p = rot_y(p, yaw_deg=yaw_deg)
    return p

# 產生 pose_2d
pose_3d = build_sidewalk_pose_3d(
    yaw_deg=10.0,      # 想更側面就調小：20；想更朝向鏡頭就調大：35
    stride=0.62,
    arm_swing=0.40,
    shoulder_w=0.26,
    hip_w=0.20,
    torso_lean=0.06,
)
pose_2d = project_to_2d(pose_3d, proj_k=0.90, persp_k=0.12)

# ====== 繪圖（沿用你原本流程） ======
fig, ax = plt.subplots(figsize=(6, 8))
ax.set_aspect('equal')
ax.axis('off')

# 1) Convex Hulls：先畫遠，再畫近（維持你的策略）
draw_order = ["R.Leg", "R.Arm", "Trunk", "Head", "L.Leg", "L.Arm"]

for part_name in draw_order:
    indices, color = PARTS[part_name]
    points = pose_2d[indices]

    if len(points) >= 3:
        try:
            hull = ConvexHull(points)
            hull_points = points[hull.vertices]
            hull_points = np.append(hull_points, [hull_points[0]], axis=0)

            alpha_val = 0.20 if "R." in part_name else 0.35
            ax.fill(hull_points[:, 0], hull_points[:, 1], color=color, alpha=alpha_val, zorder=0)
            ax.plot(hull_points[:, 0], hull_points[:, 1], color=color, linewidth=2, alpha=0.9, linestyle='--')
        except Exception:
            pass

# ====== Option A: Trunk shape difference (duplicate hull + arrow) ======
if SHOW_TRUNK_DIFF:
    trunk_indices, trunk_color = PARTS["Trunk"]
    trunk_pts = pose_2d[trunk_indices]

    if len(trunk_pts) >= 3:
        try:
            hull = ConvexHull(trunk_pts)
            hull_pts = trunk_pts[hull.vertices]
            hull_pts = np.append(hull_pts, [hull_pts[0]], axis=0)

            # shifted "next frame" trunk hull
            hull_shifted = hull_pts + DIFF_SHIFT

            # Fill (lighter)
            ax.fill(
                hull_shifted[:, 0], hull_shifted[:, 1],
                color=trunk_color, alpha=DIFF_ALPHA, zorder=0
            )

            # Border (dashed, slightly different style to show it's another frame)
            ax.plot(
                hull_shifted[:, 0], hull_shifted[:, 1],
                color=trunk_color, linewidth=DIFF_LW,
                alpha=DIFF_LINE_ALPHA, linestyle=(0, (4, 3)), zorder=0
            )

            # Arrow from original trunk centroid -> shifted trunk centroid
            c0 = trunk_pts.mean(axis=0)
            c1 = c0 + DIFF_SHIFT

            ax.annotate(
                "", xy=(c1[0], c1[1]), xytext=(c0[0], c0[1]),
                arrowprops=dict(arrowstyle="<-", linewidth=2.0, color="#444444"),
                zorder=3
            )

            # Label near arrow midpoint
            mid = (c0 + c1) / 2
            ax.text(
                mid[0] + 0.02, mid[1] + 0.02, r"$\Delta t$",
                fontsize=12, fontweight="bold", color="#444444", zorder=4
            )

        except Exception:
            pass



# 2) 骨架連線
for i, j in POSE_CONNECTIONS:
    ax.plot([pose_2d[i, 0], pose_2d[j, 0]],
            [pose_2d[i, 1], pose_2d[j, 1]],
            color='#444444', linewidth=1.5, zorder=1)

# 3) 關鍵點
for idx in range(33):
    c = '#000000'
    for name, (indices, color) in PARTS.items():
        if idx in indices:
            c = color
            break
    ax.scatter(pose_2d[idx, 0], pose_2d[idx, 1],
               s=60, c=c, edgecolors='white', linewidth=1.5, zorder=2)

# 4) Labels（位置稍微調一下，讓側面構圖更自然）

labels = {
    0:  ("Head", (pose_2d[0,0]  + 0.10, pose_2d[0,1]  + 0.12)),
    12: ("Trunk", (pose_2d[12,0] + 0.20, pose_2d[12,1] + 0.00)),
    16: ("Right Arm", (pose_2d[16,0] + 0.10, pose_2d[16,1] + 0.00)),
    15: ("Left Arm",  (pose_2d[15,0] + 0.0, pose_2d[15,1] + 0.1)),
    25: ("Left Leg",  (pose_2d[25,0] - 0.32, pose_2d[25,1] - 0.05)),
    26: ("Right Leg", (pose_2d[26,0] + 0.20, pose_2d[26,1] - 0.05)),
}

for idx, (text, pos) in labels.items():
    c = '#000000'
    for name, (indices, color) in PARTS.items():
        if idx in indices:
            c = color
            break
    ax.text(pos[0], pos[1], text, fontsize=12, fontweight='bold', color=c, zorder=5)

# ====== Legend (Body Parts) ======
legend_items = [
    ("Head",      PARTS["Head"][1]),
    ("Trunk",     PARTS["Trunk"][1]),
    ("Left Arm",  PARTS["L.Arm"][1]),
    ("Right Arm", PARTS["R.Arm"][1]),
    ("Left Leg",  PARTS["L.Leg"][1]),
    ("Right Leg", PARTS["R.Leg"][1]),
]

handles = [
    mpatches.Patch(facecolor=color, edgecolor='none', alpha=0.6, label=name)
    for name, color in legend_items
]

legend = ax.legend(
    handles=handles,
    loc="upper right",
    frameon=True,
    framealpha=0.95,
    fontsize=11,
    borderpad=0.8,
    labelspacing=0.6,
    handlelength=1.2,
    handleheight=0.8
)

if SHOW_SUBSPACE:
    # --- Draw subspace plane ---
    rect = plt.Rectangle(
        SUBSPACE_ORIGIN,
        SUBSPACE_W, SUBSPACE_H,
        facecolor="#EEEEEE",
        edgecolor="#666666",
        linewidth=1.5,
        zorder=0
    )
    ax.add_patch(rect)

    # Title
    ax.text(
        SUBSPACE_ORIGIN[0] + SUBSPACE_W / 2,
        SUBSPACE_ORIGIN[1] + SUBSPACE_H + 0.05,
        "Trunk Shape Subspace",
        ha="center", va="bottom",
        fontsize=12, fontweight="bold", color="#333333"
    )

    # --- Example projected points ---
    pts = np.array([
        [0.15, 0.20],
        [0.30, 0.28],
        [0.42, 0.22],
    ])

    pts_world = pts * np.array([SUBSPACE_W, SUBSPACE_H]) + SUBSPACE_ORIGIN

    ax.scatter(
        pts_world[:, 0], pts_world[:, 1],
        s=45, color=PARTS["Trunk"][1],
        edgecolors="white", linewidth=1.2,
        zorder=2
    )

    # --- Delta vector in subspace ---
    ax.annotate(
        "",
        xy=pts_world[2],
        xytext=pts_world[1],
        arrowprops=dict(
            arrowstyle="->",
            linewidth=2.0,
            color="#444444"
        ),
        zorder=3
    )

    mid = (pts_world[1] + pts_world[2]) / 2
    ax.text(
        mid[0] + 0.02, mid[1],
        r"$\Delta$",
        fontsize=12, color="#444444"
    )


legend.get_frame().set_linewidth(0.8)
legend.get_frame().set_edgecolor("#444444")

plt.tight_layout()




# 向量檔：最推薦（Figma/Illustrator/Inkscape 可直接編輯文字、線段、形狀）
plt.savefig("partition.svg", bbox_inches="tight")
plt.savefig("partition.pdf", dpi=300, bbox_inches="tight")  # PDF 也是向量；dpi 主要影響內嵌點陣元素

# 點陣預覽：方便快速看效果
plt.savefig("partition.png", dpi=300, bbox_inches="tight")

plt.show()
