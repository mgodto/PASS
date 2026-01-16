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

# 2. 論文定義顏色 (Paper Colors)
# Strict adherence to your text:
# Head: Red, Trunk: Green
# L.Arm: Blue, R.Arm: Navy
# L.Leg: Purple, R.Leg: Pink
PARTS = {
    "Head":      (list(range(0, 11)), "#E74C3C"),       # Red
    "Trunk":     ([11, 12, 23, 24],   "#2ECC71"),       # Green
    "L.Arm":     ([11, 13, 15, 17, 19, 21], "#3498DB"), # Blue (Left)
    "R.Arm":     ([12, 14, 16, 18, 20, 22], "#1A253A"), # Navy (Right)
    "L.Leg":     ([23, 25, 27, 29, 31],     "#9B59B6"), # Purple (Left)
    "R.Leg":     ([24, 26, 28, 30, 32],     "#FD79A8"), # Pink (Right)
}

# 3. 定義 45度立體座標 (3/4 Perspective View)
# 假設人物面向「左前方」行走
# Left Side (Blue/Purple) 是 "近端" (Near)，畫在前面
# Right Side (Navy/Pink) 是 "遠端" (Far)，畫在後面
pose_2d = np.zeros((33, 2))

# --- Head (0-10) ---
# 頭部微微轉向左
pose_2d[0] = [-0.05, 1.75] # Nose
pose_2d[1] = [ 0.00, 1.78]; pose_2d[4] = [-0.10, 1.76] # Eyes
pose_2d[2] = [ 0.02, 1.80]; pose_2d[5] = [-0.12, 1.78]
pose_2d[3] = [ 0.05, 1.82]; pose_2d[6] = [-0.14, 1.76]
pose_2d[7] = [ 0.10, 1.75]; pose_2d[8] = [-0.15, 1.72] # Ears
pose_2d[9] = [-0.03, 1.68]; pose_2d[10] = [-0.07, 1.68] # Mouth

# --- Trunk (立體梯形) ---
# 左肩(11)在近端，右肩(12)在遠端且較高(透視)
pose_2d[11] = [-0.15, 1.50]  # L.Shoulder (Near)
pose_2d[12] = [ 0.15, 1.55]  # R.Shoulder (Far)
# 左髖(23)在近端，右髖(24)在遠端
pose_2d[23] = [-0.10, 0.85]  # L.Hip (Near)
pose_2d[24] = [ 0.12, 0.90]  # R.Hip (Far)

# --- Legs (大跨步) ---
# 左腳 (Purple, Near): 向前跨出 (Toward bottom-left)
pose_2d[25] = [-0.35, 0.45]  # L.Knee
pose_2d[27] = [-0.50, 0.10]  # L.Ankle (Landing)
# 左腳掌 (落地)
pose_2d[29] = [-0.55, 0.05]; pose_2d[31] = [-0.45, 0.05]

# 右腳 (Pink, Far): 向後推蹬 (Toward top-right)
pose_2d[26] = [ 0.30, 0.50]  # R.Knee
pose_2d[28] = [ 0.45, 0.20]  # R.Ankle (High)
# 右腳掌 (腳尖離地)
pose_2d[30] = [0.45, 0.10]; pose_2d[32] = [0.55, 0.12]

# --- Arms (自然擺動) ---
# 左手 (Blue, Near): 向後擺動 (因為左腳在前)
pose_2d[13] = [ 0.00, 1.20]  # L.Elbow
pose_2d[15] = [ 0.15, 1.00]  # L.Wrist
# 左手掌
pose_2d[17] = [0.18, 0.95]; pose_2d[19] = [0.20, 0.93]; pose_2d[21] = [0.16, 0.93]

# 右手 (Navy, Far): 向前擺動 (因為右腳在後)
pose_2d[14] = [-0.10, 1.30]  # R.Elbow
pose_2d[16] = [-0.30, 1.15]  # R.Wrist
# 右手掌
pose_2d[18] = [-0.33, 1.18]; pose_2d[20] = [-0.35, 1.20]; pose_2d[22] = [-0.31, 1.20]


# --- 繪圖 ---
fig, ax = plt.subplots(figsize=(6, 8))
ax.set_aspect('equal')
ax.axis('off')

# 1. 畫部位包圍區 (Convex Hulls)
# 關鍵：先畫遠端 (Right Side)，再畫近端 (Left Side)，製造遮擋效果
draw_order = ["R.Arm", "R.Leg", "Trunk", "Head", "L.Arm", "L.Leg"]

for part_name in draw_order:
    if part_name in PARTS:
        indices, color = PARTS[part_name]
        points = pose_2d[indices]
        
        if len(points) >= 3:
            try:
                hull = ConvexHull(points)
                hull_points = points[hull.vertices]
                hull_points = np.append(hull_points, [hull_points[0]], axis=0)
                
                # 遠端(Right)透明度高一點，近端(Left)實一點
                is_far = "R." in part_name
                alpha_fill = 0.2 if is_far else 0.4
                lw = 1.5 if is_far else 2.0
                
                ax.fill(hull_points[:, 0], hull_points[:, 1], color=color, alpha=alpha_fill, zorder=0)
                ax.plot(hull_points[:, 0], hull_points[:, 1], color=color, linewidth=lw, alpha=0.9, linestyle='--')
            except: pass

# 2. 畫骨架連線
for i, j in POSE_CONNECTIONS:
    ax.plot([pose_2d[i, 0], pose_2d[j, 0]], [pose_2d[i, 1], pose_2d[j, 1]], 
            color='#444444', linewidth=1.5, zorder=1)

# 3. 畫關鍵點
for idx in range(33):
    c = '#000000'
    for name, (indices, color) in PARTS.items():
        if idx in indices:
            c = color; break
    
    # 近端點大一點，遠端點小一點
    is_left_side = idx in PARTS["L.Arm"][0] or idx in PARTS["L.Leg"][0] or idx == 11 or idx == 23
    s_size = 70 if is_left_side else 50
    z_ord = 3 if is_left_side else 2
    
    ax.scatter(pose_2d[idx, 0], pose_2d[idx, 1], s=s_size, c=c, edgecolors='white', linewidth=1.5, zorder=z_ord)

# 4. 加上 Label
labels = {
    0: ("Head", (0.05, 1.9)), 
    11: ("Trunk", (-0.4, 1.5)), # 標在左肩旁
    15: ("L.Arm", (0.22, 1.0)), # 標在左手旁
    16: ("R.Arm", (-0.50, 1.15)), # 標在右手旁
    25: ("L.Leg", (-0.6, 0.5)), # 標在左膝旁
    26: ("R.Leg", (0.55, 0.5))  # 標在右膝旁
}

for idx, (text, pos) in labels.items():
    c = '#000000'
    for name, (indices, color) in PARTS.items():
        if idx in indices:
            c = color; break
    ax.text(pos[0], pos[1], text, fontsize=12, fontweight='bold', color=c, zorder=5)

plt.tight_layout()
plt.savefig('mediapipe_3d_perspective.pdf', dpi=300, bbox_inches='tight')
plt.show()