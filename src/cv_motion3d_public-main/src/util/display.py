from ezc3d import c3d
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from scipy.linalg import eig
from config import Confing
import scipy
from matplotlib import colors
from util.preprocess import remove_nan
import os

def display_motion(path):
    c = c3d(path)
    point_data = c['data']['points'] #(XYZ1, num_mark, num_frame)
    num_frame = point_data.shape[2]
    data = point_data[0:3,:,0]
    

    x_range = np.nanmax(point_data[0,:,:]) - np.nanmin(point_data[0,:,:])
    y_range = np.nanmax(point_data[1,:,:]) - np.nanmin(point_data[1,:,:])
    z_range = np.nanmax(point_data[2,:,:]) - np.nanmin(point_data[2,:,:])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(data[0,:],data[1,:],data[2,:], s=5)
    ft = ax.set_title(f"frame num {0}")

    ax.set_box_aspect([x_range, y_range, z_range])
    ax.set_xlim(np.nanmin(point_data[0,:,:]),np.nanmax(point_data[0,:,:]))
    ax.set_ylim(np.nanmin(point_data[1,:,:]),np.nanmax(point_data[1,:,:]))
    ax.set_zlim(np.nanmin(point_data[2,:,:]),np.nanmax(point_data[2,:,:]))

    def update(frame):
        data = point_data[0:3,:,frame]
        sc._offsets3d = (data[0,:],data[1,:],data[2,:])
        ft.set_text(f"frame num {frame}")
        return sc

    ani = FuncAnimation(fig, update, frames=num_frame, interval=50, blit=False)

    plt.show()



def display_motion_score(path, x, y, save_path):
    c = c3d(path)
    point_data = c['data']['points'] #(XYZ1, num_mark, num_frame)
    point_data = np.nan_to_num(point_data)
    num_frame = point_data.shape[2]
    data = point_data[0:3,:,0]

    title = path.split('/')[2]

    x_range = np.max(point_data[0,:,:]) - np.min(point_data[0,:,:])
    y_range = np.max(point_data[1,:,:]) - np.min(point_data[1,:,:])
    z_range = np.max(point_data[2,:,:]) - np.min(point_data[2,:,:])

    fig = plt.figure(figsize=(10,4))
    ax1 = fig.add_subplot(121, projection='3d')
    sc = ax1.scatter(data[0,:],data[1,:],data[2,:], s=5)
    ft = ax1.set_title(f"frame num {0}")

    ax1.set_box_aspect([x_range, y_range, z_range])
    ax1.set_xlim(np.min(point_data[0,:,:]),np.max(point_data[0,:,:]))
    ax1.set_ylim(np.min(point_data[1,:,:]),np.max(point_data[1,:,:]))
    ax1.set_zlim(np.min(point_data[2,:,:]),np.max(point_data[2,:,:]))

    ax2 = fig.add_subplot(122)
    line, = ax2.plot([], [])
    ax2.set_xlim(np.min(x),np.max(x))
    ax2.set_ylim(np.min(y),np.max(y))
    ax2.set_xlabel('frame')
    ax2.set_ylabel('value')
    ax2.set_title(title)
    ax2.grid(True)

    def update(frame):
        data = point_data[0:3,:,frame]
        sc._offsets3d = (data[0,:],data[1,:],data[2,:])
        if x[0]<=frame and frame < x[-1]:
            line.set_data(x[0:frame-x[0]],y[0:frame-x[0]])
        ft.set_text(f"frame num {frame}")
        return sc, line

    ani = FuncAnimation(fig, update, frames=num_frame, interval=50, blit=False)

    ani.save(save_path, writer='pillow', fps=20)




def display_motion_score_contribution(path, x, y, contribution, save_path):
    config= Confing()
    title = path.split('/')[2]

    c = c3d(path)
    point_data = c['data']['points'] #(XYZ1, num_mark, num_frame)
    point_data = point_data[0:3,:,:]
    point_data = remove_nan(point_data)


    num_frame = point_data.shape[2]
    data = point_data[0:3,:,0]
    cb = contribution[0]
    norm = colors.Normalize(vmin=min(cb), vmax=max(cb))

    x_range = np.max(point_data[0,:,:]) - np.min(point_data[0,:,:])
    y_range = np.max(point_data[1,:,:]) - np.min(point_data[1,:,:])
    z_range = np.max(point_data[2,:,:]) - np.min(point_data[2,:,:])

    fig = plt.figure(figsize=(11,5))
    
    # ファイル名から拡張子を除いた部分をキーとして取得
    key = title.split('.')[0]

    # 辞書から説明文を取得（.get()を使うとキーが存在しなくてもエラーにならない）
    # 1. パスからファイル名を安全に取得 (例: 20190514_ASD_lat_V1-0007.c3d)
    title = os.path.basename(path)

    # 2. ファイル名から拡張子を除いたキーを取得 (例: 20190514_ASD_lat_V1-0007)
    # key = os.path.splitext(title)[0]

    # 3. 辞書から説明を取得。見つからない場合は、キー（ファイル名本体）をそのまま説明として使用する
    # description = config.motion_description.get(key, key)

    # 4. 最終的なタイトルを生成
    final_title = f'{title}'
    fig.suptitle(final_title)


    gs = fig.add_gridspec(9,11)
    ax1 = fig.add_subplot(gs[1:9,0:7], projection='3d')

    sc = ax1.scatter(data[0,:],data[1,:],data[2,:], c=cb, norm=norm, cmap='jet', s=5)
    ft = ax1.set_title(f"frame num {0}")

    ax1.set_box_aspect([x_range, y_range, z_range])
    ax1.set_xlim(np.min(point_data[0,:,:]),np.max(point_data[0,:,:]))
    ax1.set_ylim(np.min(point_data[1,:,:]),np.max(point_data[1,:,:]))
    ax1.set_zlim(np.min(point_data[2,:,:]),np.max(point_data[2,:,:]))

    
    ax2 = fig.add_subplot(gs[1:8,7:11])
    line, = ax2.plot([], [])
    ax2.set_xlim(np.min(x),np.max(x))
    ax2.set_ylim(np.min(y),np.max(y))
    ax2.set_xlabel('frame')
    ax2.set_ylabel('value')
    # --- f-stringを単純化し、ファイル名処理の堅牢性を向上させるための修正 ---
    filename = os.path.basename(save_path)

    # 從檔名中智慧地提取標籤
    label = "Unknown" # 預設值
    if "_first" in filename:
        label = "first"
    elif "_geodesic" in filename:
        label = "geodesic"
    elif "_second" in filename:
        label = "second"

    # 設定圖表標題
    ax2.set_title(f"Dissimilarity of {label} DS")
    ax2.grid(True)

    
    cbar = plt.colorbar(sc, pad=0.2)
    cbar.set_ticks([])


    def update(frame):
        data = point_data[0:3,:,frame]
        sc._offsets3d = (data[0,:],data[1,:],data[2,:])

        if x[0]<=frame and frame < x[-1]:
            line.set_data(x[0:frame-x[0]+1],y[0:frame-x[0]+1])
            cb = contribution[frame-x[0]+1]
            norm = colors.Normalize(vmin=min(cb), vmax=max(cb))
            sc.set_array(cb)
            sc.set_norm(norm)
        ft.set_text(f"frame num {frame}")
        cbar.set_ticks([])
        
        return sc, line, cbar

    ani = FuncAnimation(fig, update, frames=num_frame, interval=50, blit=False)

    #plt.show()
    ani.save(save_path, writer='pillow', fps=20)



#複数の波形を表示する
def display_motion_some_score(path, x, y, y_label, save_path):

    cfg = Confing()

    # yはarray
    #y[0] 値1
    #y[1] 値2
    #y_label[0] 値1のラベル
    #y_label[1] 値2のラベル
    title = os.path.basename(path)
    #title = path.split('/')[2]

    c = c3d(path)
    point_data = c['data']['points'] #(XYZ1, num_mark, num_frame)
    point_data = point_data[0:3,:,:]
    point_data = remove_nan(point_data)
    
    num_frame = point_data.shape[2]
    data = point_data[0:3,:,0]

    x_range = np.max(point_data[0,:,:]) - np.min(point_data[0,:,:])
    y_range = np.max(point_data[1,:,:]) - np.min(point_data[1,:,:])
    z_range = np.max(point_data[2,:,:]) - np.min(point_data[2,:,:])

    fig = plt.figure(figsize=(11,5))
    # --- f-stringを単純化し、ファイル名処理の堅牢性を向上させるための修正 ---
    # ファイル名から拡張子を除いた部分をキーとして取得
    key = title.split('.')[0]

    # 辞書から説明文を取得（.get()を使うとキーが存在しなくてもエラーにならない）
    # 1. パスからファイル名を安全に取得 (例: 20190514_ASD_lat_V1-0007.c3d)
    title = os.path.basename(path)

    # 2. ファイル名から拡張子を除いたキーを取得 (例: 20190514_ASD_lat_V1-0007)
    key = os.path.splitext(title)[0]

    # 3. 辞書から説明を取得。見つからない場合は、キー（ファイル名本体）をそのまま説明として使用する
    # description = config.motion_description.get(key, key)

    # 4. 最終的なタイトルを生成
    final_title = f'{title}'
    fig.suptitle(final_title)
    
    # f-stringを単純化して最終的なタイトルを作成
    gs = fig.add_gridspec(9,11)
    ax1 = fig.add_subplot(gs[0:9,0:6], projection='3d')

    sc = ax1.scatter(data[0,:],data[1,:],data[2,:], s=5)
    ft = ax1.set_title(f"frame num {0}")

    ax1.set_box_aspect([x_range, y_range, z_range])
    ax1.set_xlim(np.min(point_data[0,:,:]),np.max(point_data[0,:,:]))
    ax1.set_ylim(np.min(point_data[1,:,:]),np.max(point_data[1,:,:]))
    ax1.set_zlim(np.min(point_data[2,:,:]),np.max(point_data[2,:,:]))


    filename = os.path.basename(save_path)
    label = "Unknown" # 預設值
    if "_first" in filename:
        label = "first"
    elif "_geodesic" in filename:
        label = "geodesic"
    elif "_second" in filename:
        label = "second"

    ax2 = fig.add_subplot(gs[1:7,7:11])
    ax2.set_title(f"Subspace Dissimilarity curve")
    ax2.grid(True)

    line = []
    for i in range(len(y)):
        line.append(ax2.plot([], [], label=y_label[i]))

    ax2.set_xlim(np.min(x),np.max(x))
    ax2.set_ylim(np.min(y),np.max(y))
    ax2.set_xlabel('frame')
    ax2.set_ylabel('value')

    def update(frame):
        data = point_data[0:3,:,frame]
        sc._offsets3d = (data[0,:],data[1,:],data[2,:])

        if x[0]<=frame and frame < x[-1]:
            for i in range(len(y)):
                line[i][0].set_data(x[0:frame-x[0]+1],y[i][0:frame-x[0]+1])
        ft.set_text(f"frame num {frame}")
        
        return sc, line

    ani = FuncAnimation(fig, update, frames=num_frame, interval=50, blit=False)
    plt.legend()
    #plt.show()
    ani.save(save_path, writer='pillow', fps=20)
