import numpy as np

#data shape is (3, 41, frame_num)
#dataにnunが含まれていた場合、削除する。
#全フレーム内のポイントデータの数を共通にする(行列のサイズが合わないので)
def remove_nan(data):    
    list = [data[:,:,i] for i in range(data.shape[2])]
    new_list = []
    for i in range(len(list)):
        d = list[i]
        new_list.append(d[:,~np.isnan(d).any(axis=(0))])

    min_size = (min(new_list, key=lambda x:x.shape[1]).shape)[1]
    
    data = np.array([d[:,0:min_size] for d in new_list])
    data = np.transpose(data, (1,2,0))

    return data


if __name__ == "__main__": 
    data = np.ndarray([3,41,20])
    print(data.shape)
    data[2,0,0] = np.nan
    data[2,3,0] = np.nan
    data[2,4,5] = np.nan
    data = remove_nan(data)
    print(data.shape)

    