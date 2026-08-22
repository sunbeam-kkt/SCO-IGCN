import os
import numpy as np

sets = {
    'train', 'val'
}

datasets = {'ntu/xsub', 'ntu/xview'}

for dataset in datasets:
    for set in sets:
        print(dataset, set)
        data_jpt = np.load('F:/Desktop/SCO-HGCN/SCO-HGCN-master/data/{}/{}_data_joint.npy'.format(dataset, set))
        data_jpt = data_jpt[:, :, ::5, :, :]  # T:300->60
        data_bone = np.load('F:/Desktop/SCO-HGCN/SCO-HGCN-master/data/{}/{}_data_bone.npy'.format(dataset, set))
        data_bone = data_bone[:, :, ::5, :, :]
        N, C, T, V, M = data_jpt.shape
        # print(data_jpt.shape)
        data_jpt_bone = np.concatenate((data_jpt, data_bone), axis=1)  
        np.save('F:/Desktop/SCO-HGCN/SCO-HGCN-master/data/{}/{}_data_joint_bone.npy'.format(dataset, set), data_jpt_bone)
