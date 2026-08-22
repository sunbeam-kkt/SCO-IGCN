import sys
import random
import numpy as np
sys.path.extend(['../'])
from rotation import *  # 随机旋转，数据增强
from tqdm import tqdm  # 进度条指令
fu = 300  # Uniform number of frames：统一帧数

# def pre_normalization(data, zaxis=[0, 1], xaxis=[8, 4]):  # 预标准化
#     N, C, T, V, M = data.shape
#     s = np.transpose(data, [0, 4, 2, 3, 1])  # N, C, T, V, M  to  N, M, T, V, C
    
#     print('Execute the first stage of frame insertion')  # 前半部分插帧
#     for i_s, skeleton in enumerate(tqdm(s)):  # 遍历N，skeleton：[M, T, V, C]
#         if skeleton.sum() == 0:
#             print(i_s, ' has no skeleton')
#         for i_p, person in enumerate(skeleton):  # 遍历M，person：[T, V, C]
#             if person.sum() == 0:
#                 continue
#             if person[0].sum() == 0:        # person[0]:T,person[0].sum() == 0表示第一帧为空
#                 index = (person.sum(-1).sum(-1) != 0)
#                 tmp = person[index].copy()
#                 person *= 0
#                 person[:len(tmp)] = tmp
                
#             # keyframe localization
#             T_person, V_person, C_person = person.shape
#             s_mean = np.mean(person, axis=0, keepdims=True)   # N, C, T, V, M  to  N, M, V, C, gain mean frame
#             key_frame_ID = []
#             for t in T_person:
#                 euclidean_distance = np.linalg.norm(person[t, :, :] - s_mean, axis=0)   # compute mean euclidean distance 
#                 key_frame_ID.append(euclidean_distance)
            
#             min_value_index = np.argmin(np.abs(key_frame_ID))  # 获取绝对值最小元素的索引
#             key_frame = data[min_value_index]                  # 获取关键帧 key_frame
#             # after-keyframe
#             f_after = T - min_value_index
#             # pre-keyframe
#             f_pre = T - f_after - 1
                
#             for i_f, frame in enumerate(person):  # 遍历T，frame：[V, C]，i_f is frame ID
#                 if frame.sum() == 0:    # 如果是空白帧
#                     if person[i_f:].sum() == 0:  # 如果从i_f到最后的和为0
                        
#                         fo = i_f + 1   # fo:Original frames：每个动作的原始帧数
#                         DM = np.zeros((fo-1, V, C))  # DM:Displacement matrix:位移矩阵,生成f0-1块V行C列的全零矩阵
#                         FF = np.zeros((fu, V, C))  # FF:Fill frame matrix:填充框架矩阵，生成填充之前的600个V行C列的全零矩阵
#                         if fu % fo != 0:   # 如果fu不能整除fo，即fu不是fo的倍数
#                             ff = fu // fo  # ff:The number of frames that need to be filled between frames：帧之间需要填充的帧数，ff为整数
#                             fd = (fo * (ff + 1)) - fu  # fd:Number of frames that need to be discarded：需要丢弃的帧数
#                         else:    # 如果fu是fo的倍数
#                             ff = fu // fo - 1
#                             fd = 0  

#                         for t in range(fo):    # 用t来遍历矩阵的个数
#                             for v in range(V):  # 用v来遍历行数
#                                 for c in range(C):   # 用c来遍历列数
#                                     if t < i_f:   # 如果t在原始帧数之前
#                                         DM[t, v, c] = person[t + 1, v, c] - person[t, v, c]   # 位移矩阵等于原始矩阵中相邻两帧的差
#                                         Nid = DM[t, v, c] / (ff + 1)   # Nid:New inter-frame displacement：新的帧间位移,就是原先的帧间位移除以插帧之后的帧间距的数量，由原先的1变为ff+1
#                                     else:  # 如果t在原始帧数之后，即最后一帧
#                                         Nid = DM[t-1, v, c] / (ff + 1)  # Nid:New inter-frame displacement：新的帧间位移

#                                     for f in range(ff+1):
#                                         pnl = 0  # The proportion of noise level：噪声级比例
#                                         Nc = f * Nid * (1+pnl) + person[t, v, c]  # Nc:New coordinates：新坐标
#                                         if (fd == 0) or ((f+t*(ff+1)) <= (fu-1)): 
#                                             FF[f + t*(ff+1), v, c] = Nc
#                                         if (fd != 0) and ((f+t*(ff+1)) >= (fu-1)):
#                                             if f < (ff+1-fd):
#                                                 FF[f + t*(ff+1), v, c] = Nc         

#                         s[i_s, i_p, :] = FF
#                         break


def pre_normalization(data, zaxis=[0, 1], xaxis=[8, 4]):
    N, C, T, V, M = data.shape
    s = np.transpose(data, [0, 4, 2, 3, 1])  # N, C, T, V, M  to  N, M, T, V, C

    print('Keyframe localization and sequence interpolation preprocessing')  # 执行KL-SI处理
    
    for i_s, skeleton in enumerate(tqdm(s)):  # 遍历每个batch，skeleton：[M, C, T, V], i_s为每个样本ID
        if skeleton.sum() == 0:  # 如果是空样本则打印信息
            print(i_s, ' has no skeleton')
        
        for i_p, person in enumerate(skeleton):  # 遍历每个人，person：[T, V, C], i_p为每个人ID
            if person.sum() == 0:  # 如果当前人坐标全为0则跳过
                continue

            # 1. 计算所有帧的平均值
            mean_frame = person.mean(axis=0)  # Shape: [V, C]，所有输入帧的平均值
            
            # 2. 计算每一帧与平均帧的欧氏距离
            min_dist = float('inf')  # 预定义
            key_frame_index = -1  # same as above

            for i_f, frame in enumerate(person):
                dist = np.linalg.norm(frame - mean_frame)  # 计算欧氏距离
                if dist < min_dist:
                    min_dist = dist
                    key_frame_index = i_f  # 找到距离最小的关键帧

            # 3. 根据关键帧前后的数据进行插帧
            # 确定插帧的区间：关键帧前和关键帧后的数据
            before_key_frame = person[:key_frame_index]
            after_key_frame = person[key_frame_index+1:]

            # 插帧函数：将数据扩展到指定的帧数（150帧）
            def insert_frames(frames, target_frames=150):
                fo = len(frames)
                if fo >= target_frames:
                    drop_frame = fo - target_frames
                    frame = frames[drop_frame + 1:, :, :]
                    return frame  # 如果原始帧数已经大于等于目标帧数，直接返回原序列的后150帧

                DM = np.zeros((fo - 1, V, C))  # 帧间位移矩阵
                FF = np.zeros((300, V, C))  # 填充帧矩阵
                ff = (target_frames - 1) // (fo - 1)  # 计算每两帧之间需要插入的帧数
                fd = (target_frames - 1) % (fo - 1)  # 计算需要丢弃的帧数

                # 计算每两帧之间的位移，并填充插值数据
                for t in range(fo - 1):
                    for v in range(V):
                        for c in range(C):
                            DM[t, v, c] = frames[t + 1, v, c] - frames[t, v, c]  # 原始的帧间位移矩阵
                            Nid = DM[t, v, c] / (ff + 1)  # 新的帧间位移矩阵
 
                            for f in range(ff):
                                Nc = f * Nid + frames[t, v, c]
                                FF[f + t * (ff + 1), v, c] = Nc

                
                FF = FF[:target_frames+1]

                return FF

            # 插帧：关键帧前和关键帧后分别插帧
            before_inserted = insert_frames(before_key_frame, target_frames=150)
            after_inserted = insert_frames(after_key_frame, target_frames=150)

            # 4. 拼接前后两部分
            full_data = np.concatenate([before_inserted, after_inserted], axis=0)  # 拼接数据，去掉重复的关键帧

            # # 5. 每隔 4 帧取出一帧，得到 60 帧
            # s = s[:, :, ::5, :, :]  # 每隔4帧取出一帧

            # 更新数据
            s[i_s, i_p, :] = full_data
            



    # print('Fill null frames according to the displacement between frames')  # 根据帧之间的位移填充空帧
    # for i_s, skeleton in enumerate(tqdm(s)):  # pad
    #     if skeleton.sum() == 0:
    #         print(i_s, ' has no skeleton')
    #     for i_p, person in enumerate(skeleton):
    #         if person.sum() == 0:
    #             continue
    #         if person[0].sum() == 0:
    #             index = (person.sum(-1).sum(-1) != 0)
    #             tmp = person[index].copy()
    #             person *= 0
    #             person[:len(tmp)] = tmp
    #         for i_f, frame in enumerate(person):
    #             if frame.sum() == 0:
    #                 if person[i_f:].sum() == 0:
                        
    #                     fo = i_f + 1   # fo:Original frames：每个动作的原始帧数
    #                     DM = np.zeros((fo-1, V, C))  # DM:Displacement matrix:位移矩阵,生成f0-1块V行C列的全零矩阵
    #                     FF = np.zeros((fu, V, C))  # FF:Fill frame matrix:填充框架矩阵，生成填充之前的600个V行C列的全零矩阵
    #                     if fu % fo != 0:   # 如果fu不能整除fo，即fu不是fo的倍数
    #                         ff = fu // fo  # ff:The number of frames that need to be filled between frames：帧之间需要填充的帧数，ff为整数
    #                         fd = (fo * (ff + 1)) - fu  # fd:Number of frames that need to be discarded：需要丢弃的帧数
    #                     else:    # 如果fu是fo的倍数
    #                         ff = fu // fo - 1
    #                         fd = 0  

    #                     for t in range(fo):    # 用t来遍历矩阵的个数
    #                         for v in range(V):  # 用v来遍历行数
    #                             for c in range(C):   # 用c来遍历列数
    #                                 if t < i_f:   # 如果t在原始帧数之前
    #                                     DM[t, v, c] = person[t + 1, v, c] - person[t, v, c]   # 位移矩阵等于原始矩阵中相邻两帧的差
    #                                     Nid = DM[t, v, c] / (ff + 1)   # Nid:New inter-frame displacement：新的帧间位移,就是原先的帧间位移除以插帧之后的帧间距的数量，由原先的1变为ff+1
    #                                 else:  # 如果t在原始帧数之后，即最后一帧
    #                                     Nid = DM[t-1, v, c] / (ff + 1)  # Nid:New inter-frame displacement：新的帧间位移

    #                                 for f in range(ff+1):
    #                                     pnl = 0  # The proportion of noise level：噪声级比例
    #                                     Nc = f * Nid * (1+pnl) + person[t, v, c]  # Nc:New coordinates：新坐标
    #                                     if (fd == 0) or ((f+t*(ff+1)) <= (fu-1)): 
    #                                         FF[f + t*(ff+1), v, c] = Nc
    #                                     if (fd != 0) and ((f+t*(ff+1)) >= (fu-1)):
    #                                         if f < (ff+1-fd):
    #                                             FF[f + t*(ff+1), v, c] = Nc         

    #                     s[i_s, i_p, :] = FF
    #                     break

    print('sub the center joint #1 (spine joint in ntu and neck joint in kinetics)')
    for i_s, skeleton in enumerate(tqdm(s)):
        if skeleton.sum() == 0:
            continue
        main_body_center = skeleton[0][:, 1:2, :].copy()
        for i_p, person in enumerate(skeleton):
            if person.sum() == 0:
                continue
            mask = (person.sum(-1) != 0).reshape(T, V, 1)
            s[i_s, i_p] = (s[i_s, i_p] - main_body_center) * mask

    print('parallel the bone between hip(jpt 0) and spine(jpt 1) of the first person to the z axis')
    for i_s, skeleton in enumerate(tqdm(s)):
        if skeleton.sum() == 0:
            continue
        joint_bottom = skeleton[0, 0, zaxis[0]]
        joint_top = skeleton[0, 0, zaxis[1]]
        axis = np.cross(joint_top - joint_bottom, [0, 0, 1])
        angle = angle_between(joint_top - joint_bottom, [0, 0, 1])
        matrix_z = rotation_matrix(axis, angle)
        for i_p, person in enumerate(skeleton):
            if person.sum() == 0:
                continue
            for i_f, frame in enumerate(person):
                if frame.sum() == 0:
                    continue
                for i_j, joint in enumerate(frame):
                    s[i_s, i_p, i_f, i_j] = np.dot(matrix_z, joint)

    print(
        'parallel the bone between right shoulder(jpt 8) and left shoulder(jpt 4) of the first person to the x axis')
    for i_s, skeleton in enumerate(tqdm(s)):
        if skeleton.sum() == 0:
            continue
        joint_rshoulder = skeleton[0, 0, xaxis[0]]
        joint_lshoulder = skeleton[0, 0, xaxis[1]]
        axis = np.cross(joint_rshoulder - joint_lshoulder, [1, 0, 0])
        angle = angle_between(joint_rshoulder - joint_lshoulder, [1, 0, 0])
        matrix_x = rotation_matrix(axis, angle)
        for i_p, person in enumerate(skeleton):
            if person.sum() == 0:
                continue
            for i_f, frame in enumerate(person):
                if frame.sum() == 0:
                    continue
                for i_j, joint in enumerate(frame):
                    s[i_s, i_p, i_f, i_j] = np.dot(matrix_x, joint)

    data = np.transpose(s, [0, 4, 2, 3, 1])
    return data


if __name__ == '__main__':
    data = np.load('../data/ntu/xview/val_data.npy')
    pre_normalization(data)
    np.save('../data/ntu/xview/data_val_pre.npy', data)
