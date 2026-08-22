
import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from .activations import *
import torch.nn.functional as F

from scipy.fftpack import dct, idct

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if hasattr(m, 'bias') and m.bias is not None and isinstance(m.bias, torch.Tensor):
            nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)
    
    
class MS_TC(nn.Module): 
    def __init__(self, in_channels, kernel_size=3, expand_ratio=0.5, stride=1, dilations=[1, 3], residual=True, residual_kernel_size=1):
        super().__init__()
        inner_channel = int(in_channels * expand_ratio)
        compress_channel = int(in_channels // 4)
        self.expand_conv = nn.Sequential(
            nn.Conv2d(in_channels, inner_channel, 1, bias=True),
            nn.BatchNorm2d(inner_channel),
        )
        
        # four process branches
        self.branch1 = nn.Sequential(
            nn.Conv2d(inner_channel, inner_channel, kernel_size=kernel_size, stride=stride, padding=1),
            nn.BatchNorm2d(inner_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_channel, compress_channel, kernel_size=1, stride=stride),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(inner_channel, compress_channel, kernel_size=1, padding=0),
            nn.BatchNorm2d(compress_channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3,1), stride=(stride,1), padding=(1,0)),
            nn.BatchNorm2d(compress_channel),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(inner_channel, inner_channel, kernel_size=kernel_size, stride=stride, padding=1),
            nn.BatchNorm2d(inner_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_channel, compress_channel, kernel_size=1, stride=stride),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(inner_channel, compress_channel, kernel_size=1, padding=0),
            nn.BatchNorm2d(compress_channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3,1), stride=(stride,1), padding=(1,0)),
            nn.BatchNorm2d(compress_channel),
        )

        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(in_channels, in_channels, kernel_size=residual_kernel_size, stride=stride)
        # initialize
        self.apply(weights_init)

    def forward(self, x, y):
        # Input dim: (N,C,T,V)
        N, C, T, V = x.size()

        # print("x shape:", x.shape)  # x shape: torch.Size([64, 64, 300, 25])
        # print("y shape:", y.shape)  # y shape: torch.Size([64, 64, 300, 25])

        # 计算每一部分的大小
        split_size = C // 2   # 32

        # 切分 X, Y
        x1 = x[:, :split_size, :, :]
        x2 = x[:, split_size:, :, :]
        y1 = y[:, :split_size, :, :]
        y2 = y[:, split_size:, :, :]
        #
        # print("x1 shape:", x1.shape)  # x1 shape: torch.Size([64, 32, 300, 25])
        # print("x2 shape:", x2.shape)
        # print("y1 shape:", y1.shape)
        # print("y2 shape:", y2.shape)
        
        # retain or sampling
        # x1 = x1
        # x2 = x2[:, :, ::3, :]
        # y1 = y1
        # y2 = y2[:, :, ::3, :]

        # X branch
        x_out1 = self.branch1(x1)
        x_out2 = self.branch2(x1)
        x_out3 = self.branch3(x2)
        x_out4 = self.branch4(x2)
        
        # y branch
        y_out1 = self.branch1(y1)
        y_out2 = self.branch2(y1)
        y_out3 = self.branch3(y2)
        y_out4 = self.branch4(y2)

        # x_out1 = x_out1[:, :, ::3, :]
        # x_out2 = x_out2[:, :, ::3, :]
        # x_out3 = x_out3
        # x_out4 = x_out4

        # print("x_out1 shape:", x_out1.shape)
        # print("x_out3 shape:", x_out3.shape)

        # y_out1 = y_out1[:, :, ::3, :]
        # y_out2 = y_out2[:, :, ::3, :]
        # y_out3 = y_out3
        # y_out4 = y_out4

        # concate
        X = torch.cat((x_out1, x_out2, x_out3, x_out4), dim=1)
        Y = torch.cat((y_out1, y_out2, y_out3, y_out4), dim=1)

        res = self.residual(x)

        return X, Y


class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True):
        super(TemporalConv, self).__init__()
        pad = (kernel_size + (kernel_size-1) * (dilation-1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1),
            groups=16,
            bias=bias)

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class Zero_Layer(nn.Module):
    def __init__(self):
        super(Zero_Layer, self).__init__()

    def forward(self, x):
        return 0


class unit_gcn(nn.Module):      # spatial GCN + bn + relu
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, num_subset=3):
        super(unit_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding     
        self.inter_c = inter_channels
        self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
     
        nn.init.constant_(self.PA, 1e-6)
        self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.num_subset = num_subset

        self.conv_a = nn.ModuleList()
        self.conv_b = nn.ModuleList()
        self.conv_d = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv_a.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_b.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for i in range(self.num_subset):
            conv_branch_init(self.conv_d[i], self.num_subset)

    def forward(self, x):
        N, C, T, V = x.size()
        A = self.A.cuda(x.get_device())
        A = A + self.PA
        # A = self.PA.cuda(x.get_device())
        y = None
        for i in range(self.num_subset):
            A1 = self.conv_a[i](x).permute(0, 3, 1, 2).contiguous().view(N, V, self.inter_c * T) 
            A2 = self.conv_b[i](x).view(N, self.inter_c * T, V)        
            A1 = self.soft(torch.matmul(A1, A2) / A1.size(-1))  
            A1 = A1 + A[i]
            A2 = x.view(N, C * T, V)
            z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))
            y = z + y if y is not None else z

        y = self.bn(y)
        y += self.down(x)
        return self.relu(y)
    
def get_dct_matrix(N, is_torch=True):
    dct_m = np.eye(N)
    for k in np.arange(N):
        for i in np.arange(N):
            w = np.sqrt(2 / N)
            if k == 0:
                w = np.sqrt(1 / N)
            dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
    idct_m = np.linalg.inv(dct_m)
    if is_torch:
        dct_m = torch.from_numpy(dct_m)
        idct_m = torch.from_numpy(idct_m)
    return dct_m, idct_m

def dct2(x, norm='ortho'):
    """二维 DCT 变换"""
    return dct(dct(x.T, norm=norm).T, norm=norm)

def idct2(x, norm='ortho'):
    """二维 IDCT 变换"""
    return idct(idct(x.T, norm=norm).T, norm=norm)

def discard_high_frequencies(data, percentage=80, axis=-2):
    """
    对四维数据在指定维度(axis)上应用 DCT,并丢弃高频成分
    :param data: 输入数据，形状为 (N, C, T, V)
    :param percentage: 要保留的低频成分的百分比
    :param axis: 要进行 DCT 的维度，-1 表示对最后一个维度进行 DCT(如空间或时间维度)
    :return: 丢弃高频后的重构数据
    """
    N, C, T, V = data.shape
    
    # 选择对哪个维度进行 DCT
    if axis == -1:
        # 对最后一个维度（空间或特征维度 V）进行 DCT
        dct_data = np.array([dct2(data[n, c, t, :]) for n in range(N) for c in range(C) for t in range(T)])
        dct_data = dct_data.reshape(N, C, T, V)
    elif axis == -2:
        # 对时间维度（T）进行 DCT
        dct_data = np.array([dct2(data[n, c, :, v]) for n in range(N) for c in range(C) for v in range(V)])
        dct_data = dct_data.reshape(N, C, T, V)
    
    # 对 DCT 结果进行高频丢弃
    dct_data_flattened = np.abs(dct_data).flatten()
    sorted_indices = np.argsort(dct_data_flattened)[::-1]  # 从大到小排序
    num_to_keep = int(len(dct_data_flattened) * (percentage / 100))
    dct_data_flattened[sorted_indices[num_to_keep:]] = 0  # 将高频部分置为 0
    
    # 重构数据
    dct_data = dct_data_flattened.reshape(N, C, T, V)
    
    # if axis == -1:
    #     reconstructed_data = np.array([idct2(dct_data[n, c, t, :]) for n in range(N) for c in range(C) for t in range(T)])
    # elif axis == -2:
    #     reconstructed_data = np.array([idct2(dct_data[n, c, :, v]) for n in range(N) for c in range(C) for v in range(V)])

    # return reconstructed_data.reshape(N, C, T, V)
    return dct_data

class TS_GC(nn.Module):     
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, num_subset=3):
        super(TS_GC, self).__init__()
        inter_channels = out_channels // coff_embedding     
        self.inter_c = inter_channels
        self.out_channels = out_channels
        self.A_ske_P = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
     
        nn.init.constant_(self.A_ske_P, 1e-6)
        self.A_ske = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.num_subset = num_subset

        self.EnS = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, 1),
            nn.ReLU(),
            nn.Conv2d(out_channels // 4, out_channels // 2, 1),
            nn.ReLU(),
            nn.Conv2d(out_channels // 2, out_channels, 1),
            nn.ReLU(),
        )
        self.EnV = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.ReLU(),
        )
        self.EnM = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1)
        )
        
        self.Dec = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1)
        )

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)

    def forward(self, x, y):
        N, C, T, V = x.size()
        
        DCT_Matrix, IDCT_Matrix = get_dct_matrix(V, is_torch=True)
        
        A_ske = self.A_ske.cuda(x.get_device())   # A_skeleton
        A_ske_P = self.A_ske_P  # A_ske_P
        A_abs = A_ske_P  # adjancy matrix of ABS
        A_mot = A_ske_P + A_ske  # adjency matrix of MoT
        
        # ABS
        feature_S_ABS = self.EnS(x).permute(0, 3, 1, 2).contiguous().view(N, V,  self.out_channels * T)  # N, V, out_channel * T
        feature_V_ABS = self.EnV(x).view(N, self.out_channels * T, V)  # N, out_channel * T, V
        cross_feature_ABS = self.soft(torch.matmul(feature_S_ABS, feature_V_ABS) / feature_S_ABS.size(-1))  # N, V, V
        # print(cross_feature_ABS.shape)   # (64, 25, 25)
        # print(A_abs.shape)  # (3, 25, 25)
        for i in range(3):
            cross_feature_ABS = cross_feature_ABS + A_abs[i]
        x = x.view(N, C * T, V)
        cross_feature_ABS = self.soft(self.Dec(torch.matmul(x, cross_feature_ABS).view(N, C, T, V)))  # N, out_channel, T, V
        # print(cross_feature_ABS.shape)
        out_abs = self.EnM(cross_feature_ABS)
        out_abs = self.down(out_abs)
            
        # MoT
        feature_S_MoT = self.EnS(y).permute(0, 3, 1, 2).contiguous().view(N, V, self.out_channels * T)  # N, V, out_channel * T
        feature_V_MoT = self.EnV(y).view(N, self.out_channels * T, V)  # N, out_channel * T, V      
        cross_feature_MoT = self.soft(torch.matmul(feature_S_MoT, feature_V_MoT) / feature_S_MoT.size(-1))  # N, V, V
        for i in range(3):
            cross_feature_MoT = cross_feature_MoT + A_mot[i]
        y = y.view(N, C * T, V)
        cross_feature_MoT = self.soft(self.Dec(torch.matmul(y, cross_feature_MoT).view(N, C, T, V)))  # N, out_channel, T, V
        out_mot = self.down(self.EnM(cross_feature_MoT))
        
        # # MoT with DCT
        # y = discard_high_frequencies(y, 80, -2)  # DCT to y, retain 80% low frequencies to T dimension, [N, C, T, V]   
        # feature_S_MoT = self.EnS(y).permute(0, 3, 1, 2).contiguous()  # N, V, out_channel, T
        # feature_V_MoT = self.EnV(y)  # N, out_channel, T, V
        # n1, v1, c1, t1 = feature_S_MoT.size()
        # feature_S_MoT = np.array([idct2(feature_S_MoT[i, j, k, :]) for i in range(n1) for j in range(v1) for k in range(c1)])
        # feature_S_MoT = feature_S_MoT.reshape(n1, v1, c1, t1).view(N, V, self.out_channels * T)
        # n2, c2, t2, v2 = feature_S_MoT.size()
        # feature_V_MoT = np.array([idct2(feature_S_MoT[i, j, :, k]) for i in range(n2) for j in range(c2) for k in range(v2)])
        # feature_V_MoT = feature_V_MoT.reshape(n2, c2, t2, v2).view(N, self.out_channels * T, V)
        
        # cross_feature_MoT = self.soft(torch.matmul(feature_S_MoT, feature_V_MoT) / feature_S_MoT.size(-1))  # N, V, V
        # cross_feature_MoT = cross_feature_MoT + A_mot
        # y = y.view(N, C * T, V)
        # cross_feature_MoT = self.soft(self.Dec(torch.matmul(y, cross_feature_MoT).view(N, C, T, V)))  # N, out_channel, T, V
        # out_mot = self.down(self.EnM(out_mot))

        ABS = self.relu(self.bn(out_abs))
        MoT = self.relu(self.bn(out_mot))

        return ABS, MoT

class STE_GC_STACK(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(STE_GC_STACK, self).__init__()

        self.stegc1 = STE_GC(in_channels, out_channels // 4)
        self.conv1 = nn.Sequential(
            nn.Conv2d(out_channels // 4, out_channels // 4, 1),
            nn.ReLU(inplace=True),
        )
        self.stegc2 = STE_GC(out_channels // 4, out_channels // 2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels // 2, out_channels // 2, 1),
            nn.ReLU(inplace=True),
        )
        self.stegc3 = STE_GC(out_channels // 2, out_channels)
        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.ReLU(inplace=True),
        )
        self.stegc4 = STE_GC(out_channels, out_channels)
        self.conv4 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, y):
        x1, y1 = self.stegc1(x, y)
        x1 = self.conv1(x1)

        x2, y2 = self.stegc2(x1, y1)
        x2 = self.conv2(x2)

        x3, y3 = self.stegc3(x2, y2)
        x3 = self.conv3(x3)

        x4, y4 = self.stegc4(x3, y3)
        x4 = self.conv4(x4)

        return x4

    
class STER_GCN(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes, stride=1, residual=True):
        super(STER_GCN, self).__init__()
        # self.ste_gc_stack = nn.Sequential(
        #     STE_GC(in_channels, out_channels // 4),
        #     nn.Conv2d(out_channels // 4, out_channels // 4, 1),
        #     nn.ReLU(inplace=True),
        #     STE_GC(out_channels // 4, out_channels // 2),
        #     nn.Conv2d(out_channels // 2, out_channels // 2, 1),
        #     nn.ReLU(inplace=True),
        #     STE_GC(out_channels // 2, out_channels),
        #     nn.Conv2d(out_channels, out_channels, 1),
        #     nn.ReLU(inplace=True),
        #     STE_GC(out_channels, out_channels),
        #     nn.Conv2d(out_channels, out_channels, 1),
        #     nn.ReLU(inplace=True),
        # )

        self.ste_gc_stack = STE_GC_STACK(in_channels, out_channels)

        self.fc = nn.Sequential(
            nn.Linear(out_channels, num_classes), 
            nn.BatchNorm1d(num_classes), 
            nn.Softmax(dim=-1)
        ) 
    def forward(self, x, y):
        N, C, T, V = x.size()
        output = self.ste_gc_stack(x, y)
        output = output.mean(-1).mean(-1)
        logit_ST = self.fc(output)
        
        return logit_ST


class STE_GC(nn.Module):                           
    def __init__(self, in_channels, out_channels, residual=False):
        super(STE_GC, self).__init__()
        
        self.unit_gtcn = TCN_GCN_unit1(in_channels, out_channels)

    def forward(self, x, A):
        ST_extract = self.unit_gtcn(x, A)        
        return ST_extract, A

class TCN_GCN_unit1(nn.Module):      # spatial GCN + bn + relu
    def __init__(self, in_channels, out_channels, coff_embedding=2, num_subset=1):
        super(TCN_GCN_unit1, self).__init__()
        inter_channels = out_channels // coff_embedding     
        self.inter_c = inter_channels
        # self.PA = nn.Parameter(A)
        # nn.init.constant_(self.PA, 0.001)
        # self.A = Variable(A, requires_grad=True)
        self.num_subset = num_subset

        self.conv_a = nn.ModuleList()
        self.conv_b = nn.ModuleList()
        self.conv_d = nn.ModuleList()

        for i in range(self.num_subset):
            self.conv_a.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_b.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for i in range(self.num_subset):
            conv_branch_init(self.conv_d[i], self.num_subset)

    def forward(self, x, A):
        B, K, M, N = x.size()
        x_res = x.view(B, -1, M, N)
        A = A.cuda(x.get_device())
        PA = nn.Parameter(A)
        nn.init.constant_(PA, 0.000001)
        A = A + PA
        y = None

        # print("x shape:", x.shape)

        for i in range(self.num_subset):
            A1 = self.conv_a[i](x).view(B, N, self.inter_c * M) 
            A2 = self.conv_b[i](x).view(B, self.inter_c * M, N)        
            A1 = self.soft(torch.matmul(A1, A2) / A1.size(-1)) 
            # print(A)
            A1 = A1 + A
            z = self.conv_d[i](torch.matmul(x.view(B, K * M, N), A1).view(B, K, M, N))
            y = z + y if y is not None else z

        y = self.bn(y)
        y += self.down(x_res)
        return self.relu(y)

class CFE(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True):
        super(CFE, self).__init__()
        self.tsgc1 = TS_GC(in_channels, out_channels // 2, A)
        self.mstc1 = MS_TC(out_channels // 2, stride=stride)
        self.tsgc2 = TS_GC(out_channels // 2, out_channels, A)
        self.mstc2 = MS_TC(out_channels, stride=stride)
        self.relu = nn.ReLU()

    def forward(self, x, y):
        x_out1, y_out1 = self.tsgc1(x, y)  # N, out_channel, T, V
        x_out1, y_out1 = self.mstc1(x_out1, y_out1)  # N, out_channel, T, V
        x_out2, y_out2 = self.tsgc2(x_out1, y_out1)  # N, out_channel, T, V
        x_out2, y_out2 = self.mstc2(x_out2, y_out2)  # N, out_channel, T, V
        x_out = self.relu(x_out2)
        y_out = self.relu(y_out2)
        return x_out, y_out



class STC_Att(nn.Module):
    def __init__(self, channel, reduct_ratio, bias=True, **kwargs):
        super(STC_Att, self).__init__()

        inner_channel = channel // reduct_ratio

        self.fcn = nn.Sequential(
            nn.Conv2d(channel, inner_channel, kernel_size=1, bias=bias),
            nn.BatchNorm2d(inner_channel),
            nn.Hardswish(),
        )
        self.conv_t = nn.Conv2d(inner_channel, channel, kernel_size=1)
        self.conv_v = nn.Conv2d(inner_channel, channel, kernel_size=1)
        self.conv_c = SELayer(channel)

        self.add_attention = nn.Sequential(
            nn.BatchNorm2d(channel),
            nn.Hardswish(),
        )


    def forward(self, x, y):
        N, C, T, V = x.size()
        
        # compute x
        x_t = x.mean(3, keepdims=True)               
        x_v = x.mean(2, keepdims=True).transpose(2, 3)  
        x_att = self.fcn(torch.cat([x_t, x_v], dim=2)) 
        x_t, x_v = torch.split(x_att, [T, V], dim=2)  
        x_t_att = self.conv_t(x_t).sigmoid()          
        x_v_att = self.conv_v(x_v.transpose(2, 3)).sigmoid()
        x_att_f = x_t_att * x_v_att                    
        x_att_ff = x * x_att_f
        x_c_att = self.conv_c(x) 
        
        # compute y
        y_t = y.mean(3, keepdims=True)               
        y_v = y.mean(2, keepdims=True).transpose(2, 3)  
        y_att = self.fcn(torch.cat([y_t, y_v], dim=2)) 
        y_t, y_v = torch.split(y_att, [T, V], dim=2)  
        y_t_att = self.conv_t(y_t).sigmoid()          
        y_v_att = self.conv_v(y_v.transpose(2, 3)).sigmoid()
        y_att_f = y_t_att * y_v_att                    
        y_att_ff = y * y_att_f
        y_c_att = self.conv_c(y)                      
        
        x_att = x_att_ff + x_c_att
        x_att = self.add_attention(x_att)
        
        y_att = y_att_ff + y_c_att
        y_att = self.add_attention(y_att)
        return x_att, y_att


class SELayer(nn.Module):                           
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        N, C, _, _ = x.size()
        x_c1 = self.avg_pool(x).view(N, C)           
        x_c2 = self.fc(x_c1).view(N, C, 1, 1)        
        return x * x_c2.expand_as(x)

class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs.data)
        mse = torch.sum(diffs.pow(2)) / n
        return mse

class MC_Aug(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.d = torch.tensor(input_dim)
        self.loss = MSE()

    def forward(self, ABS_shared, MoT_shared):  
        D = self.d
        # dti_common = torch.mean(dti_common, axis=0)
        b, c, n, m = MoT_shared.size()  # N*2, 512, T, V
        B, C, M, N = ABS_shared.size()  # N*2, 512, T, V
        MoT_shared_T = MoT_shared.view(b, c, m, n)  # 
        ABS_shared_T = ABS_shared.view(B, C, N, M)

        ABS_fused = F.softmax(((ABS_shared @ (MoT_shared_T @ MoT_shared)) / D), dim=0) * ABS_shared   # N*2, 512, T, V
        MoT_fused = F.softmax(((MoT_shared @ (ABS_shared_T @ ABS_shared)) / D), dim=0) * MoT_shared    # N*2, 512, T, V

        ABS_agg_loss = torch.mean(ABS_fused, dim=1)  # N*2, T, V
        MoT_agg_loss = torch.mean(MoT_fused, dim=1)  # N*2, T, V
        Loss_shared = 0
        
        for i in range(ABS_agg_loss.shape[0]):
            Loss_common_temporal = self.loss(ABS_agg_loss[i], MoT_agg_loss[i]) + self.loss(MoT_agg_loss[i], ABS_agg_loss[i])
            Loss_shared += Loss_common_temporal

        Loss_shared = Loss_shared / ABS_agg_loss.shape[0]

        fature_shared = (ABS_fused + MoT_fused)

        return ABS_fused, MoT_fused, fature_shared, Loss_shared

class Backbone(nn.Module):
    def __init__(self, in_channels, A):
        super(Backbone, self).__init__()

        self.CFE1 = CFE(in_channels, 64, A, residual=True)
        self.STC_ATT1 = STC_Att(64, 4)

        self.CFE2 = CFE(64, 128, A, residual=True)
        self.STC_ATT2 = STC_Att(128, 4)

        self.CFE3 = CFE(128, 256, A, residual=True)
        self.STC_ATT3 = STC_Att(256, 4)

    def forward(self, abs_ske, mot_ske):

        abs1, mot1 = self.CFE1(abs_ske, mot_ske)
        abs2, mot2 = self.STC_ATT1(abs1, mot1)
        # print("abs1 shape", abs1.shape)
        abs3, mot3 = self.CFE2(abs2, mot2)
        abs4, mot4 = self.STC_ATT2(abs3, mot3)

        abs5, mot5 = self.CFE3(abs4, mot4)
        abs6, mot6 = self.STC_ATT3(abs5, mot5)

        return abs6, mot6


class Model(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3):
        super(Model, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
        self.pre_bn = nn.BatchNorm2d(in_channels)

        # self.backbone = nn.Sequential(
        #     CFE(in_channels, 64, A, residual=True),
        #     STC_Att(64, 4),
        #     CFE(64, 128, A, residual=True),
        #     STC_Att(128, 4),
        #     CFE(128, 256, A, residual=True),
        #     STC_Att(256, 4),
        # )

        self.backbone = Backbone(in_channels, A)
        
        self.ster_gcn = STER_GCN(6, 256, num_classes=num_class, residual=True)
        
         # personal features encoder 
        self.encoder_ABS = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, padding=0, bias=False)
        self.encoder_MoT = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, padding=0, bias=False)

        # shared features encoder
        self.encoder_shared = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, padding=0, bias=False)

        # features decoder 
        self.decoder_ABS = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1, padding=0, bias=False)
        self.decoder_MoT = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1, padding=0, bias=False)
        
        # Shared Modality Enhancer
        self.MC_Aug = MC_Aug(input_dim=512)
        
        # Adaptive control coefficient
        self.weights = nn.Parameter(torch.ones(3) / 3, requires_grad=True)  # 3 means three modalities: ABS,MoT,shared
        
        # reshape fuction
        self.reshape = nn.Sequential(nn.Linear(512*3, 512), nn.BatchNorm1d(512), nn.ReLU())
        
        # softmax for three modal data
        self.fc_concat = nn.Sequential(nn.Linear(512, num_class), nn.BatchNorm1d(num_class), nn.Softmax(dim=-1))
        self.fc = nn.Sequential(nn.Linear(256, num_class), nn.BatchNorm1d(num_class), nn.Softmax(dim=-1))
        # self.fc = nn.Linear(256, num_class)
        self.soft = nn.Softmax(dim=-1)
        # nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)

    def forward(self, x):
        N, C, T, V, M = x.size()    

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)  
                            
        x = x.view(N, M, V, C, T).permute(0, 3, 4, 2, 1)  # N,C,T,V,M
        x = x.permute(0, 4, 1, 2, 3).contiguous().view(N * M, C, T, V)
        
        # gain four skeleton streams
        J = x[:, :3, :, :]  # joint data: N*2, 3, 60, 25
        B = x[:, 3:, :, :]  # bone data
        J_M = torch.zeros_like(J)  # joint motion data: N*2, 3, 60, 25
        B_M = torch.zeros_like(B)  # bone motion data
        
        for t in range(T - 1):
            J_M[:, :, t, :] = J[:, :, t + 1, :] - J[:, :, t, :]
        J_M[:, :, T - 1, :] = J[:, :, T - 1, :]
        
        for t in range(T - 1):
            B_M[:, :, t, :] = B[:, :, t + 1, :] - B[:, :, t, :]
        B_M[:, :, T - 1, :] = B[:, :, T - 1, :]
        
        # ABS_SKE = self.pre_bn(x)  # N*2, 6, 60, 25
        ABS_SKE = x  # N*2, 6, 60, 25
        # MoT_SKE = self.pre_bn(torch.cat((J_M, B_M), axis=1))  # N*2, 6, 60, 25
        MoT_SKE = torch.cat((J_M, B_M), dim=1)  # N*2, 6, 60, 25
        
        # backbone process
        output_ABS, output_MoT = self.backbone(ABS_SKE, MoT_SKE)   # N*2, 256, T, V

        # gain logit
        logit_ABS = self.fc(output_ABS.mean(-1).mean(-1))
        logit_MoT = self.fc(output_MoT.mean(-1).mean(-1))
        
        # STER-GCN
        # 对每个批次单独处理
        filtered_MoT = torch.zeros_like(output_MoT)  # 初始化一个与output_MoT相同形状的全0矩阵

        def torch_percentile(tensor, q):
            k = 1 + round(float(q) * (tensor.numel() - 1))
            result = tensor.view(-1).kthvalue(k).values
            return result

        # output_MoT = output_MoT.cpu()

        for i in range(output_MoT.shape[0]):
            # 找出每个矩阵中前70%的阈值
            threshold = torch_percentile(output_MoT[i], 0.70)  # 计算第 70 百分位数
            
            # 使用阈值进行过滤处理
            # filtered_MoT[i] = np.where(output_MoT[i] >= threshold, 1, 0)
            filtered_MoT[i] = torch.where(output_MoT[i].detach() >= threshold, torch.tensor(1.0, device=output_MoT.device), torch.tensor(0.0, device=output_MoT.device))

        filtered_MoT = filtered_MoT.float()
        # logit_ST = self.ster_gcn(output_ABS, filtered_MoT)
        logit_ST = 0.0001

        # output_MoT = output_MoT.to(next(self.encoder_MoT.parameters()).device)

        # if any(param.device for param in self.encoder_MoT.parameters()):
        #     device = next(self.encoder_MoT.parameters()).device
        # else:
        #     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # output_MoT = output_MoT.to(device)

        ABS_per = self.encoder_ABS(output_ABS)   # ABS_per[N*2, 512, T, V]
        ABS_shared = self.encoder_shared(output_ABS)  # ABS_shared[N*2, 512, T, V]
        MoT_per = self.encoder_MoT(output_MoT)  # MoT_per[N*2, 512, T, V]
        MoT_shared = self.encoder_shared(output_MoT)  # MoT_common[N*2, 512, T, V]

        # SME
        ABS_agg, MoT_agg, fature_shared, Loss_shared = self.MC_Aug(ABS_shared, MoT_shared)  # input_dim = output_dim
        
        # PMR
        ABS_concat = torch.cat((ABS_agg, ABS_per), dim=1)  # [N*2, 1024, T, V]
        MoT_concat = torch.cat((MoT_agg, MoT_per), dim=1)  # [N*2, 1024, T, V]
        ABS_de = self.decoder_ABS(ABS_concat)  # [N*2, 256, T, V]
        MoT_de = self.decoder_MoT(MoT_concat)  # [N*2, 256, T, V]

        ABS_per_final = self.encoder_ABS(ABS_de)  # [N*2, 512, T, V]
        MoT_per_final = self.encoder_MoT(MoT_de)  # [N*2, 512, T, V]

        ABS_per_final_loss1 = torch.mean(torch.mean(ABS_per_final, axis=-1), axis=-1)  # [N*2, 512] 
        MoT_per_final_loss1 = torch.mean(torch.mean(MoT_per_final, axis=-1), axis=-1)  # [N*2, 512]
        fature_shared1 = torch.mean(torch.mean(fature_shared, axis=-1), axis=-1)  # [N*2, 512]            
        concat_ABS_MoT = torch.cat((self.weights[0] * ABS_per_final_loss1, self.weights[1] * MoT_per_final_loss1, self.weights[2] * fature_shared1), dim=-1)  # [N*2, 512*3]
        concat_ABS_MoT = self.reshape(concat_ABS_MoT)  # [N*2, 512]
        
        # the recognition results
        concat_ABS_MoT_logit = self.fc_concat(concat_ABS_MoT)

        logit_ABS = logit_ABS.view(N, M, -1).mean(1)
        logit_MoT = logit_MoT.view(N, M, -1).mean(1)
        # logit_ST = logit_ST.view(N, M, -1).mean(1)
        concat_ABS_MoT_logit = concat_ABS_MoT_logit.view(N, M, -1).mean(1)

        # print("logit_ABS:", logit_ABS[0], logit_ABS[1])
        # print("logit_MoT:", logit_MoT[0], logit_MoT[1])
        # print("concat_ABS_MoT_logit:", concat_ABS_MoT_logit[0], concat_ABS_MoT_logit[1])

        # logit_ABS = self.soft(logit_ABS)
        # logit_MoT = self.soft(logit_MoT)
        # concat_ABS_MoT_logit = self.soft(concat_ABS_MoT_logit)

        return logit_ABS, logit_MoT, concat_ABS_MoT_logit   # 0.5, 0.5, 0.3, 1
