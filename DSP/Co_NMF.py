import os
import torch
import torch.nn.functional as F
import soundfile as sf
import numpy as np
from scipy import signal

# 设置随机种子以保证结果可复现
torch.manual_seed(0)
np.random.seed(0)

def compute_coherence_torch(signals, fs, nfft=512, noverlap=256):
    """
    使用PyTorch计算谱平方相干系数
    注意：由于相干计算比较复杂，这里先用scipy计算，然后转换为tensor
    对于完全GPU实现，需要自定义相干计算函数
    """
    M = signals.shape[0]
    C = np.eye(M)
    
    # 由于相干计算较复杂，这里先用numpy/scipy计算再转换
    signals_np = signals.cpu().numpy()
    for i in range(M):
        for j in range(i+1, M):
            f, coh = signal.coherence(signals_np[i], signals_np[j],
                                    fs=fs, nperseg=nfft, noverlap=noverlap)
            cij = np.mean(coh)
            C[i, j] = C[j, i] = cij
    
    return C
def NMF_clustering_torch(directory, wav_files, n_speaker, device='cuda:0'):
    """
    PyTorch GPU版本的NMF聚类
    """
    # 设置设备
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    M = len(wav_files)
    if M == 0:
        raise ValueError(f"未在目录中找到音频文件")

    signals_list = []
    fs = None
    
    # 读取音频文件
    for fname in wav_files:
        data, rate = sf.read(os.path.join(directory, fname))
        if data.ndim > 1:
            data = data[:, 0]
        signals_list.append(torch.from_numpy(data.astype(np.float32)))
        if fs is None:
            fs = rate
        elif fs != rate:
            raise ValueError("不同文件的采样率不一致")

    # 将信号堆叠并移动到设备
    signals = torch.stack(signals_list).to(device)  # 形状 (M, N)

    # 计算相干矩阵 C
    print("计算相干矩阵...")
    C_cpu = compute_coherence_torch(signals, fs)
    C = torch.from_numpy(C_cpu.astype(np.float32)).to(device)
    # 设置聚类簇数 K
    K = n_speaker + 1
    
    # 随机初始化B矩阵
    B = torch.rand(M, K, device=device, requires_grad=False)
    
    # 创建掩码矩阵
    I = torch.eye(M, device=device)
    mask = 1 - I
    
    # NMF 乘法更新
    max_iter = 100
    eps = 1e-10
    
    print("开始NMF迭代...")
    for it in range(max_iter):
        # PyTorch版本的乘法更新
        numerator = torch.mm(C * mask, B)
        denominator = torch.mm(torch.mm(B, B.t()) * mask, B) + eps
        B = B * (numerator / denominator)
        
        if (it + 1) % 20 == 0:
            print(f"迭代 {it + 1}/{max_iter}")

    # 根据 B 矩阵行最大值确定聚类标签
    labels = torch.argmax(B, dim=1)
    
    return labels.cpu().numpy(), B.cpu().detach().numpy()

def NMF_clustering(directory, wav_files, n_speaker):
    M = len(wav_files)
    if M == 0:
        raise ValueError(f"未在目录中找到音频文件")

    signals = []
    fs = None
    for fname in wav_files:
        data, rate = sf.read(os.path.join(directory, fname))
        # 若为多通道，取第一个通道
        if data.ndim > 1:
            data = data[:, 0]
        signals.append(data.astype(np.float64))
        if fs is None:
            fs = rate
        elif fs != rate:
            raise ValueError("不同文件的采样率不一致")

    signals = np.array(signals)  # 形状 (M, N)

    # 计算谱平方相干系数并构建相干矩阵 C
    C = np.zeros((M, M))
    nfft = 512
    noverlap = 256
    for i in range(M):
        C[i, i] = 1.0
        for j in range(i+1, M):
            # 计算谱平方相干系数 (magnitude-squared coherence)
            f, coh = signal.coherence(signals[i], signals[j],
                                    fs=fs, nperseg=nfft, noverlap=noverlap)
            cij = np.mean(coh)  # 平均作为最终度量
            C[i, j] = C[j, i] = cij

    # 设置聚类簇数 K
    K = n_speaker + 1
    B = np.random.rand(M, K)  # 随机初始化

    I = np.eye(M)
    mask = np.ones((M, M)) - I

    # NMF 乘法更新
    max_iter = 100
    eps = 1e-10
    for it in range(max_iter):
        numerator = (C * mask).dot(B)
        denominator = ((B.dot(B.T)) * mask).dot(B) + eps
        B *= numerator / denominator

    # 根据 B 矩阵行最大值确定聚类标签
    labels = np.argmax(B, axis=1)
    return labels, B
