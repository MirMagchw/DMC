#improved version
import numpy as np
from scipy.signal import stft, istft, correlate
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import soundfile as sf
import ast, os, librosa
import os
from tools import sort_clean_wav, sort_wav
from models.tools import *
import tqdm
from config.config import opt
import torch
#from models.LAB_model.model import Model

# ECAPA2_model_file = 'exps/ecapa2.pt'
# get_ECAPA2_embedding = torch.jit.load(ECAPA2_model_file, map_location=opt.device)
# SV_embedding = get_ECAPA2_embedding(mic_signal_tensor)

# LAB_model = Model.load_from_checkpoint(checkpoint_path="./exps/LABnet.ckpt", map_location=opt.device)
# LAB_model.to(opt.device)
# LAB_model.eval() 
# LAB_model.freeze()
# LAB_DSB = LAB_model.SE
# signal_LAB_out = LAB_DSB(clustered_signals_tensor)

def compute_stft(signals, fs=16000, nperseg=512, noverlap=256):
    stft_results = []
    for signal in signals:
        f, t, Zxx = stft(signal, fs, nperseg=nperseg, noverlap=noverlap)
        stft_results.append(Zxx)
    return np.array(stft_results), f, t

def compute_istft(stft_matrix, fs=16000, nperseg=512, noverlap=256):
    separated_signals = []
    for Zxx in stft_matrix:
        _, signal = istft(Zxx, fs, nperseg=nperseg, noverlap=noverlap)
        separated_signals.append(signal)
    return np.array(separated_signals)

def groups_by_labels(datax, labels):
    unique_labels = np.unique(labels)
    k = len(unique_labels)
    
    grouped_datax = defaultdict(list)
    grouped_indices = defaultdict(list)
    
    for i, (data, label) in enumerate(zip(datax, labels)):
        grouped_datax[label].append(data)
        grouped_indices[label].append(i)
    
    for label in grouped_datax:
        grouped_datax[label] = np.array(grouped_datax[label])
        grouped_indices[label] = np.array(grouped_indices[label])

    return grouped_datax, grouped_indices

def get_delay_base(mic_signals, cluster_labels, ref_mics, fs=16000):

    M = mic_signals.shape[0]
    delays = np.zeros(M) 
    labels = np.unique(cluster_labels)
    labels = np.sort(labels)
    for cluster in labels:

        cluster_mics = np.where(cluster_labels == cluster)[0]
        ref_mic_idx = ref_mics[cluster] 
        ref_signal = mic_signals[ref_mic_idx]  
        for mic_idx in cluster_mics:
            mic_signal = mic_signals[mic_idx]
            delay_samples = gcc_phat(mic_signal, ref_signal)
            delays[mic_idx] = delay_samples
    return delays

def apply_delays(mic_signals, delays):
    num_mics, signals_length = mic_signals.shape
    shifted_signals = np.zeros_like(mic_signals)
    
    for i in range(num_mics):
        d = np.int16(delays[i])
        signal = mic_signals[i, :]
        if d > 0: 
            k = d
            if k >= signals_length:
                shifted_signal = np.zeros(signals_length)
            else:
                shifted_signal = np.concatenate([signal[k:], np.zeros(k)])
        elif d < 0: 
            m = -d
            if m >= signals_length:
                shifted_signal = np.zeros(signals_length)
            else:
                shifted_signal = np.concatenate([np.zeros(m), signal[:signals_length - m]])
        else: 
            shifted_signal = signal.copy()
        
        shifted_signals[i, :] = shifted_signal
    
    return shifted_signals

def gcc_phat(x1, x2, fs=16000):
    # Ensure inputs are numpy arrays and column vectors
    x1 = np.array(x1).reshape(-1, 1)
    x2 = np.array(x2).reshape(-1, 1)
    # Parameters
    N = len(x1) + len(x2) - 1
    NFFT = 2*(N+1)
    # Calculate FFT of input signals
    X1 = np.fft.fft(x1.flatten(), NFFT)
    X2 = np.fft.fft(x2.flatten(), NFFT)
    # Cross-correlation in frequency domain
    P = X1 * np.conj(X2)#X2作为参考信号
    # Calculate A (magnitude reciprocal)
    A = 1.0 / np.abs(P)
    # First estimation method
    R_est1 = np.fft.fftshift(np.fft.ifft(A * P))
    # Calculate range indices
    start_idx = NFFT//2 + 1 - (N-1)//2
    end_idx = NFFT//2 + 1 + (N-1)//2
    range_indices = slice(start_idx, end_idx + 1)
    # Extract relevant portions
    R_est1 = R_est1[range_indices]

    # Find maximum correlation and corresponding lag
    tau = np.argmax(np.abs(R_est1))
    delay_samples = tau - len(R_est1)//2 + 1
    
    return delay_samples

def find_closest_mic_combined(signals, fs=16000, c=343.2):
    n_mics = signals.shape[0]
    tdoa_matrix = np.zeros((n_mics, n_mics))
    for i in range(n_mics):
        for j in range(i, n_mics):
            tdoa = gcc_phat(signals[j], signals[i], fs)
            tdoa_matrix[i,j] = tdoa
            tdoa_matrix[j,i] = -tdoa
    
    nearest_mic = 0
    for i in range(n_mics):
        if tdoa_matrix[nearest_mic, i] < 0:
            nearest_mic = i
    
    energies = np.sum(signals**2, axis=1)
    closest_mic_energy = np.argmax(energies)
    
    return nearest_mic, np.int16(tdoa_matrix[nearest_mic])

def get_map(distance, cluster_labels):
    grouped_distance, grouped_indices = groups_by_labels(distance, cluster_labels)
    k = len(grouped_distance.keys())
    avg_distance = np.zeros((k, distance.shape[1]))
    sorted_labels = sorted(grouped_distance.keys())
    for i, label in enumerate(sorted_labels):
        group_data = grouped_distance[label]
        
        avg_distance[i] = np.mean(group_data, axis=0)
    cluster2source = {}
    cluster2source_2 = {}
    source2cluster = {}
    for k in range(avg_distance.shape[1]):
        cluster_index = np.argmin(avg_distance[k])
        print(cluster_index)
        if cluster_index in cluster2source:
            cluster2source[cluster_index].append(k)
        else:
            cluster2source[cluster_index] = [k]
        cluster2source_2[cluster_index] = k
        source2cluster[k] = cluster_index
    return cluster2source, cluster2source_2, source2cluster

def calculate_correlation_peaks(DSB_signals, source):

    k = DSB_signals.shape[0]
    target_source = source
    correlation_peaks = np.zeros(k)

    for i in range(k):
        DSB_signals[i] /= np.max(DSB_signals[i])
        correlation = np.correlate(DSB_signals[i], target_source, mode='full')
        correlation_peaks[i] = np.max(correlation)

    max_peak_idx = np.argmax(correlation_peaks)
    return max_peak_idx
#IBM-based SS
def create_masks(ref_mics, stft_results, B=1):
    masks = []
    N = len(ref_mics)
    K = stft_results.shape[1]
    D = stft_results.shape[2]
    
    for cluster in range(N):  
        cluster_mask = np.zeros_like(stft_results[0], dtype=bool)
        for k in range(K):  
            for b in range(D):  
                if b >= B-1:
                    flag = True
                    for mic in ref_mics:
                        if mic != ref_mics[cluster]:
                            if np.abs(stft_results[ref_mics[cluster], k, b]) - np.mean(np.abs(stft_results[mic, k, b-B+1:b+1])) < 1e-5:
                                flag = False
                    cluster_mask[k, b] = flag
        masks.append(cluster_mask)
    return np.array(masks)

def apply_masks(masks, stft_results, labels):
    enhanced_signals = []
    for mic in range(stft_results.shape[0]):
        cluster_stft = np.zeros_like(stft_results[0])
        n = labels[mic]
        cluster_stft = stft_results[mic] * masks[n]
        enhanced_signals.append(cluster_stft)
    return np.array(enhanced_signals)

def reconstruction(mic_signals_stft, cluster_labels, ref_mics, fs=16000):

    DSB_signals_stft = []
    DSB_labels = []
    labels = np.unique(cluster_labels)
    labels = np.sort(labels)
    for cluster in labels:
        cluster_mics = np.where(cluster_labels == cluster)[0]
        ref_mic_idx = ref_mics[cluster] 
        DSB_signal = np.zeros_like(mic_signals_stft[ref_mic_idx])
        for mic_idx in cluster_mics:
            DSB_signal += mic_signals_stft[mic_idx]
        DSB_signal /= cluster_mics.size
        DSB_signals_stft.append(DSB_signal)
        DSB_labels.append(cluster)
    return np.array(DSB_signals_stft), np.array(DSB_labels)

def resample_audio(audio, original_fs, target_fs=16000):
    resampled_audio = librosa.resample(audio, orig_sr=original_fs, target_sr=target_fs)
    return resampled_audio



