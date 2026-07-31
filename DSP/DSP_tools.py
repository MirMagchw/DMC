import numpy as np
from scipy.signal import stft, istft, correlate
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import soundfile as sf
import ast, os, librosa
import os
import tqdm
import torch

# For Mod-MFCC, IBM-based SS
def compute_stft(signals, fs=16000, nperseg=512, noverlap=256):
    stft_results = []
    for signal in signals:
        f, t, Zxx = stft(signal, fs, window='hann', nperseg=nperseg, noverlap=noverlap)
        stft_results.append(Zxx)
    return np.array(stft_results), f, t
def compute_istft(stft_matrix, fs=16000, nperseg=512, noverlap=256):
    istft_results= []
    for Zxx in stft_matrix:
        _, signal = istft(Zxx, fs, window='hann', nperseg=nperseg, noverlap=noverlap)
        istft_results.append(signal)
    return np.array(istft_results)

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
    P = X1 * np.conj(X2)# X2 as reference signal
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
    
def gcc_phat_windowed(x1, x2, fs=16000, win_len_sec=1.0, max_delay_ms=30, eps=1e-12):
    x1 = np.asarray(x1).flatten()
    x2 = np.asarray(x2).flatten()
    win_len = int(win_len_sec * fs)
    window = np.hanning(win_len)
    min_len = min(len(x1), len(x2))
    num_frames = min_len // win_len
    if num_frames == 0:
        raise ValueError("error")
    NFFT = 2 * win_len
    P_acc = np.zeros(NFFT, dtype=np.complex128)

    for k in range(num_frames):
        s = k * win_len
        e = s + win_len
        frame1 = x1[s:e]*window
        frame2 = x2[s:e]*window
        X1 = np.fft.fft(frame1, NFFT)
        X2 = np.fft.fft(frame2, NFFT)
        P_acc += X1 * np.conj(X2)

    P_mean = P_acc / num_frames

    P_phat = P_mean / (np.abs(P_mean) + eps)

    R = np.fft.fftshift(np.fft.ifft(P_phat))
    R = np.real(R)

    max_delay_samples = int(max_delay_ms * 1e-3 * fs)
    center = NFFT // 2
    search_region = R[
        center - max_delay_samples :
        center + max_delay_samples + 1
    ]
    tau = np.argmax(search_region)
    delay_samples = tau - max_delay_samples
    return delay_samples
    
def find_closest_mic_combined(signals, fs=16000, c=343.2):
    n_mics = signals.shape[0]
    
    tdoa_matrix = np.zeros((n_mics, n_mics))
    for i in range(n_mics):
        for j in range(i, n_mics):
            tdoa = gcc_phat_windowed(signals[j], signals[i], fs)
            tdoa_matrix[i,j] = tdoa
            tdoa_matrix[j,i] = -tdoa
    
    nearest_mic = 0
    for i in range(n_mics):
        if tdoa_matrix[nearest_mic, i] < 0:
            nearest_mic = i
    
    return nearest_mic, np.int16(tdoa_matrix[nearest_mic])




