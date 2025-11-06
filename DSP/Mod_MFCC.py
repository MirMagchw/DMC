# Classification of reverberant audio signals using clustered ad hoc distributed microphones
import numpy as np
from scipy.io import wavfile
import soundfile as sf
from scipy.fftpack import dct
import matplotlib.pyplot as plt

class Mod_MFCC():
    def __init__(self):
        self.flag = 0
    def plot_time(self, sig, fs):
        time = np.arange(0, len(sig)) * (1.0 / fs)
        plt.figure(figsize=(20, 5))
        plt.plot(time, sig)
        plt.xlabel('Time(s)')
        plt.ylabel('Amplitude')
        plt.grid()

    def plot_freq(self, sig, sample_rate, nfft=512):
        freqs = np.linspace(0, sample_rate/2, nfft//2 + 1)
        xf = np.fft.rfft(sig, nfft) / nfft
        xfp = 20 * np.log10(np.clip(np.abs(xf), 1e-20, 1e100))
        plt.figure(figsize=(20, 5))
        plt.plot(freqs, xfp)
        plt.xlabel('Freq(hz)')
        plt.ylabel('dB')
        plt.grid()
        

    def plot_spectrogram(self, spec, ylabel = 'ylabel'):
        fig = plt.figure(figsize=(20, 5))
        heatmap = plt.pcolor(spec)
        fig.colorbar(mappable=heatmap)
        plt.xlabel('Time(s)')
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.show()

    def pre_emphasis(self, b, sig):
        return np.append(sig[0], sig[1:] - b * sig[:-1])
    
    def add_window(self, frame_sig, fs, frame_len_s):
        window = np.hamming(int(round(frame_len_s * fs)))
        return frame_sig*window

    def framing(self, frame_len_s, frame_shift_s, fs, sig):
        sig_n = len(sig)
        frame_len_n, frame_shift_n = int(round(fs * frame_len_s)), int(round(fs * frame_shift_s))
        num_frame = int(np.ceil(float(sig_n - frame_len_n) / frame_shift_n) + 1)
        pad_num = frame_shift_n * (num_frame - 1) + frame_len_n - sig_n   
        pad_zero = np.zeros(int(pad_num))    
        pad_sig = np.append(sig, pad_zero)
        
        frame_inner_index = np.arange(0, frame_len_n)
        frame_index = np.arange(0, num_frame) * frame_shift_n

        frame_inner_index_extend = np.tile(frame_inner_index, (num_frame, 1))

        frame_index_extend = np.expand_dims(frame_index, 1)

        each_frame_index = frame_inner_index_extend + frame_index_extend
        each_frame_index = each_frame_index.astype(np.int64, copy=False)
        
        frame_sig = pad_sig[each_frame_index]
        return frame_sig

    def stft(self, frame_sig, nfft=512):

        frame_spec = np.fft.rfft(frame_sig, nfft, axis=1)

        frame_mag = np.abs(frame_spec)**2
        #frame_pow = (frame_mag ** 2) * 1.0 / nfft
        return frame_mag

    def mel_filter(self, frame_mag, fs, n_filter, nfft):

        mel_min = 0    
        mel_max = 2595 * np.log10(1 + fs / 2.0 / 700)   
        mel_points = np.linspace(mel_min, mel_max, n_filter + 2)   
        hz_points = 700 * (10 ** (mel_points / 2595.0) - 1)  
        filter_edge = np.floor(hz_points * (nfft) / fs)  

        fbank = np.zeros((n_filter, nfft//2 + 1))
        for m in range(1, 1 + n_filter):
            f_left = int(filter_edge[m - 1]) 
            f_center = int(filter_edge[m])  
            f_right = int(filter_edge[m + 1])  
            
            for k in range(f_left, f_center):
                fbank[m - 1, k] = (k - f_left) / (f_center - f_left)
            for k in range(f_center, f_right):
                fbank[m - 1, k] = (f_right - k) / (f_right - f_center)
        
        filter_banks = np.dot(frame_mag, fbank.T)
        filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
        filter_banks = np.log10(filter_banks)  #
        
        return filter_banks

    def mfcc_modulation_spectrum(self, Xmfcc, L=16, Q=8):

        num_frames, num_ceps = Xmfcc.shape
        modulation_spectrum = np.zeros((num_frames - L + 1, num_ceps, L // 2), dtype=complex)
        
        for eta in range(num_ceps):
            for c in range((num_frames-L)//Q + 1):
                modulation_spectrum[c, eta, :] = np.fft.fft(Xmfcc[c*Q:c*Q+L, eta])[: L // 2]
        
        return modulation_spectrum

    def avg_modulation_spectrum(self, Xmfcc_mod):

        avg_mod_spectrum = np.mean(np.abs(Xmfcc_mod), axis=0)
        return avg_mod_spectrum

    def compute_cmr(self, avg_mod_spectrum, nu1, nu2):

        avg_mod_band = np.sum(avg_mod_spectrum[:, nu1:nu2 + 1], axis = 1)

        cmr = avg_mod_band / avg_mod_spectrum[:, 0]
        return cmr

    def avg_cepstral_modulation(self, avg_mod_spectrum):

        return np.mean(avg_mod_spectrum, axis=1)
    def get_Mod_MFCC(self, wav_file):
        signal, fs = sf.read(wav_file)
        b = 0.97
        sif = self.pre_emphasis(b, signal)
        frame_len_s = 0.032
        frame_shift_s = 0.016
        frame_sig = self.framing(frame_len_s, frame_shift_s, fs, sif)

        frame_mag = self.add_window(frame_sig, fs, frame_len_s)

        nfft = 512
        frame_mag = self.stft(frame_sig, nfft)


        n_filter = 40   
        filter_banks = self.mel_filter(frame_mag, fs, n_filter, nfft)


        num_ceps = 13
        mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, :(num_ceps)]

        mfcc -= (np.mean(mfcc, axis=0) + 1e-8)

        modulation_spectrum = self.mfcc_modulation_spectrum(mfcc, L=16, Q=8)
        avg_mod_spectrum = self.avg_modulation_spectrum(modulation_spectrum)
        cmr1_1 = self.compute_cmr(avg_mod_spectrum, 1, 1)
        cmr2_8 = self.compute_cmr(avg_mod_spectrum, 2, 8)
        avg_cep_mod = self.avg_cepstral_modulation(avg_mod_spectrum)
        V = np.concatenate((avg_cep_mod, cmr1_1, cmr2_8))
        return V
