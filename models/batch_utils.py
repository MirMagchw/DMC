import numpy as np
import soundfile as sf
import torch
import librosa

def load_audio(file_path):
    audio, fs = sf.read(file_path)
    if audio.ndim > 1:
        audio = audio[:, 0]
    audio = audio.astype(np.float32, copy=False)

    return audio, fs

def prepare_waveform_batch(file_paths, device):
    audios = []
    max_len = 0

    for file_path in file_paths:
        audio, _ = load_audio(file_path)
        audios.append(audio)
        if len(audio) > max_len:
            max_len = len(audio)

    batch = np.zeros((len(audios), max_len), dtype=np.float32)
    for index, audio in enumerate(audios):
        batch[index, :len(audio)] = audio

    return torch.from_numpy(batch).to(device)

def prepare_crnn_chunk_batch(file_paths, target_seconds=4, n_fft=400, hop_length=160):
    chunk_batches = []
    chunk_counts = []

    for file_path in file_paths:
        audio, fs = load_audio(file_path)
        target_len = target_seconds * fs

        if len(audio) < target_len:
            repeat_times = int(np.ceil(target_len / len(audio)))
            audio = np.tile(audio, repeat_times)

        chunk_count = len(audio) // target_len
        audio_all = audio[:chunk_count * target_len]

        file_chunks = []
        for chunk_index in range(chunk_count):
            chunk_audio = audio_all[chunk_index * target_len:(chunk_index + 1) * target_len]
            spectrum = np.abs(librosa.stft(chunk_audio, n_fft=n_fft, hop_length=hop_length)).T

            if spectrum.shape[0] < 400:
                pad_width = ((0, 400 - spectrum.shape[0]), (0, 0))
                spectrum = np.pad(spectrum, pad_width, mode='constant', constant_values=0)
            elif spectrum.shape[0] > 400:
                spectrum = spectrum[:400, :]

            theta = np.linalg.norm(spectrum, axis=1) + np.finfo(float).eps
            spectrum /= np.mean(theta)
            file_chunks.append(torch.from_numpy(spectrum[np.newaxis, np.newaxis, ...].astype(np.float32)))

        chunk_batches.append(torch.cat(file_chunks, dim=0))
        chunk_counts.append(chunk_count)

    if len(chunk_batches) == 0:
        empty = torch.empty((0, 1, 400, 201), dtype=torch.float32)
        return empty, np.asarray(chunk_counts, dtype=np.int64)

    return torch.cat(chunk_batches, dim=0), np.asarray(chunk_counts, dtype=np.int64)
