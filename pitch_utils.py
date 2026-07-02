import math
import numpy as np
from scipy.signal import find_peaks

# Audio parameters
Fs = 44100          # Sampling rate (Hz) — standard CD quality
N = 2048            # Frame size (samples) — gives ~46ms frames, ~21 Hz freq resolution
HOP = 512           # Hop size (samples) — 75% overlap between frames
F_MIN = 50          # Minimum detectable pitch (Hz) — below bass guitar low E
F_MAX = 4000        # Maximum detectable pitch (Hz) — above highest piano notes
RMS_THRESHOLD = 0.001  # Minimum RMS energy to attempt pitch detection

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def apply_hann_window(frame):
    """
    Applies a Hann window to reduce spectral leakage before FFT.
    w[n] = 0.5(1 − cos(2πn/N))
    """
    N_frame = len(frame)
    return frame * np.hanning(N_frame)


def compute_spectrum_db(frame, Fs_val):
    """
    Returns (frequencies, magnitudes_in_dB) for plotting.
    Applies Hann window, computes rfft, converts to dB: 20*log10(|X|+1e-9)
    """
    N_frame = len(frame)
    windowed = apply_hann_window(frame)
    X = np.fft.rfft(windowed)
    freqs = np.fft.rfftfreq(N_frame, 1 / Fs_val)
    magnitudes_dB = 20 * np.log10(np.abs(X) + 1e-9)
    return freqs, magnitudes_dB


def is_voiced(frame, threshold_rms=RMS_THRESHOLD):
    """Returns True if the frame's RMS energy exceeds the silence threshold."""
    rms = np.sqrt(np.mean(frame ** 2))
    return rms > threshold_rms


def detect_pitch_fft(frame, Fs):
    """
    Detects pitch as the strongest FFT bin in [F_MIN, F_MAX].
    Fast, but limited to Fs/N bin resolution (~21.5 Hz at N=2048) and easily
    misled by strong harmonics.
    """
    N_frame = len(frame)
    windowed = frame * np.hanning(N_frame)
    spectrum = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(N_frame, 1 / Fs)

    valid_mask = (freqs >= F_MIN) & (freqs <= F_MAX)
    if not np.any(valid_mask):
        return None
    valid_spectrum = spectrum.copy()
    valid_spectrum[~valid_mask] = 0
    peak_idx = np.argmax(valid_spectrum)
    if spectrum[peak_idx] < 1e-6:
        return None
    return freqs[peak_idx]


def detect_pitch_autocorr(frame, Fs, f_min=F_MIN, f_max=F_MAX):
    """
    Detects pitch via FFT-based autocorrelation (Wiener-Khinchin theorem),
    R[τ] = IFFT(|X[f]|²), refined with parabolic interpolation for sub-sample
    accuracy. Unlike raw FFT peak-picking, this finds the period of the whole
    waveform, so it stays correct even when an upper harmonic is louder than
    the fundamental.
    """
    N_frame = len(frame)

    # DC removal is critical for accurate autocorrelation at low frequencies
    frame = frame - np.mean(frame)
    windowed = frame * np.hanning(N_frame)

    # Zero-pad to 2N so the IFFT gives linear, not circular, autocorrelation
    X = np.fft.rfft(windowed, n=2 * N_frame)
    R = np.fft.irfft(X * np.conj(X))
    R = np.real(R[:N_frame])

    r0 = R[0]
    if r0 < 1e-10:
        return None  # essentially silent
    R = R / r0

    tau_min = max(int(Fs / f_max), 1)            # shortest lag = highest freq
    tau_max = min(int(Fs / f_min), N_frame - 2)   # longest lag = lowest freq
    if tau_max <= tau_min:
        return None

    R_search = R[tau_min:tau_max + 1]

    # Dynamic threshold (30% of max, floor 0.15): a fixed height=0.4 was too
    # strict for real mic input and missed low-frequency notes.
    dynamic_threshold = max(0.15, 0.3 * np.max(R_search))
    peaks, _ = find_peaks(R_search, prominence=0.05, height=dynamic_threshold)
    if len(peaks) == 0:
        return None

    # Strongest peak = the true fundamental's period
    best_peak_idx = int(np.argmax(R_search[peaks]))
    best_tau = peaks[best_peak_idx] + tau_min

    # Parabolic interpolation for sub-sample lag accuracy
    if 0 < best_tau < N_frame - 1:
        alpha = R[best_tau - 1]
        beta = R[best_tau]
        gamma = R[best_tau + 1]
        denom = alpha - 2 * beta + gamma
        if abs(denom) > 1e-10:
            best_tau = best_tau + 0.5 * (alpha - gamma) / denom

    return Fs / best_tau


def freq_to_note(f0):
    """
    Converts frequency to (note_name, frequency_hz, cents_deviation).
    Returns ("A4", 440.2, +0.8) style tuple, or (None, None, None) if f0 is invalid.
    """
    if f0 is None or f0 <= 0:
        return None, None, None
    midi = round(12 * math.log2(f0 / 440.0) + 69)
    midi = max(21, min(108, midi))
    note_name = NOTE_NAMES[midi % 12]
    octave = (midi // 12) - 1
    ideal_freq = 440.0 * (2 ** ((midi - 69) / 12))
    cents = 1200 * math.log2(f0 / ideal_freq)
    return f"{note_name}{octave}", round(f0, 2), round(cents, 1)