import math
# pyrefly: ignore [missing-import]
import numpy as np
from scipy.signal import find_peaks

# ─── Audio parameters ──────────────────────────────────────────────────────────
Fs = 44100          # Sampling rate (Hz) — standard CD quality
N = 2048            # Frame size (samples) — gives ~46ms frames, ~21 Hz freq resolution
HOP = 512           # Hop size (samples) — 75% overlap between frames
F_MIN = 50          # Minimum detectable pitch (Hz) — below bass guitar low E
F_MAX = 4000        # Maximum detectable pitch (Hz) — above highest piano notes
RMS_THRESHOLD = 0.001  # Minimum RMS energy to attempt pitch detection

NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']


def apply_hann_window(frame):
    """
    Multiplies frame by np.hanning(N). Reduces spectral leakage (Gibbs phenomenon).

    The Hann window w[n] = 0.5(1 − cos(2πn/N)) tapers the frame to zero at both ends,
    eliminating the spectral leakage (Gibbs phenomenon) that would occur when taking
    the DFT of an abruptly truncated signal.
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
    freqs = np.fft.rfftfreq(N_frame, 1/Fs_val)
    magnitudes_dB = 20 * np.log10(np.abs(X) + 1e-9)
    return freqs, magnitudes_dB


def is_voiced(frame, threshold_rms=RMS_THRESHOLD):
    """
    Returns True if the frame has enough energy to contain a pitch.
    RMS energy check: np.sqrt(np.mean(frame**2)) > threshold_rms
    """
    rms = np.sqrt(np.mean(frame**2))
    return rms > threshold_rms


def detect_pitch_fft(frame, Fs):
    """
    Detects pitch using FFT peak frequency.
    Args: frame (np.ndarray of 2048 samples), Fs (int, sampling rate)
    Returns: frequency in Hz, or None if no clear peak

    With N = 2048 samples at Fs = 44,100 Hz, each FFT bin represents Fs/N ≈ 21.5 Hz.
    This means the FFT alone cannot distinguish two pitches closer than 21.5 Hz.
    At Fs = 44,100 Hz, the maximum detectable frequency is Fs/2 = 22,050 Hz — above
    the human hearing range of 20,000 Hz. Any frequency above 22,050 Hz would alias
    back into the audible range, appearing as a false lower frequency.
    """
    N_frame = len(frame)
    windowed = frame * np.hanning(N_frame)
    spectrum = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(N_frame, 1/Fs)
    # Only search within F_MIN to F_MAX
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
    Detects pitch using FFT-based autocorrelation (NSDF / McLeod Pitch Method inspired).
    Computes R[τ] = IFFT(|X[f]|²), normalizes properly, finds first peak above threshold.
    Args: frame (np.ndarray), Fs (int), f_min/f_max (Hz bounds)
    Returns: frequency in Hz, or None if no clear peak

    By the Wiener-Khinchin theorem, the autocorrelation R[τ] = IFFT(|X[f]|²). This is
    O(N log N) vs O(N²) for direct computation. The autocorrelation peaks at lags equal
    to the fundamental period T0 = 1/f0, making it robust to harmonics.
    Zero-padding the frame to length 2*N before computing the FFT prevents circular
    (wrap-around) correlation artifacts. Without it, the IFFT would produce circular
    autocorrelation rather than linear autocorrelation, corrupting values at large lags.
    For harmonic signals (e.g. guitar), upper harmonics (2f0, 3f0) often have higher
    energy than f0 itself. FFT would report the harmonic as the pitch. Autocorrelation
    finds the period of the entire waveform, correctly reporting f0 regardless of
    harmonic structure.
    The lag with the highest R[τ] value is an integer index, giving pitch resolution
    of Fs/lag² Hz per sample. Parabolic interpolation fits a quadratic to R[τ−1], R[τ],
    R[τ+1] and estimates the true peak at a sub-integer position, achieving sub-bin
    pitch accuracy.
    """
    N_frame = len(frame)

    # Remove DC offset before windowing — critical for low-frequency accuracy
    frame = frame - np.mean(frame)

    # Apply Hann window to reduce spectral leakage
    windowed = frame * np.hanning(N_frame)

    # Zero-pad to 2*N to avoid circular correlation artifacts
    # (without zero-padding, IFFT wraps around and corrupts long-lag values)
    X = np.fft.rfft(windowed, n=2 * N_frame)

    # Wiener-Khinchin theorem: R[τ] = IFFT(|X[f]|²)
    R = np.fft.irfft(X * np.conj(X))
    R = np.real(R[:N_frame])

    # Normalize by R[0] so peak height is a correlation coefficient in [0,1]
    r0 = R[0]
    if r0 < 1e-10:
        return None  # Frame is essentially silent
    R = R / r0

    # Convert Hz bounds to lag bounds: lag = Fs / freq
    # Searching only in [tau_min, tau_max] to restrict the fundamental search range
    tau_min = max(int(Fs / f_max), 1)    # shortest lag = highest freq
    tau_max = min(int(Fs / f_min), N_frame - 2)  # longest lag = lowest freq

    if tau_max <= tau_min:
        return None

    R_search = R[tau_min:tau_max + 1]

    # Use a low, dynamic threshold: 30% of the max correlation in range
    # This is the key fix — fixed height=0.4 was too aggressive for real microphone input,
    # causing low-frequency notes (which have lower normalized correlation) to be missed.
    dynamic_threshold = max(0.15, 0.3 * np.max(R_search))

    # find_peaks: prominence filters out noise bumps, dynamic threshold ensures
    # we only accept genuine periodicities, not just the tallest random wiggle.
    peaks, _ = find_peaks(R_search, prominence=0.05, height=dynamic_threshold)

    if len(peaks) == 0:
        return None

    # Pick the peak with the highest correlation value — this is the strongest
    # periodicity in the signal, which corresponds to the true fundamental.
    # Using the highest (not first) peak reliably resolves both harmonic signals
    # (where the fundamental lag has the tallest autocorrelation peak) and pure
    # tones (where only one peak is prominent anyway).
    best_peak_idx = int(np.argmax(R_search[peaks]))
    best_tau = peaks[best_peak_idx] + tau_min

    # Parabolic interpolation for sub-sample lag accuracy
    # (improves accuracy beyond the integer-bin limit)
    if 0 < best_tau < N_frame - 1:
        alpha = R[best_tau - 1]
        beta  = R[best_tau]
        gamma = R[best_tau + 1]
        denom = alpha - 2 * beta + gamma
        if abs(denom) > 1e-10:
            best_tau = best_tau + 0.5 * (alpha - gamma) / denom

    return Fs / best_tau


def freq_to_note(f0):
    """
    Converts frequency to (note_name, frequency_hz, cents_deviation).
    Uses MIDI formula: midi = round(12 * log2(f0/440) + 69)
    Note names: ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    Cents: 1200 * log2(f0 / ideal_freq_for_that_midi)
    Returns: ("A4", 440.2, +0.8) style tuple
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
