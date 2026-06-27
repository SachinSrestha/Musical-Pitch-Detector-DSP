import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from pitch_utils import (
    detect_pitch_fft,
    detect_pitch_autocorr,
    freq_to_note,
    compute_spectrum_db,
    Fs, N, F_MIN, F_MAX
)

def run_analysis():
    filename = "tone_harmonic_220Hz.wav"
    if not os.path.exists(filename):
        print(f"Error: {filename} not found. Run generate_test_tones.py first.")
        sys.exit(1)
    
    print(f"Loading {filename}...")
    sample_rate, data = wavfile.read(filename)
    
    # Normalize back to [-1, 1] if it's 16-bit PCM
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32767.0
        
    print(f"Frame size: {N} samples | Fs: {sample_rate} Hz\n")
    
    # Use the first N samples (or from the middle to avoid fade-in)
    # Let's take samples from 0.5s to get a stable tone
    start_idx = int(0.5 * sample_rate)
    frame = data[start_idx : start_idx + N]
    
    pitch_fft = detect_pitch_fft(frame, sample_rate)
    pitch_auto = detect_pitch_autocorr(frame, sample_rate)
    
    note_fft, f_fft, cents_fft = freq_to_note(pitch_fft)
    note_auto, f_auto, cents_auto = freq_to_note(pitch_auto)
    
    print("FFT Method:")
    print(f"  Detected frequency : {f_fft:.1f} Hz")
    print(f"  Note               : {note_fft}")
    print(f"  Cents deviation    : {cents_fft:+.1f}\n")
    
    print("Autocorrelation Method:")
    print(f"  Detected frequency : {f_auto:.1f} Hz")
    print(f"  Note               : {note_auto}")
    print(f"  Cents deviation    : {cents_auto:+.1f}\n")
    
    # Plotting
    fig, axs = plt.subplots(3, 1, figsize=(10, 10))
    fig.subplots_adjust(hspace=0.4)
    
    # Panel 1: Raw waveform
    t = np.arange(N) / sample_rate
    axs[0].plot(t, frame)
    axs[0].set_title("Raw Waveform (Time Domain)")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Amplitude")
    axs[0].grid(True)
    
    # Panel 2: FFT magnitude spectrum
    freqs, mags_db = compute_spectrum_db(frame, sample_rate)
    axs[1].plot(freqs, mags_db)
    axs[1].set_xlim([0, 1000])
    
    if pitch_fft is not None:
        axs[1].axvline(x=pitch_fft, color='r', linestyle='--', label=f'FFT Pitch ({pitch_fft:.1f} Hz)')
    if pitch_auto is not None:
        axs[1].axvline(x=pitch_auto, color='g', linestyle='--', label=f'Autocorr Pitch ({pitch_auto:.1f} Hz)')
    axs[1].legend()
    axs[1].set_title("FFT Magnitude Spectrum")
    axs[1].set_xlabel("Frequency (Hz)")
    axs[1].set_ylabel("Magnitude (dB)")
    axs[1].grid(True)
    
    # Panel 3: Autocorrelation array
    # Re-computing it to get R for plotting
    windowed = frame * np.hanning(N)
    X = np.fft.rfft(windowed, n=2*N)
    R = np.fft.irfft(X * np.conj(X))
    R = R[:N]
    R = R / (R[0] + 1e-9)
    tau_min = max(int(sample_rate / F_MAX), 1)
    tau_max = min(int(sample_rate / F_MIN), N - 1)
    
    lags = np.arange(tau_min, tau_max)
    R_search = R[tau_min:tau_max]
    axs[2].plot(lags, R_search)
    
    if pitch_auto is not None:
        best_lag = sample_rate / pitch_auto
        axs[2].axvline(x=best_lag, color='g', linestyle='--', label=f'Detected Lag ({best_lag:.1f})')
        axs[2].legend()
        
    axs[2].set_title("Autocorrelation R[τ] (Valid Lag Range)")
    axs[2].set_xlabel("Lag (samples)")
    axs[2].set_ylabel("Correlation Coefficient")
    axs[2].grid(True)
    
    plt.savefig("analysis_output.png")
    print("Analysis saved to: analysis_output.png")

if __name__ == "__main__":
    run_analysis()
