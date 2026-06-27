import numpy as np
from scipy.io import wavfile

# Audio parameters
Fs = 44100
duration = 3.0
fade_duration = 0.010  # 10ms

def apply_fade(signal, Fs, fade_duration):
    """
    Applies a linear fade-in and fade-out to the signal to avoid clicks.
    """
    fade_samples = int(Fs * fade_duration)
    fade_in = np.linspace(0, 1, fade_samples)
    fade_out = np.linspace(1, 0, fade_samples)
    
    signal[:fade_samples] *= fade_in
    signal[-fade_samples:] *= fade_out
    return signal

def save_wav(filename, signal, Fs):
    """
    Saves a signal as a 16-bit PCM wav file.
    """
    # Normalize to 16-bit PCM range
    signal = signal / np.max(np.abs(signal))
    signal_16bit = np.int16(signal * 32767)
    wavfile.write(filename, Fs, signal_16bit)
    print(f"Saved {filename}")

def generate_tones():
    t = np.linspace(0, duration, int(Fs * duration), endpoint=False)
    
    # tone_A4_440Hz.wav
    tone_A4 = np.sin(2 * np.pi * 440.0 * t)
    tone_A4 = apply_fade(tone_A4, Fs, fade_duration)
    save_wav("tone_A4_440Hz.wav", tone_A4, Fs)
    
    # tone_C4_261Hz.wav
    tone_C4 = np.sin(2 * np.pi * 261.63 * t)
    tone_C4 = apply_fade(tone_C4, Fs, fade_duration)
    save_wav("tone_C4_261Hz.wav", tone_C4, Fs)
    
    # tone_E4_330Hz.wav
    tone_E4 = np.sin(2 * np.pi * 329.63 * t)
    tone_E4 = apply_fade(tone_E4, Fs, fade_duration)
    save_wav("tone_E4_330Hz.wav", tone_E4, Fs)
    
    # tone_harmonic_220Hz.wav
    tone_harmonic = (0.3 * np.sin(2 * np.pi * 220.0 * t) + 
                     0.8 * np.sin(2 * np.pi * 440.0 * t) + 
                     0.5 * np.sin(2 * np.pi * 660.0 * t))
    tone_harmonic = apply_fade(tone_harmonic, Fs, fade_duration)
    save_wav("tone_harmonic_220Hz.wav", tone_harmonic, Fs)

if __name__ == "__main__":
    generate_tones()
