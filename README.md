# Real-Time Musical Pitch Detector


# COMP 407 — Digital Signal Processing Mini Project
CE - 4th year/ 1st semester

# Submitted By:
Sachin Shrestha , Sushma Acharya
Roll No - 3 , 59

A real-time, professional-grade musical pitch detection application written in Python. It captures live audio from your microphone and estimates the fundamental frequency using advanced Digital Signal Processing (DSP) techniques like Autocorrelation and Fast Fourier Transforms (FFT).

The application features a Studio-Grade Pro Dashboard that provides live insights into the time domain, frequency domain, volume levels, and tuning accuracy with zero lag.


## ✨ Key Features
- **Pro DSP Dashboard:** A beautifully styled dark-mode GUI with neon syntax.
- **Hardware-Accelerated Rendering:** Uses Matplotlib's advanced Blitting Engine to draw graphics at 20+ FPS with zero UI lag or CPU choking.
- **Live Oscilloscope (Time Domain):** Visually monitors the raw audio waveform in real-time.
- **Live Frequency Spectrum:** Plots the magnitude spectrum with intelligent markers identifying the fundamental pitch and up to three upper harmonics (2nd, 3rd, 4th).
- **VU Level Meter:** Tracks microphone input levels (dB) and warns of clipping.
- **Tuning Dashboard:** Displays the detected musical note, its exact frequency (Hz), and a dynamic cents-deviation bar color-coded by accuracy (Green = Perfect, Yellow = Okay, Red = Out of Tune).
- **Note History Tracking:** Automatically tracks your playing and logs a history of recently played notes and melodies at the bottom of the screen.

## System Dependencies
This project uses `sounddevice` to capture live audio, which requires PortAudio on your system.
- **Linux:** `sudo apt install libportaudio2`
- **macOS:** `brew install portaudio`
- **Windows:** No extra step needed (sounddevice ships with PortAudio on Windows).

## Requirements
The project relies on standard scientific computing libraries in Python.
- numpy>=1.24.0
- scipy>=1.10.0
- sounddevice>=0.4.6
- matplotlib>=3.7.0

Install the requirements using pip:
```bash
pip install -r requirements.txt
```

## How to Run
Follow these steps in order to test and run the project:
1. **Install dependencies:** `pip install -r requirements.txt`
2. **Run tests:** `python test_pitch_detector.py`
3. **Generate test tones (Optional):** `python generate_test_tones.py`
4. **Run static analysis (Optional):** `python static_analysis.py`
5. **Run main application:** `python pitch_detector.py`

## File Structure

| File | Description |
|------|-------------|
| `pitch_detector.py` | Main real-time application with GUI, hardware blitting, and audio callback loop. |
| `pitch_utils.py` | Core DSP functions including FFT, autocorrelation, filtering, and note mapping. |
| `test_pitch_detector.py` | Automated unit tests to verify the DSP algorithms. |
| `generate_test_tones.py` | Standalone utility script to generate `.wav` files for testing. |
| `static_analysis.py` | Offline analysis script to plot raw waveforms and pitch results. |
| `requirements.txt` | Python library dependencies. |

## Known Limitations
- The detector works best on **monophonic** (single-note) signals like a human voice or a flute. Polyphonic sounds (e.g. strumming guitar chords) may confuse the pitch tracking.
- Audio latency depends on the underlying hardware/OS and the fixed frame size (4096 samples corresponds to ~93ms at 44.1kHz).
