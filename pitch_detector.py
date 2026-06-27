import sys
import threading
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sounddevice as sd
from scipy.signal import butter, sosfilt, sosfilt_zi

from pitch_utils import (
    Fs, N,
    detect_pitch_fft,
    detect_pitch_autocorr,
    freq_to_note,
    compute_spectrum_db,
    is_voiced
)

# ─── Analysis window ──────────────────────────────────────────────────────────
# 4096 samples = ~93ms at 44100 Hz. Gives 4+ full cycles even at 50 Hz.
ANALYSIS_N = 4096

# ─── High-pass filter (stateful across blocks) ───────────────────────────────
# Eliminates DC + room rumble below 40 Hz without per-frame transient artifacts.
hp_sos = butter(2, 40, btype='highpass', fs=Fs, output='sos')
hp_zi  = sosfilt_zi(hp_sos)  # carry state between blocks — no startup transients

# ─── Circular audio buffer ────────────────────────────────────────────────────
audio_buffer = np.zeros(ANALYSIS_N)
buffer_lock  = threading.Lock()

# ─── Stability constants ──────────────────────────────────────────────────────
# Key design decision: vote on *note name* (a semitone string like "A4"), NOT
# on raw Hz. Raw Hz varies ±10 Hz frame-to-frame even for a steady held note;
# converting to semitone names before voting eliminates that fluctuation entirely.
HISTORY_LEN       = 10   # how many recent pitch estimates to keep
VOTE_THRESHOLD    = 0.55  # fraction of history that must agree on a note to display it
HOLD_FRAMES       = 8    # consecutive unvoiced frames before clearing the display

# ─── Shared state written by process_audio, read by update_plot ──────────────
state = {
    'frame'     : np.zeros(N),
    'pitch_auto': None,
    'f_display' : 0.0,    # Hz value shown in the freq box
    'note'      : '—',
    'cents'     : 0.0,
    'rms_db'    : -80.0,  # Volume level for VU meter
    'history'   : [],     # List of recently played notes
}

new_data_event = threading.Event()

# ─── Processing state (owned exclusively by process_audio thread) ─────────────
note_history   = []
hz_history     = []
no_pitch_count = 0
last_played_note = None
last_played_freq = 0.0


# ─── Audio callback (runs in PortAudio thread — keep minimal) ─────────────────
def audio_callback(indata, frames, time, status):
    global audio_buffer, hp_zi

    if status:
        print(status, file=sys.stderr)

    data = indata[:, 0].astype(np.float64)

    filtered, hp_zi = sosfilt(hp_sos, data, zi=hp_zi)

    shift = len(filtered)
    with buffer_lock:
        if shift >= ANALYSIS_N:
            audio_buffer[:] = filtered[-ANALYSIS_N:]
        else:
            audio_buffer = np.roll(audio_buffer, -shift)
            audio_buffer[-shift:] = filtered

    new_data_event.set()


# ─── Background pitch-processing thread ───────────────────────────────────────
def process_audio():
    global note_history, hz_history, no_pitch_count, last_played_note, last_played_freq

    while True:
        try:
            new_data_event.wait()
            new_data_event.clear()

            with buffer_lock:
                frame = audio_buffer.copy()

            voiced = is_voiced(frame)
            pitch_auto = None
            pitch_fft  = None

            # Calculate RMS dB for VU Meter
            rms = np.sqrt(np.mean(frame**2))
            rms_db = 20 * np.log10(rms + 1e-9)
            
            if voiced:
                pitch_auto = detect_pitch_autocorr(frame, Fs)
                pitch_fft  = detect_pitch_fft(frame[-N:], Fs)

            if voiced and pitch_auto is not None:
                no_pitch_count = 0
                note_name, f_hz, cents = freq_to_note(pitch_auto)

                if note_name is not None:
                    note_history.append(note_name)
                    hz_history.append(pitch_auto)
                    if len(note_history) > HISTORY_LEN:
                        note_history.pop(0)
                        hz_history.pop(0)

                    counts       = Counter(note_history)
                    winner_note, winner_count = counts.most_common(1)[0]
                    vote_fraction = winner_count / len(note_history)

                    if vote_fraction >= VOTE_THRESHOLD:
                        winner_hz = [hz_history[i] for i, n in enumerate(note_history)
                                     if n == winner_note]
                        median_hz = float(np.median(winner_hz))

                        _, _, stable_cents = freq_to_note(median_hz)

                        # If the note changed to a different note without silence between, log the old one
                        if last_played_note is not None and last_played_note != winner_note:
                            log_entry = f"{last_played_note} ({last_played_freq:.1f}Hz)"
                            state['history'].append(log_entry)
                            if len(state['history']) > 6:
                                state['history'].pop(0)

                        state['note']      = winner_note
                        state['f_display'] = median_hz
                        state['cents']     = stable_cents if stable_cents is not None else 0.0
                        
                        last_played_note = winner_note
                        last_played_freq = median_hz

            else:
                no_pitch_count += 1
                if no_pitch_count >= HOLD_FRAMES:
                    # Silence detected - log the note that just finished playing
                    if last_played_note is not None:
                        log_entry = f"{last_played_note} ({last_played_freq:.1f}Hz)"
                        state['history'].append(log_entry)
                        if len(state['history']) > 6:
                            state['history'].pop(0)
                        last_played_note = None

                    note_history.clear()
                    hz_history.clear()
                    state['note']      = '—'
                    state['f_display'] = 0.0
                    state['cents']     = 0.0

            state['frame']      = frame[-N:]
            state['pitch_auto'] = pitch_auto
            state['rms_db']     = rms_db

        except Exception as e:
            print(f"Audio processing error: {e}", file=sys.stderr)


# ─── Plot setup ───────────────────────────────────────────────────────────────
def init_plot():
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(14, 10), facecolor='#0f0f13')
    fig.canvas.manager.set_window_title('Pro DSP Pitch & Audio Analyzer')
    
    # 4 Rows, 3 Columns
    gs = fig.add_gridspec(4, 3, height_ratios=[1.5, 1, 1, 0.4], wspace=0.3, hspace=0.4)
    
    # ─── ROW 0: Spectrum Analyzer ───
    ax_spectrum = fig.add_subplot(gs[0, :])
    ax_spectrum.set_facecolor('#18181c')
    ax_spectrum.set_xlim(0, 4500)
    ax_spectrum.set_ylim(-80, 0)
    ax_spectrum.set_xlabel("Frequency (Hz)", color='#aaaaaa', fontsize=10)
    ax_spectrum.set_ylabel("Magnitude (dB)", color='#aaaaaa', fontsize=10)
    ax_spectrum.set_title("LIVE FREQUENCY SPECTRUM", color='#ffffff', fontsize=12, fontweight='bold', pad=10)
    ax_spectrum.grid(True, color='#2a2a35', linestyle='--', linewidth=0.5)
    ax_spectrum.tick_params(colors='#aaaaaa')
    
    line_spectrum, = ax_spectrum.plot([], [], color='#00e5ff', lw=1.2) # Cyan
    line_pitch = ax_spectrum.axvline(x=0, color='#ff0055', linestyle='-', lw=2, zorder=3) # Fundamental
    # Harmonic markers
    harmonics = [ax_spectrum.axvline(x=0, color='#ffaa00', linestyle=':', lw=1.5, zorder=2) for _ in range(3)]
    for h in harmonics: h.set_visible(False)
    line_pitch.set_visible(False)

    # ─── ROW 1: Waveform & VU Meter ───
    ax_wave = fig.add_subplot(gs[1, 0:2])
    ax_wave.set_facecolor('#18181c')
    ax_wave.set_xlim(0, N)
    ax_wave.set_ylim(-1.0, 1.0)
    ax_wave.set_title("OSCILLOSCOPE (Time Domain)", color='#ffffff', fontsize=10, fontweight='bold')
    ax_wave.axis('off')
    # Downsample by 4 to reduce rendering overhead and lag
    ds = 4 
    line_wave, = ax_wave.plot(np.arange(0, N, ds), np.zeros(N//ds), color='#b300ff', lw=1.0) # Purple

    ax_vu = fig.add_subplot(gs[1, 2])
    ax_vu.set_facecolor('#18181c')
    ax_vu.set_xlim(-60, 0)
    ax_vu.set_ylim(-0.5, 0.5)
    ax_vu.set_title("INPUT LEVEL (dB)", color='#ffffff', fontsize=10, fontweight='bold')
    ax_vu.set_yticks([])
    ax_vu.tick_params(colors='#aaaaaa')
    bar_vu = ax_vu.barh([0], [-60], height=0.6, color='#00ff88')[0] # Mint green

    # ─── ROW 2: Dashboard (Note, Freq, Tuning) ───
    ax_note = fig.add_subplot(gs[2, 0])
    ax_note.axis('off')
    rect_n = plt.Rectangle((0, 0), 1, 1, transform=ax_note.transAxes, color='#18181c', zorder=-1, fill=True)
    ax_note.add_patch(rect_n)
    text_note = ax_note.text(0.5, 0.6, '—', fontsize=72, fontweight='bold', color='#00e5ff', ha='center', va='center')
    ax_note.text(0.5, 0.15, 'DETECTED NOTE', fontsize=11, color='#888888', fontweight='bold', ha='center', va='center')

    ax_freq = fig.add_subplot(gs[2, 1])
    ax_freq.axis('off')
    rect_f = plt.Rectangle((0, 0), 1, 1, transform=ax_freq.transAxes, color='#18181c', zorder=-1, fill=True)
    ax_freq.add_patch(rect_f)
    text_freq = ax_freq.text(0.5, 0.6, '—', fontsize=32, color='#ffffff', fontweight='bold', ha='center', va='center')
    ax_freq.text(0.5, 0.15, 'FREQUENCY (Hz)', fontsize=11, color='#888888', fontweight='bold', ha='center', va='center')

    ax_cents = fig.add_subplot(gs[2, 2])
    ax_cents.set_facecolor('#18181c')
    ax_cents.set_xlim(-50, 50)
    ax_cents.set_ylim(-0.5, 0.5)
    ax_cents.set_xlabel("Cents", color='#aaaaaa')
    ax_cents.set_title("TUNING ACCURACY", color='#888888', fontsize=11, fontweight='bold', pad=10)
    ax_cents.axvline(x=0, color='#ffffff', lw=2, zorder=2)
    ax_cents.axvspan(-10, 10, color='#00ff88', alpha=0.15, zorder=0)
    ax_cents.set_yticks([])
    ax_cents.tick_params(colors='#aaaaaa')
    bar_cents = ax_cents.barh([0], [0], height=0.4, color='#00ff88', zorder=3)[0]
    
    # ─── ROW 3: Note History ───
    ax_hist = fig.add_subplot(gs[3, :])
    ax_hist.axis('off')
    rect_h = plt.Rectangle((0, 0), 1, 1, transform=ax_hist.transAxes, color='#18181c', zorder=-1, fill=True)
    ax_hist.add_patch(rect_h)
    ax_hist.text(0.02, 0.5, 'HISTORY:', fontsize=12, color='#888888', fontweight='bold', ha='left', va='center')
    text_history = ax_hist.text(0.12, 0.5, 'Waiting for notes...', fontsize=14, color='#ffffff', ha='left', va='center')
    
    # Safely adjust layout without tight_layout which caused warnings
    fig.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.08)
    
    return fig, line_spectrum, line_pitch, harmonics, line_wave, bar_vu, text_note, text_freq, bar_cents, text_history


# ─── Animation callback ───────────────────────────────────────────────────────
def update_plot(frame_idx, line_spectrum, line_pitch, harmonics, line_wave, bar_vu, text_note, text_freq, bar_cents, text_history):
    
    frame_data = state['frame']
    
    # 1. Update Spectrum
    freqs, mags_db = compute_spectrum_db(frame_data, Fs)
    line_spectrum.set_data(freqs, mags_db)

    # 2. Update Waveform Oscilloscope
    # Normalize waveform for display
    max_val = np.max(np.abs(frame_data)) + 1e-6
    ds = 4
    line_wave.set_ydata((frame_data[::ds] / max_val) * 0.8) # scale to 80% of box

    # 3. Update VU Meter
    db = max(-60.0, state['rms_db'])
    bar_vu.set_width(db - (-60)) # Width is relative to left edge
    bar_vu.set_x(-60)
    if db > -10:
        bar_vu.set_color('#ff0055') # Red if clipping/loud
    elif db > -30:
        bar_vu.set_color('#ffaa00') # Yellow
    else:
        bar_vu.set_color('#00ff88') # Green

    # 4. Update Pitch & Harmonic Markers
    f0 = state['pitch_auto']
    if f0 is not None:
        line_pitch.set_xdata([f0, f0])
        line_pitch.set_visible(True)
        # Show up to 3 harmonics (2f0, 3f0, 4f0)
        for i, h_line in enumerate(harmonics):
            h_freq = f0 * (i + 2)
            if h_freq <= 4500:
                h_line.set_xdata([h_freq, h_freq])
                h_line.set_visible(True)
            else:
                h_line.set_visible(False)
    else:
        line_pitch.set_visible(False)
        for h in harmonics: h.set_visible(False)

    # 5. Update Note & Tuning Texts
    c = state['cents']
    text_note.set_text(state['note'])
    if state['note'] == '—':
        text_note.set_color('#444444')
    elif abs(c) <= 10:
        text_note.set_color('#00ff88') # Perfect
    elif abs(c) <= 30:
        text_note.set_color('#ffaa00') # Okay
    else:
        text_note.set_color('#ff0055') # Out of tune

    f_disp = state['f_display']
    text_freq.set_text(f"{f_disp:.1f}" if f_disp > 0 else '—')

    bar_cents.set_width(c)
    if abs(c) <= 10:
        bar_cents.set_color('#00ff88')
    elif abs(c) <= 30:
        bar_cents.set_color('#ffaa00')
    else:
        bar_cents.set_color('#ff0055')
        
    # 6. Update History Text
    if len(state['history']) > 0:
        text_history.set_text("   ➜   ".join(state['history']))
    else:
        text_history.set_text("Waiting for notes...")

    return (line_spectrum, line_pitch, *harmonics, line_wave, bar_vu, text_note, text_freq, bar_cents, text_history)


# ─── Entry point ─────────────────────────────────────────────────────────────
def main():
    try:
        stream = sd.InputStream(
            device=None,
            channels=1,
            samplerate=Fs,
            blocksize=512,
            dtype='float32',
            callback=audio_callback,
        )
        stream.start()
    except Exception:
        print("Error: Could not open microphone. "
              "Check your audio input device and PortAudio installation.")
        sys.exit(1)

    processor_thread = threading.Thread(target=process_audio, daemon=True)
    processor_thread.start()

    fig, line_spectrum, line_pitch, harmonics, line_wave, bar_vu, text_note, text_freq, bar_cents, text_history = init_plot()

    ani = animation.FuncAnimation(
        fig, update_plot,
        fargs=(line_spectrum, line_pitch, harmonics, line_wave, bar_vu, text_note, text_freq, bar_cents, text_history),
        interval=40, blit=True, cache_frame_data=False
    )

    plt.show()

    stream.stop()
    stream.close()


if __name__ == "__main__":
    main()
