# Task 1.1 — Uncover the Secret Hidden Word: Documentation

## Objective
The provided audio (`task5_1.wav`) contains a spoken word masked by a loud buzz noise. The goal was to filter out the noise using digital signal processing techniques and recover the hidden word.

## Step 1: Analyzing the Signal

Before applying any filtering, I inspected the audio using a **spectrogram** (`scipy.signal.spectrogram`), which shows how the energy of the signal is distributed across frequency *and* time simultaneously. This revealed several **horizontal lines** running across the entire duration of the clip — meaning these frequencies had constant, unchanging energy from start to end. This is the signature of a **stationary tonal noise** (a buzz), as opposed to human speech, which is *non-stationary* and shows up as changing, moving patches of energy over time.

To automatically detect which frequencies belonged to the buzz, I used the **Short-Time Fourier Transform (STFT)** and computed, for every frequency bin, the **Coefficient of Variation (CV = standard deviation / mean)** across time. Frequencies with high average energy and low CV (i.e., constant over time) were flagged as buzz candidates.

## Step 2: First Attempt — Cascaded Notch Filtering

My first approach was to apply a series of **IIR notch filters** (`scipy.signal.iirnotch`) at each detected buzz frequency, one after another (cascaded), using `scipy.signal.filtfilt` to avoid phase distortion. Each notch filter removes a very narrow band of frequencies around a target frequency while leaving the rest of the signal untouched.

**Problem encountered:** The buzz turned out to be extremely rich in frequency content — over 50 distinct stationary tones spread from ~300 Hz to over 20,000 Hz (consistent with a harmonic-rich buzzer/square-wave-like sound, since its raw amplitude values were very close to ±1, i.e., near clipping). Applying 50+ notch filters in cascade caused **ringing artifacts** to accumulate, which introduced *new* audible noise rather than cleaning the signal. This taught me that stacking too many narrow filters is not always better — each filter has a small side effect, and those effects compound.

## Step 3: Second Approach — Precise FFT-Domain Filtering + Spectral Gating

To fix the ringing problem, I switched to a cleaner method:

1. **High-resolution frequency detection:** Since the buzz is stationary for the *entire* clip, I computed a single **FFT over the whole signal** (`np.fft.rfft`) instead of short windowed segments. This gives much finer frequency resolution (~0.9 Hz, versus several Hz with STFT), so each buzz tone could be pinpointed precisely instead of spreading across multiple bins.
2. **Peak detection:** I used `scipy.signal.find_peaks` with a prominence threshold to automatically locate the exact frequencies of the buzz tones from the FFT magnitude spectrum.
3. **Direct frequency-domain removal:** Instead of using IIR notch filters (which cause ringing when repeated), I directly **zeroed out** the FFT bins at each detected buzz frequency (and a couple of neighboring bins to fully capture each tone), then reconstructed the time-domain signal using the inverse FFT (`np.fft.irfft`). This removes the exact tonal components with no filter ringing, since it isn't an iterative filter — it's a direct edit of the frequency spectrum.
4. **Spectral gating cleanup:** After removing the discrete tones, I applied the `noisereduce` library (`stationary=True, prop_decrease=1.0`), which estimates the residual stationary noise floor and suppresses it across all frequencies, cleaning up any broadband hiss left behind.

## Step 4: Result

After this pipeline, the spectrogram showed a clear improvement: the constant horizontal buzz lines were gone, and in their place appeared **time-varying energy patches concentrated around 2000–4500 Hz**, consistent with the formant structure of human speech — including two distinct segments of activity separated by a brief silence, suggesting a short word or two syllables.

I also experimented with playback speed adjustment (simple resampling-based slowdown, avoiding phase-vocoder artifacts) and segment isolation to try to make the word more intelligible.

**Honest limitation:** Despite these efforts, the original signal had a very low signal-to-noise ratio (the buzz was significantly louder than the speech throughout), so full word intelligibility could not be guaranteed. Additional processing steps (AGC, band-pass tightening, more aggressive spectral gating) were tested but sometimes introduced their own artifacts (e.g., "musical noise," a known side effect of spectral gating). The final cleaned audio (`final_cleaned_audio.wav`) represents the best trade-off found between noise removal and signal distortion.

## Tools Used Summary
| Tool | Purpose |
|---|---|
| `scipy.signal.spectrogram` / `stft` | Visualize/analyze frequency content over time |
| `numpy.fft.rfft` / `irfft` | High-resolution frequency detection and direct frequency-domain filtering |
| `scipy.signal.find_peaks` | Automatically detect buzz tone frequencies |
| `scipy.signal.iirnotch` + `filtfilt` | Initial (less successful) filtering attempt |
| `noisereduce` (spectral gating) | Suppress residual broadband noise |
| `librosa` | Experimenting with playback speed for intelligibility |
