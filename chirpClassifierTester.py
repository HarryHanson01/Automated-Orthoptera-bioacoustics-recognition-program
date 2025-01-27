import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

def bandpassFilter(data, lowcut, highcut, sr, order=4):
    nyquist = 0.5 * sr
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, data)

def rmsToDb(rmsValue):
    #Convert RMS energy value to decibels
    return 20 * np.log10(rmsValue) if rmsValue > 0 else -np.inf  #avoid log(0)

def processAudioFile(file_path):
    y, sr = librosa.load(file_path, sr=None)
    
    yFiltered = bandpassFilter(y, lowcut=10000, highcut=20000, sr=sr)

    stft = np.abs(librosa.stft(yFiltered))
    frequencies = librosa.fft_frequencies(sr=sr)
    
    peakFrequency = frequencies[np.argmax(stft, axis=0)].max()
    
    energyThreshold = 0.05 * stft.max()
    significant_frequencies = frequencies[np.where(stft.max(axis=1) > energyThreshold)]
    lowestFrequency = significant_frequencies.min() if len(significant_frequencies) > 0 else 0
    
    # Chirp rate (per second)
    energy = librosa.feature.rms(y=yFiltered)
    times = librosa.times_like(energy, sr=sr)
    
    # visualize energy envelope
    plt.figure(figsize=(10, 4))
    plt.plot(times, energy[0], label="Energy Envelope")
    plt.title("Energy Envelope (Filtered Signal)")
    plt.xlabel("Time (s)")
    plt.ylabel("Energy")
    plt.legend()
    plt.show()

    # detect peaks of chirps
    chirpIndices = librosa.util.peak_pick(
        energy[0], pre_max=5, post_max=5, pre_avg=10, post_avg=10, delta=0.01, wait=20
    )
    chirpRate = len(chirpIndices) / (len(yFiltered) / sr)  # Chirps per second
    
    
    # Loudness (decibels)
    loudness_values = energy[0][chirpIndices].tolist() if len(chirpIndices) > 0 else []
    avgLoudness = np.mean(loudness_values) if loudness_values else 0
    avgLoudnessDb = rmsToDb(avgLoudness)

    # Background noise (decibels)
    nonChirpIndices = np.setdiff1d(np.arange(len(energy[0])), chirpIndices)
    backgroundNoise = np.mean(energy[0][nonChirpIndices]) if len(nonChirpIndices) > 0 else 0
    backgroundNoiseDb = rmsToDb(backgroundNoise)

    return {
            "Peak Frequency (Hz)": round(float(peakFrequency), 10),
            "Lowest Frequency (Hz)": round(float(lowestFrequency), 10),
            "Chirp Rate (chirps/sec)": round(float(chirpRate), 10),
            "Average Loudness (dB)": round(float(avgLoudnessDb), 10),
            "Background Noise (dB)": round(float(backgroundNoiseDb), 10)
        }


if __name__ == "__main__":
    audioFile = "testClips/Roesel'sBush-Cricket_Test1.wav"  # Path to .wav file
    results1 = processAudioFile(audioFile)
    print(results1)