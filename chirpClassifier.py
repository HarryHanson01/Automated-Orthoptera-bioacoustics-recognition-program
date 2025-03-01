import os
import librosa
import numpy as np
import csv
from scipy.signal import butter, filtfilt

def bandpassFilter(data, lowcut, highcut, sr, order=4):
    nyquist = 0.5 * sr
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, data)

def rmsToDb(rmsValue):
    return 20 * np.log10(rmsValue) if rmsValue > 0 else -np.inf  # Avoid log(0)

def extractSpeciesFromFilename(filename):
    species = filename.split('_')[0]  # Get the part before the first underscore
    return species

def processAudioFile(filePath):
    # Extract the species from the filename
    species = extractSpeciesFromFilename(os.path.basename(filePath))
    
    # Load audio
    y, sr = librosa.load(filePath, sr=None)
    
    # Apply bandpass filter for chirp frequencies (10–20 kHz)
    yFiltered = bandpassFilter(y, lowcut=10000, highcut=20000, sr=sr)

    # Compute spectrogram
    stft = np.abs(librosa.stft(yFiltered))
    frequencies = librosa.fft_frequencies(sr=sr)
    
    # Peak frequency
    peakFrequency = frequencies[np.argmax(stft, axis=0)].max()
    
    # Lowest frequency
    energyThreshold = 0.05 * stft.max()
    significantFrequencies = frequencies[np.where(stft.max(axis=1) > energyThreshold)]
    lowestFrequency = significantFrequencies.min() if len(significantFrequencies) > 0 else 0
    
    # Chirp rate (per second)
    energy = librosa.feature.rms(y=yFiltered)
    times = librosa.times_like(energy, sr=sr)
    
    # Detect peaks (chirps)
    chirpIndices = librosa.util.peak_pick(
        energy[0], pre_max=5, post_max=5, pre_avg=10, post_avg=10, delta=0.01, wait=20
    )
    chirpRate = len(chirpIndices) / (len(yFiltered) / sr)  # Chirps per second
    
    
    # Loudness in decibels
    loudnessValues = energy[0][chirpIndices].tolist() if len(chirpIndices) > 0 else []
    avgLoudness = np.mean(loudnessValues) if loudnessValues else 0
    avgLoudnessDb = rmsToDb(avgLoudness)

    # Background noise in decibels
    nonChirpIndices = np.setdiff1d(np.arange(len(energy[0])), chirpIndices)
    backgroundNoise = np.mean(energy[0][nonChirpIndices]) if len(nonChirpIndices) > 0 else 0
    backgroundNoiseDb = rmsToDb(backgroundNoise)
    
    # Return all features with species included
    return {
        "Species": species,
        "Peak Frequency (Hz)": round(float(peakFrequency), 10),
        "Lowest Frequency (Hz)": round(float(lowestFrequency), 10),
        "Chirp Rate (chirps/sec)": round(float(chirpRate), 10),
        "Average Loudness (dB)": round(float(avgLoudnessDb), 10),
        "Background Noise (dB)": round(float(backgroundNoiseDb), 10)
    }

def processDirectory(directoryPath, outputCsvFile):
    # Create or open the CSV file for writing
    with open(outputCsvFile, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=[
            "Species",
            "Peak Frequency (Hz)", 
            "Lowest Frequency (Hz)", 
            "Chirp Rate (chirps/sec)", 
            "Average Loudness (dB)", 
            "Background Noise (dB)"
        ])
        
        # Write the header only if the file is empty
        if file.tell() == 0:
            writer.writeheader()
        
        # Iterate over all .wav files in the directory
        for filename in os.listdir(directoryPath):
            if filename.endswith(".wav"):
                filePath = os.path.join(directoryPath, filename)
                print(f"Processing {filePath}...")
                result = processAudioFile(filePath)
                writer.writerow(result)

# Main execution
if __name__ == "__main__":
    directoryPath = "LabelledAudioChunks"  # Directory containing .wav files
    outputCsvFile = "chirp-analysis-results.csv"  # Path to the CSV output file
    processDirectory(directoryPath, outputCsvFile)
