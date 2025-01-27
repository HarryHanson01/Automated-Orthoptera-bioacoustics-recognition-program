import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import welch

# Load the .wav file
#filename = 'testClips\\4secClipNoCreatures.wav' 
filename = 'testClips\\firstSmallTestClipForDisplay.wav'  
fs, data = wavfile.read(filename)

# Calculate the Power Spectral Density using Welch's method
frequencies, psd = welch(data, fs, nperseg=1000)

# Define a power threshold (e.g., 0.001 W/Hz)
powerThreshold = 2

# Optionally, print specific frequencies and power values if above the threshold
highPowerFrequencies = frequencies[psd > powerThreshold]
highPowerValues = psd[psd > powerThreshold]

count = 0
for freq, power in zip(highPowerFrequencies, highPowerValues):
    if power > 2:
        count += 1
        print(f"Frequency: {freq:.2f} Hz, Power: {power:.5f} W/Hz")
print(count)

# Plot the Power Spectral Density
plt.figure(figsize=(10, 5))
plt.semilogy(frequencies, psd)  # Semilogarithmic scale for y-axis
plt.title('Power Spectral Density (PSD) Plot')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power/Frequency (W/Hz)')
plt.grid()
plt.xlim(0, fs / 2)  # Limit x-axis to Nyquist frequency
plt.show()