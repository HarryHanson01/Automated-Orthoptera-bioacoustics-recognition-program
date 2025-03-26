import os
import csv
import cv2
import torch
import torch.nn as nn
import numpy as np
import scipy.io.wavfile as wav
from scipy.signal import spectrogram
import matplotlib.pyplot as plt
from tqdm import tqdm  


# Sauvola thresholding 
def sauvolaThresholding(intensityNorm, windowSize=15, k=0.5, R=128):
    intensityScaled = (intensityNorm * 255).astype(np.uint8)
    mean = cv2.boxFilter(intensityScaled, ddepth=-1, ksize=(windowSize, windowSize))
    meanSq = cv2.boxFilter(intensityScaled**2, ddepth=-1, ksize=(windowSize, windowSize))
    std = np.sqrt(meanSq - mean**2)
    threshold = mean * (1 + k * ((std / R) - 1))
    binarySpectrogram = intensityScaled > threshold
    return binarySpectrogram


# Save the binary spectrogram as an image
def saveBinarySpectrogramAsImage(audioFile, outputPath):
    sampleRate, audioData = wav.read(audioFile)
    if len(audioData.shape) > 1:
        audioData = audioData[:, 0]
    nps = 2048
    frequencies, time, intensity = spectrogram(audioData, fs=sampleRate, nperseg=nps, noverlap=(nps // 2))
    mask = (frequencies >= 10000) & (frequencies <= 40000)
    frequencies = frequencies[mask]
    intensity = intensity[mask, :]
    intensityNorm = np.clip(intensity / np.max(intensity), 0, 1)
    binarySpectrogram = sauvolaThresholding(intensityNorm, windowSize=15, k=0.5)
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(time, frequencies, binarySpectrogram, cmap='gray', shading='gouraud')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(outputPath, bbox_inches='tight', pad_inches=0)
    plt.close()


# Process the binary spectrogram image 
def processImage(inputImagePath):
    image = cv2.imread(inputImagePath, 0)
    kernel = np.ones((5, 5), np.uint8)
    medianFilteredImage = cv2.medianBlur(image, 5)
    dilatedImage = cv2.dilate(medianFilteredImage, kernel, iterations=1)
    processedImage = cv2.erode(dilatedImage, kernel, iterations=1)
    return processedImage


# CNN model definition
class CNNModel(nn.Module):
    def __init__(self, numClasses):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, numClasses)
        self.dropout = nn.Dropout(0.5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# Class labels for classification - these may need updating
classLabels = ['BackgroundNoise', 'MeadowGrasshopper', 'NoID1', 'NoID10', "Roesel'sBush-Cricket"]

# Load model
model = CNNModel(5)
model.load_state_dict(torch.load('cnn_grasshopper_cricket_classifier_26-02-2025.pth', weights_only=True))
model.eval()

# The base chunk folder
baseDir = 'E:\\GCAudioChunks'

# CSV file for output
outputFile = 'GCDetections.csv'

# CSV columns
header = ['Location', 'Date', 'Time', 'Species']

with open(outputFile, mode='w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(header)

    for locationFolder in tqdm(os.listdir(baseDir), desc="Processing Location Folders", unit="location"):
        locationPath = os.path.join(baseDir, locationFolder)

        if os.path.isdir(locationPath):
            locationName = locationFolder.replace(' - chunks', '')

            chunkFolders = [f for f in os.listdir(locationPath) if os.path.isdir(os.path.join(locationPath, f))]
            
            for chunkFolder in tqdm(chunkFolders, desc=f"Processing {locationName} Chunk Folders", unit="chunk_folder"):
                chunkPath = os.path.join(locationPath, chunkFolder)

                date = chunkFolder.split('_')[0]
                baseTime = chunkFolder.split('_')[1].split(' ')[0]

                for chunkFile in os.listdir(chunkPath):
                    if chunkFile.startswith('chunk_') and chunkFile.endswith('.wav'):
                        chunkNumber = int(chunkFile.split('_')[1].split('.')[0])

                        # Calculate time: (3 * chunk_number) + base time from folder name
                        hours = int(baseTime[:2])
                        minutes = int(baseTime[2:4])
                        seconds = int(baseTime[4:6])
                        totalSeconds = (hours * 3600) + (minutes * 60) + seconds + (3 * chunkNumber)
                        newHours = (totalSeconds // 3600) % 24
                        newMinutes = (totalSeconds % 3600) // 60
                        newSeconds = totalSeconds % 60
                        time = f'{newHours:02}{newMinutes:02}{newSeconds:02}'

                        # Process the audio file and get classification
                        chunkFilePath = os.path.join(chunkPath, chunkFile)
                        outputImagePath = "temp_spectrogram.png"
                        saveBinarySpectrogramAsImage(chunkFilePath, outputImagePath)

                        # Process the image before passing it to the model
                        processedImage = processImage(outputImagePath)

                        if processedImage is not None:
                            imageDataResized = cv2.resize(processedImage, (224, 224))
                            imageTensor = torch.tensor(imageDataResized, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                            imageTensor = imageTensor / 255.0

                            with torch.no_grad():
                                output = model(imageTensor)
                                _, predicted = torch.max(output, 1)
                                species = classLabels[predicted.item()]
                        else:
                            species = 'Error loading image'

                        writer.writerow([locationName, date, time, species])

print(f'Data successfully written to {outputFile}')
