import wave
import soundfile as sf

from pydub import AudioSegment
import os

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import welch


def createNextFolder(basePath):
  existingFolders = [
    f for f in os.listdir(basePath) 
    if os.path.isdir(os.path.join(basePath, f)) and f.startswith("chunkFolder-")
  ]
  
  folderNumbers = []
  for folder in existingFolders:
    try:
      folderNum = int(folder.split("chunkFolder-")[1])
      folderNumbers.append(folderNum)
    except (ValueError, IndexError):
      continue  # Skip 
  
  nextFolderNumber = max(folderNumbers, default=0) + 1
  
  return f"chunkFolder-{nextFolderNumber}"

def splitAudioFiles(filePath):
  audio = AudioSegment.from_file(filePath)
  chunkLengthMs = 3000  # 3 seconds in milliseconds

  locationFolderName = os.path.basename(os.path.dirname(os.path.dirname(filePath)))
  baseChunkFolderName = f"{locationFolderName} - chunks"
  baseChunkFolderPath = os.path.join("E:\\audioChunks", baseChunkFolderName)
  os.makedirs(baseChunkFolderPath, exist_ok=True)
  nextFolderName = createNextFolder(baseChunkFolderPath)
  outputFolder = os.path.join(baseChunkFolderPath, nextFolderName)
  os.makedirs(outputFolder, exist_ok=True)
  print(f"Saving chunks in folder: {outputFolder}")
  
  for i in range(0, len(audio), chunkLengthMs):
    chunk = audio[i:i + chunkLengthMs]
    chunkFilename = os.path.join(outputFolder, f"chunk_{i // chunkLengthMs:03d}.wav")
    
    chunk.export(chunkFilename, format="wav")

def delQuietFiles(folderPath):
  files = os.listdir(folderPath)

  for fileName in files:
    filePath = os.path.join(folderPath, fileName)
    
    if os.path.isfile(filePath):
      if detailsOfWavFile(filePath)['countAbove'] < 30:
        os.remove(filePath)

def removeAllQuietChunks(basePath="E:\\audioChunks"):
  for mainFolder in os.listdir(basePath):
    print("main folder: ", mainFolder)
    mainFolderPath = os.path.join(basePath, mainFolder)
    if os.path.isdir(mainFolderPath):
      for subfolder in os.listdir(mainFolderPath):
        subfolderPath = os.path.join(mainFolderPath, subfolder)
        if os.path.isdir(subfolderPath):
          delQuietFiles(subfolderPath)

def goThroughAllSoundFiles(basePath="E:\Cricket Grasshopper"):
  for mainFolder in os.listdir(basePath):
    print("main folder: ", mainFolder)
    mainFolderPath = os.path.join(basePath, mainFolder)
    if os.path.isdir(mainFolderPath):
      for dateFolder in os.listdir(mainFolderPath):
        dateFolderPath = os.path.join(mainFolderPath, dateFolder)
        if os.path.isdir(dateFolderPath):
          print(" ", dateFolderPath)
          for file in os.listdir(dateFolderPath):
            filePath = os.path.join(dateFolderPath, file)
            splitAudioFiles(filePath)

def plotPsd(frequencies, psd, fs, filePath):
  plt.figure(figsize=(10, 5))
  plt.semilogy(frequencies, psd)
  plt.title(f'Power Spectral Density (PSD) Plot for: {filePath}')
  plt.xlabel('Frequency (Hz)')
  plt.ylabel('Power/Frequency (W/Hz)')
  plt.grid()
  plt.xlim(0, fs / 2)
  plt.show()

def detailsOfWavFile(filePath):
  fs, data = wavfile.read(filePath)
  frequencies, psd = welch(data, fs, nperseg=1000)

  powerThreshold = 2
  highPowerFrequencies = frequencies[psd > powerThreshold]
  highPowerValues = psd[psd > powerThreshold]
  countAbove = 0
  highestFrequency = None
  lowestFrequency = None
  for freq, power in zip(highPowerFrequencies, highPowerValues):
      if power > 2:
          countAbove += 1
          if highestFrequency is None or freq > highestFrequency:
            highestFrequency = int(freq)
          if lowestFrequency is None or freq < lowestFrequency:
            lowestFrequency = int(freq)
  
  frameDict = {
    "filePath": filePath,
    "countAbove": countAbove,
    "highestFrequency": highestFrequency,
    "lowestFrequency": lowestFrequency
  }
  return frameDict



###   main code   ###

#goThroughAllSoundFiles()
#removeAllQuietChunks()