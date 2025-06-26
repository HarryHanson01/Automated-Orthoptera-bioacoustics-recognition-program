# 🦗 Automated Orthoptera Bioacoustics Recognition 🦗
*Implemented in Python*

This project is a deep learning-based system for the **automated identification of Orthoptera** (grasshoppers and crickets) using bioacoustic data. It was developed as part of my undergraduate dissertation at Oxford Brookes University, in collaboration with Blenheim Palace, and achieved a First-Class grade.

The system processes `.wav` audio recordings, applies signal processing to detect relevant acoustic events (chirps), and uses CNN-based image classification to recognise species from spectrograms by looking at the different patterns of energy distribution.

Disclaimers:
- This project was created as part of a BSc Computer Science dissertation at Oxford Brookes University and is provided for **educational and research purposes only**.
- The trained models are **not included** due to file size limitations.
- Audio recordings used for training were provided by **Blenheim Palace** and are **not included** due to ownership restrictions.
- Accuracy is not guaranteed across all species. The model was primarily trained on Roesel's Bush-Cricket (*Roeseliana roeselii*) and Meadow Grasshopper (*Pseudochorthippus parallelus*) data found in the provided unlabeled recordings.

---

## How to Run

**1. Preprocess Audio**
use: `python event_locater_wav_files.py`

**2. Train Model**
Open `modelTraining.ipynb` and run all cells. You will need to adjust file paths to suit your system.

**3. Run Trained Model on New Data**
Use `UsingTrainedModel.ipynb` for smaller scale testing. Use the `GCDetector.py` files to run the model over a large amount of data.

**4. Combine Results**
Use `combining_GCDetections_to_one_csv.py` to get one big csv file with all your results for use in `bigDataVisualisation.ipynb`.

---

## Project Structure & Key Files

| File/Notebook                         | Description |
|--------------------------------------|-------------|
| `event_locater_wav_files.py`         | Segments raw audio into 3-second chunks and filters by signal power to isolate potential insect events. |
| `modelTraining.ipynb`                | Full model training pipeline: preprocessing, CNN design, and training from scratch. |
| `UsingTrainedModel.ipynb`            | Applies trained model to new audio samples for species identification. |
| `GCDetector.py` (+ simultaneous files) | Batch processing of multiple locations’ audio data and CSV result generation. |
| `chirpClassifier.py`                 | Initial (abandoned) feature-based classifier; replaced by image classification for better performance. |
| `FeatureVisualisation.ipynb`        | Visual analysis of extracted features and CNN results. |
| `testingMLmodels.ipynb`             | Experiments with traditional ML models (e.g. SVMs, Random Forests). |
| `fileNamer.py`                      | Quick tool for mass file renaming during manual data labelling. |
| `combining_GCDetections_to_one_csv.py` | Merges detection CSVs from multiple locations into a single dataset. |

---

## Feature Extraction Attempts

Although image-based classification ultimately proved more effective, earlier feature engineering efforts included:
- **Peak Frequency**: Dominant spectral peak
- **Lowest Frequency**: Minimum freq. with significant energy
- **Chirp Rate**: Approximate chirps/sec via RMS peak detection
- **Loudness**: RMS energy at chirp peaks in decibels
- **Background Noise**: RMS energy during non-chirp windows (approximate only)

---

## Acknowledgements

Matthias Rolf - for providing helpful insights as my project supervisor
Blenheim Palace – for their innovation team providing me field recordings and helpful advice
Oxford Brookes University –  supervision and academic support
