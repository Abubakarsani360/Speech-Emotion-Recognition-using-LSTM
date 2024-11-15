
# Speech Emotion Recognition using LSTM

This project implements a Speech Emotion Recognition (SER) system using audio data. The model classifies speech signals into seven emotions: **fear, angry, disgust, neutral, sad, ps, and happy**. The system is built with Python using **Librosa** for feature extraction and a **LSTM-based neural network** for classification.

---

## Table of Contents

1. [Dataset](#dataset)  
2. [Installation](#installation)  
3. [Workflow](#workflow)  
4. [Model Architecture](#model-architecture)  
5. [Results](#results)  
6. [Usage](#usage)  
7. [Dependencies](#dependencies)  
8. [License](#license)  

---

## Dataset

The dataset used for this project is the [TESS Toronto Emotional Speech Set](https://www.kaggle.com/datasets). It contains speech samples labeled with their respective emotions.  

The structure of the dataset after processing is as follows:  

```
|-- Dataset
    |-- fear
        |-- file1.wav
        |-- file2.wav
    |-- angry
        |-- file3.wav
        |-- file4.wav
    ...
```

Each audio file name contains the emotion label as a suffix, which is extracted for classification.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/speech-emotion-recognition.git
   cd speech-emotion-recognition
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure the dataset is available and placed in the `/kaggle/input` directory.

---

## Workflow

### 1. **Data Loading**  
   Audio files are loaded and labeled based on file names.

### 2. **Exploratory Data Analysis**  
   - Visualizations of label distribution.
   - Waveforms and spectrograms for different emotions.

### 3. **Feature Extraction**  
   - MFCC (Mel Frequency Cepstral Coefficients) features are extracted using `Librosa`.

### 4. **Model Training**  
   - LSTM model architecture:
     - Input shape: (40, 1) (MFCC features)
     - Output: 7 classes (one for each emotion)
   - Model trained using `categorical_crossentropy` loss and `adam` optimizer.

### 5. **Evaluation**  
   - Model evaluated on the validation set.
   - Accuracy and loss trends monitored for performance analysis.

---

## Model Architecture

The LSTM-based neural network consists of:

- **Input Layer**: 40 MFCC features.
- **LSTM Layer**: 256 units, outputting a single vector.
- **Dense Layers**: Two hidden layers with 128 and 64 neurons, ReLU activation.
- **Output Layer**: 7 neurons, softmax activation for multi-class classification.
- **Dropout**: Applied after every layer to prevent overfitting.

---

## Results

After 50 epochs of training:

- **Training Accuracy**: ~99.78%
- **Validation Accuracy**: ~56%  
   *(Accuracy varies depending on dataset quality and preprocessing)*  

---

## Usage

1. To train the model, run:
   ```bash
   python train_model.py
   ```

2. To visualize waveforms and spectrograms for a specific emotion, modify and execute:
   ```python
   emotion = '<emotion>'  # Example: 'happy'
   ```

3. To test on new audio files, replace the file path in:
   ```python
   librosa.load('<path_to_audio_file>')
   ```

---

## Dependencies

- Python 3.8+
- Librosa
- NumPy
- Pandas
- Seaborn
- Matplotlib
- Scikit-learn
- TensorFlow/Keras

Install them with:
```bash
pip install -r requirements.txt
```

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

---
