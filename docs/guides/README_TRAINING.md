# Gesture Recognition Model Training

This repository contains machine learning models for recognizing both static and dynamic hand gestures using MediaPipe hand landmarks.

## Overview

- **Static Gestures**: Single-frame gestures (e.g., "OK", "Delete", "Help")
  - Input: 1×63 vector (21 landmarks × 3 coordinates)
  - Model: Dense Neural Network
  
- **Dynamic Gestures**: Multi-frame gestures with motion (e.g., "Hello", "Thank You")
  - Input: 30×63 matrix (30 frames × 63 landmarks)
  - Model: 2-layer GRU (Gated Recurrent Unit) with Global Average Pooling

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Data Structure

The training script expects data in the following structure:
```
SignData/
├── hello/
│   ├── static/
│   │   ├── 0.npy
│   │   ├── 1.npy
│   │   └── ...
│   └── dynamic/
│       ├── 1766589708/
│       │   ├── 0.npy
│       │   ├── 1.npy
│       │   └── ...
│       └── 1766589719/
│           └── ...
├── thanks/
│   └── ...
└── ...
```

- **Static gestures**: Each `.npy` file contains a single 63-dimensional vector (or 126, first 63 will be used)
- **Dynamic gestures**: Each timestamped folder contains a sequence of `.npy` files, one per frame

## Training

Run the training script:
```bash
python train_gesture_model.py
```

This will:
1. Load all static gesture data and train a Dense neural network
2. Load all dynamic gesture data and train a GRU model
3. Save trained models to the `models/` directory
4. Save label mappings and normalization parameters

### Model Architecture

#### Static Gesture Model
- Input: (63,)
- Layers:
  - Dense(128, ReLU) + Dropout(0.3)
  - Dense(64, ReLU) + Dropout(0.3)
  - Dense(num_classes, Softmax)

#### Dynamic Gesture Model
- Input: (30, 63)
- Layers:
  - GRU(64, return_sequences=True) + Dropout(0.2)
  - GRU(64, return_sequences=True) + Dropout(0.2)
  - GlobalAveragePooling1D()
  - Dense(128, ReLU) + Dropout(0.3)
  - Dense(64, ReLU) + Dropout(0.3)
  - Dense(num_classes, Softmax)

### Training Parameters

- **Batch Size**: 32
- **Epochs**: 50 (with early stopping)
- **Validation Split**: 20%
- **Optimizer**: Adam
- **Loss**: Sparse Categorical Crossentropy

## Prediction

Use the prediction script to make predictions on new data:

```python
from predict_gesture import GesturePredictor
import numpy as np

# Initialize predictor
predictor = GesturePredictor()
predictor.load_models()

# Predict static gesture
landmarks = np.load('path/to/static_gesture.npy')  # Shape: (63,) or (126,)
gesture, confidence = predictor.predict_static(landmarks)
print(f"Predicted: {gesture} (confidence: {confidence:.2%})")

# Predict dynamic gesture
sequence = []  # List of frames, each frame is (63,) or (126,)
for frame_file in frame_files:
    frame = np.load(frame_file)
    sequence.append(frame[:63])  # Use first 63 landmarks
sequence = np.array(sequence)  # Shape: (n_frames, 63)

gesture, confidence = predictor.predict_dynamic(sequence)
print(f"Predicted: {gesture} (confidence: {confidence:.2%})")
```

Or run the example:
```bash
python predict_gesture.py
```

## Output Files

After training, the following files will be saved in `models/`:

- `static_gesture_model.h5` - Trained static gesture model
- `dynamic_gesture_model.h5` - Trained dynamic gesture model
- `static_label_mapping.json` - Mapping from class indices to gesture names
- `dynamic_label_mapping.json` - Mapping from class indices to gesture names
- `static_mean.npy` - Mean values for normalization (static)
- `static_std.npy` - Standard deviation for normalization (static)
- `dynamic_mean.npy` - Mean values for normalization (dynamic)
- `dynamic_std.npy` - Standard deviation for normalization (dynamic)

## Notes

- The models automatically handle sequences of different lengths (padding/truncation to 30 frames)
- If input data has 126 dimensions (2 hands), only the first 63 (single hand) are used
- Models use early stopping to prevent overfitting
- Data is automatically normalized using mean and standard deviation

## Performance

The models are optimized for edge devices (Raspberry Pi) with:
- Low computational requirements (GRU instead of LSTM)
- Fast inference time (<20ms for dynamic gestures)
- Efficient memory usage


