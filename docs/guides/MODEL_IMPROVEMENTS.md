# Sign Language Model Improvements

## Problem
The original model was only detecting "yes" and not other words. This was because the preprocessing focused on absolute coordinates rather than hand SHAPE.

## Solution
Created a new training pipeline (`train_model_v2.py`) that focuses on **hand shape features** rather than absolute coordinates.

## Key Improvements

### 1. Shape-Focused Preprocessing
- **Wrist-Centering**: Subtracts wrist position (position invariant)
- **Size Normalization**: Divides by hand size (scale invariant)
- **Shape Features**: Extracts:
  - Normalized landmark positions
  - Finger tip distances from wrist
  - Joint angles (finger bending)
  - Finger spread (distances between fingertips)
  - Thumb opposition
  - Hand orientation
  - Inter-hand features (for two-hand gestures)

### 2. Better Feature Engineering
Instead of using raw coordinates, the model now uses:
- Relative positions (normalized)
- Angles between joints
- Distances between key points
- Shape descriptors

This makes the model robust to:
- Different hand positions
- Different hand sizes
- Different camera angles
- Slight variations in gesture execution

### 3. Hyperparameter Tuning
- Uses `RandomizedSearchCV` for efficient hyperparameter tuning
- Tests 50 random combinations
- 5-fold cross-validation
- Finds optimal parameters automatically

### 4. StandardScaler
- Features are standardized (mean=0, std=1)
- Improves model performance
- Stored with the model for inference

## Files Created/Updated

1. **`train_model_v2.py`** - New training script with shape-focused preprocessing
2. **`test_model_cv2.py`** - OpenCV test script for real-time testing
3. **`Reciver/reciver.py`** - Updated to use new preprocessing and scaler
4. **`inference_helper.py`** - Updated to use new preprocessing

## How to Use

### 1. Train the Model
```bash
python train_model_v2.py
```

This will:
- Load all training data
- Extract shape features
- Train Random Forest with hyperparameter tuning
- Evaluate on test data
- Save model as `sign_model.pkl`

### 2. Test the Model
```bash
python test_model_cv2.py
```

This will:
- Load the trained model
- Open your webcam
- Show real-time predictions
- Display top 3 predictions with confidence

### 3. Use with Receiver
The receiver (`Reciver/reciver.py`) automatically uses the new preprocessing when you retrain the model.

## Expected Improvements

- **Better accuracy** across all words (not just "yes")
- **Position invariant** - works regardless of hand position
- **Scale invariant** - works with different hand sizes
- **More robust** to variations in gesture execution

## Feature Count

The new model uses **171 features** (vs 128 raw coordinates):
- ~75 features per hand (shape descriptors)
- ~20 inter-hand features (for two-hand gestures)

## Notes

- Make sure to retrain the model using `train_model_v2.py`
- The old model (`sign_model.pkl`) will be overwritten
- All inference scripts will automatically use the new preprocessing
- The model focuses on **hand shape**, not absolute coordinates

