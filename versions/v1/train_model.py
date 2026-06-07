import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# --- Configuration ---
DATA_PATH = r'C:\Users\jeeva\OneDrive\Documents\EdgeSign An IoT-Enabled Edge–Cloud Hybrid System for Real-Time\project\New folder\SignData'
MODEL_PATH = 'models'
os.makedirs(MODEL_PATH, exist_ok=True)

def load_dataset(data_path):
    """
    Load all training and testing data with proper normalization check
    """
    X_train, y_train = [], []
    X_test, y_test = [], []
    
    actions = sorted([d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))])
    
    print("=" * 60)
    print("LOADING DATASET")
    print("=" * 60)
    
    for action in actions:
        action_path = os.path.join(data_path, action)
        
        # Load training data
        train_path = os.path.join(action_path, 'train')
        if os.path.exists(train_path):
            train_files = [f for f in os.listdir(train_path) if f.endswith('.npy')]
            for file in train_files:
                data = np.load(os.path.join(train_path, file))
                X_train.append(data)
                y_train.append(action)
            print(f"  {action}: {len(train_files)} train samples")
        
        # Load test data
        test_path = os.path.join(action_path, 'test')
        if os.path.exists(test_path):
            test_files = [f for f in os.listdir(test_path) if f.endswith('.npy')]
            for file in test_files:
                data = np.load(os.path.join(test_path, file))
                X_test.append(data)
                y_test.append(action)
    
    print(f"\nTotal train samples: {len(X_train)}")
    print(f"Total test samples: {len(X_test)}")
    print(f"Number of classes: {len(actions)}")
    
    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test), actions

def augment_data(X, y, augmentation_factor=3):
    """
    Data augmentation by adding small random noise
    This helps model generalize better
    """
    X_augmented = [X]
    y_augmented = [y]
    
    for i in range(augmentation_factor):
        noise = np.random.normal(0, 0.02, X.shape)  # Small Gaussian noise
        X_noisy = X + noise
        X_augmented.append(X_noisy)
        y_augmented.append(y)
    
    X_final = np.vstack(X_augmented)
    y_final = np.concatenate(y_augmented)
    
    return X_final, y_final

def create_ensemble_model():
    """
    Create an ensemble of high-performance models
    
    Why this works:
    - Random Forest: Robust to outliers, handles non-linear relationships
    - XGBoost: Best gradient boosting, excellent for structured data
    - Gradient Boosting: Additional boosting diversity
    - Voting: Combines strengths, reduces individual weaknesses
    """
    
    # Model 1: Random Forest with optimized hyperparameters
    rf_model = RandomForestClassifier(
        n_estimators=300,        # More trees = better accuracy
        max_depth=30,            # Deep enough to capture patterns
        min_samples_split=5,     # Prevent overfitting
        min_samples_leaf=2,
        max_features='sqrt',     # Feature subset randomization
        n_jobs=-1,               # Use all CPU cores
        random_state=42,
        class_weight='balanced'  # Handle class imbalance
    )
    
    # Model 2: XGBoost - Industry standard for structured data
    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=10,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        n_jobs=-1,
        random_state=42
    )
    
    # Model 3: Gradient Boosting
    gb_model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42
    )
    
    # Ensemble: Soft voting for probability-based prediction
    ensemble = VotingClassifier(
        estimators=[
            ('rf', rf_model),
            ('xgb', xgb_model),
            ('gb', gb_model)
        ],
        voting='soft',  # Average predicted probabilities
        n_jobs=-1
    )
    
    return ensemble

def train_model(X_train, y_train, X_test, y_test, actions):
    """
    Complete training pipeline with validation
    """
    
    print("\n" + "=" * 60)
    print("TRAINING PIPELINE")
    print("=" * 60)
    
    # Step 1: Encode labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    
    # Step 2: Data augmentation
    print("\n[1/6] Applying data augmentation...")
    X_train_aug, y_train_aug = augment_data(X_train, y_train_encoded, augmentation_factor=2)
    print(f"  Augmented training set: {len(X_train_aug)} samples")
    
    # Step 3: Feature scaling (important for some models)
    print("\n[2/6] Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_aug)
    X_test_scaled = scaler.transform(X_test)
    
    # Step 4: Create and train ensemble model
    print("\n[3/6] Creating ensemble model...")
    model = create_ensemble_model()
    
    print("\n[4/6] Training ensemble (this may take a few minutes)...")
    model.fit(X_train_scaled, y_train_aug)
    
    # Step 5: Evaluate on training set
    print("\n[5/6] Evaluating on training set...")
    y_train_pred = model.predict(X_train_scaled)
    train_accuracy = accuracy_score(y_train_aug, y_train_pred)
    print(f"  Training Accuracy: {train_accuracy * 100:.2f}%")
    
    # Step 6: Evaluate on test set
    print("\n[6/6] Evaluating on test set...")
    y_test_pred = model.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test_encoded, y_test_pred)
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    
    # Detailed classification report
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(
        y_test_encoded, 
        y_test_pred, 
        target_names=label_encoder.classes_,
        digits=3
    ))
    
    # Confusion matrix analysis
    print("\n" + "=" * 60)
    print("CONFUSION MATRIX ANALYSIS")
    print("=" * 60)
    cm = confusion_matrix(y_test_encoded, y_test_pred)
    
    # Find most confused pairs
    confused_pairs = []
    for i in range(len(cm)):
        for j in range(len(cm)):
            if i != j and cm[i][j] > 0:
                confused_pairs.append((
                    label_encoder.classes_[i],
                    label_encoder.classes_[j],
                    cm[i][j]
                ))
    
    confused_pairs.sort(key=lambda x: x[2], reverse=True)
    
    if confused_pairs:
        print("\nMost confused sign pairs:")
        for true_label, pred_label, count in confused_pairs[:5]:
            print(f"  '{true_label}' confused with '{pred_label}': {count} times")
    else:
        print("  No confusion! Perfect classification!")
    
    # Step 7: Cross-validation on training set
    print("\n" + "=" * 60)
    print("CROSS-VALIDATION (5-Fold)")
    print("=" * 60)
    
    # Use a simpler model for CV to save time
    cv_model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    cv_scores = cross_val_score(cv_model, X_train_scaled, y_train_aug, cv=5)
    print(f"CV Scores: {cv_scores}")
    print(f"CV Mean: {cv_scores.mean() * 100:.2f}% (+/- {cv_scores.std() * 2 * 100:.2f}%)")
    
    # Step 8: Save model and preprocessing objects
    print("\n" + "=" * 60)
    print("SAVING MODEL")
    print("=" * 60)
    
    model_data = {
        'model': model,
        'scaler': scaler,
        'label_encoder': label_encoder,
        'actions': actions,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy
    }
    
    model_file = os.path.join(MODEL_PATH, 'sign_language_model.pkl')
    with open(model_file, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"✓ Model saved to: {model_file}")
    print(f"  File size: {os.path.getsize(model_file) / 1024 / 1024:.2f} MB")
    
    # Save lightweight version (Random Forest only) for faster inference
    lightweight_data = {
        'model': model.estimators_[0],  # Just the Random Forest
        'scaler': scaler,
        'label_encoder': label_encoder,
        'actions': actions
    }
    
    lightweight_file = os.path.join(MODEL_PATH, 'sign_language_model_lite.pkl')
    with open(lightweight_file, 'wb') as f:
        pickle.dump(lightweight_data, f)
    
    print(f"✓ Lightweight model saved to: {lightweight_file}")
    print(f"  File size: {os.path.getsize(lightweight_file) / 1024 / 1024:.2f} MB")
    
    return model, scaler, label_encoder

def analyze_dataset_quality(X, y):
    """
    Analyze dataset quality and distribution
    """
    print("\n" + "=" * 60)
    print("DATASET QUALITY ANALYSIS")
    print("=" * 60)
    
    # Class distribution
    class_counts = Counter(y)
    print("\nClass distribution:")
    for action, count in sorted(class_counts.items()):
        print(f"  {action}: {count} samples")
    
    # Check for imbalance
    max_count = max(class_counts.values())
    min_count = min(class_counts.values())
    imbalance_ratio = max_count / min_count
    
    print(f"\nImbalance ratio: {imbalance_ratio:.2f}")
    if imbalance_ratio > 3:
        print("  ⚠ WARNING: Significant class imbalance detected!")
        print("  → Consider collecting more samples for underrepresented classes")
    else:
        print("  ✓ Dataset is reasonably balanced")
    
    # Feature statistics
    print(f"\nFeature statistics:")
    print(f"  Feature dimension: {X.shape[1]}")
    print(f"  Mean feature value: {np.mean(X):.4f}")
    print(f"  Std feature value: {np.std(X):.4f}")
    
    # Check for NaN or Inf
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        print("  ⚠ WARNING: Dataset contains NaN or Inf values!")
    else:
        print("  ✓ No NaN or Inf values detected")

if __name__ == "__main__":
    # Load dataset
    X_train, y_train, X_test, y_test, actions = load_dataset(DATA_PATH)
    
    if len(X_train) == 0:
        print("\n⚠ ERROR: No training data found!")
        print("Please run data_collector_advanced.py first to collect data.")
        exit(1)
    
    if len(X_test) == 0:
        print("\n⚠ WARNING: No test data found!")
        print("Using train/test split instead...")
        X_train, X_test, y_train, y_test = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
    
    # Analyze dataset
    analyze_dataset_quality(X_train, y_train)
    
    # Train model
    model, scaler, label_encoder = train_model(X_train, y_train, X_test, y_test, actions)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Run 'real_time_inference.py' to test the model")
    print("  2. If accuracy is low, collect more data for confused signs")
    print("  3. Consider adjusting model hyperparameters")
    print("\n" + "=" * 60)
