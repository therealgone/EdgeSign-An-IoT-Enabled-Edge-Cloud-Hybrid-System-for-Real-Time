import numpy as np
import os
from collections import Counter
import matplotlib.pyplot as plt

# --- Configuration ---
DATA_PATH = 'SignData'

def analyze_dataset():
    """
    Comprehensive dataset quality analysis
    """
    print("=" * 70)
    print(" " * 20 + "DATASET ANALYSIS")
    print("=" * 70)
    
    if not os.path.exists(DATA_PATH):
        print(f"\n⚠ ERROR: {DATA_PATH} directory not found!")
        print("Please run data_collector_advanced.py first.")
        return
    
    actions = sorted([d for d in os.listdir(DATA_PATH) 
                     if os.path.isdir(os.path.join(DATA_PATH, d))])
    
    if not actions:
        print(f"\n⚠ ERROR: No sign folders found in {DATA_PATH}!")
        return
    
    print(f"\nFound {len(actions)} sign classes")
    print("\n" + "=" * 70)
    print("SAMPLE DISTRIBUTION")
    print("=" * 70)
    
    total_train = 0
    total_test = 0
    class_stats = []
    
    # Header
    print(f"{'Sign':<20} {'Train Samples':<15} {'Test Samples':<15} {'Total':<10}")
    print("-" * 70)
    
    for action in actions:
        action_path = os.path.join(DATA_PATH, action)
        
        # Count train samples
        train_path = os.path.join(action_path, 'train')
        train_count = 0
        if os.path.exists(train_path):
            train_count = len([f for f in os.listdir(train_path) if f.endswith('.npy')])
        
        # Count test samples
        test_path = os.path.join(action_path, 'test')
        test_count = 0
        if os.path.exists(test_path):
            test_count = len([f for f in os.listdir(test_path) if f.endswith('.npy')])
        
        total = train_count + test_count
        total_train += train_count
        total_test += test_count
        
        class_stats.append({
            'action': action,
            'train': train_count,
            'test': test_count,
            'total': total
        })
        
        # Status indicator
        status = "✓" if train_count >= 80 and test_count >= 30 else "⚠"
        
        print(f"{status} {action:<18} {train_count:<15} {test_count:<15} {total:<10}")
    
    print("-" * 70)
    print(f"{'TOTAL':<20} {total_train:<15} {total_test:<15} {total_train + total_test:<10}")
    
    # Quality Assessment
    print("\n" + "=" * 70)
    print("QUALITY ASSESSMENT")
    print("=" * 70)
    
    # Check balance
    train_counts = [s['train'] for s in class_stats if s['train'] > 0]
    if train_counts:
        max_train = max(train_counts)
        min_train = min(train_counts)
        imbalance_ratio = max_train / min_train if min_train > 0 else float('inf')
        
        print(f"\nClass Balance:")
        print(f"  Max samples: {max_train}")
        print(f"  Min samples: {min_train}")
        print(f"  Imbalance ratio: {imbalance_ratio:.2f}")
        
        if imbalance_ratio > 3:
            print("  ⚠ WARNING: Significant class imbalance!")
            print("  → Consider collecting more samples for underrepresented classes")
        else:
            print("  ✓ Dataset is reasonably balanced")
    
    # Recommendations
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    
    needs_more_train = [s for s in class_stats if s['train'] < 80]
    needs_more_test = [s for s in class_stats if s['test'] < 30]
    
    if needs_more_train:
        print("\n⚠ Signs needing more TRAIN data (< 80 samples):")
        for s in needs_more_train:
            print(f"  • {s['action']}: {s['train']} samples (need {80 - s['train']} more)")
    
    if needs_more_test:
        print("\n⚠ Signs needing more TEST data (< 30 samples):")
        for s in needs_more_test:
            print(f"  • {s['action']}: {s['test']} samples (need {30 - s['test']} more)")
    
    if not needs_more_train and not needs_more_test:
        print("\n✓ Dataset meets minimum requirements!")
        print("  All signs have sufficient data for training.")
    
    # Data quality check
    print("\n" + "=" * 70)
    print("DATA QUALITY CHECK")
    print("=" * 70)
    
    corrupted_files = []
    feature_dimensions = []
    
    for action in actions:
        for split in ['train', 'test']:
            split_path = os.path.join(DATA_PATH, action, split)
            if os.path.exists(split_path):
                files = [f for f in os.listdir(split_path) if f.endswith('.npy')]
                for file in files:
                    try:
                        data = np.load(os.path.join(split_path, file))
                        feature_dimensions.append(data.shape[0])
                        
                        # Check for NaN or Inf
                        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
                            corrupted_files.append(f"{action}/{split}/{file}")
                    except Exception as e:
                        corrupted_files.append(f"{action}/{split}/{file} (Error: {str(e)})")
    
    if corrupted_files:
        print(f"\n⚠ Found {len(corrupted_files)} corrupted files:")
        for f in corrupted_files[:10]:  # Show first 10
            print(f"  • {f}")
        if len(corrupted_files) > 10:
            print(f"  ... and {len(corrupted_files) - 10} more")
    else:
        print("\n✓ No corrupted files detected")
    
    # Check feature dimensions
    if feature_dimensions:
        dimension_counts = Counter(feature_dimensions)
        if len(dimension_counts) > 1:
            print(f"\n⚠ WARNING: Inconsistent feature dimensions found:")
            for dim, count in dimension_counts.items():
                print(f"  • {dim} features: {count} files")
        else:
            expected_dim = 63  # 21 landmarks × 3 coordinates
            actual_dim = list(dimension_counts.keys())[0]
            if actual_dim == expected_dim:
                print(f"\n✓ All files have correct dimension: {actual_dim}")
            else:
                print(f"\n⚠ WARNING: Expected {expected_dim} features, found {actual_dim}")
    
    # Training readiness
    print("\n" + "=" * 70)
    print("TRAINING READINESS")
    print("=" * 70)
    
    if total_train == 0:
        print("\n❌ NOT READY: No training data collected")
        print("  → Run data_collector_advanced.py to collect data")
    elif total_train < len(actions) * 50:
        print(f"\n⚠ PARTIALLY READY: Only {total_train} training samples")
        print(f"  → Recommended: {len(actions) * 100} samples ({len(actions)} signs × 100)")
    else:
        print(f"\n✓ READY FOR TRAINING: {total_train} training samples")
        if total_test > 0:
            print(f"  With {total_test} test samples for evaluation")
    
    # Final summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Total signs: {len(actions)}")
    print(f"  Total samples: {total_train + total_test}")
    print(f"  Train/Test split: {total_train}/{total_test}")
    
    if total_train > 0:
        avg_train = total_train / len(actions)
        print(f"  Average per sign: {avg_train:.1f} samples")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    analyze_dataset()
