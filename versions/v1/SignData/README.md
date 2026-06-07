# SignData Directory

Training data is **not included** in this repository (`.npy` files are gitignored due to size).

## Expected Structure

```
SignData/
├── hello/
│   ├── train/
│   │   ├── 0.npy
│   │   └── ...
│   └── test/
│       ├── 0.npy
│       └── ...
├── thanks/
│   ├── train/
│   └── test/
└── ...
```

Each `.npy` file contains 63 normalized float values (21 hand landmarks × 3 coordinates).

## How to Generate Data

```bash
cd versions/v1
pip install -r requirements.txt
python data_collector_advanced.py
```

Collect at least **100 train + 50 test** samples per sign for good accuracy.
