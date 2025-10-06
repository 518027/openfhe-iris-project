# Fully Homomorphic Encryption (FHE) SVM on Iris Dataset — OpenFHE Implementation

This project demonstrates privacy-preserving machine learning using **Fully Homomorphic Encryption (FHE)** with the [OpenFHE](https://github.com/openfheorg/openfhe-development) library.  
It implements a **Support Vector Machine (SVM)** classifier on the **Iris dataset**, performing encrypted inference for both **binary** and **multiclass** classification.

---

## Project Structure
```bash
openfhe-iris-project/
│
├── src/
│   ├── data/
│   │   └── iris.csv                 # Preprocessed Iris dataset
│   │
│   ├── log/                         # Output log
│   │   ├── analyze_results.txt 
│   │   ├── fhe_svm_iris.txt  
│   │   ├── analyze_results.txt
│   │   └── fhe_svm_iris_multiclass.txt
│   │
│   ├── analyze_results.py           # Visualization & CSV analysis (matplotlib)
│   ├── fhe_svm_iris.cpp             # Binary-class SVM inference using FHE
│   ├── fhe_svm_iris_multiclass.cpp  # Multiclass (3 classes) SVM inference using FHE
│   └── iris_utils.h                 # Dataset loader + normalization utilities
│
├── CMakeLists.txt                   # Build configuration
├── requirements.txt                 # Python dependencies
└── README.md                        # Project description (this file)
```
---

## Requirements

### System
- Ubuntu 20.04 or later (WSL2 recommended)
- g++ ≥ 11  
- cmake ≥ 3.22  
- make

### OpenFHE
Make sure OpenFHE is built and installed:
```bash
cd ~/openfhe-development
mkdir build && cd build
cmake ..
make -j$(nproc)
sudo make install
```

### Python (for visualization)
```bash
sudo apt install python3-venv python3-pip
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Build & Run

### 1️. Build project
```bash
mkdir build && cd build
cmake ..
make
```
### 2️. Run the binary-class version
```bash
./fhe_svm_iris
```
Expected output:
```bash
BFV Inner Product Correct? True
CKKS Inner Product Correct? True
```
### 3️. Run the multiclass version
```bash
./fhe_svm_iris_multiclass
```

Expected output (example):
```bash
Multiclass encrypted inference accuracy on first 150 samples: 84.67%
```

### 4️. Generate results visualization
```bash
cd ../src
python3 analyze_results.py
```

This will save:
```bash
results_plot.png
```
and print summary tables in the terminal.

---

## Results

- Encrypted inference works for both BFV and CKKS schemes.  
- Multiclass accuracy: ~84–90% on the Iris dataset.  
- All computations performed over encrypted data.  
- Visualization generated via `analyze_results.py`:
  - Bar chart comparing true vs predicted labels.
  - Accuracy table and class distributions.

Example saved file:
```bash
src/results_plot.png
```

---

## Output Log Example
```bash
Sample 0 true=0 pred=0 scores=[0.981368, 0.119921, -0.10129]
Sample 1 true=0 pred=0 scores=[0.847003, 0.344583, -0.191586]
...
Multiclass encrypted inference accuracy on first 150 samples: 84.67%
```

---

## Author
Developed by Muhammad Farhan Hanim  
For the course *Perkembangan Terbaru Sistem Komputer dan Jaringan (2025)*  
Built using OpenFHE

---

## License
This project follows the BSD 2-Clause License - same as OpenFHE.
