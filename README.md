
# 📡 Improved LUCID: DDoS Detection Model Enhancement Project

This project is part of a graduate research assignment,  
based on the paper [_"LUCID: A Practical, Lightweight Deep Learning Solution for DDoS Attack Detection"_](https://doi.org/10.1109/TNSM.2020.2971776),  
with the goal of improving the model's design, experimental process, and extending support for real-time inference.

The system detects DDoS attacks from network traffic using lightweight CNN models.  
It supports both batch inference on `.hdf5` datasets and real-time detection from live interfaces or `.pcap` files.

---

## 📁 Project Structure

```
lucid_project/
├── lucid_cnn.py               # Main script for model training and inference
├── lucid_dataset_parser.py    # Script for preprocessing .pcap files into .hdf5 datasets

├── model/                     # Model architecture and regularization settings
│   ├── builder.py             # Defines the CNN architecture
│   └── regularizer.py         # L1/L2 regularization configuration

├── utils/                     # Utility functions used across the project
│   ├── constants.py           # Global constants (e.g. SEED, PATIENCE)
│   ├── logger.py              # Logs prediction results and performance metrics
│   ├── minmax_utils.py        # Provides min/max values for normalization
│   ├── path_utils.py          # Path and folder management helpers
│   ├── preprocessing.py       # Padding and normalization functions
│   ├── seed_utils.py          # Random seed configuration
│   └── prediction_utils.py    # Model loading, metadata parsing, warm-up, etc.

├── data/                      # Dataset parsing, loading, and preparation
│   ├── args.py
│   ├── data_loader.py
│   ├── ddos_specs.py
│   ├── flow_utils.py
│   ├── parser.py
│   ├── process_pcap.py
│   ├── runner.py
│   └── split.py

├── core/                      # Core logic for training and prediction
│   ├── args.py
│   ├── helpers.py
│   ├── trainer.py
│   ├── predictor.py
│   ├── live_predictor.py
│   └── prediction_runner.py

└── requirements.txt           # Required Python packages
```

---

## 🧭 How to Use This Project

This project is fully modular and allows you to go from raw `.pcap` files to real-time inference in 4 steps:

---

### 1. Installation

You need:

- Python 3.9
- TensorFlow 2.13 (Tested with CUDA 11.8 and cuDNN 8.6 for GPU use)
- Required packages in `requirements.txt`
- `tshark` installed (required for `pyshark`)

Install dependencies using:

```bash
conda create -n lucid-env python=3.9
conda activate lucid-env
pip install -r requirements.txt
sudo apt install tshark
```

---

### 2. Data Preprocessing (pcap → hdf5)

Raw network captures (`.pcap`) must be transformed into structured feature sets for training.

Preprocessing happens in **three internal stages**, but you only need to run **two commands**:

#### Stage 1: Convert `.pcap` to `.data`  
- Parses packet-level features and groups them into bidirectional flows  
- Flows are fragmented using fixed time windows  
- Labels are assigned as `benign` or `ddos`

```bash
python3 lucid_dataset_parser.py \
  --dataset_type DOS2019 \
  --dataset_folder ./sample-dataset/ \
  --packets_per_flow 10 \
  --time_window 10
```

This creates a file like:  
`10t-10n-DOS2019-preprocess.data`

#### Stage 2: Convert `.data` to `.hdf5`  
- Loads `.data` files into memory  
- Applies normalization (min-max), padding, and splits into train / val / test sets  
- Saves final `.hdf5` datasets

```bash
python3 lucid_dataset_parser.py --preprocess_folder ./sample-dataset/
```

At the end, you’ll see a summary like:

> Total samples: 7486 (3743 benign, 3743 ddos)  
> Train / Val / Test sizes: (6060, 677, 749)

You will get:  
- `10t-10n-DOS2019-dataset-train.hdf5`  
- `10t-10n-DOS2019-dataset-val.hdf5`  
- `10t-10n-DOS2019-dataset-test.hdf5`

---

### 3. Model Training (Train CNN on Flow Data)

The script `lucid_cnn.py` uses a CNN-based model built with Keras.  
You can train models using one or multiple datasets.

```bash
python3 lucid_cnn.py --train ./sample-dataset/ --epochs 30
```

Under the hood, it:

- Automatically loads all `*-train.hdf5` and `*-val.hdf5` files
- Performs grid search with early stopping
- Saves the best model to `.h5` and logs performance in `.csv`

Model filename format:

```
10t-10n-DOS2019-LUCID.h5
10t-10n-DOS2019-LUCID.csv
```

Example log summary:

> Accuracy: 0.9261, F1 Score: 0.9256  
> Samples: 677 (validation set)

---

### 4. Inference (Batch or Live)

Once training is complete, you can evaluate the model on test sets or live traffic.

#### 🧪 Batch Prediction

```bash
python3 lucid_cnn.py \
  --predict ./sample-dataset/ \
  --model ./output/10t-10n-DOS2019-LUCID.h5
```

It will scan all test `.hdf5` files in the directory and output predictions with performance metrics such as:

- Accuracy
- F1 Score
- False Positive Rate
- Total packets and flows

#### 🌐 Real-Time Prediction (Live Interface or `.pcap`)

LUCID also supports live detection:

```bash
python3 lucid_cnn.py \
  --predict_live eth0 \
  --model ./output/10t-10n-DOS2019-LUCID.h5 \
  --dataset_type DOS2019
```

You can also use a pre-recorded `.pcap`:

```bash
python3 lucid_cnn.py \
  --predict_live ./sample-dataset/sample.pcap \
  --model ./output/10t-10n-DOS2019-LUCID.h5 \
  --dataset_type DOS2019
```

If your traffic does not match the default IP schema (e.g., from CIC-DDoS2019), you can manually define:

```bash
--attack_net 11.0.0.0/24 --victim_net 10.42.0.0/24
```

---

## ✅ Summary

- This project enhances the original LUCID DDoS detection model by improving modularity, usability, and real-time inference capabilities.
- The full pipeline—from raw `.pcap` files to real-time classification—is integrated and executable using a minimal set of commands.
- It supports both offline evaluation and live deployment environments with clear configuration and modular structure.

---

## 🙏 Acknowledgements

This project builds upon the original LUCID framework developed by  
Fondazione Bruno Kessler (FBK), Queen’s University Belfast, and the University of Catania,  
as introduced in the paper:

> R. Doriguzzi-Corin, S. Millar, S. Scott-Hayward, J. Martínez-del-Rincón, and D. Siracusa,  
> *"Lucid: A Practical, Lightweight Deep Learning Solution for DDoS Attack Detection"*,  
> IEEE Transactions on Network and Service Management, vol. 17, no. 2, pp. 876–889, June 2020.  
> [https://doi.org/10.1109/TNSM.2020.2971776](https://doi.org/10.1109/TNSM.2020.2971776)

We thank the original authors for making their work open-source and reproducible.  
This project aims to enhance the LUCID pipeline with clearer modularization, support for real-time inference, and improved experimental usability.

---

## 🛡 License

This project is licensed under the terms of the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).  
You are free to use, modify, and distribute the code for academic or commercial purposes,  
as long as you include proper attribution and retain the original license.

> © 2025 Youngin Shin, Yonsei University Graduate School of Engineering  
> Based on code and research originally published by the LUCID authors under the Apache License 2.0.
