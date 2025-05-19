# AIBID-SCSA Pipeline

## Overview
This repository implements the AIBID-SCSA framework described in the article. It includes:
- **Preprocessing**: Min-Max normalization (`preprocessing.py`)
- **Feature Selection**: Improved Sparrow Search Algorithm (`issa.py`)
- **Classification**: MIX_LSTM model with BiLSTM layers (`model.py`)
- **Hyperparameter Tuning**: Rabbit Optimization Algorithm (`roa.py`)
- **Orchestration**: Full pipeline in `train.py`

## Requirements
- Python 3.8+
- TensorFlow 2.x
- scikit-learn
- pandas
- numpy

## Installation
```bash
pip install tensorflow scikit-learn pandas numpy
```

## Usage
1. Configure `config.py` with your dataset path and hyperparameters.
2. Ensure the TON_IoT dataset CSV files are downloaded to `DATA_DIR`.
3. Run the pipeline:
```bash
python train.py
```

## File Structure
```
├── config.py
├── data_loader.py
├── preprocessing.py
├── issa.py
├── model.py
├── roa.py
├── utils.py
├── train.py
└── README.md
```

## Citation
Please cite the original article when using this code.