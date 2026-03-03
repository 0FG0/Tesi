# Optimization of Theoretical Production Times and Inefficiency Analysis through Machine Learning

This project develops machine learning models to improve manufacturing processing time estimation. It analyzes production inefficiencies compared to company standards. Through regression and classification techniques, the system identifies operational anomalies and redefines theoretical times more consistently with actual times. The goal is to support industrial decision-making with quantitative and replicable predictive tools.

## 1. Features 📊

- Prediction of the **Inefficiency Index** using a regression model.
- Prediction of **Processing Time (HOURS)** with comparison to theoretical time (AS400).
- Standard classification into 3 classes:
  - **NORMAL**
  - **WARNING**
  - **ANOMALY**
- Anomaly-oriented classification optimized to detect anomalous cases.
- Dedicated feature engineering pipeline:
  - Temporal, Ratio, Lag and Rolling window features.
- Automatic saving of results to CSV (`predictions.csv` or custom path).
- Separate scripts for model training/tuning and a single script (`main.py`) for inference.

---

## 2. Installation 📦

### Prerequisites

- Python 3.10+
- Pip aggiornato
- Ambiente virtuale (venv) consigliato

### - Clone the repository

```bash
    git clone <URL_REPOSITORY>
    cd tesi_efficienza_macchine
```

#### - Create and activate the virtual environment

```bash
    python -m venv venv
```

#### Windows:

```bash
    .\venv\Scripts\Activate.ps1
```

#### Linux/macOS:

```bash
    source venv/bin/activate
```

#### - Install dependencies

```bash
    pip install --upgrade pip
    pip install -r requirements.txt
```

## 3. Usage 🚀

#### - Run predictions (inference)

- Use the default dataset

```bash
    python main.py
```

- Use a custom CSV

```bash
    python main.py --input data\processed\koepfer_160_2.csv
```

- CSV input + custom output path

```bash
    python main.py --input data\processed\koepfer_160_2.csv --output outputs\predizioni_custom.csv
```

#### - Model training (optional)

- python src\regression_inefficiency_models.py
- python src\regression_time_models.py
- python src\classification_models_standard.py
- python src\classification_models_anomaly_oriented.py
