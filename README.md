# Optimization of Theoretical Production Times and Inefficiency Analysis through Machine Learning

### This project develops machine learning models to improve manufacturing processing time estimation. It analyzes production inefficiencies compared to company standards. Through regression and classification techniques, the system identifies operational anomalies and redefines theoretical times more consistently with actual times. The goal is to support industrial decision-making with quantitative and replicable predictive tools.

- ## Features 📊

- Prediction of the **Inefficiency Index** using a regression model.
- Prediction of **Processing Time (HOURS)** with comparison to theoretical time (AS400).
- Standard classification into 3 classes:
  - NORMAL
  - WARNING
  - ANOMALY
- Anomaly-oriented classification optimized to detect anomalous cases.
- Dedicated feature engineering pipeline:
  - Temporal features
  - Ratio features
  - Lag features
  - Rolling window features
- Automatic saving of results to CSV (`predictions.csv` or custom path).
- Separate scripts for model training/tuning and a single script (`main.py`) for inference.

- ## Installation 📦

  - Python 3.10+
  - Updated pip
  - Virtual environment (venv)

  1. #### Clone the repository
   
     ```bash
      git clone <URL_REPOSITORY>
      cd tesi_efficienza_macchine
     
  2. #### Create and activate the virtual environment
   
    'python -m venv venv'
    '.\venv\Scripts\Activate.ps1'
  
  3. #### Install dependencies
   
    'pip install --upgrade pip'
    'pip install -r requirements.txt'

- ## Usage 🚀
  
  1. #### Run predictions (inference)

     - Use the default dataset
       
       'python main.py'
       
     - Use a custom CSV
       
       'python main.py --input data\processed\koepfer_160_2.csv'
       
     - CSV input + custom output path
       
       'python main.py --input data\processed\koepfer_160_2.csv --output outputs\predizioni_custom.csv'
       
  2. #### Model training (optional)
     
     python src\regression_inefficiency_models.py 
     python src\regression_time_models.py 
     python src\classification_models_standard.py 
     python src\classification_models_anomaly_oriented.py
