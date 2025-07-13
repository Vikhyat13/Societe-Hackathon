
Societe-Hackathon
=================

A machine learning project for intrusion detection and risk analysis, featuring data preprocessing, exploratory data analysis, model development, risk scoring, root cause analysis, and a dashboard for visual insights.

Features
--------

- Data Preprocessing: Clean and prepare network intrusion data.
- Exploratory Data Analysis (EDA): Visualize and understand trends in the data.
- Model Development: Train and evaluate a Random Forest model for intrusion detection.
- Risk Scoring: Assign risk scores to network events.
- Root Cause Analysis: Identify causes behind detected intrusions.
- Dashboard: Visualize results and insights interactively.

Project Structure
-----------------

Societe-Hackathon/
├── data_preprocessing.py
├── eda_analysis.py
├── model_development.py
├── risk_scoring.py
├── root_cause_analysis.py
├── dashboard.py
├── main.py
├── intrusion_data.csv
├── best_model_randomforest.joblib
├── requirements.txt
├── logs/
├── reports/

Installation
------------

1. Clone the repository:
   git clone https://github.com/Vikhyat13/Societe-Hackathon.git
   cd Societe-Hackathon

2. Install dependencies:
   pip install -r requirements.txt

Usage
-----

Run each module as needed:

- Data Preprocessing:
  python data_preprocessing.py

- EDA:
  python eda_analysis.py

- Model Training:
  python model_development.py

- Risk Scoring:
  python risk_scoring.py

- Root Cause Analysis:
  python root_cause_analysis.py

- Dashboard:
  python dashboard.py

- Main Pipeline:
  python main.py

File Descriptions
-----------------

File/Folder                  | Description
----------------------------|--------------------------------------------------
data_preprocessing.py        | Data cleaning and preparation
eda_analysis.py              | Exploratory data analysis
model_development.py         | Model training and evaluation
risk_scoring.py              | Assigns risk scores to events
root_cause_analysis.py       | Analyzes causes of intrusions
dashboard.py                 | Dashboard for visualization
main.py                      | Main pipeline script
intrusion_data.csv           | Dataset for intrusion detection
best_model_randomforest.joblib | Saved Random Forest model
logs/                        | Logs generated during execution
reports/                     | Generated reports
requirements.txt             | Python dependencies

Contributing
------------

Pull requests and suggestions are welcome! Open an issue or submit a PR to help improve the project.

License
-------

This project is open source. Please add your preferred license here.

