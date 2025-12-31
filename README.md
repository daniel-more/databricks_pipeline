# Favorita Store Retail Forecasting

## Overview
This project focuses on **retail demand forecasting** for Favorita stores using historical sales data.  
The goal is to build a **production-style forecasting system** that supports experimentation, scenario analysis, and automated deployment.

The project combines **data engineering**, **machine learning**, and **MLOps** practices using Databricks, MLflow, and CI/CD pipelines.

> **Note on authorship**  
> This repository originated as a group project.  
> This fork represents continued, independent development after the final shared milestone, including pipeline improvements, forecasting experiments, and MLOps enhancements.

---

## Problem Statement
Accurate sales forecasting is critical for:
- inventory planning
- supply chain optimization
- reducing waste and stockouts

Retail demand is affected by seasonality, promotions, holidays, and external factors, making it a strong candidate for machine learning–based time series models.

---

## Architecture & Workflow

### 1. Data Pipeline (Databricks)
- Ingest raw sales, store, and calendar data
- Clean and transform data using Spark
- Store processed data in structured tables for modeling

### 2. Forecasting & Modeling
- Time series forecasting models trained on historical sales
- Experiment tracking with MLflow
- Model comparison and versioning

### 3. What-If Scenario Analysis
- Simulate changes in:
  - promotions
  - holidays
  - demand shifts
- Evaluate forecast sensitivity under different assumptions

### 4. MLOps & CI/CD
- Automated testing for data and code changes
- Model training and validation pipelines
- Reproducible deployments using CI/CD workflows

---

## Technologies Used

- **Databricks** – distributed data processing and pipeline orchestration  
- **Apache Spark** – large-scale data transformations  
- **MLflow** – experiment tracking, model registry, and reproducibility  
- **Python** – data processing and modeling  
- **CI/CD Pipelines** – automated testing and deployment  
- **Git/GitHub** – version control and collaboration  

---

## Key Features

- End-to-end data pipeline from raw data to forecasts  
- Experiment tracking and model versioning with MLflow  
- What-if forecasting scenarios for business analysis  
- Production-oriented project structure  
- CI/CD pipelines to support reliable iteration  

---

## Project Structure (High Level)

.
├── databricks.yml
├── notebooks
│   ├── 01-download-dataset-kaggle.ipynb
│   ├── 02-bronze-tier.ipynb
│   ├── 03-Oil_data_Streaming_Silver.ipynb
│   ├── 03-silver-tier.ipynb
│   ├── 04-gold-tier-data-insights.ipynb
│   ├── 04-mlflow-select-best-model-predict.ipynb
│   ├── 04-mlflow-training-tunning.ipynb
│   ├── 04-mlflow-training.ipynb
│   ├── 05-platinum-tier.ipynb
│   └── 06-permissions.ipynb
├── README.md
├── requirements.txt
├── resources
│   ├── clusters.yml
│   ├── jobs.yml
└── src
    └── main.py


---

## Results & Learnings
- Built scalable data pipelines suitable for real retail workloads  
- Improved forecasting reliability through experiment tracking  
- Gained hands-on experience with MLOps practices in a cloud environment  
- Learned how scenario analysis supports business decision-making  

---

## Future Improvements
- Add additional forecasting models (e.g., hierarchical or probabilistic models)
- Incorporate external data sources (weather, macro trends)
- Improve monitoring and alerting for model performance
- Deploy forecasts as an API or dashboard

---

## Acknowledgments
This project was initially developed as a collaborative effort.  
Thanks to the original team members for their contributions to the shared foundation of this work.

Adding CI/CD
