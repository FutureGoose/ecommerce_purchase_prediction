# 📊 E-Commerce Purchase Prediction Analysis

## Overview 🌐
This project explores the challenge of predicting customer purchase decisions in an e-commerce context. Utilizing a dataset with a range of user activities, our analysis aims to identify key behaviors and factors leading to conversions.

## Data Cleaning 🔍
We began with thorough data cleaning, addressing missing values, outliers, and ensuring the integrity and quality of our dataset. This formed the basis for reliable model inputs and accurate insights.

## Feature Engineering 🔧
A range of feature transformations was applied to handle the diversity in data distribution, including log and square root transformations, to prepare the features for optimal modeling performance.

## Analytical Approach 📈
- **Exploratory Data Analysis (EDA):** A comprehensive EDA provided insights into feature distributions, correlations, and their relationships with our target variable—conversion.
- **Sampling and SMOTE:** To counter the dataset imbalance, techniques such as subsampling, oversampling, and SMOTE were evaluated for their effect on model accuracy.

## Model Development 🛠️
We experimented with various machine learning models, with a special focus on:
- **Explainable Boosting Machine (EBM):** Known for its interpretability and performance.
- **LightGBM:** Renowned for its efficiency and scalability.

## Pipeline Implementation 🔄
Implemented robust pre-processing and imputation pipelines, ensuring a systematic approach to handling and transforming the data inputs for our models.

## Interpret Dashboard 📊
Integrated the InterpretML dashboard providing a user-friendly interface to visualize and understand model behaviors and feature importances.

## Stakeholder Presentation 🎤
Created a narrative presentation summarizing the model's strengths, weaknesses, and business implications for stakeholders' strategic decision-making.

## Metrics & Goals 🎯
Carefully selected metrics guided our modeling focus:
- **Precision:** To ensure the accurate identification of true converters.
- **Recall:** To capture a comprehensive sample of actual converters.

## Project Structure 📁
```
ecommerce_purchase_prediction/
├── assets/                 # Project assets
├── .gitignore              # Specifies intentionally untracked files to ignore
├── LICENSE                 # Project license
├── README.md               # Project readme (you are here)
├── Uppdragsbeskrivning.pdf # Assignment description
├── ecommerce_purchase_prediction.ipynb # Main project notebook
├── goose_helpers.py        # Helper functions script
├── project_data.csv        # Dataset used for the project
└── requirements.txt        # Project dependencies
```

## Installation & Usage 🛠️
For those interested in running the analysis locally, please refer to `requirements.txt` for the necessary dependencies. The main analysis can be found in `ecommerce_purchase_prediction.ipynb`.

## License 📜
This project is licensed under the terms of the MIT license.

## Acknowledgements 🙏
Special thanks to the UC Irvine Machine Learning Repository for providing the initial dataset and to Ali Leylani, the course facilitator, for both presenting a formidable challenge by intentionally complicating the dataset and for formulating a comprehensive assignment that truly tested our skills.