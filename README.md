# CaliHousingML: Advanced Feature Analysis

## Overview
This project explores advanced feature selection and dimensionality reduction techniques on the California Housing Dataset, using a Linear Regression model. We aim to determine the impact of these techniques on the model's performance.

## Techniques and Results
We applied various feature selection methods and PCA for dimensionality reduction. The performance of each method is evaluated based on R2 Score and Mean Squared Error (MSE).

### Implemented Techniques
1. Original Dataset
2. SelectKBest with F-Regression
3. SelectKBest with Mutual Information Regression
4. Principal Component Analysis (PCA)

### Performance Metrics
Below are the R2 and MSE values for the Linear Regression model with each technique:

- **Original Dataset**:
  - R2 Score: 0.5758
  - MSE: 0.5559
- **SelectKBest with F-Regression**:
  - R2 Score: 0.5129
  - MSE: 0.6383
- **SelectKBest with Mutual Information Regression**:
  - R2 Score: 0.5733
  - MSE: 0.5592
- **PCA**:
  - R2 Score: 0.4329
  - MSE: 0.7431

## Installation
To run this project, clone the repository and install the required dependencies.
```bash
mkdir CaliHousingFeatSelectionDimReduction
git clone https://github.com/mayankbaluni/CaliHousingFeatSelectionDimReduction.git
cd CaliHousingFeatSelectionDimReduction
python CaliHousingFeatSelectionDimReduction.py

