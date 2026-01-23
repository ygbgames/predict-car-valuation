# 🚗 Used Car Valuation Prediction Report

This repository contains a comprehensive data science project focused on predicting the market value of used cars. By comparing seven different machine learning architectures, we identify the most robust model for automated vehicle appraisal.

**Repository Link:** [ygbgames/predict-car-valuation](https://github.com/ygbgames/predict-car-valuation)

---

## 1. Define the Problem Statement
The used car market suffers from pricing inconsistency due to the subjective nature of manual appraisals. A vehicle's value is influenced by a complex mix of age, mileage, and technical specifications.

**Objective:** To develop a high-performance regression model that accurately estimates car prices, providing a transparent and data-driven valuation tool for buyers, sellers, and dealerships.

---

## 2. Model Outcomes or Predictions
After testing various algorithms, the **Stacking Regressor** emerged as the superior model for this task.

* **Best Model:** Stacking Regressor (Combining Random Forest and XGBoost).
* **Accuracy:** Achieved an **R-squared ($R^2$) of 0.77**, meaning the model explains 77% of the variance in car pricing.
* **Inference:** Tree-based ensemble methods significantly outperformed Linear Regression and Neural Networks for this dataset.

---

## 3. Data Acquisition
The dataset provides a diverse set of car listings with features critical to market value:
* **Vehicle Identity:** Brand and Model.
* **Technical Specs:** Fuel Type, Transmission, and Engine details.
* **Usage History:** Kilometers driven and number of years since manufacture.
* **Target Variable:** Selling Price.

---

## 4. Data Preprocessing/Preparation
Before modeling, the raw data underwent a rigorous pipeline to ensure high-quality inputs:

1.  **Feature Engineering:** Transformed the "Year" attribute into a "Vehicle Age" feature to better capture depreciation.
2.  **Categorical Encoding:** Applied encoding to non-numeric features (e.g., Fuel Type, Transmission) for model compatibility.
3.  **Exploratory Data Analysis (EDA):** Visualized price distributions and feature correlations to understand market drivers.

> **Visual Insight:**
> ![Correlation Heatmap](https://github.com/ygbgames/predict-car-valuation/blob/main/Correlation.png)
> *Figure 1: Correlation analysis showing the relationship between car age, mileage, and price.*

---

## 5. Modeling
We implemented and compared seven distinct models to find the optimal solution:

* **Linear Regression:** Established as the baseline ($R^2$: 0.38).
* **K-Nearest Neighbors (KNN):** Simple instance-based learning ($R^2$: 0.52).
* **Decision Tree:** A basic tree-based approach ($R^2$: 0.63).
* **Neural Network (Keras):** Deep learning approach ($R^2$: 0.41).
* **Tuned XGBoost:** Gradient boosting ensemble ($R^2$: 0.69).
* **Tuned Random Forest:** Bagging ensemble ($R^2$: 0.76).
* **Stacking Regressor:** The final selected model, leveraging the strengths of both Random Forest and XGBoost.

---

## 6. Model Evaluation
The models were evaluated based on their performance on the test set. While the Neural Network and Linear models struggled, the **Stacking Regressor** provided the most reliable results.

### Performance Summary Table
| Model | R-squared ($R^2$) |
| :--- | :--- |
| **Stacking Regressor** | **0.77** |
| Random Forest (Tuned) | 0.76 |
| XGBoost (Tuned) | 0.69 |
| Decision Tree | 0.63 |
| K-Nearest Neighbors | 0.52 |
| Neural Network (Keras) | 0.41 |
| Linear Regression | 0.38 |

### Final Model Error Metrics
For the Stacking Regressor, the error metrics are as follows:
* **Mean Absolute Error (MAE):** 3751.67
* **Root Mean Squared Error (RMSE):** 7298.27

> **Actual vs. Predicted Visualization:**
> ![Model Performance Plot](https://via.placeholder.com/800x400.png?text=Placeholder:+Actual+vs+Predicted+Price+Scatter+Plot)
> *Figure 2: Comparing the model's predicted prices against actual market values.*

---

## 🚀 How to Access the Repository

### View Online
The entire analysis, including code and dynamic charts, can be viewed directly in the browser:
[car_valuation_final.ipynb](https://github.com/ygbgames/predict-car-valuation/blob/main/car_valuation_final.ipynb)

### Local Setup & Execution
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/ygbgames/predict-car-valuation.git](https://github.com/ygbgames/predict-car-valuation.git)
    cd predict-car-valuation
    ```
2.  **Install dependencies:**
    ```bash
    pip install pandas numpy scikit-learn xgboost tensorflow matplotlib seaborn
    ```
3.  **Run the Notebook:**
    Launch Jupyter and open `car_valuation_final.ipynb` to execute the full pipeline.

---
**Author:** [ygbgames](https://github.com/ygbgames)  
