# 📘 Student Final Grade Prediction (STAT 385 Project)

This repository contains our STAT 385 final project completed at the University of Illinois Chicago.  
The goal of this project is to **predict final grades (G3)** for Math and Portuguese classes using statistical and machine learning methods.

We analyze academic, personal, and social factors to determine which variables are the strongest predictors of student performance.

---

## 🚀 Project Overview

- **Dataset Source:** UCI Machine Learning Repository (Cortez, 2008)
- **Students:** Portuguese secondary school students
- **Variables:** 33+ features (demographics, study habits, grades, lifestyle, etc.)
- **Targets:**  
  - `mathfinal`  
  - `porfinal`

---

## 📊 Methods Used

### 🔹 Exploratory Data Analysis (EDA)
- Histograms, boxplots, QQ plots  
- Correlation heatmaps  
- Summary statistics such as mean, SD, quartiles  
  (See pages 8–12 for Math and pages 18–25 for Portuguese in the presentation)  
  :contentReference[oaicite:3]{index=3}

### 🔹 Models Implemented
- **Multiple Linear Regression**
- **Stepwise AIC & BIC selection**
- **Lasso Regression (glmnet)**
- **Ridge (via glmnet)**
- **Principal Component Regression (PCR)**
- **Partial Least Squares (PLS)**
- **Random Forest**
- **Bagging**
- **Boosting (gbm)**
- **kNN**
- **SVM**

All models were evaluated using a consistent training/testing split with seed `38520251`.

---

## 📈 Model Performance Summary

Across both subjects:

- **Portuguese models outperform Math models**  
  due to fewer extreme outliers and more stable distributions.
- **Top-performing models (lowest RMSE):**  
  - Math → Stepwise AIC  
  - Portuguese → Stepwise AIC  
  (Detailed tables found on pages 17 and 26 of the presentation)  
  :contentReference[oaicite:4]{index=4}

---

## 🧠 Key Predictors Identified

### For Math:
- `mathp2` (strongest predictor)
- `mathp1`
- `famrel`
- alcohol use variables (`Walc`, `Dalc`)
- some demographic factors such as `age`

### For Portuguese:
- `porp2` (strongest)
- `porp1`
- `schoolMS`
- `internetyes`
- reasons for choosing school

Lasso coefficient tables appear in the report:  
:contentReference[oaicite:5]{index=5}

---

## 📁 Repository Structure

