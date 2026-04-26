
# 🩺 Lifestyle Disease Risk Prediction & Health Monitoring System

## 🔴 Live Dashboard
👉 [Click here to view the live dashboard](https://health-monitoring-system-s6wr65wid.vercel.app)

---

## 📌 Project Overview
This project analyzes health-related data to predict the **risk of cardiovascular and lifestyle diseases** based on an individual's health parameters. Users input health information such as **age, BMI, blood pressure, smoking status, waist circumference, and other lifestyle indicators**, and the system uses a trained machine learning model to estimate disease risk.

The project also generates **visual insights through a dashboard** to help users better understand their health patterns and risk factors.

The goal of this project is to demonstrate how **data analysis and predictive modeling can assist in early detection of lifestyle diseases and promote preventive healthcare.**

---

## 🎯 Objectives
* Analyze healthcare datasets to identify patterns related to lifestyle diseases
* Build and compare multiple machine learning models for disease risk prediction
* Provide accurate risk assessments based on comprehensive health parameters
* Identify contributing risk factors for individual users
* Create an interactive web interface for risk assessment
* Help users understand how lifestyle factors affect their health

---

## 🧠 Key Features
✔ Health data preprocessing and cleaning from NHANES dataset
✔ Lifestyle disease risk prediction using machine learning (Gradient Boosting)
✔ Feature engineering (BMI categories, BP stages, waist-to-height ratio, etc.)
✔ User input system for personal health parameters
✔ Risk probability calculation and factor identification
✔ Interactive web interface with real-time predictions
✔ API endpoint for integration with other systems
✔ Model performance validation (99.91% accuracy)

---

## 🤖 ML Model Results
The backend trains and compares **6 machine learning models** on the NHANES dataset (5,735 participants):

| Model | Accuracy | F1 Score | AUC-ROC | CV Mean |
|---|---|---|---|---|
| Logistic Regression | 95.12% | 0.9712 | 0.9777 | 0.9571 |
| Decision Tree | 99.91% | 0.9995 | 0.9973 | 0.9993 |
| Random Forest | 99.74% | 0.9984 | 1.0000 | 0.9928 |
| **Gradient Boosting** ✅ | **99.91%** | **0.9995** | **1.0000** | **0.9993** |
| SVM | 97.04% | 0.9825 | 0.9943 | 0.9693 |
| KNN | 96.34% | 0.9783 | 0.9881 | 0.9629 |

**Best Model: Gradient Boosting (Perfect AUC-ROC = 1.0000)**

### Training Details
- **Dataset:** NHANES (National Health and Nutrition Examination Survey)
- **Samples:** 5,735 participants after preprocessing
- **Features:** 17 (including 5 engineered features)
- **Target Classes:** Low Risk (15.8%) vs High Risk (84.2%)
- **Train/Test Split:** 80% / 20% stratified
- **Cross-Validation:** 5-Fold Stratified KFold
- **Model Type:** Gradient Boosting Classifier

---

## 📊 Input Parameters
The system analyzes the following user inputs:

### Demographic & Clinical Factors
* **Age** (years, 18–120)
* **Gender** (Male/Female)
* **Smoking Status** (Non-smoker / Smoker)

### Physical Measurements
* **Body Mass Index (BMI)** (kg/m²)
* **Waist Circumference** (cm) — Abdominal obesity indicator
* **Arm Circumference** (cm)
* **Blood Pressure (Systolic & Diastolic)** (mmHg)

### Lifestyle & Socioeconomic
* **Alcohol Use** (Drinks / Doesn't drink)
* **Average Drinks per Session**
* **Income-to-Poverty Ratio**
* **Household Size**

Using these parameters, the system estimates the **risk probability of cardiovascular and lifestyle-related diseases.**

---

## 📈 Output
The system provides:
* **Risk Level**: Low Risk or High Risk classification
* **Risk Probability**: 0–100% likelihood of disease
* **Contributing Factors**: Identified risk factors for the individual
* **Visual Dashboard**: Health indicators and risk distribution

---

## 🛠 Project Structure

```
Health-Monitoring-System/
│
├── lifestyle_risk_model.py          # ML training pipeline
├── new_dataset_NHANES__1_.csv       # NHANES dataset (5,735 samples)
├── dataset_updated.csv              # Legacy dataset
│
├── lifestyle_risk_gb_model.pkl      # Trained Gradient Boosting model
├── lifestyle_risk_scaler.pkl        # Feature scaler
├── lifestyle_risk_config.json       # Model configuration & features
│
├── web_app/
│   ├── app.py                       # Flask backend
│   ├── templates/
│   │   └── index.html               # Web interface
│   ├── static/
│   │   └── style.css                # Styling
│   └── health_risk_rf_model.pkl     # Legacy model (deprecated)
│
├── outputs/                         # Visualization outputs
│   ├── fig1_eda_overview.png
│   ├── fig2_model_comparison.png
│   ├── fig3_best_model_analysis.png
│   └── fig4_tree_analysis.png
│
├── dashboard/                       # React frontend (optional)
├── AIML_PROJECT (1).ipynb          # Legacy Jupyter notebook
└── README.md                        # This file
```

---

## 🚀 How to Run Locally

### Prerequisites
```bash
pip install pandas numpy scikit-learn matplotlib seaborn scipy flask joblib
```

### 1. Train the Model
```bash
python lifestyle_risk_model.py
```
This will:
- Load and preprocess the NHANES dataset
- Engineer 5 new features
- Train 6 models and select the best one
- Save model, scaler, and configuration files
- Generate EDA visualizations

### 2. Run the Web App
```bash
cd web_app
python app.py
```
Then visit: **http://127.0.0.1:5000**

---

## 📐 Feature Engineering
The model includes these engineered features:

| Feature | Description | Example Values |
|---------|-------------|-----------------|
| **BMI_Category** | WHO BMI classifications | 0=Underweight, 1=Normal, 2=Overweight, 3=Obese1, 4=Obese2+ |
| **BP_Stage** | ACC/AHA hypertension stages | 0=Normal, 1=Elevated, 2=Stage1, 3=Stage2 |
| **Age_Group** | Age brackets for population analysis | 0=<30, 1=30–45, 2=45–60, 3=>60 |
| **WHtR** | Waist-to-Height Ratio (visceral obesity) | Continuous: >0.5 indicates high risk |
| **Pulse_Pressure** | Systolic − Diastolic pressure | Continuous: Cardiovascular health indicator |

---

## 🔍 Risk Factor Identification
The system automatically identifies contributing risk factors:

✗ **Hypertension** — BP Stage ≥ 1  
✗ **Obesity** — BMI ≥ 30  
✗ **Smoking** — Active smoker  
✗ **Age** — 45 years or older  
✗ **Abdominal Obesity** — WHtR > 0.5

---

## 📡 API Endpoints

### GET `/`
Renders the web interface.

### POST `/`
Form submission for web predictions.
```
Parameters: age, gender, bmi, systolic_bp, diastolic_bp, smoking_status, waist_circumference, arm_circumference, income_pir, alcohol_use, avg_drinks, household_size
```

### POST `/api/predict`
JSON API for programmatic predictions.
```json
{
  "age": 62,
  "gender": 1,
  "bmi": 33,
  "systolic_bp": 145,
  "diastolic_bp": 92,
  "smoking_status": 2,
  "waist_circumference": 110,
  "arm_circumference": 33,
  "income_pir": 2.5,
  "alcohol_use": 1,
  "avg_drinks": 2,
  "household_size": 3
}
```

**Response:**
```json
{
  "status": "success",
  "risk_level": "High Risk",
  "risk_probability": 1.0,
  "risk_percentage": 100.0,
  "contributing_factors": [
    "Hypertension (Stage 1+)",
    "Obesity (BMI ≥ 30)",
    "Smoker",
    "Age ≥ 45 years",
    "High Abdominal Obesity (WHtR > 0.5)"
  ],
  "model": "Gradient Boosting"
}
```

### GET `/health`
Health check endpoint.
```json
{
  "status": "ok",
  "model": "Gradient Boosting",
  "auc_roc": 1.0,
  "accuracy": 0.9991,
  "f1_score": 0.9995
}
```

---

## 🛠 Technologies Used

### Backend / ML
* **Python 3**
* **Pandas** – Data preprocessing and analysis
* **NumPy** – Numerical operations
* **Scikit-learn** – Machine learning models
* **Matplotlib / Seaborn** – Data visualization
* **SciPy** – Statistical analysis
* **Joblib** – Model serialization
* **Flask** – Web framework

### Frontend / Dashboard
* **HTML5** – Structure
* **CSS3** – Styling with gradients & responsive design
* **JavaScript** – Form handling
* **Jinja2** – Template rendering
* **React** (optional) – Advanced dashboard

### Data Source
* **NHANES** – National Health and Nutrition Examination Survey

---

## 📝 License
This project is open source and available for educational purposes.

---

## 👥 Contributing
Contributions are welcome! Feel free to fork, improve, and submit pull requests.

---

## 📧 Contact
For questions or issues, please open an issue on the repository.

---

**Last Updated:** April 25, 2026  
**Model Version:** Gradient Boosting v1.0  
**Dataset Version:** NHANES (5,735 samples)

---

## 🚀 Run the ML Backend Locally

### 1. Clone the repo
```bash
git clone https://github.com/sam-coolshrestha/Health-Monitoring-System.git
cd Health-Monitoring-System
```

### 2. Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

### 3. Run the pipeline
```bash
python lifestyle_risk_model.py
```

This will print all model results and generate 4 plots in an `outputs/` folder.

---

## 🖥 Run the Dashboard Locally

```bash
cd dashboard
npm install
npm run dev
```
Open `http://localhost:5173` in your browser.

---

## 📂 Project Workflow
1️⃣ Data Collection
2️⃣ Data Preprocessing
3️⃣ Exploratory Data Analysis (EDA)
4️⃣ Feature Engineering
5️⃣ Model Training & Evaluation
6️⃣ Dashboard Visualization
7️⃣ User Health Input System (Risk Calculator)

---

## 📊 Dataset Analysis
The NHANES dataset contains multiple health and lifestyle parameters analyzed to identify correlations between **lifestyle habits and disease risks.**

Key findings:
* **59.4%** of participants are smokers — largest single risk factor
* **71.2%** are overweight or obese (BMI ≥ 25)
* Hypertension rate jumps from **13%** (age 18–30) to **59%** (age 60+)
* BMI and waist circumference are nearly perfectly correlated **(r = 0.91)**

---

## 🔮 Future Scope

### 1️⃣ Computer Vision Based Rehabilitation Assistant
A future version aims to integrate **OpenCV-based rehabilitation exercise monitoring**, where:
* Patients perform rehabilitation exercises
* The system detects body posture using computer vision
* It provides feedback on whether exercises are performed correctly

This feature is **currently proposed as a future enhancement and has not yet been implemented.**

### 2️⃣ Machine Learning Model Improvements
* Train deep learning models (Neural Networks)
* Improve accuracy of disease risk prediction

### 3️⃣ Real-Time Health Monitoring
Integration with wearable devices for real-time health data.

### 4️⃣ Backend API
Build a REST API so the dashboard can call the ML model directly in real time.

---

## 📜 License
This project is open source and available under the **MIT License**.


