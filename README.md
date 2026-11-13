# Startup Success Prediction

A machine learning classification project that predicts whether a startup is likely to **succeed or fail** based on its funding, team, market, product traction, revenue, investor type, sector, and founder background.

The project includes exploratory data analysis (EDA), data preprocessing, model training and evaluation, hyperparameter tuning, and a Streamlit web application for interactive predictions.

## Project Overview

Startup success can depend on several factors, including funding, market conditions, team composition, product traction, and founder experience. This project uses historical startup-related features to build a classification model that predicts one of two outcomes:

- **Success:** The startup reaches an IPO or is acquired.
- **Failure:** The startup does not reach either of those outcomes.

The final tuned XGBoost model is used by the Streamlit application to generate a prediction and display the model's probability for the predicted class.

## Dataset

The dataset contains **100,000 rows and 11 columns** (10 input features and 1 target column).

| Feature | Description |
|---|---|
| `funding_rounds` | Number of funding rounds |
| `founder_experience_years` | Years of founder experience |
| `team_size` | Number of team members |
| `market_size_billion` | Market size in billions of dollars |
| `product_traction_users` | Number of product users |
| `burn_rate_million` | Startup burn rate in millions |
| `revenue_million` | Startup revenue |
| `investor_type` | Investor category |
| `sector` | Startup business sector |
| `founder_background` | Founder background category |
| `outcome` | Target variable: `success` or `failure` |

### Target distribution

| Outcome | Records |
|---|---:|
| Failure | 55,610 |
| Success | 44,390 |
| **Total** | **100,000** |

The raw dataset contains no missing values according to the preprocessing notebook's null-value check.

## Project Workflow

### 1. Exploratory Data Analysis (EDA)

The preprocessing notebook explores the dataset using:

- Data types and descriptive statistics
- Missing-value checks
- Target outcome distribution
- Box plots for numerical features

### 2. Data Preprocessing

The preprocessing workflow includes:

- Converting the original outcomes (`IPO` and `Acquisition`) into `success`; other outcomes are mapped to `failure`.
- Detecting and capping outliers using the Interquartile Range (IQR) method for:
  - `funding_rounds`
  - `market_size_billion`
  - `product_traction_users`
  - `burn_rate_million`
  - `revenue_million`
- Converting categorical columns to categorical data types.
- Exporting the processed dataset as a CSV file.

### 3. Model Preparation

The modeling notebook performs the following steps:

- Separates input features and the target variable.
- Encodes the target using `LabelEncoder`.
- Applies one-hot encoding to categorical input features using `pd.get_dummies()`.
- Splits the data into training and testing sets using an 80:20 split, `random_state=42`, and stratification.

Standard scaling and SMOTE are present in the notebook as commented-out steps; they are not applied in the executed workflow shown.

### 4. Model Training and Comparison

The following models are explored:

- Random Forest Classifier
- XGBoost Classifier
- Voting Classifier combining Logistic Regression, Random Forest, and XGBoost

### 5. Hyperparameter Tuning

`GridSearchCV` with 5-fold cross-validation is used to tune the XGBoost model.

The best parameters recorded in the notebook are:

| Hyperparameter | Value |
|---|---:|
| `n_estimators` | 100 |
| `max_depth` | 3 |
| `learning_rate` | 0.2 |
| `subsample` | 0.9 |

The tuned XGBoost model is then evaluated on the test set.

## Model Evaluation

The following test accuracies are recorded in the modeling notebook:

| Model | Accuracy |
|---|---:|
| Random Forest | 76.12% |
| XGBoost | 75.795% |
| Voting Classifier | 75.995% |
| **Tuned XGBoost** | **76.255%** |

The tuned XGBoost model achieved the highest recorded test accuracy among these models.

Random Forest also has a recorded mean 5-fold cross-validation accuracy of approximately **75.57%**.

The notebook additionally uses a confusion matrix and classification report to examine model performance through precision, recall, and F1-score.

## Streamlit Web Application

The Streamlit application allows users to enter startup details and receive a prediction.

### Input features

- Funding rounds
- Founder experience
- Team size
- Market size
- Product traction
- Burn rate
- Revenue
- Investor type
- Sector
- Founder background

### Application features

- Interactive sidebar inputs
- Startup success/failure prediction
- Prediction probability using `predict_proba()`
- Plotly gauge chart showing the predicted class probability
- Success or failure result message

The app loads the saved model, label encoder, and feature-column list from the `models/` directory. It applies one-hot encoding to the categorical inputs and aligns the resulting columns with the model's training features.

## Tech Stack

- **Python**
- **Pandas** – Data manipulation and preprocessing
- **NumPy** – Numerical operations
- **Scikit-learn** – Model training, evaluation, encoding, cross-validation, and hyperparameter tuning
- **XGBoost** – Gradient boosting classification
- **Imbalanced-learn** – SMOTE support (present but commented out in the modeling notebook)
- **Matplotlib** – Exploratory data visualization
- **Plotly** – Prediction probability gauge
- **Streamlit** – Interactive web application
- **Joblib** – Saving and loading model artifacts
- **Jupyter Notebook** – Development and experimentation
- **Git & GitHub** – Version control

## Project Structure

```text
startup-success-prediction/
│
├── app.py
├── README.md
├── requirements.txt
│
├── data/
│   ├── startup_success_raw.csv
│   └── startup_success_cleaned.csv
│
├── notebooks/
│   ├── Pre_Processing.ipynb
│   └── Modeling.ipynb
│
└── models/
    ├── model.pkl
    ├── encoder.pkl
    └── feature_columns.pkl
```

The model artifact files are generated by the final export cell in `Modeling.ipynb`. The directory and file names must match the paths used in `app.py`.

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/startup-success-prediction.git
cd startup-success-prediction
```

Replace the repository URL with your GitHub repository URL.

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

**Windows (PowerShell)**

```powershell
.\venv\Scripts\Activate.ps1
```

**Windows (Command Prompt)**

```cmd
venv\Scripts\activate
```

**macOS / Linux**

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Generate the model files

Run the preprocessing notebook if you want to reproduce the cleaned dataset. Then run `Modeling.ipynb` to train and export the model artifacts into the `models/` directory.

Ensure the notebook's input and output paths match your repository structure.

### 5. Run the Streamlit application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal to use the application.

## Making a Prediction

1. Launch the Streamlit application.
2. Enter the startup details in the sidebar.
3. Click **Predict Startup Success**.
4. View the predicted outcome and its probability.


## Disclaimer

This project is intended for educational and demonstration purposes. Its predictions are based on the features and patterns represented in the dataset. A prediction is not a guarantee of a startup's actual future performance.

---
