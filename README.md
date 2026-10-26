# SpendWise — Student Expense Tracker

A Streamlit application for student expense tracking, budgeting, visual analysis, and an educational machine-learning comparison workflow.

## Run locally

1. Extract the ZIP file.
2. Open a terminal in the `SpendWise_Student_Expense_Tracker` folder.
3. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

4. Start the app:

   ```bash
   python -m streamlit run app.py
   ```

## Main application features

- Dashboard with spending KPIs, monthly trend, category breakdown, recent activity, and budget status
- Add, search, filter, export, and delete expenses
- Budget planning, personal trend-based next-month estimate, financial insights, and payment reminders
- SQLite storage in `data/student_expenses.db`
- Responsive, polished Streamlit UI with direct navigation and interactive Plotly charts

## EDA and machine learning

Open **Model Performance** from the sidebar or use the Dashboard shortcut. It includes:

- Dataset overview and original-column inspection
- Numeric distributions with box plots, categorical distributions, correlations, missing-value checks, and potential outlier review
- Four regression algorithms: Linear Regression, Random Forest Regression, K-Nearest Neighbors (KNN) Regression, and Support Vector Regression (SVR)
- A fixed 80/20 train/test split, numeric scaling and imputation inside the preprocessing pipeline, and one-hot encoding for categorical fields
- Model comparison using MAE, RMSE, and R², plus actual-versus-predicted visualisation, downloadable results, and permutation importance

### Dataset and limitations

The CSV at `reference_dataset/student_spending (1).csv` is read as-is; the application does not modify or save new columns into it. The temporary target is calculated in memory as the sum of existing expense-component columns. Those expense components are excluded from model inputs to prevent target leakage. The models estimate the combined total from the remaining demographic and financial-context fields.

The included dataset may yield low or negative R² values. The app reports the actual test results and explains this limitation; it does not claim that a model is accurate merely because it trained successfully. The separate personal next-month estimate uses a simple trend from the user's own logged monthly expenses and is not the same as the reference-dataset model comparison.

## Project structure

```text
SpendWise_Student_Expense_Tracker/
├── app.py
├── requirements.txt
├── README.md
├── reference_dataset/
│   └── student_spending (1).csv
└── data/
    └── student_expenses.db
```
