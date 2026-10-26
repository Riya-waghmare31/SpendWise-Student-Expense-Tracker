
import os
import sqlite3
from datetime import date, datetime

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

st.set_page_config(page_title="SpendWise", page_icon="💳", layout="wide")

DB_PATH = os.path.join("data", "student_expenses.db")
REFERENCE_CSV = os.path.join("reference_dataset", "student_spending (1).csv")

EXPENSE_CATEGORIES = [
    "Food", "Rent / Hostel", "Electricity", "Transport",
    "Stationery / Study", "Shopping", "Entertainment",
    "Healthcare / Medicine", "Mobile / Internet", "Subscriptions",
    "Other", "Unnecessary / Impulse"
]
PAYMENT_METHODS = ["UPI", "Cash", "Card", "Bank Transfer", "Other"]


def inject_css():
    st.markdown("""
    <style>
    .stApp { background: #f6f8fc; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f3f6ff 100%);
        border-right: 1px solid #e5eaf5;
    }
    [data-testid="stSidebar"] > div:first-child { padding-top: 1.1rem; }
    [data-testid="stSidebar"] [data-testid="stRadio"] > label { display: none; }
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] { gap: 5px; }
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label {
        border: 1px solid transparent; border-radius: 11px; padding: 9px 12px;
        transition: background .18s ease, transform .18s ease, border-color .18s ease, box-shadow .18s ease;
        cursor: pointer; margin: 1px 0;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:hover {
        background: #eaf0ff; border-color: #d7e2ff; transform: translateX(3px);
        box-shadow: 0 4px 12px rgba(57, 86, 160, .08);
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) {
        background: linear-gradient(100deg, #e7edff, #f3f6ff); border-color: #cbd7ff;
        box-shadow: inset 3px 0 0 #536dfe;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label p { font-weight: 550; }
    .block-container { padding-top: 1.5rem; }
    h1, h2, h3 { letter-spacing: -0.025em; }
    div[data-testid="stPlotlyChart"], div[data-testid="stDataFrame"] {
        border-radius: 14px; overflow: hidden;
    }
    div[data-testid="stButton"] button { transition: all .18s ease; border-radius: 10px; }
    div[data-testid="stButton"] button:hover { transform: translateY(-1px); box-shadow: 0 5px 14px rgba(55, 75, 130, .15); }
    .card {
        background: white; padding: 18px; border-radius: 16px;
        border: 1px solid #e8eaf2; box-shadow: 0 4px 18px rgba(50,60,90,.06);
        margin-bottom: 12px;
    }
    .card:hover { box-shadow: 0 8px 24px rgba(50,60,90,.11); transform: translateY(-1px); }
    .small { color: #697386; font-size: 0.9rem; }
    .metric { font-size: 1.55rem; font-weight: 700; color: #27324a; }
    .pill { display:inline-block; padding:6px 10px; border-radius:20px;
            background:#eef1ff; color:#4956a5; font-size:.82rem; }
    </style>
    """, unsafe_allow_html=True)


def init_db():
    os.makedirs("data", exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS expenses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                category TEXT NOT NULL,
                amount REAL NOT NULL,
                payment_method TEXT NOT NULL,
                note TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS reminders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                payment_name TEXT NOT NULL,
                amount REAL NOT NULL,
                due_date TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'Upcoming'
            )
        """)


def add_expense(d, category, amount, method, note):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO expenses(date, category, amount, payment_method, note) VALUES (?, ?, ?, ?, ?)",
            (str(d), category, float(amount), method, note)
        )


def get_expenses():
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(
            "SELECT id, date, category, amount, payment_method, note "
            "FROM expenses ORDER BY date DESC, id DESC", conn
        )


def delete_expense(expense_id):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM expenses WHERE id = ?", (int(expense_id),))


def get_reminders():
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(
            "SELECT id, payment_name, amount, due_date, status "
            "FROM reminders ORDER BY due_date", conn
        )


def add_reminder(name, amount, due, status):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO reminders(payment_name, amount, due_date, status) VALUES (?, ?, ?, ?)",
            (name, float(amount), str(due), status)
        )


def complete_reminder(rid):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("UPDATE reminders SET status='Completed' WHERE id=?", (int(rid),))


def read_reference_dataset():
    if not os.path.exists(REFERENCE_CSV):
        return pd.DataFrame(), []
    df = pd.read_csv(REFERENCE_CSV)
    return df, list(df.columns)


def find_expense_columns(df):
    """Select only columns that actually exist and are numeric expense-like columns."""
    keywords = [
        "expense", "spend", "tuition", "housing", "food", "transport",
        "book", "supply", "entertainment", "personal", "technology",
        "health", "wellness", "miscellaneous", "rent", "electricity",
        "shopping", "subscription"
    ]
    found = []
    for col in df.columns:
        name = str(col).lower()
        if any(k in name for k in keywords):
            numeric = pd.to_numeric(df[col], errors="coerce")
            if numeric.notna().sum() > 0:
                found.append(col)
    return found


@st.cache_data
def train_reference_model():
    df, columns = read_reference_dataset()
    if df.empty:
        return None

    # Temporary working copy only. Original CSV is never changed.
    work = df.copy()
    expense_cols = find_expense_columns(work)

    # Prefer an existing total-expense column if one is explicitly present.
    total_candidates = [
        c for c in work.columns
        if "total" in str(c).lower() and
        ("expense" in str(c).lower() or "spend" in str(c).lower())
    ]

    if total_candidates:
        target = pd.to_numeric(work[total_candidates[0]], errors="coerce")
        target_name = total_candidates[0]
        target_source = "existing total-expense column"
    elif expense_cols:
        numeric_expenses = work[expense_cols].apply(pd.to_numeric, errors="coerce")
        target = numeric_expenses.sum(axis=1, min_count=1)
        target_name = "temporary_total_monthly_expense"
        target_source = "calculated from actual expense columns"
    else:
        return {
            "error": "No usable expense/total-expense column was found.",
            "columns": columns,
            "expense_cols": []
        }

    # Remove target source from features to avoid leakage.
    exclude = set(total_candidates)
    if target_name in work.columns:
        exclude.add(target_name)

    feature_df = work.drop(columns=list(exclude), errors="ignore").copy()
    feature_df = feature_df.loc[target.notna()].copy()
    y = target.loc[feature_df.index]

    # Remove empty columns and obvious row-index columns.
    drop_cols = []
    for col in feature_df.columns:
        if feature_df[col].isna().all():
            drop_cols.append(col)
        if str(col).lower().startswith("unnamed:"):
            drop_cols.append(col)
    feature_df = feature_df.drop(columns=list(set(drop_cols)), errors="ignore")

    if feature_df.shape[1] == 0 or len(feature_df) < 5:
        return {
            "error": "Not enough usable feature data for training.",
            "columns": columns,
            "expense_cols": expense_cols
        }

    numeric_cols = feature_df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = [c for c in feature_df.columns if c not in numeric_cols]

    transformers = []
    if numeric_cols:
        transformers.append(("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median"))
        ]), numeric_cols))
    if categorical_cols:
        transformers.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical_cols))

    if not transformers:
        return {"error": "No usable features after preprocessing.", "columns": columns}

    preprocessor = ColumnTransformer(transformers=transformers)
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=200, random_state=42
        ))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        feature_df, y, test_size=0.2, random_state=42
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)

    return {
        "model": model,
        "mae": mae,
        "r2": r2,
        "columns": columns,
        "expense_cols": expense_cols,
        "target_name": target_name,
        "target_source": target_source,
        "feature_cols": list(feature_df.columns),
        "training_rows": len(feature_df)
    }


def monthly_data(df):
    if df.empty:
        return pd.DataFrame(columns=["month", "amount"])
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d["amount"] = pd.to_numeric(d["amount"], errors="coerce").fillna(0)
    d = d.dropna(subset=["date"])
    out = d.groupby(d["date"].dt.to_period("M"))["amount"].sum().reset_index()
    out["month"] = out["date"].astype(str)
    return out[["month", "amount"]]


def next_month_prediction(df):
    m = monthly_data(df)
    if len(m) == 0:
        return 0.0, 0.0, 0.0, "Add expenses to create a personal prediction."
    values = m["amount"].astype(float).tolist()
    avg = float(np.mean(values))
    last = float(values[-1])

    if len(values) == 1:
        estimate = last
    elif len(values) == 2:
        estimate = 0.4 * values[-2] + 0.6 * values[-1]
    else:
        x = np.arange(1, len(values) + 1).reshape(-1, 1)
        y = np.array(values)
        lr = LinearRegression().fit(x, y)
        trend = float(lr.predict([[len(values) + 1]])[0])
        recent = float(np.mean(values[-3:]))
        estimate = 0.5 * trend + 0.5 * recent

    return max(0.0, estimate), avg, last, (
        "The estimate is based on your previous monthly spending and recent trend."
    )


def budget_suggestion(df, income, savings):
    available = max(0.0, income - savings)
    if df.empty:
        defaults = {
            "Food": .25, "Rent / Hostel": .25, "Transport": .10,
            "Stationery / Study": .06, "Shopping": .07,
            "Entertainment": .06, "Healthcare / Medicine": .05,
            "Subscriptions": .04, "Other": .12
        }
        return {k: available * v for k, v in defaults.items()}, available

    d = df.copy()
    d["amount"] = pd.to_numeric(d["amount"], errors="coerce").fillna(0)
    cat = d.groupby("category")["amount"].sum()
    total = cat.sum()
    if total <= 0:
        return {}, available
    return {k: available * float(v / total) for k, v in cat.items()}, available


def current_month_total(df):
    if df.empty:
        return 0.0
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d["amount"] = pd.to_numeric(d["amount"], errors="coerce").fillna(0)
    now = pd.Timestamp.today()
    return float(d.loc[
        (d["date"].dt.year == now.year) &
        (d["date"].dt.month == now.month), "amount"
    ].sum())


def previous_month_total(df):
    if df.empty:
        return 0.0
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d["amount"] = pd.to_numeric(d["amount"], errors="coerce").fillna(0)
    now = pd.Timestamp.today()
    prev = now - pd.DateOffset(months=1)
    return float(d.loc[
        (d["date"].dt.year == prev.year) &
        (d["date"].dt.month == prev.month), "amount"
    ].sum())


def card(title, value, subtitle=""):
    st.markdown(
        f'<div class="card"><div class="small">{title}</div>'
        f'<div class="metric">{value}</div><div class="small">{subtitle}</div></div>',
        unsafe_allow_html=True
    )


def dashboard(df, prediction, budget_limit=0):
    st.markdown("# Good to see you!")
    st.write("Track your spending, stay within budget and understand your financial habits.")

    total = float(df["amount"].sum()) if not df.empty else 0.0
    this_month = current_month_total(df)
    m = monthly_data(df)
    avg_month = float(m["amount"].mean()) if not m.empty else 0.0
    remaining = max(0.0, budget_limit - this_month) if budget_limit else 0.0

    cols = st.columns(5)
    vals = [
        ("Total Spent", f"₹{total:,.0f}", "All saved expenses"),
        ("This Month", f"₹{this_month:,.0f}", "Current month"),
        ("Average Monthly", f"₹{avg_month:,.0f}", "Personal history"),
        ("Remaining Budget", f"₹{remaining:,.0f}", "Based on budget"),
        ("Next Month", f"₹{prediction:,.0f}", "Personal estimate")
    ]
    for c, (t, v, s) in zip(cols, vals):
        with c:
            card(t, v, s)

    left, right = st.columns(2)
    with left:
        st.markdown("### Monthly Spending Trend")
        if not m.empty:
            fig = px.line(m, x="month", y="amount", markers=True)
            fig.update_layout(height=320, margin=dict(l=10,r=10,t=20,b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Add expenses to see your monthly trend.")

    with right:
        st.markdown("### Category Spending Breakdown")
        if not df.empty:
            cat = df.groupby("category", as_index=False)["amount"].sum().sort_values("amount", ascending=False)
            fig = px.bar(cat, x="category", y="amount")
            fig.update_layout(height=320, margin=dict(l=10,r=10,t=20,b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Add expenses to see category spending.")

    st.markdown("### Recent Expenses")
    st.dataframe(df.head(8), use_container_width=True, hide_index=True)

    a, b = st.columns(2)
    with a:
        st.markdown("### Budget Suggestion")
        st.write("Use the Budget Maker to set income and savings goals.")
    with b:
        st.markdown("### Notifications")
        if budget_limit and this_month >= 0.8 * budget_limit:
            st.warning(f"Your spending has reached {this_month / budget_limit:.0%} of your budget.")
        else:
            st.success("No budget alert currently.")

    st.markdown("### How It Works")
    st.write("Add daily expenses → SQLite saves them → monthly totals update → charts, budget suggestions, predictions and insights update.")

    st.markdown("### Key Features")
    st.write("Daily tracking • Personal history • Smart budgeting • Next-month estimate • Visual analytics • Neutral spending insights")


def add_expense_page():
    st.title("Add Expense")
    with st.form("expense_form"):
        d = st.date_input("Date", value=date.today(), help="Select the date when you spent the money.")
        category = st.selectbox("Category", EXPENSE_CATEGORIES, help="Choose the category that best matches your expense.")
        amount = st.number_input("Amount (₹)", min_value=0.0, step=10.0, help="Enter the amount you actually spent.")
        method = st.selectbox("Payment method", PAYMENT_METHODS)
        note = st.text_input("Note", placeholder="Example: College canteen")
        submitted = st.form_submit_button("Save Expense")

    if submitted:
        if amount <= 0:
            st.error("Amount must be greater than ₹0.")
        else:
            add_expense(d, category, amount, method, note)
            st.success("Expense saved successfully.")
            st.cache_data.clear()


def history_page(df):
    st.title("Expense History")
    if df.empty:
        st.info("No expenses saved yet.")
        return

    search = st.text_input("Search")
    cats = ["All"] + sorted(df["category"].dropna().unique().tolist())
    cat = st.selectbox("Category filter", cats)
    min_date = pd.to_datetime(df["date"]).min().date()
    max_date = pd.to_datetime(df["date"]).max().date()
    dates = st.date_input("Date filter", value=(min_date, max_date))

    view = df.copy()
    if search:
        text_cols = view[["category", "payment_method", "note"]].fillna("").astype(str).agg(" ".join, axis=1)
        view = view[text_cols.str.contains(search, case=False, na=False)]
    if cat != "All":
        view = view[view["category"] == cat]
    if isinstance(dates, tuple) and len(dates) == 2:
        view = view[(pd.to_datetime(view["date"]).dt.date >= dates[0]) &
                    (pd.to_datetime(view["date"]).dt.date <= dates[1])]

    st.dataframe(view, use_container_width=True, hide_index=True)

    csv = view.to_csv(index=False).encode("utf-8")
    st.download_button("Export CSV", csv, "spendwise_expenses.csv", "text/csv")

    st.markdown("### Delete Expense")
    rid = st.number_input("Expense ID", min_value=1, step=1)
    if st.button("Delete"):
        delete_expense(rid)
        st.success("Expense deleted.")
        st.rerun()


def visualize_page(df):
    st.title("Visualize Spending")
    if df.empty:
        st.info("Add expenses to view charts.")
        return

    m = monthly_data(df)
    cat = df.groupby("category", as_index=False)["amount"].sum()

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(px.line(m, x="month", y="amount", markers=True, title="Monthly Spending Trend"), use_container_width=True)
    with c2:
        st.plotly_chart(px.pie(cat, names="category", values="amount", hole=.45, title="Spending Distribution"), use_container_width=True)

    st.plotly_chart(px.bar(cat.sort_values("amount", ascending=False), x="category", y="amount", title="Category Spending"), use_container_width=True)

    current = current_month_total(df)
    previous = previous_month_total(df)
    compare = pd.DataFrame({"Month": ["Previous", "Current"], "Amount": [previous, current]})
    st.plotly_chart(px.bar(compare, x="Month", y="Amount", title="Current Month vs Previous Month"), use_container_width=True)


def budget_page(df):
    st.title("Smart Budget Maker")
    income = st.number_input("Monthly income / allowance (₹)", min_value=0.0, step=500.0)
    savings = st.number_input("Savings goal (₹)", min_value=0.0, step=500.0)

    if st.button("Create Budget"):
        suggestions, available = budget_suggestion(df, income, savings)
        spent = current_month_total(df)
        st.metric("Available after savings", f"₹{available:,.0f}")
        st.metric("Suggested spending limit", f"₹{available:,.0f}")
        st.metric("Remaining money", f"₹{max(0, available - spent):,.0f}")

        if income > 0:
            st.write(f"Spending percentage: {spent / income:.1%}")
            st.write(f"Savings percentage: {savings / income:.1%}")

        if spent > available:
            st.error("Over Budget")
        elif available > 0 and spent >= 0.8 * available:
            st.warning("Watch Spending")
        else:
            st.success("Within Budget")

        if suggestions:
            st.markdown("### Category-wise Suggested Limits")
            bdf = pd.DataFrame(list(suggestions.items()), columns=["Category", "Suggested Limit"])
            bdf["Suggested Limit"] = bdf["Suggested Limit"].round(2)
            st.dataframe(bdf, use_container_width=True, hide_index=True)


def prediction_page(df):
    st.title("Next Month Prediction")
    estimate, avg, last, explanation = next_month_prediction(df)
    c1, c2, c3 = st.columns(3)
    c1.metric("Estimated next month spending", f"₹{estimate:,.0f}")
    c2.metric("Historical average", f"₹{avg:,.0f}")
    c3.metric("Last month", f"₹{last:,.0f}")
    st.info(explanation)

    m = monthly_data(df)
    if len(m) >= 3:
        st.caption("With 3 or more personal months, the app uses Linear Regression with recent spending trend.")


def insights_page(df):
    st.title("Financial Insights")
    if df.empty:
        st.info("Add expenses to generate insights.")
        return

    d = df.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d["amount"] = pd.to_numeric(d["amount"], errors="coerce").fillna(0)
    cat = d.groupby("category")["amount"].sum()
    if not cat.empty:
        st.write(f"**{cat.idxmax()}** is currently your highest spending category.")

    current = current_month_total(d)
    previous = previous_month_total(d)
    if previous > 0:
        if current > previous:
            st.write("Your total spending increased compared with the previous month.")
        elif current < previous:
            st.write("Your total spending decreased compared with the previous month.")
        else:
            st.write("Your current and previous month spending are the same.")

    impulse = float(d.loc[
        (d["date"].dt.year == pd.Timestamp.today().year) &
        (d["date"].dt.month == pd.Timestamp.today().month) &
        (d["category"] == "Unnecessary / Impulse"), "amount"
    ].sum())
    st.metric("Impulse spending this month", f"₹{impulse:,.0f}")
    st.write(f"Impulse spending percentage: {(impulse / current * 100) if current else 0:.1f}%")
    if impulse > 0:
        st.info("You may want to review repeated impulse purchases.")


def reminders_page():
    st.title("Payment Reminders")
    with st.form("reminder_form"):
        name = st.text_input("Payment name")
        amount = st.number_input("Amount (₹)", min_value=0.0, step=100.0)
        due = st.date_input("Due date", value=date.today())
        status = st.selectbox("Status", ["Upcoming", "Due Soon", "Completed"])
        ok = st.form_submit_button("Add Reminder")
    if ok and name and amount >= 0:
        add_reminder(name, amount, due, status)
        st.success("Reminder added.")
        st.rerun()

    r = get_reminders()
    if not r.empty:
        st.dataframe(r, use_container_width=True, hide_index=True)
        rid = st.number_input("Reminder ID to mark completed", min_value=1, step=1)
        if st.button("Mark Completed"):
            complete_reminder(rid)
            st.rerun()


def import_page():
    st.title("Import Expenses")
    st.write("Expected columns: date, category, amount, payment_method, note")
    file = st.file_uploader("Choose a CSV", type=["csv"])
    if file:
        imported = pd.read_csv(file)
        st.markdown("### Preview")
        st.dataframe(imported.head(10), use_container_width=True, hide_index=True)

        required = {"date", "category", "amount", "payment_method", "note"}
        if not required.issubset(set(imported.columns)):
            st.error("CSV must contain: date, category, amount, payment_method, note")
        elif st.button("Import to Personal History"):
            count = 0
            for _, row in imported.iterrows():
                try:
                    add_expense(
                        pd.to_datetime(row["date"]).date(),
                        str(row["category"]),
                        float(row["amount"]),
                        str(row["payment_method"]),
                        str(row["note"])
                    )
                    count += 1
                except Exception:
                    pass
            st.success(f"Imported {count} expense(s) into SQLite.")
            st.rerun()



def eda_ml_lab_page():
    """Exploratory analysis and multiple ML models using the unchanged reference CSV."""
    st.title("EDA & Machine Learning Lab")
    raw, actual_columns = read_reference_dataset()
    if raw.empty:
        st.warning("Reference dataset was not found.")
        return

    st.markdown("Explore the original dataset and compare four core regression algorithms using clear visualizations.")
    st.markdown("### 1. Dataset overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Records", f"{len(raw):,}")
    c2.metric("Original columns", len(actual_columns))
    c3.metric("Missing cells", f"{int(raw.isna().sum().sum()):,}")
    with st.expander("Show exact original dataset columns"):
        st.code("\\n".join(actual_columns))
    st.dataframe(raw.head(10), use_container_width=True, hide_index=True)

    st.markdown("### 2. Exploratory Data Analysis (EDA)")
    numeric = raw.select_dtypes(include=np.number).drop(
        columns=[c for c in raw.columns if str(c).lower().startswith("unnamed:")],
        errors="ignore"
    )
    categorical = [c for c in raw.columns if c not in numeric.columns]
    e1, e2 = st.columns(2)
    with e1:
        if not numeric.empty:
            selected_num = st.selectbox("Numeric column distribution", numeric.columns.tolist(), key="eda_num")
            fig = px.histogram(raw, x=selected_num, nbins=30, title=f"Distribution of {selected_num}",
                               color_discrete_sequence=["#4969d8"])
            fig.update_layout(height=340)
            st.plotly_chart(fig, use_container_width=True)
    with e2:
        if categorical:
            selected_cat = st.selectbox("Categorical column counts", categorical, key="eda_cat")
            counts = raw[selected_cat].fillna("Missing").astype(str).value_counts().head(15).rename_axis(selected_cat).reset_index(name="Count")
            fig = px.bar(counts, x=selected_cat, y="Count", title=f"Counts by {selected_cat}",
                         color="Count", color_continuous_scale="Blues")
            fig.update_layout(height=340)
            st.plotly_chart(fig, use_container_width=True)
    if len(numeric.columns) >= 2:
        st.plotly_chart(px.imshow(numeric.corr(numeric_only=True), text_auto=".2f",
                                  title="Numeric Feature Correlation", color_continuous_scale="RdBu_r",
                                  aspect="auto"), use_container_width=True)

    st.markdown("### 3. Regression model comparison")
    expense_cols = find_expense_columns(raw)
    # For this dataset, total spending is calculated in memory from the existing expense columns.
    if len(expense_cols) < 1:
        st.warning("No numeric expense columns were detected. Regression models cannot be trained.")
    else:
        expense_values = raw[expense_cols].apply(pd.to_numeric, errors="coerce")
        target = expense_values.sum(axis=1, min_count=1)
        # Exclude component expenses from features to avoid target leakage.
        excluded = set(expense_cols)
        excluded.update(c for c in raw.columns if str(c).lower().startswith("unnamed:"))
        features = raw.drop(columns=list(excluded), errors="ignore").loc[target.notna()].copy()
        target = target.loc[features.index]
        num_cols = features.select_dtypes(include=np.number).columns.tolist()
        cat_cols = [c for c in features.columns if c not in num_cols]
        transformers = []
        if num_cols:
            transformers.append(("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median"))
            ]), num_cols))
        if cat_cols:
            transformers.append(("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]), cat_cols))
        if not transformers or len(features) < 10:
            st.warning("Not enough usable feature data for model training.")
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                features, target, test_size=0.2, random_state=42
            )
            prep = ColumnTransformer(transformers=transformers)
            models = {
                "Linear Regression": LinearRegression(),
                "Random Forest": RandomForestRegressor(n_estimators=150, random_state=42),
                "KNN Regression": KNeighborsRegressor(n_neighbors=5),
                "Support Vector Regression (SVR)": SVR(kernel="rbf")
            }
            results = []
            for name, estimator in models.items():
                pipe = Pipeline([("preprocessor", prep), ("model", estimator)])
                try:
                    pipe.fit(X_train, y_train)
                    prediction = pipe.predict(X_test)
                    results.append({
                        "Algorithm": name,
                        "MAE": round(mean_absolute_error(y_test, prediction), 2),
                        "R² Score": round(r2_score(y_test, prediction), 3)
                    })
                except Exception as exc:
                    st.warning(f"{name} could not be trained: {exc}")
            if results:
                result_df = pd.DataFrame(results).sort_values("MAE")
                st.dataframe(result_df, use_container_width=True, hide_index=True)
                r1, r2 = st.columns(2)
                with r1:
                    st.plotly_chart(px.bar(result_df, x="Algorithm", y="MAE",
                                           title="Model Comparison — Lower MAE is Better",
                                           color="Algorithm"), use_container_width=True)
                with r2:
                    st.plotly_chart(px.bar(result_df, x="Algorithm", y="R² Score",
                                           title="Model Comparison — Higher R² is Better",
                                           color="Algorithm"), use_container_width=True)
                st.caption("Target: temporary total spending calculated from existing expense columns in memory. It is not saved to or added to the original CSV.")
                st.caption("Features: available demographic/income/payment columns. Expense components are excluded to prevent target leakage.")

    st.markdown("### Algorithms included")
    st.write("Linear Regression • Random Forest Regression • K-Nearest Neighbors Regression • Support Vector Regression (SVR)")


def about_project():
    st.title("About Project")
    st.write("""
    SpendWise is a student-focused expense tracking application. It combines daily
    expense storage, visual analysis, budgeting, personal spending forecasting and
    a reference-dataset machine learning component.
    """)
    st.markdown("### Technology")
    st.write("Python • Streamlit • Pandas • NumPy • Scikit-learn • Plotly • SQLite")


def about_ml(model_info):
    st.title("About ML")
    st.write("The project includes four core regression algorithms: Linear Regression, Random Forest Regression, K-Nearest Neighbors Regression, and Support Vector Regression (SVR). Open EDA & ML Lab to explore the dataset and compare their performance.")
    st.markdown("### Exact reference dataset columns")
    if model_info:
        st.code("\n".join(model_info.get("columns", [])))
    if model_info and "error" not in model_info:
        st.write(f"**Target:** {model_info['target_name']}")
        st.write(f"**Target source:** {model_info['target_source']}")
        st.write(f"**Training rows:** {model_info['training_rows']}")
        st.metric("MAE", f"{model_info['mae']:,.2f}")
        st.metric("R²", f"{model_info['r2']:.3f}")
        st.write("Model: RandomForestRegressor(n_estimators=200, random_state=42)")
    elif model_info:
        st.warning(model_info.get("error", "ML model could not be trained."))


def main():
    inject_css()
    init_db()

    df = get_expenses()
    model_info = train_reference_model()

    st.sidebar.markdown("# 💳 SpendWise")
    st.sidebar.caption("STUDENT EXPENSE TRACKER")
    st.sidebar.markdown("---")
    st.sidebar.markdown("<div style='font-size:11px;font-weight:700;letter-spacing:1.4px;color:#7b8499;padding:0 0 8px'>WORKSPACE</div>", unsafe_allow_html=True)

    # Simple direct-click navigation. Data import, About pages, and the ML lab
    # are intentionally not shown in the sidebar.
    nav_items = [
        "Dashboard",
        "Add Expense",
        "Expense History",
        "Visualize Spending",
        "Budget Maker",
        "Next Month Prediction",
        "Financial Insights",
        "Payment Reminders",
    ]
    previous_page = st.session_state.get("selected_page", "Dashboard")
    if previous_page not in nav_items:
        previous_page = "Dashboard"
    selected = st.sidebar.radio(
        "Navigate",
        nav_items,
        index=nav_items.index(previous_page),
        label_visibility="collapsed",
        key="selected_page",
    )
    st.sidebar.markdown("<div style='margin-top:22px;padding:13px 12px;border:1px solid #e1e7f5;border-radius:13px;background:rgba(255,255,255,.8)'><div style='font-size:11px;font-weight:700;color:#536dfe;letter-spacing:1px'>SMART MONEY HABITS</div><div style='font-size:12px;color:#657087;margin-top:5px;line-height:1.5'>Track thoughtfully.<br>Plan confidently.<br>Spend intentionally.</div><div style='margin-top:10px;font-size:10px;color:#8a93a7'>● LOCAL DATABASE ACTIVE</div></div>", unsafe_allow_html=True)

    estimate, _, _, _ = next_month_prediction(df)

    if selected == "Dashboard":
        dashboard(df, estimate)
    elif selected == "Add Expense":
        add_expense_page()
    elif selected == "Expense History":
        history_page(df)
    elif selected == "Visualize Spending":
        visualize_page(df)
    elif selected == "Budget Maker":
        budget_page(df)
    elif selected == "Next Month Prediction":
        prediction_page(df)
    elif selected == "Financial Insights":
        insights_page(df)
    elif selected == "Payment Reminders":
        reminders_page()
    elif selected == "Import Expenses":
        import_page()
    elif selected == "About Project":
        about_project()
    else:
        about_ml(model_info)


if __name__ == "__main__":
    main()
