
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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
    :root { --sw-ink:#17233c; --sw-muted:#6d7890; --sw-blue:#4968e8; --sw-teal:#0f9d91; }
    .stApp { background: linear-gradient(180deg,#f7f9ff 0%,#f4f7fc 55%,#f8faff 100%); color:var(--sw-ink); }
    [data-testid="stSidebar"] { background:linear-gradient(180deg,#ffffff 0%,#f1f5ff 100%); border-right:1px solid #e4eaf6; }
    [data-testid="stSidebar"] > div:first-child { padding-top:1.25rem; }
    [data-testid="stSidebar"] [data-testid="stRadio"] > label { display:none; }
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] { gap:6px; }
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label { border:1px solid transparent; border-radius:12px; padding:10px 12px; transition:all .18s ease; cursor:pointer; margin:1px 0; }
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:hover { background:#eaf0ff; border-color:#d7e2ff; transform:translateX(3px); box-shadow:0 5px 15px rgba(57,86,160,.09); }
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) { background:linear-gradient(100deg,#e6edff,#f4f6ff); border-color:#cbd7ff; box-shadow:inset 3px 0 0 var(--sw-blue); }
    [data-testid="stSidebar"] [data-testid="stRadio"] label p { font-weight:560; color:#283550; }
    .block-container { padding-top:1.45rem; padding-bottom:3rem; max-width:1500px; }
    h1,h2,h3 { letter-spacing:-.035em; color:#17233c; }
    h1 { font-weight:780 !important; }
    [data-testid="stMetric"] { background:#fff; border:1px solid #e7ecf6; padding:17px 18px; border-radius:16px; box-shadow:0 5px 20px rgba(35,54,95,.045); }
    [data-testid="stMetricLabel"] { color:#68758e !important; }
    [data-testid="stMetricValue"] { color:#17233c !important; font-weight:760; }
    div[data-testid="stPlotlyChart"],div[data-testid="stDataFrame"] { border-radius:15px; overflow:hidden; }
    div[data-testid="stButton"] button,div[data-testid="stFormSubmitButton"] button { border-radius:11px; font-weight:650; transition:all .18s ease; }
    div[data-testid="stButton"] button:hover,div[data-testid="stFormSubmitButton"] button:hover { transform:translateY(-1px); box-shadow:0 7px 18px rgba(55,75,130,.15); }
    div[data-testid="stTextInput"] input,div[data-testid="stNumberInput"] input,div[data-testid="stSelectbox"] div[data-baseweb="select"] { border-radius:10px; }
    .sw-hero { background:radial-gradient(circle at 88% 12%,rgba(255,255,255,.23),transparent 25%),linear-gradient(120deg,#253c8e 0%,#4968e8 54%,#3989c9 100%); color:white; border-radius:22px; padding:26px 28px; margin-bottom:20px; box-shadow:0 13px 32px rgba(52,80,174,.18); }
    .sw-hero h1 { color:white !important; margin:0 0 7px 0; font-size:2rem; }
    .sw-hero p { color:#e8edff; margin:0; font-size:1rem; }
    .sw-kicker { font-size:.72rem; letter-spacing:1.5px; text-transform:uppercase; font-weight:800; opacity:.82; margin-bottom:9px; }
    .sw-panel { background:rgba(255,255,255,.92); border:1px solid #e5ebf6; border-radius:17px; padding:18px 20px; box-shadow:0 5px 18px rgba(35,54,95,.04); margin-bottom:12px; }
    .sw-panel-title { font-weight:750; font-size:1.03rem; color:#1d2a45; margin-bottom:5px; }
    .sw-muted { color:#6d7890; font-size:.9rem; }
    .sw-tag { display:inline-block; padding:5px 10px; border-radius:20px; background:#eaf0ff; color:#4058bd; font-size:.78rem; font-weight:700; margin-right:5px; }
    hr { border-color:#e7ecf5 !important; }
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
    exclude.update(expense_cols)
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


def go_to_model_performance():
    """Navigate via a callback before the sidebar widget is recreated."""
    st.session_state["selected_page"] = "Model Performance"


def dashboard(df, prediction, budget_limit=0):
    st.markdown("""
    <div class="sw-hero">
      <div class="sw-kicker">YOUR PERSONAL MONEY SPACE</div>
      <h1>Make every rupee count.</h1>
      <p>Understand your habits, plan your month, and make confident spending decisions.</p>
    </div>
    """, unsafe_allow_html=True)

    total = float(pd.to_numeric(df["amount"], errors="coerce").fillna(0).sum()) if not df.empty else 0.0
    this_month = current_month_total(df)
    m = monthly_data(df)
    avg_month = float(m["amount"].mean()) if not m.empty else 0.0
    remaining = max(0.0, budget_limit - this_month) if budget_limit else 0.0

    # Four readable KPI cards avoid overcrowding smaller screens.
    cols = st.columns(4)
    vals = [
        ("ALL-TIME SPENDING", f"₹{total:,.0f}", "Across saved expenses"),
        ("THIS MONTH", f"₹{this_month:,.0f}", "Spending so far"),
        ("MONTHLY AVERAGE", f"₹{avg_month:,.0f}", f"Based on {len(m)} month(s)"),
        ("NEXT-MONTH ESTIMATE", f"₹{prediction:,.0f}", "Trend-based estimate")
    ]
    for c, (t, v, sub) in zip(cols, vals):
        with c:
            st.metric(t, v, help=sub)

    st.markdown("## Your spending at a glance")
    left, right = st.columns([1.25, 1])
    with left:
        st.markdown("<div class='sw-panel-title'>Monthly spending trend</div><div class='sw-muted'>See how your spending changes over time.</div>", unsafe_allow_html=True)
        if not m.empty:
            fig = px.area(m, x="month", y="amount", markers=True)
            fig.update_traces(line=dict(color="#4968e8", width=3), fillcolor="rgba(73,104,232,.13)", marker=dict(size=7, color="#4968e8"))
            fig.update_layout(height=330, margin=dict(l=8,r=8,t=16,b=8), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis_title="Month", yaxis_title="Amount (₹)", hovermode="x unified")
            fig.update_xaxes(showgrid=False)
            fig.update_yaxes(gridcolor="#edf1f8", zeroline=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Add your first expense to unlock your spending trend.")
    with right:
        st.markdown("<div class='sw-panel-title'>Where your money goes</div><div class='sw-muted'>Category-wise share of saved expenses.</div>", unsafe_allow_html=True)
        if not df.empty:
            cat = df.groupby("category", as_index=False)["amount"].sum().sort_values("amount", ascending=False)
            fig = px.pie(cat, names="category", values="amount", hole=.62, color_discrete_sequence=px.colors.qualitative.Safe)
            fig.update_traces(textposition="inside", textinfo="percent", hovertemplate="%{label}<br>₹%{value:,.0f}<extra></extra>")
            fig.update_layout(height=330, margin=dict(l=8,r=8,t=16,b=8), paper_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h",yanchor="top",y=-.08,xanchor="center",x=.5), showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Your category breakdown will appear after you save expenses.")

    st.markdown("## Your money snapshot")
    a, b = st.columns([1, 1])
    with a:
        st.markdown("<div class='sw-panel-title'>Recent activity</div>", unsafe_allow_html=True)
        if df.empty:
            st.info("No expenses yet. Add one to start building your personal history.")
        else:
            recent = df.head(6).copy()
            recent["amount"] = recent["amount"].map(lambda x: f"₹{x:,.2f}")
            st.dataframe(recent[[c for c in ["date","category","amount","payment_method","note"] if c in recent.columns]], use_container_width=True, hide_index=True)
    with b:
        st.markdown("<div class='sw-panel-title'>Budget pulse</div><div class='sw-muted'>Set a monthly budget in Budget Maker for a personalised status.</div>", unsafe_allow_html=True)
        if budget_limit > 0:
            progress = min(1.0, this_month / budget_limit)
            st.progress(progress, text=f"{progress:.0%} of budget used")
            st.metric("Budget remaining", f"₹{max(0, budget_limit-this_month):,.0f}")
            if this_month > budget_limit:
                st.error("Your recorded spending is above the current budget.")
            elif progress >= .8:
                st.warning("You're getting close to your budget limit.")
            else:
                st.success("You're currently within your budget.")
        else:
            st.info("Budget tracking becomes active after you create a budget. Your expense history remains saved locally.")
        st.button(
            "Open ML Model Studio",
            use_container_width=True,
            type="primary",
            on_click=go_to_model_performance,
        )

    st.markdown("## A simple routine that works")
    c1,c2,c3 = st.columns(3)
    with c1:
        st.markdown("<div class='sw-panel'><div class='sw-panel-title'>01 · Record</div><div class='sw-muted'>Log daily expenses with a category and payment method.</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='sw-panel'><div class='sw-panel-title'>02 · Understand</div><div class='sw-muted'>Use charts and insights to identify patterns in your spending.</div></div>", unsafe_allow_html=True)
    with c3:
        st.markdown("<div class='sw-panel'><div class='sw-panel-title'>03 · Plan ahead</div><div class='sw-muted'>Set a budget and review your data-driven spending estimate.</div></div>", unsafe_allow_html=True)


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
        st.session_state["monthly_budget_limit"] = max(0.0, income - savings)
        st.session_state["monthly_income"] = income
        st.session_state["monthly_savings_goal"] = savings
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
    """EDA and fair evaluation of four regression algorithms on the unchanged reference CSV."""
    st.markdown("# Model Performance Studio")
    st.caption("Explore the original dataset, validate the modelling approach, and compare four regression algorithms.")
    raw, actual_columns = read_reference_dataset()
    if raw.empty:
        st.warning("Reference dataset was not found. Keep the original CSV in reference_dataset/.")
        return

    raw = raw.copy()
    numeric_all = raw.select_dtypes(include=np.number)
    total_missing = int(raw.isna().sum().sum())
    k1,k2,k3,k4 = st.columns(4)
    k1.metric("Dataset records", f"{len(raw):,}")
    k2.metric("Original columns", f"{len(actual_columns)}")
    k3.metric("Missing cells", f"{total_missing:,}")
    k4.metric("Numeric features", f"{len(numeric_all.columns)}")
    with st.expander("Inspect original columns and sample records"):
        st.write("The CSV is read-only. No columns are added to or removed from the source file.")
        st.code("\n".join(actual_columns))
        st.dataframe(raw.head(12), use_container_width=True, hide_index=True)

    st.markdown("## 1 · Exploratory data analysis")
    tab1,tab2,tab3,tab4 = st.tabs(["Distributions", "Category patterns", "Relationships", "Data quality"])
    numeric = numeric_all.drop(columns=[c for c in numeric_all.columns if str(c).lower().startswith("unnamed:")], errors="ignore")
    categorical = [c for c in raw.columns if c not in numeric.columns and not str(c).lower().startswith("unnamed:")]
    with tab1:
        if not numeric.empty:
            selected_num = st.selectbox("Choose a numeric field", numeric.columns.tolist(), key="studio_num")
            series = pd.to_numeric(raw[selected_num], errors="coerce").dropna()
            fig = px.histogram(series.to_frame(name=selected_num), x=selected_num, nbins=28, marginal="box", title=f"Distribution of {selected_num}", color_discrete_sequence=["#4968e8"])
            fig.update_layout(height=360, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=10,r=10,t=55,b=10))
            fig.update_yaxes(gridcolor="#edf1f8")
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"Valid values: {len(series):,} · Mean: {series.mean():,.2f} · Median: {series.median():,.2f} · Std. deviation: {series.std():,.2f}")
    with tab2:
        if categorical:
            selected_cat = st.selectbox("Choose a categorical field", categorical, key="studio_cat")
            counts = raw[selected_cat].fillna("Missing").astype(str).value_counts().head(15).rename_axis(selected_cat).reset_index(name="Records")
            fig = px.bar(counts.sort_values("Records"), x="Records", y=selected_cat, orientation="h", title=f"Most common values in {selected_cat}", color="Records", color_continuous_scale=["#cdd8ff", "#4968e8"])
            fig.update_layout(height=370, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=10,r=10,t=55,b=10), coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No categorical fields were found.")
    with tab3:
        if len(numeric.columns) >= 2:
            corr = numeric.corr(numeric_only=True)
            fig = px.imshow(corr, text_auto=".2f", title="Numeric feature correlation", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
            fig.update_layout(height=max(400, min(720, 28*len(corr.columns))), margin=dict(l=10,r=10,t=55,b=10))
            st.plotly_chart(fig, use_container_width=True)
            pairs=[]
            cols=corr.columns.tolist()
            for i in range(len(cols)):
                for j in range(i+1,len(cols)):
                    pairs.append((cols[i],cols[j],float(corr.iloc[i,j])))
            pairs.sort(key=lambda x:abs(x[2]),reverse=True)
            if pairs:
                a,b,v=pairs[0]
                st.info(f"Strongest absolute numeric correlation in this dataset: **{a}** and **{b}** (r = {v:.2f}). Correlation does not by itself prove causation.")
        else:
            st.info("At least two numeric fields are required for correlation analysis.")
    with tab4:
        quality = pd.DataFrame({"Column":raw.columns,"Data type":[str(x) for x in raw.dtypes],"Missing":raw.isna().sum().values,"Missing %":(raw.isna().mean().values*100).round(2),"Unique values":raw.nunique(dropna=True).values})
        st.dataframe(quality.sort_values(["Missing","Column"],ascending=[False,True]), use_container_width=True, hide_index=True)
        st.caption("Rows with entirely empty or index-like columns are not used as model features. The source CSV remains unchanged.")
        if not numeric.empty:
            outlier_rows=[]
            for col in numeric.columns:
                values=pd.to_numeric(raw[col],errors="coerce").dropna()
                if len(values) < 4:
                    continue
                q1,q3=values.quantile(.25),values.quantile(.75)
                iqr=q3-q1
                count=int(((values < q1-1.5*iqr) | (values > q3+1.5*iqr)).sum()) if iqr > 0 else 0
                outlier_rows.append({"Numeric column":col,"Potential outliers (IQR rule)":count})
            if outlier_rows:
                st.markdown("**Potential numeric outliers**")
                st.dataframe(pd.DataFrame(outlier_rows).sort_values("Potential outliers (IQR rule)",ascending=False),use_container_width=True,hide_index=True)
                st.caption("The IQR rule flags values for review; it does not automatically mean the records are incorrect.")

    st.markdown("## 2 · Train and compare the four algorithms")
    expense_cols = find_expense_columns(raw)
    if not expense_cols:
        st.warning("No numeric expense columns were detected. Review the source column names before modelling.")
        return
    expense_values = raw[expense_cols].apply(pd.to_numeric, errors="coerce")
    target = expense_values.sum(axis=1, min_count=1)
    excluded = set(expense_cols)
    excluded.update(c for c in raw.columns if str(c).lower().startswith("unnamed:"))
    features = raw.drop(columns=list(excluded), errors="ignore").loc[target.notna()].copy()
    target = target.loc[features.index]
    # Remove all-empty and constant features to avoid noisy/invalid model inputs.
    features = features.drop(columns=[c for c in features.columns if features[c].isna().all() or features[c].nunique(dropna=True) <= 1], errors="ignore")
    num_cols = features.select_dtypes(include=np.number).columns.tolist()
    cat_cols = [c for c in features.columns if c not in num_cols]
    if not len(features) >= 10 or not (num_cols or cat_cols):
        st.warning("There are not enough usable feature rows to train the models safely.")
        return

    st.markdown("<div class='sw-panel'><div class='sw-panel-title'>What the model predicts</div><div class='sw-muted'>A temporary total of the existing expense-component columns, calculated in memory only. Those component columns are excluded from model inputs to avoid target leakage. No target column is written into the CSV.</div></div>", unsafe_allow_html=True)
    with st.expander("Review target and feature choices"):
        st.write("**Target components:**", expense_cols)
        st.write("**Model input features:**", features.columns.tolist())
        st.write("**Rows used:**", len(features))
        st.write("**Validation:**", "80% training / 20% held-out test set, fixed random seed 42")

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regression": RandomForestRegressor(n_estimators=200, random_state=42, min_samples_leaf=2),
        "KNN Regression": KNeighborsRegressor(n_neighbors=min(5, max(1, len(X_train)//3))),
        "Support Vector Regression (SVR)": SVR(kernel="rbf", C=10, epsilon=0.1)
    }
    results=[]
    fitted={}
    for name, estimator in models.items():
        transformers=[]
        if num_cols:
            # Scaling is important for distance- and margin-based models and is safe for the tree model too.
            transformers.append(("num", Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler())]),num_cols))
        if cat_cols:
            transformers.append(("cat",Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]),cat_cols))
        prep=ColumnTransformer(transformers=transformers)
        pipe=Pipeline([("preprocessor",prep),("model",estimator)])
        try:
            pipe.fit(X_train,y_train)
            pred=pipe.predict(X_test)
            rmse=float(np.sqrt(mean_squared_error(y_test,pred)))
            mae=float(mean_absolute_error(y_test,pred))
            r2=float(r2_score(y_test,pred)) if len(y_test)>1 else float("nan")
            results.append({"Algorithm":name,"MAE (₹)":mae,"RMSE (₹)":rmse,"R² Score":r2})
            fitted[name]=(pipe,pred)
        except Exception as exc:
            st.warning(f"{name} could not be trained: {exc}")
    if not results:
        st.error("No model could be trained. Check the dataset and selected feature columns.")
        return

    result_df=pd.DataFrame(results).sort_values("MAE (₹)").reset_index(drop=True)
    best_name=str(result_df.iloc[0]["Algorithm"])
    best_mae=float(result_df.iloc[0]["MAE (₹)"])
    best_rmse=float(result_df.iloc[0]["RMSE (₹)"])
    best_r2=float(result_df.iloc[0]["R² Score"])
    m1,m2,m3=st.columns(3)
    m1.metric("Best model by test MAE",best_name)
    m2.metric("Best model MAE",f"₹{best_mae:,.2f}",help="Lower is better on the held-out test set.")
    m3.metric("Best model R²",f"{best_r2:.3f}" if np.isfinite(best_r2) else "N/A",help="R² can be negative when predictions are worse than a mean baseline.")
    st.dataframe(result_df.style.format({"MAE (₹)":"₹{:,.2f}","RMSE (₹)":"₹{:,.2f}","R² Score":"{:.3f}"}), use_container_width=True, hide_index=True)
    st.download_button("Download model comparison (CSV)", result_df.to_csv(index=False).encode("utf-8"), "spendwise_model_comparison.csv", "text/csv")
    left,right=st.columns(2)
    with left:
        fig=px.bar(result_df.sort_values("MAE (₹)",ascending=True),x="MAE (₹)",y="Algorithm",orientation="h",title="Test error · lower is better",color="MAE (₹)",color_continuous_scale=["#bdece5","#0f9d91"])
        fig.update_layout(height=350,margin=dict(l=10,r=10,t=55,b=10),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",coloraxis_showscale=False)
        st.plotly_chart(fig,use_container_width=True)
    with right:
        r2df=result_df.copy()
        r2df["R² Score"]=r2df["R² Score"].replace([np.inf,-np.inf],np.nan)
        fig=px.bar(r2df,x="Algorithm",y="R² Score",title="Explained variance · higher is better",color="R² Score",color_continuous_scale=["#ffd9a8","#4968e8"])
        fig.update_layout(height=350,margin=dict(l=10,r=10,t=55,b=10),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",coloraxis_showscale=False,xaxis_tickangle=-15)
        st.plotly_chart(fig,use_container_width=True)

    st.markdown("### Prediction quality check")
    chosen=st.selectbox("Inspect model predictions",list(fitted.keys()),index=list(fitted.keys()).index(best_name),key="inspect_model")
    chosen_pipe,pred=fitted[chosen]
    actual_pred=pd.DataFrame({"Actual total (₹)":np.asarray(y_test),"Predicted total (₹)":np.asarray(pred)})
    actual_pred["Residual (₹)"]=actual_pred["Actual total (₹)"]-actual_pred["Predicted total (₹)"]
    fig=px.scatter(actual_pred,x="Actual total (₹)",y="Predicted total (₹)",title=f"Actual vs predicted · {chosen}",color_discrete_sequence=["#4968e8"],hover_data=["Residual (₹)"])
    low=float(min(actual_pred["Actual total (₹)"].min(),actual_pred["Predicted total (₹)"].min()))
    high=float(max(actual_pred["Actual total (₹)"].max(),actual_pred["Predicted total (₹)"].max()))
    fig.add_shape(type="line",x0=low,y0=low,x1=high,y1=high,line=dict(color="#0f9d91",dash="dash",width=2))
    fig.update_layout(height=390,margin=dict(l=10,r=10,t=55,b=10),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig,use_container_width=True)
    with st.expander("View held-out predictions"):
        st.dataframe(actual_pred.round(2),use_container_width=True,hide_index=True)
        st.download_button("Download held-out predictions",actual_pred.to_csv(index=False).encode("utf-8"),"spendwise_test_predictions.csv","text/csv")

    st.markdown("### Which inputs matter most? · model explainability")
    try:
        importance=permutation_importance(chosen_pipe,X_test,y_test,n_repeats=5,random_state=42,scoring="neg_mean_absolute_error",n_jobs=1)
        imp_df=pd.DataFrame({"Input feature":X_test.columns,"MAE importance":importance.importances_mean,"Variation":importance.importances_std}).sort_values("MAE importance",ascending=False)
        fig=px.bar(imp_df.head(10).sort_values("MAE importance"),x="MAE importance",y="Input feature",orientation="h",title="Permutation importance on held-out test data",color="MAE importance",color_continuous_scale=["#cdd8ff","#4968e8"])
        fig.update_layout(height=360,margin=dict(l=10,r=10,t=55,b=10),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",coloraxis_showscale=False)
        st.plotly_chart(fig,use_container_width=True)
        st.caption("Permutation importance measures how much test MAE changes when one input column is shuffled. Negative values can occur and do not prove a feature is harmful.")
    except Exception as exc:
        st.info(f"Feature-importance explanation is unavailable for this run: {exc}")

    st.markdown("### What the results mean")
    st.write(f"**{best_name}** has the lowest MAE on this one held-out test split. MAE is the average absolute prediction error; RMSE penalizes larger errors more; R² compares the model with a simple mean baseline.")
    if best_r2 < 0:
        st.warning("The best model's R² is negative on this split. Treat the predictions cautiously and compare against a baseline before presenting it as useful forecasting.")
    elif best_r2 < .25:
        st.info("The model explains only a limited share of variation on this test split. Present this honestly as an experimental baseline, not a highly accurate forecast.")
    else:
        st.info("These are test-set results, not a guarantee of future accuracy. Results may vary with another dataset or split.")
    st.caption("Important limitation: the target is constructed from existing expense components, while those components are excluded from inputs. The models estimate this total from demographic and financial context, so this is an educational regression experiment rather than a verified personal next-month forecast.")
    st.markdown("### Algorithms included")
    st.write("Linear Regression · Random Forest Regression · K-Nearest Neighbors Regression · Support Vector Regression (SVR)")


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
        "Model Performance",
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
        dashboard(df, estimate, st.session_state.get("monthly_budget_limit", 0.0))
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
    elif selected == "Model Performance":
        eda_ml_lab_page()
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
