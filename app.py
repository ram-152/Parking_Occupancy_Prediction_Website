"""Parking occupancy prediction dashboard for the Parking Birmingham CSV."""

from __future__ import annotations

import io
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(page_title="Parking Occupancy Predictor", page_icon="🅿️", layout="wide")

REQUIRED = ["SystemCodeNumber", "Capacity", "Occupancy", "LastUpdated"]
FEATURES = ["SystemCodeNumber", "Capacity", "Year", "Month", "Day", "Hour",
            "DayOfWeek", "IsWeekend", "HourSin", "HourCos", "WeekdaySin", "WeekdayCos"]


def load_and_prepare(file) -> tuple[pd.DataFrame, dict]:
    data = pd.read_csv(file)
    # Tolerate extra whitespace in CSV header names.
    data.columns = [str(c).strip() for c in data.columns]
    missing = [c for c in REQUIRED if c not in data.columns]
    if missing:
        raise ValueError(f"Required columns not found: {', '.join(missing)}. Check the CSV header.")

    before = len(data)
    data = data.drop_duplicates().copy()
    duplicate_count = before - len(data)
    data["SystemCodeNumber"] = data["SystemCodeNumber"].astype("string").str.strip()
    data["Capacity"] = pd.to_numeric(data["Capacity"], errors="coerce")
    data["Occupancy"] = pd.to_numeric(data["Occupancy"], errors="coerce")
    data["LastUpdated"] = pd.to_datetime(data["LastUpdated"], errors="coerce", dayfirst=True)
    before = len(data)
    data = data.dropna(subset=REQUIRED).copy()
    missing_count = before - len(data)
    data = data[(data["Capacity"] > 0) & (data["SystemCodeNumber"] != "")].copy()
    if data.empty:
        raise ValueError("No valid records remain after cleaning. Check capacity, timestamps, and missing values.")

    data["Occupancy"] = np.minimum(data["Occupancy"].clip(lower=0), data["Capacity"])
    data["OccupancyPercentage"] = (100 * data["Occupancy"] / data["Capacity"]).clip(0, 100)
    data["AvailableSpaces"] = (data["Capacity"] - data["Occupancy"]).clip(lower=0)
    stamp = data["LastUpdated"]
    data["Year"], data["Month"], data["Day"] = stamp.dt.year, stamp.dt.month, stamp.dt.day
    data["Hour"], data["DayOfWeek"] = stamp.dt.hour, stamp.dt.dayofweek
    data["IsWeekend"] = (stamp.dt.dayofweek >= 5).astype(int)
    data["HourSin"] = np.sin(2 * np.pi * data["Hour"] / 24)
    data["HourCos"] = np.cos(2 * np.pi * data["Hour"] / 24)
    data["WeekdaySin"] = np.sin(2 * np.pi * data["DayOfWeek"] / 7)
    data["WeekdayCos"] = np.cos(2 * np.pi * data["DayOfWeek"] / 7)
    return data.sort_values("LastUpdated").reset_index(drop=True), {
        "duplicates": duplicate_count, "missing": missing_count,
    }


class LinearRegressor:
    """Batch-gradient-descent linear regression implemented with NumPy."""
    def fit(self, x, y):
        self.weights = np.zeros(x.shape[1], dtype=float)
        y = np.asarray(y, dtype=float)
        for _ in range(350):
            error = x @ self.weights - y
            self.weights -= 0.08 * ((x.T @ error) / len(y) + 1e-4 * self.weights)
        return self

    def predict(self, x):
        return x @ self.weights


class DecisionTreeRegressor:
    """Small CART-style regression tree using variance reduction splits."""
    def __init__(self, max_depth=9, min_leaf=20, candidates=10):
        self.max_depth, self.min_leaf, self.candidates = max_depth, min_leaf, candidates

    def fit(self, x, y):
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.root = self._grow(np.arange(len(self.y)), 0)
        return self

    def _grow(self, indices, depth):
        values = self.y[indices]
        node = {"value": float(values.mean())}
        if depth >= self.max_depth or len(indices) < 2 * self.min_leaf or np.all(values == values[0]):
            return node
        best_loss, best_feature, best_threshold, best_mask = float("inf"), None, None, None
        for feature in range(self.x.shape[1]):
            column = self.x[indices, feature]
            low, high = float(column.min()), float(column.max())
            if low >= high:
                continue
            thresholds = np.linspace(low, high, self.candidates + 2)[1:-1]
            for threshold in thresholds:
                mask = column <= threshold
                left_n, right_n = int(mask.sum()), int((~mask).sum())
                if left_n < self.min_leaf or right_n < self.min_leaf:
                    continue
                left_y, right_y = values[mask], values[~mask]
                loss = ((left_y - left_y.mean()) ** 2).sum() + ((right_y - right_y.mean()) ** 2).sum()
                if loss < best_loss:
                    best_loss, best_feature, best_threshold, best_mask = loss, feature, threshold, mask
        if best_feature is None:
            return node
        node.update({"feature": best_feature, "threshold": best_threshold,
                     "left": self._grow(indices[best_mask], depth + 1),
                     "right": self._grow(indices[~best_mask], depth + 1)})
        return node

    def predict(self, x):
        x = np.asarray(x, dtype=float)
        result = np.empty(len(x), dtype=float)
        def visit(node, rows):
            if not len(rows):
                return
            if "feature" not in node:
                result[rows] = node["value"]
                return
            mask = x[rows, node["feature"]] <= node["threshold"]
            visit(node["left"], rows[mask])
            visit(node["right"], rows[~mask])
        visit(self.root, np.arange(len(x)))
        return result


class KNNRegressor:
    """K-nearest-neighbor regression with inverse-distance weighting."""
    def __init__(self, neighbors=5, train_limit=8000):
        self.neighbors, self.train_limit = neighbors, train_limit

    def fit(self, x, y):
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        if len(y) > self.train_limit:
            selected = np.linspace(0, len(y) - 1, self.train_limit, dtype=int)
            x, y = x[selected], y[selected]
        self.x, self.y = x, y
        return self

    def predict(self, x):
        x = np.asarray(x, dtype=float)
        output = np.empty(len(x), dtype=float)
        # Batches keep temporary distance arrays small for browser-based runs.
        for start in range(0, len(x), 32):
            batch = x[start:start + 32]
            distances = ((batch[:, None, :] - self.x[None, :, :]) ** 2).sum(axis=2)
            k = min(self.neighbors, self.x.shape[0])
            nearest = np.argpartition(distances, k - 1, axis=1)[:, :k]
            near_dist = np.take_along_axis(distances, nearest, axis=1)
            weights = 1 / np.maximum(np.sqrt(near_dist), 1e-9)
            output[start:start + len(batch)] = (weights * self.y[nearest]).sum(axis=1) / weights.sum(axis=1)
        return output


class ModelBundle:
    """Stores feature encoding/scaling and a fitted estimator for predictions."""
    def __init__(self, estimator, columns, means, scales, numeric_columns):
        self.estimator, self.columns = estimator, columns
        self.means, self.scales, self.numeric_columns = means, scales, numeric_columns

    def transform(self, frame):
        encoded = pd.get_dummies(frame[FEATURES], columns=["SystemCodeNumber"], dtype=float)
        encoded = encoded.reindex(columns=self.columns, fill_value=0)
        encoded = _scale_feature_frame(encoded, self.numeric_columns, self.means, self.scales, self.columns)
        return np.column_stack([np.ones(len(encoded)), encoded.to_numpy(dtype=float)])

    def predict(self, frame):
        return self.estimator.predict(self.transform(frame))


def _regression_metrics(actual, predicted):
    residual = np.asarray(actual, dtype=float) - np.asarray(predicted, dtype=float)
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    total = float(np.sum((np.asarray(actual, dtype=float) - np.mean(actual)) ** 2))
    r_squared = 1 - float(np.sum(residual ** 2)) / total if total else 0.0
    return mae, rmse, r_squared


def _scale_feature_frame(frame, numeric_columns, means, scales, column_order=None):
    """Scale numeric fields by building fresh float columns, never in-place casting."""
    numeric_part = (frame[numeric_columns].astype("float64") - means) / scales
    other_part = frame.drop(columns=numeric_columns).astype("float64")
    scaled = pd.concat([numeric_part, other_part], axis=1)
    if column_order is not None:
        scaled = scaled.reindex(columns=column_order, fill_value=0.0)
    return scaled.astype("float64")


def train_models(data: pd.DataFrame):
    if len(data) < 10:
        raise ValueError("Please upload at least 10 valid rows so the models can be evaluated.")
    cut = int(len(data) * 0.8)
    train, test = data.iloc[:cut], data.iloc[cut:]
    numeric = [f for f in FEATURES if f != "SystemCodeNumber"]
    train_encoded = pd.get_dummies(train[FEATURES], columns=["SystemCodeNumber"], dtype=float)
    test_encoded = pd.get_dummies(test[FEATURES], columns=["SystemCodeNumber"], dtype=float)
    columns = list(train_encoded.columns)
    means = train_encoded[numeric].mean()
    scales = train_encoded[numeric].std().replace(0, 1).fillna(1)
    train_encoded = _scale_feature_frame(train_encoded, numeric, means, scales, columns)
    test_encoded = test_encoded.reindex(columns=columns, fill_value=0.0)
    test_encoded = _scale_feature_frame(test_encoded, numeric, means, scales, columns)
    x_train = np.column_stack([np.ones(len(train_encoded)), train_encoded.to_numpy(dtype=float)])
    x_test = np.column_stack([np.ones(len(test_encoded)), test_encoded.to_numpy(dtype=float)])
    estimators = {
        "Linear Regression": LinearRegressor(),
        "Decision Tree Regression": DecisionTreeRegressor(max_depth=9, min_leaf=20),
        "KNN Regression": KNNRegressor(neighbors=5),
    }
    fitted, scored = {}, []
    for name, estimator in estimators.items():
        estimator.fit(x_train, train["Occupancy"].to_numpy())
        model = ModelBundle(estimator, columns, means, scales, numeric)
        pred = np.clip(model.predict(test[FEATURES]), 0, test["Capacity"].to_numpy())
        fitted[name] = model
        mae, rmse, r2 = _regression_metrics(test["Occupancy"].to_numpy(), pred)
        scored.append({"Model": name, "MAE (spaces)": mae, "RMSE (spaces)": rmse, "R²": r2})
    scores = pd.DataFrame(scored).sort_values("RMSE (spaces)").reset_index(drop=True)
    return fitted, scores


def prediction_features(area: str, capacity: float, when: datetime) -> pd.DataFrame:
    h, dow = when.hour, when.weekday()
    return pd.DataFrame([{
        "SystemCodeNumber": area, "Capacity": capacity, "Year": when.year,
        "Month": when.month, "Day": when.day, "Hour": h, "DayOfWeek": dow,
        "IsWeekend": int(dow >= 5), "HourSin": np.sin(2*np.pi*h/24),
        "HourCos": np.cos(2*np.pi*h/24), "WeekdaySin": np.sin(2*np.pi*dow/7),
        "WeekdayCos": np.cos(2*np.pi*dow/7),
    }], columns=FEATURES)


st.title("🅿️ Parking Occupancy Prediction")
st.write("Upload a Parking Birmingham CSV to explore parking usage, compare DWDM regression models, and predict availability.")
st.info("Required CSV columns: `SystemCodeNumber`, `Capacity`, `Occupancy`, and `LastUpdated`. Algorithms: Linear Regression, Decision Tree Regression, and KNN Regression.")

uploaded = st.file_uploader("Upload your Parking Birmingham CSV", type=["csv"])
if uploaded is None:
    st.markdown("#### Getting started\nSelect the dataset CSV above. The file is processed for this session and is not saved by the app.")
    st.stop()

try:
    parking, cleaning = load_and_prepare(io.BytesIO(uploaded.getvalue()))
    models, scores = train_models(parking)
except Exception as exc:
    st.error(str(exc))
    st.stop()

st.success(f"Loaded and cleaned {len(parking):,} records from **{uploaded.name}**.")
metric_cols = st.columns(4)
metric_cols[0].metric("Valid records", f"{len(parking):,}")
metric_cols[1].metric("Parking areas", f"{parking['SystemCodeNumber'].nunique():,}")
metric_cols[2].metric("Duplicate rows removed", f"{cleaning['duplicates']:,}")
metric_cols[3].metric("Rows with missing required data removed", f"{cleaning['missing']:,}")

with st.expander("Preview cleaned data"):
    st.dataframe(parking[[*REQUIRED, "OccupancyPercentage", "AvailableSpaces"]].head(20), use_container_width=True)

st.header("Explore the data")
left, right = st.columns(2)
with left:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(parking["Occupancy"], bins=30, color="#277da1", edgecolor="white")
    ax.set(title="Distribution of occupied spaces", xlabel="Occupied spaces", ylabel="Records")
    st.pyplot(fig)
    plt.close(fig)
with right:
    hourly = parking.groupby("Hour")["OccupancyPercentage"].mean()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(hourly.index, hourly.values, marker="o", color="#f8961e")
    ax.set(title="Average occupancy by hour", xlabel="Hour of day", ylabel="Average occupancy (%)", xticks=range(0, 24, 2))
    ax.set_ylim(0, 100)
    st.pyplot(fig)
    plt.close(fig)

st.header("Model results")
st.caption("The latest 20% of records by timestamp are held out for evaluation. MAE and RMSE are in parking spaces; lower is better. R² closer to 1 is generally better.")
st.dataframe(scores.style.format({"MAE (spaces)": "{:.2f}", "RMSE (spaces)": "{:.2f}", "R²": "{:.3f}"}), use_container_width=True, hide_index=True)
fig, ax = plt.subplots(figsize=(9, 4))
scores.set_index("Model")[["MAE (spaces)", "RMSE (spaces)"]].plot(kind="bar", ax=ax, color=["#43aa8b", "#f94144"])
ax.set_ylabel("Error (spaces)")
ax.set_xlabel("")
ax.set_title("Model error comparison")
ax.tick_params(axis="x", rotation=15)
st.pyplot(fig)
plt.close(fig)
best_name = scores.iloc[0]["Model"]
st.caption(f"Lowest test RMSE: **{best_name}**")

st.header("Predict parking availability")
areas = sorted(parking["SystemCodeNumber"].astype(str).unique())
default_idx = 0
col1, col2, col3, col4 = st.columns([2, 1, 2, 2])
with col1:
    selected_area = st.selectbox("Parking area", areas, index=default_idx)
with col2:
    typical_capacity = int(round(parking.loc[parking["SystemCodeNumber"].astype(str) == selected_area, "Capacity"].median()))
    capacity = st.number_input("Total spaces", min_value=1, value=max(1, typical_capacity), step=1)
with col3:
    prediction_date = st.date_input("Prediction date", value=(parking["LastUpdated"].max() + timedelta(days=1)).date())
with col4:
    prediction_time = st.time_input("Prediction time", value=datetime.strptime("12:00", "%H:%M").time())
selected_model = st.selectbox("Model", list(models), index=list(models).index(best_name))

if st.button("Predict occupancy", type="primary"):
    requested = datetime.combine(prediction_date, prediction_time)
    x_new = prediction_features(selected_area, capacity, requested)
    estimate = float(models[selected_model].predict(x_new)[0])
    occupied = int(np.clip(np.rint(estimate), 0, capacity))
    free = int(capacity) - occupied
    percent = 100 * occupied / capacity
    status = "Full" if free == 0 else ("Limited availability" if percent >= 80 else "Available")
    st.subheader("Prediction result")
    out = st.columns(4)
    out[0].metric("Predicted occupancy", f"{occupied:,} spaces")
    out[1].metric("Available spaces", f"{free:,}")
    out[2].metric("Occupancy", f"{percent:.1f}%")
    out[3].metric("Status", status)
    st.caption(f"{selected_area} · {requested:%d %b %Y, %I:%M %p} · {selected_model}")

with st.expander("About this project"):
    st.markdown("""
    **Goal:** Predict occupancy from historical parking observations and provide availability guidance.

    **Preprocessing:** Exact duplicate records and rows missing required fields are removed. Numeric values are converted, occupancy is bounded by capacity, and date/time fields are transformed into model features. Numeric predictors are standardized before fitting.

    **Status guide:** Full = no free spaces; Limited availability = at least 80% occupied; Available = below 80% occupied.

    **Limitations:** Predictions reflect patterns in the uploaded history. Special events, closures, and other unrecorded factors are not known to the models. The chronological holdout score may change with the selected dataset.
    """)
