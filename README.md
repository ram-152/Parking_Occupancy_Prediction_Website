# Parking Occupancy Prediction Website

A small upload-and-predict dashboard for the DWDM Parking Birmingham project. It uses Linear Regression, Decision Tree Regression, and KNN Regression; it does not use XGBoost.

## Start the website

1. Install Python 3.10 or later.
2. Open a terminal in this folder and install the packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Start the app:

   ```bash
   streamlit run app.py
   ```

4. Open the local address shown in the terminal (usually `http://localhost:8501`) and upload your CSV.

## Dataset format

The uploaded CSV must contain these columns:

| Column | Meaning |
| --- | --- |
| `SystemCodeNumber` | Parking area identifier |
| `Capacity` | Total number of spaces |
| `Occupancy` | Occupied spaces in the observation |
| `LastUpdated` | Date and time of the observation |

The app removes exact duplicates and rows missing required values, derives time features and availability, and then trains the three models. It evaluates on the latest 20% of timestamp-sorted records. Uploads are processed in memory for the current app session.

## What the website shows

- Cleaned dataset preview and cleaning counts
- Occupancy distribution and average occupancy by hour
- MAE, RMSE, and R² model comparison table and chart
- A prediction form for area, capacity, date, time, and model
- Predicted occupied spaces, available spaces, occupancy percentage, and status

This is a college project demonstration. Predictions depend on the history in the uploaded CSV and do not account for unrecorded events or closures.
