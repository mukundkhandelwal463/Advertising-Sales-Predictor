import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# ----------- Load and Train Model -----------
data = pd.read_csv("6 advertising.csv")

x = data.iloc[:, :-1]
y = data["Sales"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

lr = LinearRegression()
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)

# ----------- Streamlit Web App -----------
st.set_page_config(page_title="Ad Budget Sales Predictor", layout="centered")
st.title("📊 Advertising Budget → Sales Prediction")

# Interactive numeric input fields
tv = st.number_input("📺 TV Budget")
radio = st.number_input("📻 Radio Budget")
newspaper = st.number_input("📰 Newspaper Budget")


# Prediction function
def predict_sale(tv_budget, radio_budget, newspaper_budget):
    features = pd.DataFrame([[tv_budget, radio_budget, newspaper_budget]],columns=["TV", "Radio", "Newspaper"])
    result = lr.predict(features)
    return result[0]


# Predict and display
if st.button("🔮 Predict Sales"):
    sales = predict_sale(tv, radio, newspaper)
    st.success(f"📈 Predicted Sales: {sales:.2f} units")

# ----------- Model Performance ----------
st.subheader("📉 Model Performance (on Test Set)")
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
col1, col2 = st.columns(2)
col1.metric("Mean Absolute Error", f"{mae:.2f}")
col2.metric("R² Score", f"{r2:.4f}")

# ----------- Actual vs. Predicted Plot -----------
st.subheader("📊 Actual vs. Predicted Sales")

fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, color='blue', label='Predictions')
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction Line')
ax.set_xlabel("Actual Sales")
ax.set_ylabel("Predicted Sales")
ax.set_title("Actual vs. Predicted Sales")
ax.legend()
st.pyplot(fig)

# ----------- Optional: Show Data -----------
with st.expander("🔍 View Training Data"):
    st.dataframe(data)
