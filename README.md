# 📈 Advertising Sales Predictor (Linear Regression Model)

A simple and interactive web application built using **Streamlit** that predicts product sales based on advertising budgets for **TV**, **Radio**, and **Newspaper** using a **Linear Regression** model.

---

## 🔗 Live Demo

👉 **[Click here to try the app](https://advertising-sales-predictor-detqnqfxp7qytmmh4pffqk.streamlit.app/)**

---

## 📂 Dataset

- **Dataset Name:** Advertising Dataset  
- **Source:** Provided in the project folder  
- **Features:**
  - `TV`: Budget allocated to TV advertising  
  - `Radio`: Budget allocated to Radio advertising  
  - `Newspaper`: Budget allocated to Newspaper advertising  
  - `Sales`: Resulting product sales

---

## ⚙️ Features

- Accepts interactive inputs for advertising budgets
- Predicts sales using a trained Linear Regression model
- Shows model performance (R² score)
- Visualizes **Actual vs Predicted Sales** using plots

---

## 🧰 Tech Stack

- **Frontend:** Streamlit
- **Backend/ML:** Pandas, NumPy, Scikit-learn
- **Visualization:** Matplotlib, Seaborn
- **Deployment:** Streamlit Cloud

---

## 📁 Project Structure

advertising-sales-predictor/
│
├── advertising.csv # Advertising dataset
├── model.pkl # Trained Linear Regression model
├── streamlit_app.py # Main Streamlit app
├── requirements.txt # Python dependencies
├── screenshots/ # App screenshots (optional)
│ ├── input.png
│ └── output.png
└── README.md # Project documentation

yaml
Copy
Edit

---

## ▶️ Run Locally

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/advertising-sales-predictor.git
   cd advertising-sales-predictor
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
Run the Streamlit app

bash
Copy
Edit
streamlit run streamlit_app.py


👨‍💻 Author
Mukund Khandelwal
🔗 LinkedIn: https://www.linkedin.com/posts/mukund-khandelwal-6a8663283_machinelearning-datascience-python-activity-7353326279393226753-2J5H?utm_source=share&utm_medium=member_desktop&rcm=ACoAAET5diABs7bbZlDnVTGZ4DnPgeKxnEmHsgA

💬 Feedback
If you find this project helpful or have suggestions to improve it, feel free to open an issue or contribute to the repository.
