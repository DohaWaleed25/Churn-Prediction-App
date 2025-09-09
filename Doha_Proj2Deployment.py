import streamlit as st  # pyright: ignore[reportMissingImports]
import pandas as pd  # pyright: ignore[reportMissingModuleSource]
import numpy as np  # pyright: ignore[reportMissingImports]
import matplotlib.pyplot as plt  # pyright: ignore[reportMissingModuleSource]
import joblib  # pyright: ignore[reportMissingImports]
from sklearn.naive_bayes import GaussianNB # pyright: ignore[reportMissingModuleSource]

# Background 
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(to right, #11998e, #4aa02c);
}
[data-testid="stHeader"] {
    background-color: rgba(0,0,0,0);
}
[data-testid="stSidebar"] {
    background: #6afb92;
    color: black; /* يجعل النصوص باللون الأسود */
}
[data-testid="stSidebar"] * {
    color: white !important; /* يضمن أن كل العناصر جوه الـ sidebar تبقى باللون الابيض */
}
/* تعديل الـ selectbox (القيمة المختارة) */
.css-1wa3eu0-placeholder, .css-1uccc91-singleValue {
    color: Black !important;   /* يخلي التكست اسود */
    font-weight: bold;         /* ممكن تخليه بولد */
}
/* تعديل اللستة المنسدلة */
.css-26l3qy-menu {
    background: #000000 !important; /* لون الخلفية */
    color: White !important;        /* لون التكست */
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# Title 
st.title("📊 Churn Prediction 📊") 
st.write("This Simple app uses a Naive Bayes Model to predict whether a customer is likely to **churn** or not.")

st.markdown(
    """
    ✅ **Churn** means when a customer stops using a company’s services.  
    ✅ **Tenure** means how long the customer has been with the company (in months). 

    """
)

df = pd.read_excel("churn_dataset.xlsx")
st.subheader("Show Dataset Preview")
#st.write(df)


# Load Model 
model = joblib.load("Gaussian_Model.pkl")


# User Input 
st.sidebar.header("🔧 Input Customer Information")
Age = st.sidebar.slider ("Age (Years)" , 0 , 60 , 30)
Gender = st.sidebar.selectbox ("Gender" , ["Male 🚹" , "Female 🚺"], index = 1)
Tenure = st.sidebar.slider ("Tenure (Months)" , 0 , 75 , 37)
gender_val = 1 if Gender == "Male" else 0

input_data = np.array([[Age,Tenure, gender_val]])

st.write(f"You selected age: {Age}")
st.write(f"You selected tenure: {Tenure}")
st.write(f"You selected gender: {Gender}")

# Predict
st.markdown("---")
prediction = model.predict(input_data)[0]
pred_proba = model.predict_proba(input_data)[0]
churn_labels = {0: "No, will stay ✅", 
                1: "Yes, will leave ❌"}


# Show prediction
st.subheader("🔮 Prediction Result")
st.write(f"** Churn Predicted:** {churn_labels [prediction]}")

# Show prediction probabilities
st.markdown("---")
st.subheader("📈 Prediction Probabilities")

col1, col2 = st.columns([3, 2])  

with col1:
    st.write(f"🟢 Stay: {pred_proba[0]:.2%}")
    st.write(f"🔺 Leave: {pred_proba[1]:.2%}")

with col2:
    fig, ax = plt.subplots(figsize=(2, 2))  # smaller chart
    labels = ["Stay (No)", "Leave (Yes)"]
    colors = ["#4CAF50", "#FF5252"]
    ax.pie(pred_proba, labels=labels, autopct="%1.1f%%",
           startangle=90, colors=colors, textprops={"fontsize": 8})
    ax.axis("equal")

    st.pyplot(fig)


















