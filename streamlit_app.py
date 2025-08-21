import streamlit as st
import joblib
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    # page_icon="🧠",
    layout="centered"
)



# Load all models
binary_model = joblib.load("./models/binary_model.pkl")
ternary_model = joblib.load("./models/ternary_model.pkl")
star_model = joblib.load("./models/star_LogisticRegression_model.pkl")

# Class mappings
ternary_labels = ["Negative", "Neutral", "Positive"]
ternary_colors = {"Negative": "tomato", "Neutral": "skyblue", "Positive": "lightgreen"}
star_emojis = ["⭐", "⭐⭐", "⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐⭐"]




# App title
st.title("Opinion Mining on Web Reviews (Sentiment Analysis)")

st.markdown(
    """
    **Welcome to the Opinion Mining Dashboard!** 🧠  

    This web application demonstrates sentiment analysis on user reviews, built as part of a **Web Mining** course project. Using **Yelp reviews** as the dataset, we trained multiple traditional machine learning models to predict sentiments in three different ways:

    1. **Binary Sentiment (2-class)** – Classifies text as **Positive** or **Negative**.
    2. **Ternary Sentiment (3-class)** – Classifies text as **Negative**, **Neutral**, or **Positive**.
    3. **Star Rating (5-class)** – Predicts a **1 to 5 star rating** based on the review text.

    You can enter any review in the text box, choose a model, and see the predicted sentiment or rating visually displayed.  
    This project showcases how machine learning can automatically understand and interpret opinions expressed in text, which is widely useful in customer feedback analysis, recommendation systems, and social media monitoring.
    """,
    unsafe_allow_html=True
)
# Model selector
model_choice = st.selectbox(
    "Choose a model",
    ("Binary Sentiment (2-class)", "Ternary Sentiment (3-class)", "Star Rating (5-class)")
)

# Text input
user_input = st.text_area("Enter your review text below:")

if st.button("Predict"):
    if not user_input.strip():
        st.warning("Please enter some text before predicting.")
    else:
        if model_choice == "Binary Sentiment (2-class)":
            probs = binary_model.predict_proba([user_input])[0]
            labels = ["Negative", "Positive"]

            fig, ax = plt.subplots()
            ax.pie(probs, labels=labels, autopct='%1.1f%%', colors=["tomato", "lightgreen"], startangle=90)
            ax.axis('equal')
            st.pyplot(fig)

        elif model_choice == "Ternary Sentiment (3-class)":
            pred_idx = ternary_model.predict([user_input])[0]
            pred_label = ternary_labels[int(pred_idx)]  # convert number to label
            color = ternary_colors[pred_label]
            st.markdown(f"<h3 style='color:{color}'>{pred_label}</h3>", unsafe_allow_html=True)


        elif model_choice == "Star Rating (5-class)":
            pred = star_model.predict([user_input])[0]
            try:
                stars = star_emojis[int(pred)-1]
                st.markdown(f"<h3>{int(pred)} / 5 {stars}</h3>", unsafe_allow_html=True)
            except:
                st.error(f"Invalid prediction: {pred}")

# streamlit run streamlit_app.py