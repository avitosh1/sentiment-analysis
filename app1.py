import os
import pathlib
import platform
import streamlit as st
import pandas as pd

# ------------------------------
# Windows Fix for FastAI Exports
# ------------------------------
if platform.system() == "Windows":
    pathlib.PosixPath = pathlib.WindowsPath

from fastai.text.all import load_learner

# ------------------------------
# Page Config
# ------------------------------
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="🎬",
    layout="wide"
)

# ------------------------------
# Metrics
# ------------------------------
comparison_df = pd.DataFrame({
    "Model": ["Custom LSTM", "AWD-LSTM", "BERT"],
    "Accuracy": [0.8512, 0.8948, 0.8705],
    "Precision": [0.8641, 0.8819, 0.8299],
    "Recall": [0.8335, 0.9057, 0.9320],
    "F1 Score": [0.8485, 0.8937, 0.8780]
})

# ------------------------------
# Model Finder
# ------------------------------
def find_model():
    for root, dirs, files in os.walk("."):
        for file in files:
            if file == "awd_lstm_classifier.pkl":
                return os.path.join(root, file)
    return None

@st.cache_resource
def load_model():
    model_path = find_model()
    if model_path is None:
        return None
    return load_learner(model_path)

# ------------------------------
# Sidebar
# ------------------------------
page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Sentiment Predictor",
        "Model Performance",
        "Model Comparison",
        "Confusion Matrices",
        "Classification Reports",
        "About Project"
    ]
)

# ------------------------------
# Home
# ------------------------------
if page == "Home":
    st.title("🎬 Sentiment Analysis Using Deep Learning")
    st.markdown("""
    ### Dataset
    IMDb Movie Reviews

    ### Models
    - Custom LSTM
    - AWD-LSTM (ULMFiT)
    - BERT

    ### Objective
    Compare deep learning models for sentiment classification.
    """)

# ------------------------------
# Predictor
# ------------------------------
elif page == "Sentiment Predictor":

    st.title("🎯 Sentiment Predictor")

    try:
        learn = load_model()

        if learn is None:
            st.error("awd_lstm_classifier.pkl not found.")
        else:
            st.success("Model Loaded Successfully")

            review = st.text_area(
                "Enter Movie Review",
                height=200
            )

            if st.button("Analyze Sentiment"):

                if review.strip() == "":
                    st.warning("Please enter a review.")
                else:

                    pred_class, pred_idx, probs = learn.predict(review)

                    confidence = float(probs[pred_idx]) * 100

                    if int(pred_class) == 1:
                        st.success("😊 Positive Review")
                    else:
                        st.error("😞 Negative Review")

                    st.metric(
                        "Confidence",
                        f"{confidence:.2f}%"
                    )

                    st.write({
                        "Negative": round(float(probs[0]), 4),
                        "Positive": round(float(probs[1]), 4)
                    })

    except Exception as e:
        st.error(f"Model Loading Error:\n{e}")

# ------------------------------
# Performance
# ------------------------------
elif page == "Model Performance":

    st.title("📊 Model Performance")

    st.dataframe(
        comparison_df,
        use_container_width=True
    )

# ------------------------------
# Comparison
# ------------------------------
elif page == "Model Comparison":

    st.title("📈 Model Comparison")

    st.dataframe(
        comparison_df,
        use_container_width=True
    )

    st.subheader("Accuracy")
    st.bar_chart(
        comparison_df.set_index("Model")["Accuracy"]
    )

    st.subheader("Precision")
    st.bar_chart(
        comparison_df.set_index("Model")["Precision"]
    )

    st.subheader("Recall")
    st.bar_chart(
        comparison_df.set_index("Model")["Recall"]
    )

    st.subheader("F1 Score")
    st.bar_chart(
        comparison_df.set_index("Model")["F1 Score"]
    )

# ------------------------------
# Confusion Matrices
# ------------------------------
elif page == "Confusion Matrices":

    st.title("🔍 Confusion Matrices")

    matrices = {
        "Custom LSTM":
            "results/confusion_matrix.png",

        "AWD-LSTM":
            "awd_lstm_results/awd_lstm_confusion_matrix.png",

        "DistilBERT":
            "results/distilbert_confusion_matrix.png"
    }

    for model, file in matrices.items():

        st.subheader(model)

        if os.path.exists(file):

            st.image( file, 
                     width=500)

        else:

            st.warning(
                f"{file} not found"
            )
# ------------------------------
# Classification Reports
# ------------------------------
elif page == "Classification Reports":

    st.title("📋 Classification Reports")

    reports = {
        "Custom LSTM":
            "results/classification_report.txt",

        "AWD-LSTM":
            "awd_lstm_results/awd_lstm_classification_report.txt",

        "DistilBERT":
            "results/distilbert_classification_report.txt"
    }

    for model, report_file in reports.items():

        st.subheader(model)

        if os.path.exists(report_file):

            with open(
                report_file,
                "r",
                encoding="utf-8"
            ) as f:

                st.text(
                    f.read()
                )

        else:

            st.warning(
                f"{report_file} not found"
            )
# ------------------------------
# About
# ------------------------------
elif page == "About Project":

    st.title("ℹ️ About Project")

    st.markdown("""
    ### Sentiment Analysis Using Deep Learning

    Models Compared:
    - Custom LSTM
    - AWD-LSTM (ULMFiT)
    - BERT

    Dataset:
    IMDb Movie Reviews

    Metrics:
    - Accuracy
    - Precision
    - Recall
    - F1 Score

    Built using:
    - Streamlit
    - FastAI
    - PyTorch
    - Transformers
    """)
