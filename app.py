import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from datetime import datetime
import random

# ---------------------------------------------------
# SAMPLE EMAIL GENERATOR
# ---------------------------------------------------
examples = [
    ("Order not received",
     "I placed an order last week but still haven't received my package. Please check the status immediately."),

    ("Great service",
     "I really liked your product quality and fast delivery. Keep up the good work!"),

    ("Password reset help",
     "I am unable to reset my password and cannot login into my account. Please assist me."),

    ("Limited time offer!!!",
     "Congratulations! You have won a gift voucher. Click this link now to claim your reward."),

    ("Feature suggestion",
     "It would be great if your app supports dark mode in the next update.")
]

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(page_title="AI Email Triage System", layout="wide")

# ---------------- TITLE COLOR STYLE ----------------
st.markdown("""
<style>
h1 {color:#4CAF50;}
h2 {color:#03A9F4;}
h3 {color:#FFC107;}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("email_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

# ---------------------------------------------------
# PRIORITY LOGIC
# ---------------------------------------------------
def get_priority(label):
    label = label.lower()
    if label in ["complaint", "support"]:
        return "High", "red"
    elif label == "feedback":
        return "Medium", "orange"
    else:
        return "Low", "green"

# ---------------------------------------------------
# SESSION STORAGE
# ---------------------------------------------------
if "emails" not in st.session_state:
    st.session_state.emails = []

# ---------------------------------------------------
# SIDEBAR NAVIGATION (NO DROPDOWN)
# ---------------------------------------------------
st.sidebar.title("🎯 Navigation")

st.sidebar.subheader("🔎 Search")
search_query = st.sidebar.text_input("Search emails...")

if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

if st.sidebar.button("🏠 Dashboard"):
    st.session_state.page = "Dashboard"
if st.sidebar.button("📨 Classify Email"):
    st.session_state.page = "Classify Email"
if st.sidebar.button("📈 Analytics"):
    st.session_state.page = "Analytics"
if st.sidebar.button("🧠 Model Info"):
    st.session_state.page = "Model Info"

page = st.session_state.page

# ===================================================
# DASHBOARD
# ===================================================
if page == "Dashboard":

    st.title("📊 Email Classification Dashboard")

    df = pd.DataFrame(st.session_state.emails)

    total = len(df)
    spam = len(df[df["label"]=="spam"]) if total>0 else 0
    complaint = len(df[df["label"]=="complaint"]) if total>0 else 0
    confidence = round(df["confidence"].mean(),2) if total>0 else 0

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Emails", total)
    c2.metric("Complaints", complaint)
    c3.metric("Avg Confidence", f"{confidence}%")
    c4.metric("Spam Blocked", spam)

    if total>0:
        high=len(df[df["priority"]=="High"])
        medium=len(df[df["priority"]=="Medium"])
        low=len(df[df["priority"]=="Low"])

        st.subheader("📌 Priority Overview")
        pc1,pc2,pc3=st.columns(3)
        pc1.metric("High 🔴",high)
        pc2.metric("Medium 🟡",medium)
        pc3.metric("Low 🟢",low)

        st.divider()

        colA, colB = st.columns(2)

        with colA:
            dist = df["label"].value_counts().reset_index()
            dist.columns=["Category","Count"]
            fig=px.pie(dist,names="Category",values="Count",title="Email Category Distribution")
            st.plotly_chart(fig,use_container_width=True)

        with colB:
            df["date"]=df["time"].dt.date
            trend=df.groupby("date").size().reset_index(name="Emails")
            fig2=px.line(trend,x="date",y="Emails",title="Daily Incoming Emails")
            st.plotly_chart(fig2,use_container_width=True)

    else:
        st.info("No emails classified yet.")

# ===================================================
# CLASSIFY EMAIL
# ===================================================
elif page=="Classify Email":

    st.title("📨 Classify New Email")

    if "subject_text" not in st.session_state:
        st.session_state.subject_text = ""
    if "content_text" not in st.session_state:
        st.session_state.content_text = ""

    # INPUT FIELDS FIRST
    subject = st.text_input("Subject", value=st.session_state.subject_text)
    content = st.text_area("Email Content", height=220, value=st.session_state.content_text)

    # BUTTONS BELOW CONTENT
    colA, spacer, colB = st.columns([1,1,1])

    with colA:
        classify_clicked = st.button("🔍 Classify")

    with colB:
        if st.button("🎲 Load Example"):
            sample = random.choice(examples)
            st.session_state.subject_text = sample[0]
            st.session_state.content_text = sample[1]
            st.rerun()

    if classify_clicked:
        if content.strip()=="":
            st.warning("Enter email content")
        else:
            text=subject+" "+content
            vec=vectorizer.transform([text])
            pred=model.predict(vec)[0]

            try:
                prob=max(model.predict_proba(vec)[0])*100
            except:
                prob=90

            priority, color = get_priority(pred)

            st.success(f"Prediction: {pred}")
            st.markdown(f"### Priority: :{color}[{priority}]")
            st.info(f"Confidence: {prob:.2f}%")

            st.session_state.emails.append({
                "time":datetime.now(),
                "label":str(pred).lower(),
                "confidence":prob,
                "priority":priority
            })

    st.subheader("💡 Classification Guide")

    with st.expander("😡 Complaint — 🔴 High Priority"):
        st.write("Customer dissatisfaction requiring urgent attention")

    with st.expander("🛠 Support — 🔴 High Priority"):
        st.write("Technical help or service request")

    with st.expander("💬 Feedback — 🟡 Medium Priority"):
        st.write("Suggestions and opinions")

    with st.expander("🚫 Spam — 🟢 Low Priority"):
        st.write("Promotional or unwanted content")

    with st.expander("📄 Other — 🟢 Low Priority"):
        st.write("General communication")

# ===================================================
# ANALYTICS
# ===================================================
elif page=="Analytics":

    st.title("📈 Advanced Analytics")

    df=pd.DataFrame(st.session_state.emails)

    if len(df)>0:
        df["date"]=df["time"].dt.date
        trend=df.groupby("date").size().reset_index(name="Emails")
        fig=px.line(trend,x="date",y="Emails")
        st.plotly_chart(fig,use_container_width=True)

# ===================================================
# MODEL INFO
# ===================================================
elif page=="Model Info":

    st.title("🧠 Model Information")

    st.subheader("About the Email Classification System")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### ⚙ Model Architecture")
        st.write("Algorithm: Logistic Regression")
        st.write("Vectorization: TF-IDF")
        st.write("Max Iterations: 300")
        st.write("Type: Supervised Machine Learning - Classification")
        st.write("Purpose: Multi-class email categorization")

    with col2:
        st.markdown("### 📊 Dataset Information")
        st.write("Classes: Complaint, Feedback, Other, Spam, Support")
        st.write("Split: 80% Training / 20% Testing")
        st.write("Preprocessing: Lowercase, punctuation removal, stopwords removal")

    st.divider()

    st.subheader("📈 Performance Metrics")
    st.success("Model Accuracy: ~99%")
    st.write("TF-IDF + Logistic Regression classifier")

    st.divider()

    st.subheader("🔄 How It Works")
    st.write("""
    1. User enters email text
    2. Text preprocessing applied
    3. TF-IDF converts text into numeric vectors
    4. Logistic Regression predicts category
    5. Priority assigned automatically
    6. Dashboard updates analytics in real-time
    """)

    st.subheader("Algorithm Used")
    st.write("Logistic Regression with TF-IDF vectorization")

    st.subheader("Why Logistic Regression (Model Comparison)?")

    comparison = pd.DataFrame({
    "Criteria": [
        "Training Speed",
        "High-Dimensional Sparse Text",
        "Probability Estimates",
        "Memory Usage",
        "Real-time Suitability"
    ],

    "Logistic Regression": [
        "Fast",
        "Very suitable",
        "Well calibrated",
        "Low",
        "Highly suitable"
    ],

    "Linear SVM": [
        "Moderate",
        "Very suitable",
        "Requires extra calibration",
        "Medium",
        "Suitable"
    ],

    "Naive Bayes": [
        "Very fast",
        "Suitable but simplistic assumptions",
        "Less reliable probabilities",
        "Very low",
        "Very suitable but lower accuracy"
    ],

    "Random Forest": [
        "Slow on text data",
        "Not optimal for sparse features",
        "Moderate quality",
        "High",
        "Not ideal for real-time"
    ]
})


    st.dataframe(comparison, use_container_width=True)
