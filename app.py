import streamlit as st
import pickle
import PyPDF2
import docx
import re

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Resume Classifier", layout="wide")

# -----------------------------
# CUSTOM CSS (DARK UI)
# -----------------------------
st.markdown("""
<style>
body {
    background-color: #0b1a33;
}
.big-title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
    color: #ffcc00;
}
.subtitle {
    text-align: center;
    color: #cccccc;
}
.card {
    background-color: #132347;
    padding: 20px;
    border-radius: 10px;
}
.prediction-box {
    background-color: #ffcc00;
    color: black;
    padding: 20px;
    border-radius: 10px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD MODEL
# -----------------------------
model, tfidf = pickle.load(open("model.pkl", "rb"))

# -----------------------------
# TITLE
# -----------------------------
st.markdown('<div class="big-title">RESUME CLASSIFIER</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Analyze your resume with Machine Learning & NLP</div>', unsafe_allow_html=True)

# -----------------------------
# FUNCTIONS
# -----------------------------
def read_pdf(file):
    text = ""
    reader = PyPDF2.PdfReader(file)
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def read_docx(file):
    doc = docx.Document(file)
    return " ".join([para.text for para in doc.paragraphs])

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# Simple skill keywords (you can expand)
skills_db = ["sql", "python", "java", "aws", "etl", "react", "html", "css", "oracle"]

# -----------------------------
# LAYOUT (2 COLUMNS)
# -----------------------------
col1, col2 = st.columns(2)

# LEFT SIDE - Upload
with col1:
    st.subheader("Upload Resume")
    uploaded_file = st.file_uploader("Upload your resume")

# RIGHT SIDE - Prediction
with col2:
    st.subheader("Prediction")

# -----------------------------
# PROCESS FILE
# -----------------------------
if uploaded_file is not None:

    if uploaded_file.name.endswith(".pdf"):
        text = read_pdf(uploaded_file)
    else:
        text = read_docx(uploaded_file)

    clean = clean_text(text)

    vector = tfidf.transform([clean])
    prediction = model.predict(vector)[0]
    probs = model.predict_proba(vector)[0]

    confidence = max(probs) * 100

    # Prediction UI
    col2.markdown(f"""
    <div class="prediction-box">
        <h2>{prediction}</h2>
        <p>Confidence: {confidence:.2f}%</p>
        <p>Resume Match Score: {int(confidence*2.5)}%</p>
    </div>
    """, unsafe_allow_html=True)

    # -----------------------------
    # RESUME PREVIEW
    # -----------------------------
    st.subheader("Resume Preview")
    st.text_area("", text[:1500], height=200)

    # -----------------------------
    # SKILL DETECTION
    # -----------------------------
    detected = [skill for skill in skills_db if skill in clean]
    missing = [skill for skill in skills_db if skill not in clean]

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Detected Skills")
        for skill in detected:
            st.write(f"• {skill}")

    with col4:
        st.subheader("Missing Skills")
        for skill in missing[:5]:
            st.write(f"• {skill}")

    # -----------------------------
    # SUGGESTION
    # -----------------------------
    st.subheader("Suggestion")

    if confidence > 70:
        st.success("Your resume is strong for this role.")
    elif confidence > 40:
        st.warning("Your resume is moderately matching. Improve skills.")
    else:
        st.error("Your resume needs improvement for this role.")

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.markdown("### Features")
col5, col6, col7 = st.columns(3)

with col5:
    st.write("Model Type")
    st.subheader("ML Classifier")

with col6:
    st.write("Text Vectorizer")
    st.subheader("TF-IDF")

with col7:
    st.write("Prediction")
    st.subheader("Multi-Category")