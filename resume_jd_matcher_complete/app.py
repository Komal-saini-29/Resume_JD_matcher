import re
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Resume–JD Matcher", page_icon="📄", layout="wide")
st.title("📄 Resume–Job Description Matcher")
st.caption("NLP-based candidate-to-job matching using TF-IDF and cosine similarity.")

SKILL_PATTERNS = {
    "Python": [r"\bpython\b"], "Machine Learning": [r"\bmachine learning\b"],
    "Deep Learning": [r"\bdeep learning\b"], "NLP": [r"\bnlp\b", r"\bnatural language processing\b"],
    "PyTorch": [r"\bpytorch\b"], "TensorFlow": [r"\btensorflow\b"],
    "LLMs": [r"\bllms?\b", r"\blarge language models?\b"],
    "RAG": [r"\brag\b", r"\bretrieval[- ]augmented generation\b"],
    "Transformers": [r"\btransformers?\b"], "Embeddings": [r"\bembeddings?\b"],
    "Git": [r"\bgit\b"], "GitHub": [r"\bgithub\b"],
    "Scikit-learn": [r"\bscikit[- ]learn\b"], "NumPy": [r"\bnumpy\b"],
    "Pandas": [r"\bpandas\b"], "SQL": [r"\bsql\b"],
    "Computer Vision": [r"\bcomputer vision\b"], "Speech Processing": [r"\bspeech processing\b", r"\baudio\b"]
}

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s+#.-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def extract_skills(text):
    text = text.lower()
    return [skill for skill, patterns in SKILL_PATTERNS.items()
            if any(re.search(pattern, text) for pattern in patterns)]

def similarity_score(resume, jd):
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform([clean_text(resume), clean_text(jd)])
    return cosine_similarity(matrix[0:1], matrix[1:2])[0][0]

col1, col2 = st.columns(2)
with col1:
    st.subheader("Candidate Resume")
    resume = st.text_area("Paste resume text", height=360, placeholder="I'll paste my resume here...")
with col2:
    st.subheader("Job Description")
    jd = st.text_area("Paste job description", height=360, placeholder="I'll paste the job description here...")

if st.button("🔍 Analyze Match", type="primary", use_container_width=True):
    if not resume.strip() or not jd.strip():
        st.error("Please paste both a resume and a job description.")
    else:
        score = similarity_score(resume, jd)
        resume_skills, jd_skills = set(extract_skills(resume)), set(extract_skills(jd))
        matched = sorted(resume_skills & jd_skills)
        missing = sorted(jd_skills - resume_skills)
        st.divider(); st.subheader("Match Analysis")
        m1, m2, m3 = st.columns(3)
        m1.metric("Text Similarity", f"{score * 100:.1f}%")
        m2.metric("Matched Skills", len(matched))
        m3.metric("Potential Skill Gaps", len(missing))
        st.progress(float(score))
        left, right = st.columns(2)
        with left:
            st.markdown("### ✅ Skills found in both")
            st.write(" • ".join(matched) if matched else "No predefined skills matched.")
        with right:
            st.markdown("### 📌 Skills in JD but not detected in resume")
            st.write(" • ".join(missing) if missing else "No skill gaps detected from the current skill dictionary.")
        st.divider(); st.subheader("How to interpret the result")
        st.write("The score is a textual-alignment signal from TF-IDF and cosine similarity, not a hiring decision or a measure of candidate quality.")

st.sidebar.header("About this project")
st.sidebar.write("Demonstrates text preprocessing, TF-IDF vectorization, cosine similarity, rule-based skill extraction, and an interactive Streamlit interface.")
