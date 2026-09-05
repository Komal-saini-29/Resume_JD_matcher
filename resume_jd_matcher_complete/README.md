# Resume–Job Description Matcher

A beginner NLP project that compares a candidate resume with a job description using TF-IDF and cosine similarity, then reports matched skills and potential skill gaps.

## Tech stack
Python, Scikit-learn, TF-IDF, cosine similarity, regular expressions, Streamlit.

## Run
```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## How it works
1. Cleans and normalizes text.
2. Converts the resume and JD to TF-IDF vectors.
3. Calculates cosine similarity.
4. Detects selected technical skills using a rule-based dictionary.
5. Displays the results in a Streamlit web interface.

## Limitations
TF-IDF mainly captures lexical overlap; it does not fully understand semantic meaning. The skill dictionary is intentionally small, and the score should not be used as an automated hiring decision.

## Future research improvements
- Compare TF-IDF with transformer embeddings.
- Add stronger skill/entity extraction.
- Add weighted skill matching.
- Evaluate on labeled resume/JD pairs.
- Track experiments and compare retrieval/similarity methods.
