# NLP Grammar & Perplexity Analyzer

This project is a **Streamlit-based NLP application** that analyzes a given sentence for **spelling errors**, **grammatical correctness**, and selects the **most probable corrected sentence** using a **statistical N-gram language model with perplexity**.

It demonstrates a **classical NLP approach** without using deep learning models.

---

## 🔍 Features

- Spell checking using edit distance
- Grammar correction using candidate sentence generation
- Part-of-Speech (POS) tagging
- N-gram Language Model
- Sentence selection using **Perplexity**
- Interactive **Streamlit web interface**

---

## 🛠️ Technologies Used

- Python
- Streamlit
- NLP concepts (Tokenization, POS tagging)
- N-gram Language Model
- Perplexity scoring
- Brown Corpus (for language modeling)

---

## 📁 Project Structure
NLP/
│
├── src/
│ ├── app.py # Streamlit app
│ ├── spell_checker.py
│ ├── grammar_candidates.py
│ ├── language_model.py
│ ├── pos_tagger.py
│ ├── tokenizer.py
│ └── main.py
│
├── data/
│ └── brown_corpus.txt
│
├── create_corpus.py
├── requirements.txt
└── README.md

---

## ▶️ How to Run Locally

### 1. Install dependencies

pip install -r requirements.txt

2. Run the Streamlit app
streamlit run src/app.py

3. Open in browser
http://localhost:8501

How It Works :

User enters a sentence

Spell checker detects possible spelling mistakes

Grammar module generates multiple corrected sentence candidates

Each candidate is evaluated using an N-gram Language Model

Perplexity is calculated for each sentence

The sentence with the lowest perplexity is selected as the best correction

Output is displayed in the Streamlit interface
