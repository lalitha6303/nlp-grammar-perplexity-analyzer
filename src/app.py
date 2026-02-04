import streamlit as st
import os

from tokenizer import tokenize
from spell_checker import load_dictionary, correct_spelling
from grammar_candidates import generate_candidates
from language_model import train_lm, perplexity
from pos_tagger import pos_tag

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="NLP Grammar & Perplexity Analyzer",
    layout="wide"
)

# -------------------------------------------------
# PROFESSIONAL STYLING (SUBTLE & ACADEMIC)
# -------------------------------------------------
st.markdown("""
<style>
.block-container {
    max-width: 1100px;
    padding-top: 2rem;
}

h1, h2, h3 {
    color: #1f3a5f;
    font-weight: 600;
}

pre {
    background-color: #f1f3f6 !important;
    border-radius: 8px;
    border: 1px solid #e1e5eb;
}

div[data-testid="stSuccess"] {
    background-color: #e8f5ee !important;
    border-left: 6px solid #2e7d32;
}

div[data-testid="stInfo"] {
    background-color: #eaf2fb !important;
    border-left: 6px solid #1565c0;
}

div[data-testid="stWarning"] {
    background-color: #fff8e1 !important;
    border-left: 6px solid #f9a825;
}

div[data-testid="stExpander"] {
    border-radius: 10px;
    border: 1px solid #dde3ea;
    background-color: #ffffff;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# TITLE
# -------------------------------------------------
st.markdown("## NLP Grammar & Perplexity Analyzer")
st.markdown("Statistical Language Modeling · Grammar Correction · PoS Tagging")
st.markdown("---")

# -------------------------------------------------
# LOAD CORPUS
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = os.path.join(BASE_DIR, "data", "brown_corpus.txt")

unigram, bigram = train_lm(CORPUS)
vocab = load_dictionary(CORPUS)

# -------------------------------------------------
# INPUT
# -------------------------------------------------
col_input, col_btn = st.columns([4, 1])

with col_input:
    sentence = st.text_input(
        "Enter an English sentence",
        placeholder="they is learning langauge processing"
    )

with col_btn:
    analyze = st.button("Analyze")

# -------------------------------------------------
# PROCESS
# -------------------------------------------------
if analyze and sentence.strip():

    original_tokens = tokenize(sentence.lower())

    # Spelling correction
    corrected_tokens = correct_spelling(original_tokens, vocab)

    # Grammar candidates
    candidates = generate_candidates(corrected_tokens)

    # Best sentence using perplexity
    best_sentence = min(
        candidates,
        key=lambda s: perplexity(s, unigram, bigram)
    )

    # ---------------- RESULTS ----------------
    st.markdown("---")
    st.markdown("### Results")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Original Sentence**")
        st.code(sentence)

    with col2:
        st.markdown("**Corrected Sentence**")
        st.success(best_sentence)

    # ---------------- PERPLEXITY + POS ----------------
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("**Perplexity Scores**")
        for s in candidates:
            st.write(f"{round(perplexity(s, unigram, bigram), 3)} → {s}")

    with col4:
        st.markdown("**Part-of-Speech Tags**")
        for w, t in pos_tag(best_sentence):
            st.write(f"{w} → `{t}`")

    # ---------------- DYNAMIC EXPLANATION ----------------
    with st.expander("Why this output is correct"):

        # ---- Perplexity explanation (always valid) ----
        st.info("""
**Perplexity Behavior**

❌ Less natural or ungrammatical sentence → **Higher perplexity**  
✅ More natural and grammatical sentence → **Lower perplexity**

Perplexity measures how likely a sentence is according to the trained language model.
""")

        # ---- Spelling explanation (dynamic) ----
        spelling_changes = []
        for o, c in zip(original_tokens, corrected_tokens):
            if o != c:
                spelling_changes.append((o, c))

        st.success("**Spelling Correction**")
        if spelling_changes:
            for o, c in spelling_changes:
                st.write(f"- `{o}` → `{c}` (Minimum Edit Distance)")
        else:
            st.write("- No spelling errors detected")

        # ---- Grammar explanation (dynamic) ----
        st.warning("**Grammar Correction**")
        if " ".join(corrected_tokens) != best_sentence:
            st.write(f"- Sentence adjusted to: `{best_sentence}`")
            st.write("- Corrected using subject–verb agreement")
            st.write("- Best sentence selected using statistical language modeling")
        else:
            st.write("- No grammatical correction required")

        # ---- Conclusion ----
        st.markdown("""
**Conclusion**

The system generates multiple sentence candidates and selects the most probable one
using an **N-gram Language Model and Perplexity**, which is the classical approach
used in traditional NLP systems.
""")

else:
    st.caption("Enter a sentence and click **Analyze** to see the output")
