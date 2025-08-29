import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 🔁 Cache model so it doesn't reload on every run
@st.cache_resource(show_spinner=True)
def load_model():
    model_name = "facebook/m2m100_418M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# Common languages (add more if you like)
LANGS = {
    "English": "en",
    "Urdu": "ur",
    "Arabic": "ar",
    "French": "fr",
    "German": "de",
    "Spanish": "es",
    "Chinese (Simplified)": "zh",
    "Hindi": "hi",
    "Italian": "it",
    "Portuguese": "pt",
    "Turkish": "tr",
    "Russian": "ru",
    "Japanese": "ja",
    "Korean": "ko"
}

st.set_page_config(page_title="AI Translator", page_icon="🌍", layout="centered")
st.title("🌍 AI Translator (M2M100)")

col1, col2 = st.columns(2)
with col1:
    src_name = st.selectbox("Source language", list(LANGS.keys()), index=list(LANGS.keys()).index("English"))
with col2:
    tgt_name = st.selectbox("Target language", list(LANGS.keys()), index=list(LANGS.keys()).index("Urdu"))

src = LANGS[src_name]
tgt = LANGS[tgt_name]

text = st.text_area("Enter text to translate", height=160, placeholder="Type or paste your text here...")

def translate(text: str, src_lang: str, tgt_lang: str) -> str:
    # set source language
    tokenizer.src_lang = src_lang
    # tokenize
    encoded = tokenizer(text, return_tensors="pt", truncation=True)
    # set target language BOS token
    forced_bos_token_id = tokenizer.get_lang_id(tgt_lang)
    generated_tokens = model.generate(
        **encoded,
        forced_bos_token_id=forced_bos_token_id,
        max_new_tokens=256
    )
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

btn = st.button("Translate")
if btn and text.strip():
    with st.spinner("Translating..."):
        try:
            out = translate(text.strip(), src, tgt)
            st.subheader("Translation")
            st.write(out)
        except Exception as e:
            st.error(f"Oops, something went wrong: {e}")

st.caption("Powered by transformers • Model: facebook/m2m100_418M")
