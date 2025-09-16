import os

import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Настройки
st.set_page_config(page_title="Toxicity Detector", page_icon="🧪", layout="centered")

# Путь к модели: по умолчанию используем best из ноутбука
# Можно переопределить через переменную окружения MODEL_DIR
DEFAULT_MODEL_DIR = "notebooks/civil_comments_distilbert/best"
MODEL_DIR = os.environ.get("MODEL_DIR", DEFAULT_MODEL_DIR)
THRESHOLD = float(os.environ.get("THRESHOLD", 0.5))


@st.cache_resource(show_spinner=True)
def load_pipeline(model_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


tokenizer, model, device = load_pipeline(MODEL_DIR)

st.title("Модерация токсичных комментариев")
st.caption("DistilBERT binary toxicity classifier (HF Transformers)")

with st.sidebar:
    st.header("Настройки")
    MODEL_DIR = st.text_input("Путь к модели", MODEL_DIR)
    THRESHOLD = st.slider("Порог токсичности", 0.0, 1.0, THRESHOLD, 0.01)
    st.write("GPU:", torch.cuda.is_available())

user_text = st.text_area(
    "Введите комментарий", height=150, placeholder="Write your comment here…"
)

col1, col2 = st.columns([1, 1])
with col1:
    run = st.button("Оценить токсичность", type="primary")
with col2:
    clear = st.button("Очистить")

if clear:
    st.rerun()

if run and user_text.strip():
    with st.spinner("Считаем…"):
        enc = tokenizer(
            user_text, return_tensors="pt", truncation=True, max_length=256
        ).to(device)
        with torch.no_grad():
            logits = model(**enc).logits  # [1, 2]
            probs = F.softmax(logits, dim=-1).detach().cpu().numpy()[0]
            toxic_score = float(probs[1])
            is_toxic = toxic_score >= THRESHOLD

    st.subheader("Результат")
    st.metric("Вероятность токсичности", f"{toxic_score:.4f}")

    if is_toxic:
        st.error("Комментарий классифицирован как токсичный")
    else:
        st.success("Комментарий классифицирован как не токсичный")

    with st.expander("Технические детали"):
        st.json(
            {
                "model_dir": MODEL_DIR,
                "threshold": THRESHOLD,
                "probs": {"non_toxic": probs[0], "toxic": probs[1]},
            }
        )

st.markdown("---")
st.caption("Подсказка: можно задать MODEL_DIR и THRESHOLD как переменные окружения.")
