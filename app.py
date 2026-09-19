import streamlit as st
import numpy as np
import soundfile as sf
import io
import torch
import torch.nn.functional as F
from model import load_model, fix_audio_length, audio_to_mel, LABELS

st.set_page_config(page_title="Language Voice Classifier", page_icon="🎙️", layout="centered")

LANG_INFO = {
    "English": {"flag": "🇬🇧", "country": "the United Kingdom", "grad": "linear-gradient(135deg, #12145c, #b91d47, #ffffff)"},
    "Spanish": {"flag": "🇪🇸", "country": "Spain", "grad": "linear-gradient(135deg, #aa151b, #f1bf00)"},
    "French":  {"flag": "🇫🇷", "country": "France", "grad": "linear-gradient(135deg, #0055a4, #ffffff, #ef4135)"},
    "Arabic":  {"flag": "🇪🇬", "country": "Egypt", "grad": "linear-gradient(135deg, #ce1126, #000000, #c09a2e)"},
}

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&display=swap');

html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }

.stApp {
    background: radial-gradient(circle at 20% 20%, #1a1a2e 0%, #0d0d17 50%, #050508 100%);
    color: #f0f0f5;
}

#MainMenu, footer, header { visibility: hidden; }

.hero {
    text-align: center;
    padding: 40px 0 10px 0;
}
.hero h1 {
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(90deg, #7f5af0, #2cb67d, #7f5af0);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shine 4s linear infinite;
    margin-bottom: 4px;
}
@keyframes shine {
    to { background-position: 200% center; }
}
.hero p {
    color: #9a9ab0;
    font-size: 0.95rem;
}

.credit {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 10px;
    margin-top: 10px;
    margin-bottom: 30px;
}
.credit-text {
    color: #b0b0c5;
    font-size: 0.85rem;
}
.ig-btn {
    background: linear-gradient(135deg, #f58529, #dd2a7b, #8134af, #515bd4);
    color: white !important;
    padding: 6px 16px;
    border-radius: 20px;
    text-decoration: none !important;
    font-size: 0.8rem;
    font-weight: 500;
    box-shadow: 0 4px 15px rgba(221,42,123,0.4);
    transition: transform 0.2s ease;
}
.ig-btn:hover { transform: scale(1.05); }

.glass-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 24px;
    padding: 30px;
    backdrop-filter: blur(10px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    margin-bottom: 20px;
}

.result-card {
    border-radius: 28px;
    padding: 50px 30px;
    text-align: center;
    animation: fadeInUp 0.7s ease;
    box-shadow: 0 20px 60px rgba(0,0,0,0.5);
    margin-top: 10px;
}
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(25px); }
    to { opacity: 1; transform: translateY(0); }
}
.result-flag { font-size: 5rem; }
.result-title { font-size: 1.8rem; font-weight: 700; color: white; margin-top: 10px; }
.result-sub { font-size: 1rem; color: rgba(255,255,255,0.85); margin-top: 4px; }

.conf-bar-bg {
    background: rgba(255,255,255,0.15);
    border-radius: 10px;
    height: 8px;
    width: 100%;
    margin-top: 20px;
    overflow: hidden;
}
.conf-bar-fill {
    background: white;
    height: 100%;
    border-radius: 10px;
    animation: growBar 1s ease;
}
@keyframes growBar {
    from { width: 0%; }
}
.prob-row {
    display: flex;
    justify-content: space-between;
    font-size: 0.85rem;
    color: #9a9ab0;
    margin-top: 6px;
}
.prob-bar-bg {
    background: rgba(255,255,255,0.08);
    border-radius: 8px;
    height: 6px;
    width: 100%;
    margin-top: 4px;
    margin-bottom: 10px;
}
.prob-bar-fill {
    background: linear-gradient(90deg, #7f5af0, #2cb67d);
    height: 100%;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>🎙️ Language Voice Classifier</h1>
    <p>Speak a few seconds. I'll guess the language.</p>
</div>
<div class="credit">
    <span class="credit-text">Made by Yasser</span>
    <a class="ig-btn" href="https://instagram.com/m1deey" target="_blank">📸 @m1deey</a>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def get_model():
    return load_model("best_model.pt")

model = get_model()

st.markdown('<div class="glass-card">', unsafe_allow_html=True)
audio_value = st.audio_input("🎤 Record your voice")
st.markdown('</div>', unsafe_allow_html=True)

if audio_value is not None:
    audio_bytes = audio_value.read()
    audio_array, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")

    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=1)

    if sr != 16000:
        import librosa
        audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)

    fixed = fix_audio_length(audio_array)
    mel = audio_to_mel(fixed).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        out = model(mel)
        probs = F.softmax(out, dim=1)[0]
    idx = probs.argmax().item()
    label = LABELS[idx]
    confidence = probs[idx].item()
    info = LANG_INFO[label]

    st.markdown(f"""
    <div class="result-card" style="background: {info['grad']};">
        <div class="result-flag">{info['flag']}</div>
        <div class="result-title">You sound like you're from {info['country']}!</div>
        <div class="result-sub">Detected language: {label} — {confidence:.0%} confidence</div>
        <div class="conf-bar-bg"><div class="conf-bar-fill" style="width:{confidence*100}%;"></div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("**All predictions**")
    for lang, p in zip(LABELS, probs):
        pct = p.item() * 100
        st.markdown(f"""
        <div class="prob-row"><span>{LANG_INFO[lang]['flag']} {lang}</span><span>{pct:.1f}%</span></div>
        <div class="prob-bar-bg"><div class="prob-bar-fill" style="width:{pct}%;"></div></div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
