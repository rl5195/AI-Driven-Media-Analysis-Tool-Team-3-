
# -----------------------------------------------------------------------------
# Unified Media Analysis Platform - Enhanced Version with Dynamic Multi-language OCR Support
# -----------------------------------------------------------------------------

import whisper
import torch
import streamlit as st
from langdetect import detect
from transformers import MarianMTModel, MarianTokenizer
import re
import librosa
import scipy.fft
import cv2
import os
import shutil
from transformers import pipeline
from moviepy.editor import VideoFileClip
from torchvision import models, transforms
from fpdf import FPDF
import base64
import matplotlib.pyplot as plt
import easyocr
import langid

# Streamlit page config
st.set_page_config(page_title="Unified Media Analysis Platform", layout="wide")

def detect_language(text):
    lang, _ = langid.classify(text)
    return lang

@st.cache_resource
def load_models():
    whisper_model = whisper.load_model("base")
    text_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    vision_model = models.resnet50(pretrained=True)
    vision_model.eval()
    return whisper_model, text_classifier, vision_model

def get_ocr_reader_by_lang(lang_code):
    lang_map = {
        "en": ["en"],
        "es": ["en", "es"],
        "fr": ["en", "fr"],
        "ar": ["en", "ar"],
        "ru": ["en", "ru"],
        "zh": ["en", "ch_sim"]
    }
    langs = lang_map.get(lang_code, ["en"])
    return easyocr.Reader(langs, gpu=False)

def translate_to_english(text):
    try:
        lang = detect_language(text)
        if lang.startswith('en'):
            return text, lang
        model_name = f"Helsinki-NLP/opus-mt-{lang}-en"
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        translated = model.generate(**tokenizer(text, return_tensors='pt', padding=True))
        english_text = tokenizer.decode(translated[0], skip_special_tokens=True)
        return english_text, lang
    except Exception as e:
        print(f"exception {e}")
        return text, "und"

# Audio transcription
def transcribe_audio(file_path, model):
    result = model.transcribe(file_path)
    return result['text']

# Text analysis
def analyze_text(text, classifier):
    labels = ["misinformation", "biased language", "xenophobic statement", "neutral"]
    return classifier(text, labels)

def detect_xenophobia(text, classifier):
    labels = ["neutral", "xenophobic", "biased"]
    return classifier(text, labels)

# Audio analysis
def audio_analysis(file_path):
    audio, sr = librosa.load(file_path)
    fft = scipy.fft.fft(audio)
    fft_freq = scipy.fft.fftfreq(len(audio), 1 / sr)
    return {
        "fft": fft,
        "fft_freq": fft_freq,
        "mean": audio.mean(),
        "max": audio.max(),
        "min": audio.min(),
        "std": audio.std(),
        "emotion": "neutral"
    }

def create_pdf_report(transcript, translated_text, text_analysis_result, xenophobia_result, ocr_text, ocr_analysis_result, audio_result):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    def add_section(title, content):
        pdf.set_font("Arial", 'B', size=14)
        pdf.cell(200, 10, txt=title, ln=True)
        pdf.set_font("Arial", size=12)
        for line in content.split('\n'):
            pdf.multi_cell(0, 10, txt=line)
        pdf.ln()

    add_section("Transcription", transcript)
    add_section("Translated Text", translated_text)
    add_section("Text Analysis Result", str(text_analysis_result))
    add_section("Xenophobia Detection", str(xenophobia_result))
    add_section("OCR Extracted Text", ocr_text)
    add_section("OCR Text Analysis Result", str(ocr_analysis_result))
    add_section("Audio Features", f"Mean: {audio_result['mean']}\nMax: {audio_result['max']}\nMin: {audio_result['min']}\nStd: {audio_result['std']}")

    output_path = "media_analysis_report.pdf"
    pdf.output(output_path)
    return output_path

def download_pdf_button(file_path, label="Download PDF"):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    href = f'{label}'
    st.markdown(href, unsafe_allow_html=True)

# Frame extraction and analysis
def extract_frames(video_path, output_folder="frames", interval=1):
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)
    vidcap = cv2.VideoCapture(video_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    success, image = vidcap.read()
    count, frame_count = 0, 0
    while success:
        if count % int(fps * interval) == 0:
            frame_path = os.path.join(output_folder, f"frame{frame_count}.jpg")
            cv2.imwrite(frame_path, image)
            frame_count += 1
        success, image = vidcap.read()
        count += 1
    vidcap.release()

def analyze_frames(folder, model):
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    frame_data = []
    for filename in sorted(os.listdir(folder)):
        if filename.endswith(".jpg"):
            image = cv2.imread(os.path.join(folder, filename))
            input_tensor = preprocess(image).unsqueeze(0)
            with torch.no_grad():
                output = model(input_tensor)
            preds = torch.nn.functional.softmax(output[0], dim=0)
            top_prob, top_class = torch.max(preds, 0)
            frame_data.append((filename, top_class.item(), top_prob.item()))
    return frame_data

# OCR helpers
def clean_ocr_text(raw_text):
    return re.sub(r"[^a-zA-Z0-9一-龥\s.,:;!?%\-]", "", raw_text)

def is_frame_usable(image_path, threshold=30):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return img.mean() > threshold

def perform_ocr_on_frames(folder, reader):
    seen = set()
    texts = []
    for filename in sorted(os.listdir(folder)):
        if filename.endswith(".jpg"):
            image_path = os.path.join(folder, filename)
            if is_frame_usable(image_path):
                results = reader.readtext(image_path, detail=0)
                for line in results:
                    cleaned = clean_ocr_text(line.strip())
                    if cleaned and cleaned not in seen and len(cleaned) > 20:
                        seen.add(cleaned)
                        texts.append(cleaned)
    return " ".join(texts)

# Main logic
whisper_model, text_classifier, vision_model = load_models()

st.sidebar.header("Upload File")
uploaded_file = st.sidebar.file_uploader("Upload an Audio/Video file", type=["mp4", "wav", "mp3"])
frame_interval = st.sidebar.slider("Frame Extraction Interval (seconds)", 1, 10, 1)

if uploaded_file:
    file_path = "temp_uploaded_file"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    audio_path = file_path
    if uploaded_file.name.endswith(".mp4"):
        video = VideoFileClip(file_path)
        audio_path = "extracted_audio.wav"
        video.audio.write_audiofile(audio_path, verbose=False, logger=None)

    transcript = transcribe_audio(audio_path, whisper_model)
    translated_text, detected_language = translate_to_english(transcript)
    text_analysis_result = analyze_text(translated_text, text_classifier)
    xenophobia_result = detect_xenophobia(translated_text, text_classifier)

    ocr_text, ocr_analysis_result, frame_analysis_result = "", {}, []
    if uploaded_file.name.endswith(".mp4"):
        extract_frames(file_path, interval=frame_interval)
        frame_analysis_result = analyze_frames("frames", vision_model)
        ocr_reader = get_ocr_reader_by_lang(detected_language)
        ocr_text = perform_ocr_on_frames("frames", ocr_reader)
        translated_ocr, _ = translate_to_english(ocr_text)
        ocr_analysis_result = analyze_text(translated_ocr, text_classifier)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎵 Audio", "📝 Transcript", "🌎 Xenophobia", "🎥 Video", "🔍 OCR"])

    with tab1:
        audio_result = audio_analysis(audio_path)
        st.header("Audio Features")
        st.metric("Mean", round(audio_result["mean"], 4))
        st.metric("Max", round(audio_result["max"], 4))
        st.metric("Min", round(audio_result["min"], 4))
        st.metric("Std Dev", round(audio_result["std"], 4))
        st.metric("Emotion Prediction", audio_result["emotion"])
        fig, ax = plt.subplots()
        ax.plot(audio_result["fft_freq"][:500], abs(audio_result["fft"][:500]))
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Amplitude")
        st.pyplot(fig)

    with tab2:
        st.header("Transcription and NLP")
        st.info(f"Detected Language: {detected_language}")
        st.text_area("Original Transcript", transcript, height=200)
        st.text_area("Translated to English", translated_text, height=200)
        st.subheader("Text Analysis")
        st.json(text_analysis_result)

    with tab3:
        st.header("Xenophobia Detection")
        st.json(xenophobia_result)

    with tab4:
        st.header("Video Frame Analysis")
        if uploaded_file.name.endswith(".mp4"):
            for frame_name, cls, prob in frame_analysis_result[:10]:
                st.image(f"frames/{frame_name}", caption=f"Class: {cls}, Confidence: {prob:.2f}", width=300)

    with tab5:
        st.header("OCR Result")
        if ocr_text:
            st.text_area("Extracted Text", ocr_text, height=200)
            st.json(ocr_analysis_result)
        else:
            st.info("No OCR results available.")

    if st.button("📄 Generate PDF Report"):
        pdf_file = create_pdf_report(
            transcript,
            translated_text,
            text_analysis_result,
            xenophobia_result,
            ocr_text,
            ocr_analysis_result,
            audio_result
        )
        download_pdf_button(pdf_file, label="📥 Click to Download Report")