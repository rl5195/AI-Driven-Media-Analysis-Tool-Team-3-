# AI-Driven Media Analysis Tool - Team 3

This platform provides a unified, AI-powered solution for analyzing multimedia content such as audio, video, and text. It is designed to support multilingual inputs and deliver automated insights across several modalities.

## 🧠 Key Features

- **🎙️ Multilingual Audio Transcription & Translation**  
  Upload an MP4, MP3, or WAV file. The system extracts audio and transcribes speech using Whisper. It detects the language and translates non-English input into English for further analysis.

- **📝 NLP-Based Content Analysis**  
  The transcript is classified into four categories using a zero-shot classifier:
  - `biased language`
  - `misinformation`
  - `xenophobic statement`
  - `neutral`  
  Each classification returns a score (0–1) indicating the likelihood of the label.

- **🌍 Xenophobia Detection Module**  
  This dedicated module flags content that may include xenophobic or biased remarks, with confidence scores for each.

- **🎥 Video Frame Classification**  
  The system extracts video frames at regular intervals and analyzes them using ResNet to identify visual context. Each frame is labeled with a predicted class and confidence score.

- **🔍 OCR Text Detection**  
  Text embedded in video frames (e.g., captions, banners) is detected using EasyOCR. Extracted text is translated into English and fed into the same NLP pipeline for classification.

- **📄 PDF Report Generation**  
  Users can export a detailed PDF report summarizing all components: audio analysis, transcript, translation, NLP results, frame analysis, and OCR findings.

## 🚀 How to Use

1. Upload your audio/video file using the sidebar uploader.
2. Wait for automatic transcription, translation, analysis, and visualization.
3. Navigate tabs (`Audio`, `Transcript`, `Xenophobia`, `Video`, `OCR`) to review each module’s output.
4. Click “Generate PDF Report” to export a summary.

## 🛠️ Built With

- Python
- Streamlit
- HuggingFace Transformers (`facebook/bart-large-mnli`, `MarianMT`)
- OpenAI Whisper
- torchvision / ResNet
- EasyOCR
- moviepy / librosa / scipy

## 👥 Team 3

Project developed as part of the NYU UNICC Capstone 2025.

# AI-Driven-Media-Analysis-Tool-Team-3-
