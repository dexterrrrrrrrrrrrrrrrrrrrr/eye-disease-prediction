# 🤖 AI Retinal Analyzer - Eye Disease Prediction

[![Streamlit App](https://img.shields.io/badge/Streamlit-LIVE-brightgreen)](https://eye-disease-prediction-ffdexterrrrrrrrrrrrr.streamlit.app/)
[![GitHub Repo](https://img.shields.io/badge/GitHub-Repo-blue?logo=github)](https://github.com/dexterrrrrrrrrrrrrrrrrrrrr/eye-disease-prediction)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

## ✨ Features
- **Auto Eye Detection** 👁️ (Left/Right - Optic disc rule)
- **Retinal Analysis** (Optic disc, vessels, macula)
- **Gemini AI Reports** 🤖 (Medical insights)
- **Dark Modern UI**
- **Demo Images** (Normal/Myopic/Severe)

## 🚀 Quick Start
```bash
git clone https://github.com/dexterrrrrrrrrrrrrrrrrrrrr/eye-disease-prediction.git
cd skin-eye-disease-prediction
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## 🔑 Gemini AI Setup
1. Copy `.env.example → .env`
2. Add key: `GEMINI_API_KEY=AIza...`
3. Refresh app

## 📁 Structure
```
├── app.py           # Main Streamlit app
├── requirements.txt # Dependencies
├── .env.example     # API key template
├── README.md        # This file
└── .gitignore       # venv ignored
```

## 🎯 Medical Features
- Optic disc sizing (1.5-2.0mm normal)
- Blood vessel density/tortuosity
- Macula health analysis
- Myopia severity scoring
- AI-powered second opinions

## 🛠️ Tech Stack
- **Frontend**: Streamlit 1.28+
- **AI**: Google Gemini  
- **CV**: OpenCV 4.8+
- **Data**: Pandas/Plotly

## 📈 Demo Results
| Feature | Status |
|---------|--------|
| Eye Detection | ✅ Auto |
| Reports | ✅ PDF-ready |
| AI Analysis | 🔑 Key needed |
| Dark UI | ✅ |

## 🤝 Contributing
1. Fork repo
2. Create feature branch
3. PR to `main`

## ⚠️ Disclaimer
Educational tool only. Consult ophthalmologist.



