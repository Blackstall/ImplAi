# 🔧 Intelligent Customization Implant — Streamlit App

A modern, **shadcn-inspired** Streamlit interface for implant size prediction using machine-learning models.  
This project features a fully redesigned 50/50 split-screen layout, clean typography, animated gradients, and a professional workflow for patient-data input.

Perfect for researchers, engineers, or clinicians who want a fast, elegant interface for experimentation and prediction.

---

## 🎨 Features

✅ Modern UI inspired by **shadcn**, **Inter**, **Playfair**, and minimal grid layouts  
✅ Clean split-screen design (Coordinator panel + Prediction form)  
✅ Auto-calculated BMI + structured clinical inputs  
✅ TensorFlow model loading with graceful error handling  
✅ Scaler + label encoder support  
✅ Animated background gradient  
✅ Fully customizable CSS layer  

---

## 🚀 Demo

**Private Hosted**

---

## 🧠 Tech Stack

- **Streamlit** — UI and layout  
- **TensorFlow / Keras** — implant prediction model  
- **scikit-learn / joblib** — scaler + label encoder  
- **NumPy** — data transformations  
- **Custom CSS** — animations, typography, gradients  

---

## 📦 Installation

### 1️⃣ Install dependencies

```bash
pip install -r requirements.txt


streamlit run train_model.py


project/
│
├── train_model.py          # Main Streamlit UI + prediction pipeline
├── combined_model.h5       # Trained TensorFlow model (add manually)
├── scaler.pkl              # Feature scaler
├── label_encoder_size.pkl  # Label encoder for implant size
└── requirements.txt        # Dependencies
