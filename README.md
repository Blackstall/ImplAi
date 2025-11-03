#🔧 Intelligent Customization Implant — Streamlit App

A modern, shadcn-inspired Streamlit interface for implant size prediction using machine-learning models.
This project features a fully redesigned 50/50 split-screen layout, clean typography, animated gradients, and a professional workflow for patient-data input.

Perfect for researchers, engineers, or clinicians who want a fast, elegant interface for experimentation and prediction.

🎨 Features

✅ Modern UI inspired by shadcn, Inter, Playfair, and minimal grid layouts
✅ Clean split-screen design (Coordinator panel + Prediction form)
✅ Auto-calculated BMI + structured clinical inputs
✅ TensorFlow model loading with graceful error handling
✅ Scaler + label encoder support
✅ Animated background gradient
✅ Fully customizable CSS layer

🚀 Demo

Private Hosted

🧠 Tech Stack

Streamlit — UI and layout

TensorFlow / Keras — implant prediction model

scikit-learn / joblib — scaler + label encoder

Numpy — data transformations

Custom CSS — animations, typography, gradients

📦 Installation
1️⃣ Install dependencies
pip install -r requirements.txt

2️⃣ Run the app
streamlit run train_model.py

📁 Project Structure
project/
│
├── train_model.py          # Main Streamlit UI + prediction pipeline
├── combined_model.h5       # Trained TensorFlow model (add manually)
├── scaler.pkl              # Feature scaler
├── label_encoder_size.pkl  # Label encoder for implant size
└── requirements.txt        # Dependencies

⚠️ Missing Model Files?

If you clone this repo and don’t see predictions, you may be missing:

combined_model.h5

scaler.pkl

label_encoder_size.pkl

Add them to the root folder to enable predictions.

🎯 Purpose

This project aims to streamline implant planning by giving clinicians and engineers a structured UI to input patient metrics and instantly visualize predicted implant size and wall thickness.

🤝 Contributing

Contributions are welcome! Submit a PR or open an issue.

📄 License

MIT License — free to use, modify, and distribute.

If you want, I can also:
✅ Add badges (Python version, license, last update, etc.)
✅ Create a more “corporate medical tech” tone
✅ Add diagrams or architecture illustrations
