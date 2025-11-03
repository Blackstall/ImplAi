# train_model.py
import os
import warnings
from typing import Optional, Tuple


import streamlit as st
import numpy as np


# Optional heavy deps — degrade gracefully
try:
    import joblib
except Exception:
    joblib = None


try:
    import tensorflow as tf
except Exception:
    tf = None


# basics
warnings.filterwarnings("ignore")
st.set_page_config(page_title="INTELLIGENT CUSTOMIZATION IMPLANT", layout="wide", initial_sidebar_state="collapsed")


# cache_resource compatibility
try:
    cache_resource = st.cache_resource
except Exception:
    def cache_resource(fn):
        return st.cache(allow_output_mutation=True)(fn)




@cache_resource
def load_model_and_tools() -> Tuple[Optional[object], Optional[object], Optional[object], str]:
    model_path = "combined_model.h5"
    scaler_path = "scaler.pkl"
    le_path = "label_encoder_size.pkl"


    if tf is None:
        return None, None, None, "TensorFlow not installed. Install: pip3 install tensorflow"
    if joblib is None:
        return None, None, None, "joblib / scikit-learn not installed. Install: pip3 install joblib scikit-learn"


    missing = [p for p in (model_path, scaler_path, le_path) if not os.path.exists(p)]
    if missing:
        return None, None, None, f"Missing files: {', '.join(missing)}"


    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        return None, None, None, f"Failed to load model: {e}"


    try:
        scaler = joblib.load(scaler_path)
    except Exception as e:
        return model, None, None, f"Failed to load scaler.pkl: {e}"


    try:
        label_encoder = joblib.load(le_path)
    except Exception as e:
        return model, scaler, None, f"Failed to load label_encoder_size.pkl: {e}"


    return model, scaler, label_encoder, "OK"




model, scaler, label_encoder, load_status = load_model_and_tools()
model_loaded = model is not None and scaler is not None and label_encoder is not None


# CSS: shadcn-like + Tailwind scale, neon accent + gradients + animated emoji
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:wght@400;600;700&display=swap');


    :root{
      --outer-bg: #4a5568;
      --panel-right: #1a1a1a;
      --neon: #c4ff00;
      --cyan: #06b6d4;
      --muted: #a1a1a1;
      --text: #e6edf3;
      --gap: 64px;
    }


    html, body, .stApp { background: var(--outer-bg); color:var(--text); font-family: Inter, system-ui, -apple-system, "Segoe UI", Roboto, Arial; }


    /* header - modern hero treatment */
    .hero-wrap {
      text-align: center;
      padding-top: 34px;
      padding-bottom: 8px;
    }
    .page-header {
      display: inline-flex;
      align-items: center;
      gap: 12px;
      padding: 8px 18px;
      border-radius: 999px;
      background: linear-gradient(90deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02));
      box-shadow: none;
      letter-spacing: .26em;
      font-weight: 300;
      font-size: 12px;
      text-transform: uppercase;
      font-family: "Playfair Display", Inter, serif;
      color: rgba(230,247,255,0.95);
      margin: 0 auto 6px auto;
    }
    .page-badge {
      display:inline-flex;
      align-items:center;
      justify-content:center;
      width:34px;height:34px;border-radius:8px;
      background: linear-gradient(180deg, rgba(196,255,0,0.96), rgba(6,182,212,0.9));
      color:#07101a;
      font-weight:700;
      box-shadow: 0 6px 18px rgba(6,182,212,0.06);
    }
    .hero-title {
      font-family: "Playfair Display", Inter, serif;
      font-size: 34px;
      font-weight: 600;
      margin: 6px 0 4px 0;
      letter-spacing: 0.02em;
      line-height:1.02;
      background: linear-gradient(90deg, rgba(230,247,255,0.98), rgba(160,230,255,0.95) 45%, rgba(120,255,180,0.9) 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      display: block;
    }
    .hero-sub {
      font-size: 14px;
      color: rgba(230,247,255,0.88);
      margin-bottom: 18px;
      font-weight: 500;
      letter-spacing: 0.01em;
    }
    .title-underline {
      width:92px;
      height:4px;
      margin: 10px auto 0;
      border-radius: 6px;
      background: linear-gradient(90deg, var(--neon), var(--cyan));
      opacity: 0.95;
    }
    .header-row { display:flex; align-items:center; justify-content:center; gap:12px; }


    /* floating emoji */
    .emoji { display:inline-block; transform-origin:center; font-size:20px; }
    @keyframes floaty { 0% { transform: translateY(0) } 50% { transform: translateY(-8px) rotate(5deg) } 100% { transform: translateY(0) } }
    .emoji-float { animation: floaty 4s ease-in-out infinite; }


    /* layout */
    .cols { display:flex; gap:var(--gap); align-items:start; width:100%; box-sizing:border-box; padding: 0 32px 48px 32px; }
    .col-left, .col-right { flex:1; min-width:300px; box-sizing:border-box; padding: 32px; }


    /* left gradient accent — modern */
    .col-left {
      background: linear-gradient(180deg, rgba(196,255,0,0.95) 0%, rgba(6,182,212,0.95) 100%);
      color: #07101a;
      display:flex;
      flex-direction:column;
      justify-content:flex-start;
      gap:18px;
      min-height:420px;
      border-radius: 8px;
    }
    .coord .name { font-weight:800; font-size:22px; margin-bottom:6px; }
    .coord .role { color: rgba(7,16,26,0.75); font-weight:500; margin-bottom:12px; }
    .contact { display:flex; flex-direction:column; gap:10px; color: rgba(7,16,26,0.88); font-size:14px; }
    .contact .row { display:flex; gap:10px; align-items:center; }
    .icon { width:28px; height:28px; border-radius:50%; background:#07101a; color:var(--neon); display:inline-flex; align-items:center; justify-content:center; font-weight:700; }


    /* right panel */
    .col-right {
      background: linear-gradient(180deg, rgba(16,18,20,1) 0%, rgba(26,26,26,1) 100%);
      color: var(--text);
      border-radius: 8px;
      min-height:420px;
      padding-top: 48px;
    }
    .section-title { font-size:20px; font-weight:300; color:var(--muted); margin-bottom:12px; padding-left:4px; }


    /* form fields - underline only */
    .stTextInput>div>div>input,
    .stNumberInput>div>div>input,
    .stSelectbox>div>div>div[role="button"] {
      background: transparent !important;
      color: var(--text) !important;
      border: none !important;
      border-bottom: 1px solid rgba(255,255,255,0.06) !important;
      padding: 12px 10px !important;
      border-radius: 0 !important;
      transition: border-color .14s ease, box-shadow .14s ease;
      font-weight:500;
    }
    .stTextInput>div>div>input:focus,
    .stNumberInput>div>div>input:focus,
    .stSelectbox>div>div>div[role="button"]:focus {
      outline: none !important;
      border-bottom: 2px solid var(--neon) !important;
      box-shadow: 0 8px 28px rgba(196,255,0,0.06);
    }
    label.form-label { font-size:12px; color:var(--muted); margin-bottom:6px; font-weight:500; }


    /* button */
    .stButton>button {
      background: linear-gradient(90deg, var(--neon), var(--cyan)) !important;
      color: #07101a !important;
      font-weight:800 !important;
      width:100% !important;
      padding:12px 16px !important;
      border-radius:8px !important;
      transition: transform .12s ease, filter .12s ease, box-shadow .12s ease;
      box-shadow: 0 12px 36px rgba(6,182,212,0.08);
    }
    .stButton>button:hover { transform: scale(1.02); filter:brightness(0.95); }


    /* results */
    .results { margin-top:12px; color:var(--neon); font-weight:700; font-size:16px; }


    /* footer */
    .footer { text-align:center; margin-top:28px; color:var(--muted); font-size:12px; }


    /* responsive */
    @media (max-width: 980px) {
      .cols { flex-direction:column; gap:28px; padding: 0 18px 28px 18px; }
      .col-right { padding-top:24px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Header with floating emoji
st.markdown(
    """
    <div class="hero-wrap">
      <div class="page-header">
        <div class="page-badge">IC</div>
        <div style="font-size:12px;font-weight:600;color:rgba(230,247,255,0.95);letter-spacing:.18em;">INTELLIGENT CUSTOMIZATION IMPLANT</div>
      </div>


      <div class="hero-title"><span class="emoji emoji-float">🧠</span>&nbsp;Smart implant predictions — designed for precision</div>
      <div class="hero-sub">Precision-driven predictions. Clear inputs. Confident outcomes.</div>
      <div class="title-underline" aria-hidden="true"></div>
    </div>
    """,
    unsafe_allow_html=True,
)


# Columns layout (guaranteed side-by-side)
st.markdown('<div class="cols">', unsafe_allow_html=True)


# left content HTML
left_html = """
<div class="col-left">
  <div class="coord">
    <div class="name">NURASYRANI BINTI RABUAN</div>
    <div class="role">Project Coordinator, can guide your implant prediction process</div>
    <div class="contact">
      <div class="row"><div class="icon">@</div><div>nurasyranirbn@gmail.com</div></div>
      <div class="row"><div class="icon">☎</div><div>017-777 2373</div></div>
      <div class="row"><div class="icon">#</div><div>IC: 950616-01-7038</div></div>
    </div>
  </div>
</div>
"""


# right column wrapper start
right_start = '<div class="col-right">'
right_end = '</div>'


# Render columns using Streamlit columns to keep widgets interactive and aligned
col1, col2 = st.columns([1, 1])
with col1:
    st.markdown(left_html, unsafe_allow_html=True)


with col2:
    # Render form directly in the Streamlit column (remove the manual <div class="col-right"> wrapper)
    st.markdown('<div style="max-width:720px;padding-left:8px;padding-right:8px;">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">What patient data do you need to predict?</div>', unsafe_allow_html=True)


    with st.form("input_form"):
        age = st.number_input("Age", min_value=10, max_value=120, value=35, step=1, format="%d")
        gender = st.selectbox("Gender", ["Male", "Female"])
        height = st.number_input("Height (cm)", min_value=80.0, max_value=250.0, value=180.0, step=0.1, format="%.1f")
        weight = st.number_input("Weight (kg)", min_value=20.0, max_value=300.0, value=75.0, step=0.1, format="%.1f")


        # BMI display (muted)
        try:
            bmi = weight / ((height / 100) ** 2)
        except Exception:
            bmi = 0.0
        st.markdown(f'<div style="margin-top:6px;color:var(--muted);font-weight:500;">Calculated BMI: {bmi:.2f}</div>', unsafe_allow_html=True)


        tibial_width = st.number_input("Tibial Width (mm)", min_value=20.0, max_value=200.0, value=85.0, step=0.1, format="%.1f")
        submitted = st.form_submit_button("PREDICT")


    st.markdown('</div>', unsafe_allow_html=True)


    if submitted:
        input_data = np.array([[age, 1 if gender == "Male" else 0, height, weight, bmi, tibial_width]], dtype=float)


        if not model_loaded:
            st.markdown(f'<div class="results">Model not available: {load_status}</div>', unsafe_allow_html=True)
            st.info("To enable predictions, install dependencies and add: combined_model.h5, scaler.pkl, label_encoder_size.pkl")
        else:
            try:
                Xs = scaler.transform(input_data)
                preds = model.predict(Xs)


                if isinstance(preds, (list, tuple)) and len(preds) == 2:
                    class_pred, reg_pred = preds
                else:
                    preds = np.asarray(preds)
                    if preds.ndim == 2 and preds.shape[1] == 1:
                        class_pred = np.array([[0]])
                        reg_pred = preds
                    elif preds.ndim == 2 and preds.shape[1] > 1:
                        class_pred = preds
                        reg_pred = np.array([[np.nan]])
                    else:
                        class_pred = np.array([[0]])
                        reg_pred = preds.reshape(1, -1)


                try:
                    class_index = int(np.argmax(class_pred, axis=1)[0])
                    predicted_size = label_encoder.inverse_transform([class_index])[0]
                except Exception:
                    predicted_size = "N/A"


                try:
                    predicted_wall_thickness = float(np.squeeze(reg_pred))
                    wall_text = f"{predicted_wall_thickness:.2f} mm"
                except Exception:
                    wall_text = "N/A"


                st.markdown(f'<div class="results">Predicted Implant Size: {predicted_size}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="results">Predicted Wall Thickness: {wall_text}</div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Prediction failed: {e}")


st.markdown('</div>', unsafe_allow_html=True)  # close .cols


# footer
st.markdown('<div class="footer">2023</div>', unsafe_allow_html=True)




