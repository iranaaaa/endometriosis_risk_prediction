import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import os
import sqlite3
from datetime import datetime
from src.data_preprocessing import load_and_preprocess_data
from src.model import train_xgboost, explain_model_shap

# Keep layout wide and keep our pink theme applied
st.set_page_config(page_title="Endometriosis Risk Predictor", layout="wide", page_icon="🌸")

# --- DATABASE SETUP ---
def init_db():
    # This creates a local file called patients.db to store records
    conn = sqlite3.connect('patients.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS assessments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            name TEXT,
            age INTEGER,
            risk_level TEXT,
            probability REAL,
            symptoms TEXT,
            family_history TEXT
        )
    ''')
    try:
        c.execute('ALTER TABLE assessments ADD COLUMN family_history TEXT')
    except sqlite3.OperationalError:
        pass
    conn.commit()
    conn.close()

init_db()
# ----------------------

st.title("🌸 Endometriosis Patient Screening Portal")
st.markdown("Please fill out the patient's details and currently active symptoms below for a real-time risk assessment.")

# --- BACKGROUND DATA LOADING & TRAINING ---
# We cache this so the heavy AI processing only runs once when the app boots up behind the scenes!
@st.cache_resource(show_spinner="Booting AI Model behind the scenes...")
def load_and_train_model_v3():
    dataset_path = "dataset.csv"
    if not os.path.exists(dataset_path):
        return None, None, None, False
    
    # Load dataset totally behind the scenes (auto-detects if Excel used semicolons instead of commas)
    try:
        df = pd.read_csv(dataset_path, sep=';', encoding='utf-8')
        if len(df.columns) < 5:  # If it failed to separate, try commas
            df = pd.read_csv(dataset_path, sep=',', encoding='utf-8')
    except:
        df = pd.read_csv(dataset_path, sep=None, engine='python')
        
    df.columns = df.columns.str.strip()
    X_train, X_test, y_train, y_test, scaler, feature_names = load_and_preprocess_data(df, target_col='label')
    model = train_xgboost(X_train, y_train)
    return model, scaler, feature_names, True

model, scaler, feature_names, is_ready = load_and_train_model_v3()

if not is_ready:
    st.error("System Error: Could not find `dataset.csv` in the system folder! Please place your dataset file into the VS Code folder so the web app can train the model automatically.")
    st.stop()
# ----------------------------------

# --- QUICK PATIENT INFO SECTION ---
st.markdown("### 1. Patient Details")
col1, col2, col3, col4 = st.columns(4)
with col1:
    patient_name = st.text_input("Full Name", placeholder="Jane Doe")
with col2:
    patient_age = st.number_input("Age", min_value=10, max_value=120, value=25)
with col3:
    patient_gender = st.selectbox("Biological Sex", ["Female", "Other"])
with col4:
    family_history = st.selectbox("Family History of Endometriosis", ["Unknown", "Yes", "No"])

st.markdown("---")

# --- SYMPTOM CHECKER ---
st.markdown("### 2. Clinical Symptom Checker")
st.markdown("Select **only** the physical symptoms the patient is currently experiencing. Any symptom left unselected is automatically logged as absent.")

selected_symptoms = st.pills(
    "Select all present symptoms (click to toggle):",
    options=feature_names,
    selection_mode="multi",
    default=[]
)

st.markdown("---")

# --- FINAL RESULTS ---
colA, colB, colC = st.columns([1, 2, 1])
with colB:
    generate_clicked = st.button("Generate Complete Risk Assessment", type="primary", use_container_width=True)

if generate_clicked:
    if not patient_name.strip():
        st.warning("Please enter the patient's name to generate the assessment report.")
    else:
        st.snow()
        
        # We turn the selected symptoms into machine code (1s and 0s)
        input_data = {f: (1 if f in selected_symptoms else 0) for f in feature_names}
        input_df = pd.DataFrame([input_data])
        # Scale input
        input_scaled = pd.DataFrame(scaler.transform(input_df), columns=feature_names)
        
        # Get AI decision
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]
        
        # --- SAVE TO DATABASE ---
        risk_level = "High Risk" if prediction == 1 else "Low Risk"
        try:
            conn = sqlite3.connect('patients.db')
            c = conn.cursor()
            c.execute('''
                INSERT INTO assessments (timestamp, name, age, risk_level, probability, symptoms, family_history)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
                patient_name, 
                patient_age, 
                risk_level, 
                float(probability), 
                ", ".join(selected_symptoms) if selected_symptoms else "None",
                family_history
            ))
            conn.commit()
            conn.close()
            st.toast("Patient assessment securely logged to local database!")
        except Exception as e:
            st.warning(f"Database Error: Could not save record - {e}")
        # ------------------------
        
        st.markdown("---")
        report_card = st.container(border=True)
        
        with report_card:
            st.markdown(f"<h2 style='text-align: center; color: #d63384; margin-bottom: 0px;'>Official Clinical Assessment Report</h2>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align: center; font-size: 1.1em; color: gray;'><b>Patient Name:</b> {patient_name.upper()} &nbsp;&nbsp;|&nbsp;&nbsp; <b>Age:</b> {patient_age} &nbsp;&nbsp;|&nbsp;&nbsp; <b>Date:</b> {datetime.now().strftime('%B %d, %Y')}</p>", unsafe_allow_html=True)
            st.markdown("<hr style='margin-top: 5px; margin-bottom: 20px;'>", unsafe_allow_html=True)
            
            res_col1, res_col2 = st.columns([1, 2])
            
            with res_col1:
                st.markdown("#### Diagnostic Conclusion")
                if prediction == 1:
                    st.error(f"**EVALUATION: HIGH RISK**\n\n**Confidence Score:** {probability:.1%}\n\n*Recommendation: Immediate specialist consultation and further diagnostic imaging are strongly advised.*")
                else:
                    st.success(f"**EVALUATION: LOW RISK**\n\n**Confidence Score:** {probability:.1%}\n\n*Recommendation: Routine clinical monitoring advised. Re-evaluate if symptom severity increases.*")
                
                st.markdown("#### Patient Snapshot")
                if selected_symptoms:
                    symptoms_bullet = "\n- ".join(selected_symptoms)
                    st.markdown(f"**Presenting Clinical Symptoms:**\n- {symptoms_bullet}")
                else:
                    st.markdown("**Presenting Clinical Symptoms:** None reported")
                st.markdown(f"**Family History of Endometriosis:** {family_history}")
                    
            with res_col2:
                st.markdown("#### Mathematical Feature Analysis")
                st.caption("Displaying the top 8 clinical indicators computationally driving this evaluation.")
            
            explainer = shap.TreeExplainer(model)
            shap_values_single = explainer.shap_values(input_scaled)
            
            plt.clf()
            fig, ax = plt.subplots(figsize=(8, 4))
            fig.patch.set_alpha(0.0)
            ax.patch.set_alpha(0.0)
            
            vals = shap_values_single[0]
            sorted_idx = np.argsort(np.abs(vals))[-8:] # Grab the Top 8 defining traits
            top_vals = vals[sorted_idx]
            
            feature_labels = []
            for i in sorted_idx:
                status = "Present" if input_data[feature_names[i]] == 1 else "Absent"
                feature_labels.append(f"{feature_names[i]} ({status})")
                
            # Matching the requested pink theme logic: pinkish red for pushing high impact, silver for pushing low impact
            colors = ['#d63384' if v > 0 else '#a2a9b1' for v in top_vals]
            ax.barh(feature_labels, top_vals, color=colors, edgecolor='white')
            ax.set_xlabel('Mathematical Impact on Final Risk Probability')
            
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor='#d63384', label='Pushes towards High Risk'),
                               Patch(facecolor='#a2a9b1', label='Pushes towards Low Risk')]
            ax.legend(handles=legend_elements, loc='lower right')
            
            st.pyplot(fig)

        # --- DETAILED DOCTOR VIEW ---
        st.markdown("---")
        with st.expander("Detailed Clinical Analysis (For Medical Professionals)"):
            st.markdown("""
            This section provides a complete, transparent breakdown of how the AI arrived at the risk probability using SHAP (SHapley Additive exPlanations). 
            It details every evaluated symptom's mathematical contribution to the final risk score.
            """)
            
            # Create a dataframe of all SHAP values for the doctor to review
            shap_df = pd.DataFrame({
                'Symptom': feature_names,
                'Patient Status': ["Present" if input_data[f] == 1 else "Absent" for f in feature_names],
                'Mathematical Impact': vals
            })
            
            # Sort by absolute impact
            shap_df['Abs Impact'] = shap_df['Mathematical Impact'].abs()
            shap_df = shap_df.sort_values(by='Abs Impact', ascending=False).drop(columns=['Abs Impact'])
            
            # Add a human readable direction
            shap_df['Effect'] = shap_df['Mathematical Impact'].apply(
                lambda x: "Increases Risk" if x > 0.005 else ("Decreases Risk" if x < -0.005 else "Neutral")
            )
            
            def color_effect(val):
                if 'Increases' in str(val):
                    return 'color: #d63384; font-weight: bold;'
                elif 'Decreases' in str(val):
                    return 'color: #6c757d;'
                return ''
                
            try:
                styled_df = shap_df.style.map(color_effect, subset=['Effect'])
            except AttributeError:
                styled_df = shap_df.style.applymap(color_effect, subset=['Effect'])
                
            st.dataframe(
                styled_df,
                use_container_width=True,
                hide_index=True
            )
            
            # Give a brief explanation of base value
            if isinstance(explainer.expected_value, (list, np.ndarray)):
                base_value = explainer.expected_value[1] if len(explainer.expected_value) > 1 else explainer.expected_value[0]
            else:
                base_value = explainer.expected_value
                
            st.info(f"**Interpretation Guide:** The model has a baseline starting score of **{base_value:.3f}** (representing the average risk). "
                    f"By adding up the specific 'Mathematical Impact' of all present and absent symptoms for this patient, "
                    f"the model pushes that baseline score to a final total. That total is then mathematically converted into the final **{probability:.1%}** risk percentage.")
