import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import os
import base64
import glob
from datetime import datetime
from src.data_preprocessing import load_and_preprocess_data
from src.model import train_xgboost, explain_model_shap

st.set_page_config(page_title="Endometriosis Risk Predictor", layout="wide", page_icon="🌸")

CSV_FILE = r"C:\Users\hp\.gemini\antigravity\scratch\endometriosis_risk_prediction\patients data\patients.csv"
os.makedirs(os.path.dirname(CSV_FILE), exist_ok=True)

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def get_latest_image_path(prefix):
    artifact_dir = r"C:\Users\hp\.gemini\antigravity\brain\703d9bce-e680-49c5-b365-885d43863141"
    files = glob.glob(os.path.join(artifact_dir, f"{prefix}*.png"))
    if not files:
        return None
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]

if 'page' not in st.session_state:
    st.session_state.page = "main"

trans = {
    "English": {
        "admin": "Admin Panel",
        "edu_title": "Understanding Endometriosis",
        "edu_sub": "A clinical overview of the female reproductive condition.",
        "edu_p1": "The uterus is lined with a layer of tissue called the **endometrium**. Each month, this lining organically thickens in preparation for a potential pregnancy.",
        "edu_p2": "If pregnancy does not occur, the thickened lining breaks down and exits the body as a menstrual period.",
        "edu_p3": "In **Endometriosis**, tissue that is structurally similar to this lining begins growing *outside* the uterus—commonly on the ovaries, fallopian tubes, or pelvic tissue.",
        "edu_p4": "During the menstrual cycle, these external patches also thicken and bleed. However, because they are trapped outside the uterus, the blood has no way to exit the body. This continuous cycle leads to severe inflammation, scarring, and intense pelvic pain.",
        "title": "🌸 Endometriosis Patient Screening Portal",
        "tab1": "New Patient Assessment",
        "tab2": "Patient History Registry",
        "tab3": "Doctor's Analysis Pane",
        "desc": "Please fill out the patient's details and currently active symptoms below for a real-time risk assessment.",
        "sec1": "### 1. Patient Details",
        "name": "Full Name",
        "age": "Age",
        "sex": "Biological Sex",
        "fam_hist": "Family History of Endometriosis",
        "sec2": "### 2. Clinical Symptom Checker",
        "sec2_desc": "Select **only** the physical symptoms the patient is currently experiencing. Any symptom left unselected is automatically logged as absent.",
        "pills": "Select all present symptoms (click to toggle):",
        "btn": "Generate Complete Risk Assessment",
        "report_title": "Official Clinical Assessment Report",
        "conclusion": "#### Diagnostic Conclusion",
        "snapshot": "#### Patient Snapshot",
        "recommendation_high": "*Recommendation: Immediate specialist consultation and further diagnostic imaging are strongly advised.*",
        "recommendation_low": "*Recommendation: Routine clinical monitoring advised. Re-evaluate if symptom severity increases.*",
        "high_risk": "**EVALUATION: HIGH RISK**",
        "low_risk": "**EVALUATION: LOW RISK**",
        "doc_title": "### Detailed Clinical Analysis (For Medical Professionals)",
        "doc_desc": "This suite uses SHAP (SHapley Additive exPlanations) to provide a transparent breakdown of the AI's logic. It details every evaluated symptom's mathematical contribution to the final risk score.",
        "doc_wait": "Please generate a risk assessment in Tab 1 first to view the SHAP diagnostic analysis.",
        "top_chart": "**Top Clinical Indicators Chart**",
        "next_steps": "**Suggested Medical Next Steps**",
        "feature_table": "#### Complete Feature Impact Table"
    },
    "हिन्दी (Hindi)": {
        "admin": "व्यवस्थापक पैनल (Admin)",
        "edu_title": "एंडोमेट्रियोसिस को समझना",
        "edu_sub": "मादा प्रजनन स्थिति का नैदानिक ​​अवलोकन।",
        "edu_p1": "गर्भाशय **एंडोमेट्रियम** नामक ऊतक की एक परत से पंक्तिबद्ध होता है। हर महीने, यह अस्तर संभावित गर्भावस्था की तैयारी में मोटा हो जाता है।",
        "edu_p2": "यदि गर्भावस्था नहीं होती है, तो यह मोटा अस्तर टूट जाता है और मासिक धर्म (माहवारी) के रूप में शरीर से बाहर निकल जाता है।",
        "edu_p3": "**एंडोमेट्रियोसिस** में, इस अस्तर के समान ऊतक गर्भाशय के *बाहर* बढ़ने लगता है - आमतौर पर अंडाशय, फैलोपियन ट्यूब या श्रोणि ऊतक पर।",
        "edu_p4": "मासिक धर्म चक्र के दौरान, ये बाहरी धब्बे भी मोटे होते हैं और खून बहता है। हालाँकि, क्योंकि वे गर्भाशय के बाहर फंसे हुए हैं, रक्त के शरीर से बाहर निकलने का कोई रास्ता नहीं है। इस लगातार चक्र से गंभीर सूजन, घाव और तीव्र श्रोणि दर्द होता है।",
        "title": "🌸 एंडोमेट्रियोसिस रोगी स्क्रीनिंग पोर्टल",
        "tab1": "नया रोगी मूल्यांकन",
        "tab2": "रोगी इतिहास रजिस्ट्री",
        "tab3": "डॉक्टर का विश्लेषण फलक (Doctor's Analysis)",
        "desc": "वास्तविक समय में जोखिम मूल्यांकन के लिए कृपया नीचे रोगी का विवरण और वर्तमान में सक्रिय लक्षण भरें।",
        "sec1": "### 1. रोगी का विवरण",
        "name": "पूरा नाम",
        "age": "उम्र",
        "sex": "जैविक लिंग",
        "fam_hist": "एंडोमेट्रियोसिस का पारिवारिक इतिहास",
        "sec2": "### 2. नैदानिक लक्षण चेकर",
        "sec2_desc": "केवल उन्हीं शारीरिक लक्षणों का चयन करें जिनका रोगी वर्तमान में अनुभव कर रहा है। कोई भी लक्षण जिसे नहीं चुना गया है, उसे स्वचालित रूप से अनुपस्थित माना जाएगा।",
        "pills": "सभी मौजूद लक्षणों का चयन करें (टॉगल करने के लिए क्लिक करें):",
        "btn": "संपूर्ण जोखिम मूल्यांकन उत्पन्न करें",
        "report_title": "आधिकारिक नैदानिक मूल्यांकन रिपोर्ट",
        "conclusion": "#### नैदानिक निष्कर्ष",
        "snapshot": "#### रोगी स्नैपशॉट",
        "recommendation_high": "*सिफारिश: तत्काल विशेषज्ञ परामर्श और आगे की नैदानिक इमेजिंग की दृढ़ता से सलाह दी जाती है।*",
        "recommendation_low": "*सिफारिश: नियमित नैदानिक निगरानी की सलाह दी जाती है। लक्षणों की गंभीरता बढ़ने पर पुनर्मूल्यांकन करें।*",
        "high_risk": "**मूल्यांकन: उच्च जोखिम (HIGH RISK)**",
        "low_risk": "**मूल्यांकन: निम्न जोखिम (LOW RISK)**",
        "doc_title": "### विस्तृत नैदानिक विश्लेषण (चिकित्सा पेशेवरों के लिए)",
        "doc_desc": "यह सूट एआई (AI) के तर्क का पारदर्शी ब्रेकडाउन प्रदान करने के लिए SHAP का उपयोग करता है।",
        "doc_wait": "SHAP डायग्नोस्टिक विश्लेषण देखने के लिए कृपया पहले टैब 1 में जोखिम मूल्यांकन उत्पन्न करें।",
        "top_chart": "**शीर्ष नैदानिक संकेतक चार्ट**",
        "next_steps": "**सुझाए गए चिकित्सा अगले कदम**",
        "feature_table": "#### संपूर्ण फ़ीचर प्रभाव तालिका"
    },
    "தமிழ் (Tamil)": {
        "admin": "நிர்வாக குழு",
        "edu_title": "எண்டோமெட்ரியோசிஸ் புரிந்துகொள்வது",
        "edu_sub": "பெண் இனப்பெருக்க நிலை பற்றிய மருத்துவ கண்ணோட்டம்.",
        "edu_p1": "கருப்பை **எண்டோமெட்ரியம்** எனப்படும் திசுக்களின் ஒரு அடுக்கால் வரிசையாக உள்ளது. ஒவ்வொரு மாதமும், சாத்தியமான கர்ப்பத்திற்குத் தயாராகும் வகையில் இந்த புறணி தடிமனாகிறது.",
        "edu_p2": "கர்ப்பம் ஏற்படவில்லை என்றால், தடிமனான புறணி உடைந்து மாதவிடாய் காலமாக உடலை விட்டு வெளியேறுகிறது.",
        "edu_p3": "**எண்டோமெட்ரியோசிஸில்**, இந்த புறணிக்கு ஒத்த திசு கருப்பைக்கு *வெளியே* வளரத் தொடங்குகிறது - பொதுவாக கருப்பைகள், ஃபலோபியன் குழாய்கள் அல்லது இடுப்பு திசுக்களில்.",
        "edu_p4": "மாதவிடாய் சுழற்சியின் போது, இந்த வெளிப்புறத் திட்டுகளும் தடிமனாகி இரத்தப்போக்கு ஏற்படுகின்றன. இருப்பினும், அவை கருப்பைக்கு வெளியே சிக்கியிருப்பதால், இரத்தம் உடலை விட்டு வெளியேற வழியில்லை. இது கடுமையான வீக்கம் மற்றும் தீவிர இடுப்பு வலிக்கு வழிவகுக்கிறது.",
        "faq_title": "அடிக்கடி கேட்கப்படும் கேள்விகள்",
        "faq_sub": "எண்டோமெட்ரியோசிஸ் பற்றிய பொதுவான கேள்விகள்.",
        "treatments_title": "அறியப்பட்ட சிகிச்சைகள் மற்றும் குணப்படுத்துதல்கள்",
        "treatments_sub": "மேலாண்மை மற்றும் சிகிச்சை விருப்பங்களைப் புரிந்துகொள்வது.",
        "title": "🌸 எண்டோமெட்ரியோசிஸ் நோயாளி திரையிடல் போர்ட்டல்",
        "tab1": "புதிய நோயாளி மதிப்பீடு",
        "tab2": "நோயாளி வரலாற்று பதிவேடு",
        "tab3": "மருத்துவரின் பகுப்பாய்வு",
        "desc": "நிகழ்நேர ஆபத்து மதிப்பீட்டிற்கு நோயாளியின் விவரங்கள் மற்றும் தற்போதுள்ள அறிகுறிகளை கீழே நிரப்பவும்.",
        "sec1": "### 1. நோயாளி விவரங்கள்",
        "name": "முழு பெயர்",
        "age": "வயது",
        "sex": "பாலினம்",
        "fam_hist": "எண்டோமெட்ரியோசிஸின் குடும்ப வரலாறு",
        "sec2": "### 2. மருத்துவ அறிகுறி சரிபார்ப்பு",
        "sec2_desc": "நோயாளி தற்போது அனுபவிக்கும் உடல் அறிகுறிகளை மட்டும் தேர்ந்தெடுக்கவும்.",
        "pills": "தற்போதுள்ள அனைத்து அறிகுறிகளையும் தேர்ந்தெடுக்கவும்:",
        "btn": "முழுமையான ஆபத்து மதிப்பீட்டை உருவாக்கு",
        "report_title": "அதிகாரப்பூர்வ மருத்துவ மதிப்பீட்டு அறிக்கை",
        "conclusion": "#### மருத்துவ முடிவு",
        "snapshot": "#### நோயாளி ஸ்னாப்ஷாட்",
        "recommendation_high": "*பரிந்துரை: உடனடி சிறப்பு ஆலோசனை மற்றும் கூடுதல் நோயறிதல் இமேஜிங் கடுமையாக அறிவுறுத்தப்படுகிறது.*",
        "recommendation_low": "*பரிந்துரை: வழக்கமான மருத்துவ கண்காணிப்பு அறிவுறுத்தப்படுகிறது.*",
        "high_risk": "**மதிப்பீடு: அதிக ஆபத்து (HIGH RISK)**",
        "low_risk": "**மதிப்பீடு: குறைந்த ஆபத்து (LOW RISK)**",
        "doc_title": "### விரிவான மருத்துவ பகுப்பாய்வு (மருத்துவர்களுக்கானது)",
        "doc_desc": "AI இன் தர்க்கத்தின் வெளிப்படையான முறிவை வழங்க இது SHAP ஐப் பயன்படுத்துகிறது.",
        "doc_wait": "SHAP பகுப்பாய்வைக் காண முதலில் தாவல் 1-ல் ஆபத்து மதிப்பீட்டை உருவாக்கவும்.",
        "top_chart": "**சிறந்த மருத்துவ குறிகாட்டிகள் விளக்கப்படம்**",
        "next_steps": "**பரிந்துரைக்கப்பட்ட மருத்துவ அடுத்த படிகள்**",
        "feature_table": "#### முழுமையான அம்ச प्रभाव அட்டவணை"
    }
}

# --- SIDEBAR ---
with st.sidebar:
    icon_path = get_latest_image_path("happy_uterus")
    if icon_path and os.path.exists(icon_path):
        img_base64 = get_base64_of_bin_file(icon_path)
        st.markdown(f'''
            <style>
            @keyframes bounce {{
                0%, 100% {{ transform: translateY(0); }}
                50% {{ transform: translateY(-15px); }}
            }}
            .animated-uterus {{
                animation: bounce 3s ease-in-out infinite;
                display: block;
                margin-left: auto;
                margin-right: auto;
                width: 140px;
                margin-bottom: 25px;
            }}
            </style>
            <img src="data:image/png;base64,{img_base64}" class="animated-uterus">
        ''', unsafe_allow_html=True)
    
    dark_mode = st.toggle("🌙 Dark Mode")
    lang_choice = st.selectbox("Language / भाषा / மொழி", ["English", "हिन्दी (Hindi)", "தமிழ் (Tamil)"])
    t = trans[lang_choice]
    
    st.title(t["admin"])
    st.markdown("---")
    
    if st.session_state.page != "education":
        if st.button("Understanding Endometriosis (Overview)", use_container_width=True):
            st.session_state.page = "education"
            st.rerun()
    if st.session_state.page != "pcod":
        if st.button("Understanding PCOD vs PCOS", use_container_width=True):
            st.session_state.page = "pcod"
            st.rerun()
    if st.session_state.page != "faq":
        if st.button("Frequently Asked Questions", use_container_width=True):
            st.session_state.page = "faq"
            st.rerun()
            
    if st.session_state.page != "treatments":
        if st.button("Known Treatments & Cures", use_container_width=True):
            st.session_state.page = "treatments"
            st.rerun()

    if st.session_state.page != "main":
        st.markdown("---")
        if st.button("Back to Patient Portal", use_container_width=True):
            st.session_state.page = "main"
            st.rerun()

    st.markdown("---")
    st.markdown("### 🎗️ Patient Support")
    st.markdown("""
    <div style='background-color: rgba(255, 75, 139, 0.1); padding: 15px; border-radius: 10px; border-left: 4px solid #ff4b8b; font-size: 0.9em; margin-bottom: 20px;'>
    <p style='margin-bottom: 10px;'>Endometriosis carries a heavy emotional burden. <b>You are not alone.</b></p>
    <b>Communities & Resources:</b>
    <ul style='margin-top: 5px; padding-left: 20px;'>
    <li><a href='https://www.endofound.org/' target='_blank' style='color: #ff4b8b;'>EndoFound</a></li>
    <li><a href='https://www.endowhat.com/' target='_blank' style='color: #ff4b8b;'>Endo What?</a></li>
    <li><a href='https://www.myendometriosisteam.com/' target='_blank' style='color: #ff4b8b;'>MyEndometriosisTeam</a></li>
    </ul>
    <p style='margin-top: 10px; font-size: 0.85em; opacity: 0.9;'>If struggling with trauma or infertility, please reach out to local chronic illness counseling.</p>
    </div>
    """, unsafe_allow_html=True)



# --- DYNAMIC CSS INJECTION FOR THEMES ---
if dark_mode:
    st.markdown("""
        <style>
        [data-testid="stAppViewContainer"] { background-color: #1a1216; color: #f8f9fa; }
        [data-testid="stSidebar"] { background-color: #2b1c23; }
        [data-testid="stHeader"] { background-color: rgba(0,0,0,0); }
        .report-card { padding: 20px; border-radius: 10px; box-shadow: 0 4px 15px rgba(255, 75, 139, 0.2); background-color: #2b1c23; border-left: 5px solid #ff4b8b; }
        .metric-value { font-size: 2.5rem !important; font-weight: bold; color: #ff4b8b !important; }
        
        /* Main Typography: Force Headings, Markdown, and Labels to be white */
        h1, h2, h3, h4, h5, h6, label, label p { color: #f8f9fa !important; }
        .stMarkdown p, .stMarkdown span { color: #f8f9fa !important; }
        
        /* Inputs & Pills: Force interior text to be explicitly BLACK to contrast with the light pink background */
        input { color: #000000 !important; }
        [data-baseweb="select"] span { color: #000000 !important; }
        ul[data-baseweb="menu"] * { color: #000000 !important; }
        [data-testid="stPills"] button p, [data-testid="stPills"] button span { color: #000000 !important; }
        
        /* Streamlit Buttons (e.g. Generate button, custom sidebar buttons) - text should be black */
        .stButton button p { color: #000000 !important; }
        .stDownloadButton button p { color: #000000 !important; }
        
        [data-testid="stMetricValue"] div { color: #ff4b8b !important; }
        hr { border-top: 1px solid #4a343e !important; }
        </style>
    """, unsafe_allow_html=True)
    report_title_color = "#ff4b8b"
    report_text_color = "#a2a9b1"
else:
    st.markdown("""
        <style>
        .report-card { padding: 20px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); background-color: #f8f9fa; border-left: 5px solid #d63384; }
        .metric-value { font-size: 2.5rem !important; font-weight: bold; color: #d63384 !important; }
        hr { border-top: 1px solid #e0e0e0 !important; }
        </style>
    """, unsafe_allow_html=True)
    report_title_color = "#d63384"
    report_text_color = "gray"

if st.session_state.page == "education":
    st.title(t["edu_title"])
    st.markdown(f"### {t['edu_sub']}")
    st.markdown("---")
    st.markdown("""
    <style>
    @keyframes thicken { 0%, 100% { opacity: 0.5; } 50% { opacity: 1; } }
    @keyframes shed { 0%, 40% { transform: translateY(0); opacity: 0; } 50% { transform: translateY(10px); opacity: 1; } 70% { transform: translateY(40px); opacity: 0; } 100% { transform: translateY(0); opacity: 0; } }
    @keyframes inflame { 0%, 100% { transform: scale(1); opacity: 0.6; } 50% { transform: scale(1.3); opacity: 1; filter: drop-shadow(0 0 3px red); } }
    .anim1 .lining { animation: thicken 4s infinite; }
    .anim1 .drop1 { animation: shed 4s infinite; animation-delay: 0s; }
    .anim1 .drop2 { animation: shed 4s infinite; animation-delay: 0.3s; }
    .anim1 .drop3 { animation: shed 4s infinite; animation-delay: 0.6s; }
    .anim2 .lining2 { animation: thicken 4s infinite; }
    .anim2 .lesion circle { animation: inflame 4s infinite; box-sizing: border-box;}
    </style>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1,1])
    with col1:
        st.markdown(t["edu_p1"])
        st.markdown(t["edu_p2"])
        
        svg1 = '''
        <svg class="anim1" viewBox="0 0 200 200" width="300" height="300" xmlns="http://www.w3.org/2000/svg">
          <path d="M 40 50 Q 80 50 100 80 Q 120 50 160 50" stroke="#f8b4c4" stroke-width="10" fill="none" stroke-linecap="round"/>
          <circle cx="30" cy="55" r="12" fill="#ffd1dc"/>
          <circle cx="170" cy="55" r="12" fill="#ffd1dc"/>
          <path d="M 75 70 C 75 30 125 30 125 70 C 125 110 110 140 110 170 L 90 170 C 90 140 75 110 75 70 Z" fill="#f8b4c4"/>
          <path class="lining" d="M 85 75 C 85 50 115 50 115 75 C 115 100 105 130 105 160 L 95 160 C 95 130 85 100 85 75 Z" fill="#d63384"/>
          <circle class="drop drop1" cx="100" cy="150" r="4" fill="#d63384" opacity="0"/>
          <circle class="drop drop2" cx="95" cy="160" r="3" fill="#d63384" opacity="0"/>
          <circle class="drop drop3" cx="105" cy="140" r="3" fill="#d63384" opacity="0"/>
        </svg>
        '''
        st.markdown(f"<div style='text-align: center; margin-top:20px;'>{svg1}</div>", unsafe_allow_html=True)

    with col2:
        st.markdown(t["edu_p3"])
        st.markdown(t["edu_p4"])
        
        svg2 = '''
        <svg class="anim2" viewBox="0 0 200 200" width="300" height="300" xmlns="http://www.w3.org/2000/svg">
          <path d="M 40 50 Q 80 50 100 80 Q 120 50 160 50" stroke="#f8b4c4" stroke-width="10" fill="none" stroke-linecap="round"/>
          <circle cx="30" cy="55" r="12" fill="#ffd1dc"/>
          <circle cx="170" cy="55" r="12" fill="#ffd1dc"/>
          <path d="M 75 70 C 75 30 125 30 125 70 C 125 110 110 140 110 170 L 90 170 C 90 140 75 110 75 70 Z" fill="#f8b4c4"/>
          <path class="lining2" d="M 85 75 C 85 50 115 50 115 75 C 115 100 105 130 105 160 L 95 160 C 95 130 85 100 85 75 Z" fill="#d63384"/>
          <g class="lesion">
            <circle cx="35" cy="50" r="5" fill="#d63384" style="transform-origin: 35px 50px;"/>
            <circle cx="160" cy="65" r="4" fill="#d63384" style="transform-origin: 160px 65px;"/>
            <circle cx="65" cy="110" r="6" fill="#d63384" style="transform-origin: 65px 110px;"/>
            <circle cx="130" cy="90" r="4" fill="#d63384" style="transform-origin: 130px 90px;"/>
          </g>
        </svg>
        '''
        st.markdown(f"<div style='text-align: center; margin-top:20px;'>{svg2}</div>", unsafe_allow_html=True)

elif st.session_state.page == "pcod":
    st.title("Understanding PCOD vs PCOS")
    st.markdown("### A clinical comparative overview of ovarian conditions.")
    st.markdown("---")
    
    st.markdown("""
    <style>
    @keyframes cyst-pulse { 0%, 100% { transform: scale(1); } 50% { transform: scale(1.3); } }
    @keyframes pcos-flare { 0%, 100% { filter: drop-shadow(0 0 2px #ff4b8b); transform: scale(1); opacity:0.5;} 50% { filter: drop-shadow(0 0 10px #d63384); transform: scale(1.05); opacity:1;} }
    .anim-pcod circle { animation: cyst-pulse 4s infinite; transform-origin: center; box-sizing: border-box; }
    .anim-pcos circle { animation: cyst-pulse 2.5s infinite; transform-origin: center; box-sizing: border-box; }
    .anim-pcos .pcos-glow { animation: pcos-flare 3s infinite; transform-origin: center; box-sizing: border-box; }
    </style>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1,1])
    with col1:
        st.markdown("#### PCOD (Polycystic Ovarian Disease)")
        st.markdown("In PCOD, the ovaries release many **immature or partially mature eggs** which eventually form small cysts.")
        st.markdown("It is very common, primarily managed by lifestyle and diet changes, and usually **does not severely impact fertility** if managed early. Hormone production remains mostly stable, avoiding severe systemic issues.")
        
        svg_pcod = '''
        <svg class="anim-pcod" viewBox="0 0 200 200" width="300" height="300" xmlns="http://www.w3.org/2000/svg">
            <ellipse cx="100" cy="100" rx="70" ry="45" fill="#ffd1dc" />
            <circle cx="65" cy="100" r="5" fill="#f8b4c4" style="transform-origin: 65px 100px; animation-delay: 0s;"/>
            <circle cx="90" cy="80" r="4" fill="#f8b4c4" style="transform-origin: 90px 80px; animation-delay: 0.5s;"/>
            <circle cx="120" cy="90" r="6" fill="#f8b4c4" style="transform-origin: 120px 90px; animation-delay: 1.0s;"/>
            <circle cx="135" cy="110" r="5" fill="#f8b4c4" style="transform-origin: 135px 110px; animation-delay: 1.5s;"/>
            <circle cx="100" cy="120" r="4" fill="#f8b4c4" style="transform-origin: 100px 120px; animation-delay: 2.0s;"/>
        </svg>
        '''
        st.markdown(f"<div style='text-align: center; margin-top:20px;'>{svg_pcod}</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("#### PCOS (Polycystic Ovary Syndrome)")
        st.markdown("PCOS is a more severe **metabolic and endocrine disorder**. Ovaries produce excess **androgens** (male hormones).")
        st.markdown("This leads to irregular periods, systemic weight gain, hair thinning, and significantly larger inflammatory cysts. **Fertility is often compromised**, requiring dedicated medical interventions.")
        
        svg_pcos = '''
        <svg class="anim-pcos" viewBox="0 0 200 200" width="300" height="300" xmlns="http://www.w3.org/2000/svg">
            <ellipse class="pcos-glow" cx="100" cy="100" rx="75" ry="50" fill="none" stroke="#ff4b8b" stroke-width="4" style="transform-origin: 100px 100px;"/>
            <ellipse cx="100" cy="100" rx="70" ry="45" fill="#ffb6c1" />
            <circle cx="55" cy="100" r="8" fill="#d63384" style="transform-origin: 55px 100px; animation-delay: 0s;"/>
            <circle cx="80" cy="75" r="9" fill="#d63384" style="transform-origin: 80px 75px; animation-delay: 0.3s;"/>
            <circle cx="115" cy="70" r="8" fill="#d63384" style="transform-origin: 115px 70px; animation-delay: 0.6s;"/>
            <circle cx="145" cy="95" r="9" fill="#d63384" style="transform-origin: 145px 95px; animation-delay: 0.9s;"/>
            <circle cx="130" cy="120" r="8" fill="#d63384" style="transform-origin: 130px 120px; animation-delay: 1.2s;"/>
            <circle cx="95" cy="125" r="9" fill="#d63384" style="transform-origin: 95px 125px; animation-delay: 1.5s;"/>
            <circle cx="65" cy="120" r="7" fill="#d63384" style="transform-origin: 65px 120px; animation-delay: 1.8s;"/>
        </svg>
        '''
        st.markdown(f"<div style='text-align: center; margin-top:20px;'>{svg_pcos}</div>", unsafe_allow_html=True)

elif st.session_state.page == "faq":
    st.title("Frequently Asked Questions")
    st.markdown("### Common questions and clinical answers about Endometriosis.")
    st.markdown("---")
    
    with st.expander("1. What are the main symptoms?"):
        st.markdown("Severe pelvic pain during periods, heavy bleeding, pain during intercourse, chronic fatigue, and pervasive bowel or urinary issues.")
        
    with st.expander("2. How is it officially diagnosed?"):
        st.markdown("While an ultrasound or MRI can sometimes show deep, large cysts (endometriomas), the only definitive way to correctly diagnose and confirm regular endometriosis is through a minimally invasive **laparoscopic surgery** and tissue biopsy.")
        
    with st.expander("3. Can it cause infertility?"):
        st.markdown("Yes. Endometriosis is one of the leading causes of infertility in women, affecting up to 50% of those diagnosed. However, many women with the condition still successfully conceive naturally or with medical assistance (like IVF).")
        
    with st.expander("4. At what age does it typically start?"):
        st.markdown("It typically begins years after a person's first menstrual period. Tragically, diagnosis is often delayed for 7 to 10 years due to a societal lack of awareness and the normalization of extreme menstrual pain.")

elif st.session_state.page == "treatments":
    st.title("Known Treatments & Medical Management")
    st.markdown("### Current clinical protocols for treating Endometriosis.")
    st.markdown("---")
    st.markdown("""
        <div style="font-size: 1.1em; line-height: 1.6;">
        <b style="font-size: 1.3em;">Is there a cure?</b><br>
        Currently, there is <b style="color:#ff4b8b; font-size: 1.1em;">no absolute cure</b> for endometriosis. However, the symptoms can be effectively managed with the right specialized medical team.<br><br>
        
        <b style="color:#ff4b8b;">1. Laparoscopic Excision Surgery:</b><br>
        This is considered the "gold standard" treatment. A specialized endo-excision surgeon meticulously cuts out the endometriosis lesions from their root, rather than superficially burning (ablating) them. This provides the longest-lasting relief.<br><br>
        
        <b style="color:#ff4b8b;">2. Hormonal Suppression Therapies:</b><br>
        Current treatments like continuous birth control pills, hormonal IUDs (like Mirena), or GnRH agonists (like Orilissa or Lupron) are used to proactively stop ovulation. This minimizes internal bleeding and aggressively slows lesion growth.<br><br>
        
        <b style="color:#ff4b8b;">3. Pain Management & Physical Therapy:</b><br>
        Medical management often relies on NSAIDs (like Ibuprofen or Naproxen). Additionally, dedicated <b>Pelvic Floor Physical Therapy</b> is highly recommended to relieve muscular tension and manage the chronic, radiating pelvic pain.
        </div>
    """, unsafe_allow_html=True)

else:
    # --- MAIN APP ---
    st.title(t["title"])
    tab1, tab2, tab3 = st.tabs([t["tab1"], t["tab2"], t["tab3"]])

    with tab1:
        st.markdown(t["desc"])

        @st.cache_resource(show_spinner="Booting AI Model behind the scenes...")
        def load_and_train_model_v3():
            dataset_path = "dataset.csv"
            if not os.path.exists(dataset_path):
                return None, None, None, False
            try:
                df = pd.read_csv(dataset_path, sep=';', encoding='utf-8')
                if len(df.columns) < 5:
                    df = pd.read_csv(dataset_path, sep=',', encoding='utf-8')
            except:
                df = pd.read_csv(dataset_path, sep=None, engine='python')
            df.columns = df.columns.str.strip()
            X_train, X_test, y_train, y_test, scaler, feature_names = load_and_preprocess_data(df, target_col='label')
            model = train_xgboost(X_train, y_train)
            return model, scaler, feature_names, True

        model, scaler, feature_names, is_ready = load_and_train_model_v3()
        if not is_ready:
            st.error("System Error: Could not find `dataset.csv` in the system folder!")
            st.stop()

        with st.container(border=True):
            st.markdown(t["sec1"])
            col1, col2, col3 = st.columns(3)
            with col1:
                patient_name = st.text_input(t["name"], placeholder="Jane Doe")
            with col2:
                patient_age = st.number_input(t["age"], min_value=10, max_value=120, value=25)
            with col3:
                patient_gender = st.selectbox(t["sex"], ["Female", "Other"])

        st.markdown("<br>", unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown(t["sec2"])
            st.markdown(t["sec2_desc"])
            selected_symptoms = st.pills(t["pills"], options=feature_names, selection_mode="multi", default=[])

        st.markdown("<br>", unsafe_allow_html=True)

        colA, colB, colC = st.columns([1, 2, 1])
        with colB:
            generate_clicked = st.button(t["btn"], type="primary", use_container_width=True)

        if generate_clicked:
            if not patient_name.strip():
                st.warning("Please enter the patient's name to generate the assessment report.")
            else:
                with st.spinner('Analyzing clinical data vectors...'):
                    input_data = {f: (1 if f in selected_symptoms else 0) for f in feature_names}
                    input_df = pd.DataFrame([input_data])
                    input_scaled = pd.DataFrame(scaler.transform(input_df), columns=feature_names)
                    
                    prediction = model.predict(input_scaled)[0]
                    probability = model.predict_proba(input_scaled)[0][1]
                    
                    explainer = shap.TreeExplainer(model)
                    shap_values_single = explainer.shap_values(input_scaled)
                    vals = shap_values_single[0]
                    
                    st.session_state['assessment'] = {
                        'prediction': prediction,
                        'probability': probability,
                        'vals': vals,
                        'input_data': input_data,
                        'input_scaled': input_scaled,
                        'name': patient_name,
                        'age': patient_age,
                        'symptoms': selected_symptoms,
                        'base_value': explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) and len(explainer.expected_value) > 1 else (explainer.expected_value[0] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value)
                    }

                    risk_level = "High Risk" if prediction == 1 else "Low Risk"
                    try:
                        data = {
                            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "Name": patient_name,
                            "Age": patient_age,
                            "Risk Level": risk_level,
                            "Probability": float(probability),
                            "Symptoms": ", ".join(selected_symptoms) if selected_symptoms else "None"
                        }
                        df_to_save = pd.DataFrame([data])
                        if os.path.exists(CSV_FILE):
                            df_to_save.to_csv(CSV_FILE, mode='a', header=False, index=False)
                        else:
                            df_to_save.to_csv(CSV_FILE, mode='w', header=True, index=False)
                        st.toast("Patient assessment securely logged to local CSV file!")
                    except Exception as e:
                        pass
                    
        if 'assessment' in st.session_state:
            st.markdown("---")
            a = st.session_state['assessment']
            prediction = a['prediction']
            probability = a['probability']
            name = a['name']
            age = a['age']
            symptoms = a['symptoms']
            
            st.markdown(f"<div class='report-card'>", unsafe_allow_html=True)
            st.markdown(f"<h2 style='text-align: center; color: {report_title_color}; margin-bottom: 0px;'>{t['report_title']}</h2>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align: center; font-size: 1.1em; color: {report_text_color};'><b>Patient Name:</b> {name.upper()} &nbsp;&nbsp;|&nbsp;&nbsp; <b>Age:</b> {age} &nbsp;&nbsp;|&nbsp;&nbsp; <b>Date:</b> {datetime.now().strftime('%B %d, %Y')}</p>", unsafe_allow_html=True)
            st.markdown("<hr style='margin-top: 5px; margin-bottom: 20px;'>", unsafe_allow_html=True)
            
            st.markdown(t["conclusion"])
            st.metric(label="Calculated Risk Probability", value=f"{probability:.1%}", delta="High Risk" if prediction == 1 else "Low Risk", delta_color="inverse" if prediction == 1 else "normal")
            
            if prediction == 1:
                st.error(f"{t['high_risk']}\n\n{t['recommendation_high']}")
            else:
                st.success(f"{t['low_risk']}\n\n{t['recommendation_low']}")
            
            st.markdown(t["snapshot"])
            if symptoms:
                symptoms_bullet = "\n- ".join(symptoms)
                st.markdown(f"**Presenting Clinical Symptoms:**\n- {symptoms_bullet}")
            else:
                st.markdown("**Presenting Clinical Symptoms:** None reported")
            st.markdown("</div>", unsafe_allow_html=True)

    with tab2:
        st.markdown("### Historical Patient Records")
        if os.path.exists(CSV_FILE):
            try:
                history_df = pd.read_csv(CSV_FILE)
                history_df = history_df.iloc[::-1].reset_index(drop=True)
                if "Probability" in history_df.columns:
                    history_df["Probability"] = history_df["Probability"].apply(lambda x: f"{float(x):.1%}" if pd.notnull(x) else x)

                search_query = st.text_input("Search by Patient Name:", "")
                if search_query:
                    filtered_df = history_df[history_df["Name"].str.contains(search_query, case=False, na=False)]
                    st.dataframe(filtered_df, use_container_width=True, hide_index=True)
                else:
                    st.dataframe(history_df, use_container_width=True, hide_index=True)
            except Exception as e:
                st.error(f"Error loading registry: {e}")
        else:
            st.info("No patient records have been saved yet.")

    with tab3:
        st.markdown(t["doc_title"])
        if 'assessment' not in st.session_state:
            st.info(t["doc_wait"])
        else:
            a = st.session_state['assessment']
            st.markdown("#### Excerpt of Model Diagnostics")
            st.markdown(t["doc_desc"])
            
            res_col1, res_col2 = st.columns([1, 1])
            with res_col1:
                st.markdown(t["top_chart"])
                vals = a['vals']
                sorted_idx = np.argsort(np.abs(vals))[-8:] 
                top_vals = vals[sorted_idx]
                
                feature_labels = []
                for i in sorted_idx:
                    status = "Present" if a['input_data'][feature_names[i]] == 1 else "Absent"
                    feature_labels.append(f"{feature_names[i]} ({status})")
                    
                fig, ax = plt.subplots(figsize=(6, 4))
                fig.patch.set_alpha(0.0)
                ax.patch.set_alpha(0.0)
                
                primary_chart_color = '#ff4b8b' if dark_mode else '#d63384'
                sec_chart_color = '#a2a9b1' if dark_mode else '#a2a9b1'
                colors = [primary_chart_color if v > 0 else sec_chart_color for v in top_vals]
                ax.barh(feature_labels, top_vals, color=colors, edgecolor='white' if dark_mode else '#6c757d')
                ax.set_xlabel('Mathematical Impact')
                if dark_mode:
                    ax.xaxis.label.set_color('white')
                    ax.tick_params(axis='x', colors='white')
                    ax.tick_params(axis='y', colors='white')
                st.pyplot(fig)
                
            with res_col2:
                st.markdown(t["next_steps"])
                if a['prediction'] == 1:
                    st.warning("""
                    **HIGH RISK PROFILE DETECTED**
                    - **Primary Action**: Schedule patient for a transvaginal ultrasound or pelvic MRI.
                    - **Secondary Action**: Refer to a specialized endometriosis surgeon or gynecologist.
                    - **Counseling**: Discuss potential minimally invasive laparoscopy if imaging is inconclusive but symptoms are severe.
                    - **Pain Management**: Review current analgesics and consider hormonal suppressive therapies as a bridge.
                    """)
                else:
                    st.success("""
                    **LOW RISK PROFILE DETECTED**
                    - **Primary Action**: Routine monitoring. Assess whether pain management strategies are adequately controlling symptoms.
                    - **Secondary Action**: If symptoms persist or worsen over 3-6 months, escalate diagnostic protocols.
                    - **Differential Diagnosis**: Consider other causes of pelvic pain (e.g., IBS, PID, interstitial cystitis).
                    """)

            st.markdown("---")
            st.markdown(t["feature_table"])
            shap_df = pd.DataFrame({
                'Symptom': feature_names,
                'Patient Status': ["Present" if a['input_data'][f] == 1 else "Absent" for f in feature_names],
                'Mathematical Impact': vals
            })
            shap_df['Abs Impact'] = shap_df['Mathematical Impact'].abs()
            shap_df = shap_df.sort_values(by='Abs Impact', ascending=False).drop(columns=['Abs Impact'])
            shap_df['Effect'] = shap_df['Mathematical Impact'].apply(
                lambda x: "Increases Risk" if x > 0.005 else ("Decreases Risk" if x < -0.005 else "Neutral")
            )
            def color_effect(val):
                if 'Increases' in str(val):
                    return f"color: {primary_chart_color}; font-weight: bold;"
                elif 'Decreases' in str(val):
                    return f"color: {sec_chart_color};"
                return ''
            try:
                styled_df = shap_df.style.map(color_effect, subset=['Effect'])
            except:
                styled_df = shap_df.style.applymap(color_effect, subset=['Effect'])
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
            
            base_value = a['base_value']
            st.info(f"**Interpretation Guide:** Baseline starting score: **{base_value:.3f}**. "
                    f"Symptom impacts mathematically shift this baseline to the final probability **{a['probability']:.1%}**.")
