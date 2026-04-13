import os
import joblib
import pandas as pd
import streamlit as st
from fpdf import FPDF

# -----------------------------
# Feature Validation Data for Dengue Fever Hematological Assessment
# -----------------------------
VALIDATION_FEATURES_DENGUE = {
    "abs_lymphocytes": {
        "description": "Absolute lymphocyte count. According to WHO, early leukopenia and altered lymphocyte counts are common hematological findings in suspected dengue.",
        "role": "WBC Index",
    },
    "RDW-CV(%)": {
        "description": "Red Cell Distribution Width. Variations can reflect changes in RBC morphology during the febrile phase.",
        "role": "RBC Index",
    },
    "HCT(%)": {
        "description": "WHO Warning Sign. An increase in hematocrit concurrent with a rapid decrease in platelet count indicates plasma leakage and progression to severe dengue.",
        "role": "Hemoconcentration Marker",
    },
    "MPV(fl)": {
        "description": "Mean Platelet Volume. Often altered during rapid platelet turnover and destruction in dengue virus infection.",
        "role": "Platelet Index",
    },
    "MCHC(g/dl)": {
        "description": "Mean Corpuscular Hemoglobin Concentration. A standard RBC index.",
        "role": "RBC Index",
    },
    "Neutrophils(%)": {
        "description": "Neutropenia is a very common laboratory finding during the acute phase of dengue fever.",
        "role": "WBC Index",
    },
    "MCV(fl)": {
        "description": "Mean Corpuscular Volume. Used to assess overall red blood cell status.",
        "role": "RBC Index",
    },
    "MCH(pg)": {
        "description": "Mean Corpuscular Hemoglobin. Assesses hemoglobin weight per red cell.",
        "role": "RBC Index",
    },
    "PCT(%)": {
        "description": "Plateletcrit. Total platelet mass. Significantly drops due to hallmark thrombocytopenia.",
        "role": "Platelet Index",
    },
    "Age": {
        "description": "Demographic factor. Different age groups may have varying risks of severe dengue manifestations.",
        "role": "Demographic",
    },
    "PDW(%)": {
        "description": "Platelet Distribution Width. Increased PDW indicates high variation in platelet size, often seen due to reactive bone marrow during dengue thrombocytopenia.",
        "role": "Platelet Index",
    },
    "platelet_Moderate_Thrombocytopenia": {
        "description": "Moderate Thrombocytopenia (Platelets 50k-100k). Thrombocytopenia is a hallmark of dengue fever and a critical marker for systemic involvement.",
        "role": "Platelet Marker",
    },
    "Hemoglobin(g/dl)": {
        "description": "May initially appear normal or elevated due to hemoconcentration and plasma leakage.",
        "role": "Hemoconcentration Marker",
    },
    "platelet_Mild_Thrombocytopenia": {
        "description": "Mild Thrombocytopenia (Platelets 100k-150k). Early indicator of platelet drop associated with dengue infection.",
        "role": "Platelet Marker",
    },
    "Monocytes(%)": {
        "description": "Monocytosis can be observed, particularly during the recovery phase of dengue.",
        "role": "WBC Index",
    }
}


# -----------------------------
# Utilities
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    try:
        model = joblib.load(model_path)
        if not hasattr(model, 'predict'):
            raise AttributeError("Loaded object is not a valid model.")
        return model
    except FileNotFoundError:
        st.error(f"**Fatal Error:** Model file not found at '{model_path}'.")
        st.stop()
    except Exception as e:
        st.error(f"**Fatal Error:** Could not load model. Details: {e}")
        st.stop()


def risk_label_from_proba(p_high: float) -> str:
    if p_high < 0.30:
        return "Low Risk (Unlikely/Mild)"
    elif p_high < 0.70:
        return "Moderate Risk (Dengue w/o Warning Signs)"
    else:
        return "High Risk (Dengue with Warning Signs)"


# -----------------------------
# Preprocessing
# -----------------------------
def preprocess_for_prediction(df_raw):
    """
    Prepares raw user inputs for the Dengue model.
    """
    expected_features = [
        "abs_lymphocytes",
        "RDW-CV(%)",
        "HCT(%)",
        "MPV(fl)",
        "MCHC(g/dl)",
        "Neutrophils(%)",
        "MCV(fl)",
        "MCH(pg)",
        "PCT(%)",
        "Age",
        "PDW(%)",
        "platelet_Moderate_Thrombocytopenia",
        "Hemoglobin(g/dl)",
        "platelet_Mild_Thrombocytopenia",
        "Monocytes(%)"
    ]

    processed_df = pd.DataFrame(index=df_raw.index)
    binary_map = {'Yes': 1, 'No': 0}

    for feature in expected_features:
        if feature in df_raw.columns:
            if "Thrombocytopenia" in feature:
                processed_df[feature] = df_raw[feature].map(binary_map).fillna(0)
            else:
                processed_df[feature] = pd.to_numeric(df_raw[feature])
        else:
            st.error(f"Critical Error: Form input missing for '{feature}'.")
            return None

    return processed_df[expected_features]


# -----------------------------
# Explanations
# -----------------------------
def explain_risk_factors_dengue(patient_data_text):
    table_data = []

    for feat_key, val in patient_data_text.items():
        # Identify if this feature is a critical marker based on value
        is_flagged = False
        risk_level = "Moderate"

        # Thrombocytopenia indicators
        if "Thrombocytopenia" in feat_key and val == "Yes":
            is_flagged = True
            risk_level = "High" if "Moderate" in feat_key else "Moderate"

        # Hematocrit elevation indicator (generalized example threshold > 48)
        if feat_key == "HCT(%)" and float(val) > 48.0:
            is_flagged = True
            risk_level = "High"

        if is_flagged:
            info = VALIDATION_FEATURES_DENGUE.get(feat_key, {})
            feat_display = feat_key.replace('_', ' ').title()

            table_data.append({
                "Feature": feat_display,
                "Value": str(val),
                "Risk": risk_level,
                "Role": info.get("role", "Hematological Marker"),
                "Interpretation": info.get("description", "")
            })

    if not table_data:
        return None, None

    df_table = pd.DataFrame(table_data)
    df_table.drop_duplicates(subset=['Feature'], keep='first', inplace=True)

    table_html = df_table.to_html(index=False, escape=False)

    styled_html = f"""
    <div style='overflow-x:auto;'>
        <style>
            table {{ border-collapse: collapse; width: 100%; margin-top: 10px;}}
            th, td {{ border: 1px solid #ccc; padding: 8px; text-align: left !important; color: #000000 !important; }}
            th {{ background-color: #f2f2f2; font-weight: bold; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
        </style>
        {table_html}
    </div>
    """
    return df_table, styled_html


# -----------------------------
# PDF Report
# -----------------------------
def generate_pdf_report(user_vals, risk_label, percent, df_table=None):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_left_margin(15)
    pdf.set_right_margin(15)

    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "HemaDengue Pro - Hematological Risk Report", ln=True, align="C")
    pdf.ln(5)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "1. Patient Clinical Profile", ln=True)
    pdf.set_font("Arial", "", 10)

    col_width = 90
    for key, val in user_vals.items():
        display_name = key.replace('_', ' ')
        pdf.cell(col_width, 6, f"{display_name}: {val}", border=1)
        if list(user_vals.keys()).index(key) % 2 != 0:
            pdf.ln()
    pdf.ln(10)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "2. Risk Assessment", ln=True)
    pdf.set_font("Arial", "B", 14)

    color = (220, 53, 69) if "High Risk" in risk_label else (255, 193, 7) if "Moderate Risk" in risk_label else (76,
                                                                                                                 175,
                                                                                                                 80)
    pdf.set_text_color(*color)
    pdf.cell(0, 8, f"Status: {risk_label} (Probability: {percent:.2f}%)", ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(5)

    if df_table is not None and not df_table.empty:
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, "3. Key Risk Factors identified", ln=True)
        pdf.set_font("Arial", "", 8)
        headers = ["Feature", "Role", "Interpretation"]
        widths = [40, 40, 100]
        for i, h in enumerate(headers):
            pdf.cell(widths[i], 6, h, border=1)
        pdf.ln()
        for _, row in df_table.iterrows():
            pdf.cell(widths[0], 6, str(row['Feature']), border=1)
            pdf.cell(widths[1], 6, str(row['Role']), border=1)
            pdf.cell(widths[2], 6, str(row['Interpretation'])[:65] + "...", border=1, ln=True)

    pdf.ln(10)
    pdf.set_font("Arial", "I", 8)
    pdf.cell(0, 8, "Generated by HemaDengue Pro - Clinical AI Prototype. Ref guidelines by WHO.", ln=True, align="R")
    return bytes(pdf.output())


# -----------------------------
# Main UI
# -----------------------------
st.set_page_config(page_title="HemaDengue Pro", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');

    .stApp { background-color: #f8f9fa !important; font-family: 'Roboto', sans-serif !important; }
    h1, h2, h3 { font-family: 'Roboto', sans-serif !important; color: #202124 !important; font-weight: 500;}

    /* Clean form box */
    div[data-testid="stForm"] {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 24px;
        box-shadow: 0 1px 2px 0 rgba(60,64,67,0.3), 0 1px 3px 1px rgba(60,64,67,0.15);
    }

    /* Subdued professional button */
    .stButton>button { 
        background-color: #1a73e8 !important; 
        color: white !important; 
        border-radius: 4px; 
        width: 100%; 
        height: 48px; 
        font-weight: 500; 
        border: none;
        transition: background-color 0.2s;
    }
    .stButton>button:hover { 
        background-color: #1557b0 !important;
    }

    /* Override number inputs for clean look */
    .stNumberInput > div > div > input, .stSelectbox > div > div {
        border-radius: 4px !important;
        border: 1px solid #dadce0 !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("⚕️ HemaDengue Pro")
st.markdown(
    "<p style='font-size: 1.1rem; color: #5f6368; margin-top: -15px; margin-bottom: 30px;'>Clinical Decision Support System for Dengue Hematological Assessment</p>",
    unsafe_allow_html=True)

model = load_model("best_global_stacking_model.pkl")

tabs = st.tabs(["Risk Assessment", "Feature Knowledge Base"])

with tabs[0]:
    st.subheader("Patient Hematology Data Entry (CBC)")
    st.info("Please enter the patient's Complete Blood Count (CBC) and Platelet indices.")

    with st.form("dengue_pred_form"):
        col1, col2, col3 = st.columns(3)
        inputs = {}

        with col1:
            st.markdown("**WBC & Demographics**")
            inputs['Age'] = st.number_input("Age (Years)", min_value=1, max_value=120, value=30)
            inputs['abs_lymphocytes'] = st.number_input("Absolute Lymphocytes (x10^9/L)", min_value=0.0, max_value=50.0,
                                                        value=2.0, format="%.2f")
            inputs['Neutrophils(%)'] = st.number_input("Neutrophils (%)", min_value=0.0, max_value=100.0, value=55.0,
                                                       format="%.1f")
            inputs['Monocytes(%)'] = st.number_input("Monocytes (%)", min_value=0.0, max_value=100.0, value=5.0,
                                                     format="%.1f")

        with col2:
            st.markdown("**RBC Indices & Hemoconcentration**")
            inputs['Hemoglobin(g/dl)'] = st.number_input("Hemoglobin (g/dl)", min_value=0.0, max_value=25.0, value=14.0,
                                                         format="%.1f")
            inputs['HCT(%)'] = st.number_input("Hematocrit (%)", min_value=0.0, max_value=80.0, value=42.0,
                                               format="%.1f")
            inputs['MCV(fl)'] = st.number_input("MCV (fl)", min_value=0.0, max_value=150.0, value=90.0, format="%.1f")
            inputs['MCH(pg)'] = st.number_input("MCH (pg)", min_value=0.0, max_value=50.0, value=30.0, format="%.1f")
            inputs['MCHC(g/dl)'] = st.number_input("MCHC (g/dl)", min_value=0.0, max_value=50.0, value=33.0,
                                                   format="%.1f")
            inputs['RDW-CV(%)'] = st.number_input("RDW-CV (%)", min_value=0.0, max_value=30.0, value=13.0,
                                                  format="%.1f")

        with col3:
            st.markdown("**Platelet Indices & Markers**")
            inputs['MPV(fl)'] = st.number_input("MPV (fl)", min_value=0.0, max_value=20.0, value=10.0, format="%.1f")
            inputs['PCT(%)'] = st.number_input("Plateletcrit PCT (%)", min_value=0.0, max_value=1.0, value=0.25,
                                               format="%.3f")
            inputs['PDW(%)'] = st.number_input("PDW (%)", min_value=0.0, max_value=30.0, value=12.0, format="%.1f")
            inputs['platelet_Mild_Thrombocytopenia'] = st.selectbox("Mild Thrombocytopenia (100k-150k)?", ["No", "Yes"],
                                                                    index=0)
            inputs['platelet_Moderate_Thrombocytopenia'] = st.selectbox("Moderate Thrombocytopenia (50k-100k)?",
                                                                        ["No", "Yes"], index=0)

        submit = st.form_submit_button("Predict Dengue Risk", type="primary")

    if submit:
        raw_df = pd.DataFrame([inputs])
        X_processed = preprocess_for_prediction(raw_df)

        if X_processed is not None:
            try:
                # Assuming standard predict_proba exists
                probs = model.predict_proba(X_processed)[0]
                p_pos = probs[1]
                risk_lbl = risk_label_from_proba(p_pos)
                pct = p_pos * 100

                st.markdown("---")
                st.subheader("Diagnostic Assessment Results")
                colA, colB = st.columns([1, 2])

                with colA:
                    st.metric(label="Calculated Risk Probability", value=f"{pct:.1f}%")
                    if "Low" in risk_lbl:
                        st.success(f"Status: {risk_lbl}")
                    elif "Moderate" in risk_lbl:
                        st.warning(f"Status: {risk_lbl}")
                    else:
                        st.error(f"Status: {risk_lbl}")

                with colB:
                    st.markdown("**Risk Severity Index**")
                    st.progress(int(pct) if int(pct) <= 100 else 100)
                    st.caption(
                        "A higher percentage indicates a stronger correlation with severe hematological dengue markers based on clinical training data.")

                st.markdown("---")

                st.subheader("Feature Contribution & WHO Markers")
                df_exp, html_exp = explain_risk_factors_dengue(inputs)
                if html_exp:
                    st.markdown(html_exp, unsafe_allow_html=True)
                else:
                    st.success(
                        "No significant WHO hemoconcentration or thrombocytopenia warning signs detected heavily from the inputs.")

                st.subheader("Download Clinical Report")
                pdf_data = generate_pdf_report(inputs, risk_lbl, pct, df_exp)
                st.download_button("Download Dengue Clinical Report (PDF)", data=pdf_data,
                                   file_name="HemaDengue_Pro_Clinical_Report.pdf", mime="application/pdf")

            except Exception as e:
                st.error(f"Prediction Error: {str(e)}")

with tabs[1]:
    st.header("Clinical Reference: Dengue Hematological Indicators")
    st.markdown("Consistent with the 2009 WHO guidelines for Dengue diagnosis and warning signs.")

    for feat, meta in VALIDATION_FEATURES_DENGUE.items():
        with st.expander(f"{feat.replace('_', ' ').title()} ({meta['role']})"):
            st.write(meta['description'])