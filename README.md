# ⚕️ HemaDengue Pro

**HemaDengue Pro** is an open-source clinical decision support framework developed to assess the risk of Dengue Fever using routine Complete Blood Count (CBC) and hematological markers. 

By leveraging a robust **Stacking Ensemble Machine Learning approach**, the application evaluates 15 critical blood indices against established World Health Organization (WHO) 2009 criteria for clinical Dengue Warning Signs.

## 🎯 Features
- **Explainable AI:** Identifies and flags the critical hematological markers driving the risk prediction (e.g., dropping platelets, rising hematocrit).
- **Clinical Dashboard UI:** The secure, minimalist interface mimics institutional Electronic Health Record (EHR) systems, completely avoiding consumer-app distractions.
- **Diagnostic Export:** Auto-generates standardized, downloadable Clinical PDF Reports for patient files.
- **WHO Alignment:** Incorporates an integrated knowledge-base dynamically aligned with WHO warning indicators for plasma leakage and systemic hematological severity.

## 🧠 Data & Model Architecture 
This predictive pipeline is powered by `best_global_stacking_model.pkl`. The model parameters were refined through advanced computational optimization, specifically yielding an elite subset of 15 features extracted via *Recursive Feature Elimination (RFE)*, *Gain Ratio*, and *Information Gain*.

**Key Input Vectors Include:**
*   **WBC Differentials:** Absolute Lymphocytes, Neutrophils, Monocytes.
*   **Hemoconcentration Flags:** HCT, Hemoglobin, MCV, MCH, MCHC, RDW-CV.
*   **Platelet Destruction Markers:** MPV, PDW, PCT, and binned Mild/Moderate Thrombocytopenia classifications.

## 🚀 Local Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/shamsulalam1114/HemaDengue-Pro.git
   cd HemaDengue-Pro
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Boot the Front-End**
   ```bash
   streamlit run app.py
   ```

## ⚖️ Disclaimer
*This software is an explicitly experimental research prototype formulated for clinical decision support engineering. It has not been subjected to medical regulatory approval and under no circumstances operates as a diagnostic substitute for licensed medical supervision or clinical judgment.*
