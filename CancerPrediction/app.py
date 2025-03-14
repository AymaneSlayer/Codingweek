import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import os
import base64

# --- Fonction pour charger et encoder l'image locale en base64 ---
def get_base64_of_bin_file(bin_file):
    with open(bin_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# --- Fonction pour définir l'image de fond avec une image WebP locale ---
def set_background(webp_file):
    bin_str = get_base64_of_bin_file(webp_file)
    page_bg_img = f"""
    <style>
    .stApp {{
        background-image: url("data:image/webp;base64,{bin_str}");
        background-size: cover;
        background-position: center;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Définition de l'image de fond
set_background("CancerPrediction/Background.webp")

# --- Injection de CSS pour améliorer le design de la sidebar et des éléments de saisie ---
st.markdown(
    """
    <style>
    /* Style général de la sidebar */
    .css-1d391kg {
        background: rgba(255, 255, 255, 0.85);
        border-radius: 10px;
        padding: 10px;
    }
    /* Style personnalisé pour les sliders : dégradé sur la barre */
    div[data-baseweb="slider"] > div > div > div {
        background: linear-gradient(90deg, #FF5733, #FFC300);
    }
    /* Personnalisation des champs numériques */
    .stNumberInput > div > input {
        border: 2px solid #FF5733;
        border-radius: 5px;
        padding: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =============================
# 1. CHARGEMENT DU MODÈLE XGBOOST
# =============================
@st.cache_data
def load_xgb_model(path: str):
    model = xgb.XGBClassifier()
    model.load_model(path)
    return model

MODEL_PATH = "CancerPrediction/models/xgboost_trained_model.json"
model = load_xgb_model(MODEL_PATH)

# =============================
# 2. CONFIGURATION DE L'EXPLAINER SHAP
# =============================
explainer = shap.TreeExplainer(model)

# =============================
# 3. INTERFACE STREAMLIT
# =============================
st.title("🩺 Interface de dépistage du cancer du col de l'utérus")

# Barre latérale pour la saisie des informations du patient
st.sidebar.header("📝 Informations du patient")

# Champs de saisie pour chaque variable avec des emojis dans les libellés
age = st.sidebar.number_input("👤 Âge", min_value=15, max_value=100, value=30)
smokes = st.sidebar.selectbox("🚬 Fumeur", ["Non", "Oui"])
num_sexual_partners = st.sidebar.number_input("💑 Nombre de partenaires sexuels", min_value=0, max_value=50, value=1)
first_sexual_intercourse = st.sidebar.number_input("📅 Âge au premier rapport", min_value=10, max_value=30, value=18)
num_pregnancies = st.sidebar.number_input("🤰 Nombre de grossesses", min_value=0, max_value=20, value=0)
smokes_years = st.sidebar.number_input("⌛ Durée de tabagisme (années)", min_value=0, max_value=50, value=5)
smokes_packs_per_year = st.sidebar.number_input("🚬 Paquets/an", min_value=0, max_value=100, value=1)
hormonal_contraceptives_years = st.sidebar.number_input("💊 Contraceptifs hormonaux (années)", min_value=0, max_value=50, value=0)
iud_years = st.sidebar.number_input("🩺 IUD (années)", min_value=0, max_value=50, value=0)
stds = st.sidebar.selectbox("🦠 Antécédents de MST (STDs)", options=[0, 1], format_func=lambda x: "Non" if x == 0 else "Oui")
stds_number = st.sidebar.number_input("🔢 Nombre de MST diagnostiquées", min_value=0, max_value=10, value=0)
schiller = st.sidebar.selectbox("🔬 Test Schiller", options=[0, 1], format_func=lambda x: "Négatif" if x == 0 else "Positif")
hinselmann = st.sidebar.selectbox("🔬 Test Hinselmann", options=[0, 1], format_func=lambda x: "Négatif" if x == 0 else "Positif")
citology = st.sidebar.selectbox("🧪 Test de cytologie", options=[0, 1], format_func=lambda x: "Négatif" if x == 0 else "Positif")
dx_cancer = st.sidebar.selectbox("⚠️ Diagnostic Cancer (Dx:Cancer)", options=[0, 1], format_func=lambda x: "Négatif" if x == 0 else "Positif")
dx_cin = st.sidebar.selectbox("⚠️ Diagnostic CIN (Dx:CIN)", options=[0, 1], format_func=lambda x: "Négatif" if x == 0 else "Positif")

# Liste des colonnes attendues par le modèle
model_columns = [
    'Number of sexual partners', 'Num of pregnancies', 'Smokes', 'Smokes (years)', 'Smokes (packs/year)', 
    'Hormonal Contraceptives', 'Hormonal Contraceptives (years)', 'IUD', 'IUD (years)', 'STDs', 
    'STDs:cervical condylomatosis', 'STDs:vaginal condylomatosis', 'STDs:syphilis', 'STDs:pelvic inflammatory disease', 
    'STDs:genital herpes', 'STDs:molluscum contagiosum', 'STDs:AIDS', 'STDs:HIV', 'STDs:Hepatitis B', 
    'STDs:HPV', 'Dx:Cancer', 'Dx:CIN', 'Dx', 'Hinselmann', 'Schiller', 'Citology', 'Age_log', 
    'First sexual intercourse_log'
]

# Bouton de validation
if st.sidebar.button("✅ Valider les données"):

    # Créer un DataFrame avec les données saisies
    input_data = {
        'Age': [age],
        'Smokes': [smokes == 'Oui'],
        'Number of sexual partners': [num_sexual_partners],
        'First sexual intercourse': [first_sexual_intercourse],
        'Num of pregnancies': [num_pregnancies],
        'Smokes (years)': [smokes_years],
        'Smokes (packs/year)': [smokes_packs_per_year],
        'Hormonal Contraceptives (years)': [hormonal_contraceptives_years],
        'IUD (years)': [iud_years],
        'STDs': [stds],
        'STDs (number)': [stds_number],
        'Schiller': [schiller],
        'Hinselmann': [hinselmann],
        'Citology': [citology],
        'Dx:Cancer': [dx_cancer],
        'Dx:CIN': [dx_cin],
    }
    df_input = pd.DataFrame(input_data)

    # Aligner les colonnes de df_input avec celles attendues par le modèle
    df_input_aligned = df_input.reindex(columns=model_columns, fill_value=0)

    st.subheader("📝 Données saisies :")
    st.write(df_input_aligned)

    # =============================
    # 4. FAIRE LA PRÉDICTION
    # =============================
    proba = model.predict_proba(df_input_aligned)[0, 1]
    pred_class = model.predict(df_input_aligned)[0]

    st.subheader("🔮 Résultat de la prédiction :")
    pourcentage = round(proba * 100, 2)
    st.write(f"**Probabilité estimée d'avoir un cancer du col de l'utérus : {pourcentage}%**")

    if pred_class == 1:
        st.error("Le modèle prédit un risque élevé (Biopsy = 1).")
    else:
        st.success("Le modèle prédit un risque faible (Biopsy = 0).")

    # =============================
    # 5. INTERPRÉTATION LOCALE AVEC SHAP
    # =============================
    shap_values = explainer.shap_values(df_input_aligned)

    st.subheader("📊 Interprétation locale (SHAP) :")
    # Pour le cas binaire, si shap_values est une liste, on prend la composante de la classe positive
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values

    fig, ax = plt.subplots(figsize=(8, 6))
    shap.waterfall_plot(
        shap.Explanation(
            values=sv[0],
            base_values=explainer.expected_value,
            data=df_input_aligned.iloc[0, :],
            feature_names=df_input_aligned.columns
        ),
        max_display=10,
        show=False
    )
    st.pyplot(fig)
