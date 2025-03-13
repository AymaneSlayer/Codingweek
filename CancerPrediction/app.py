import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import os

# =============================
# 1. CHARGEMENT DU MODÈLE XGBOOST
# =============================
@st.cache_data
def load_xgb_model(path: str):
    model = xgb.XGBClassifier()
    model.load_model(path)  # Charge le modèle XGBoost depuis un fichier
    return model

# Chemin vers votre modèle XGBoost pré-entraîné (mettre le bon chemin)
MODEL_PATH = "CancerPrediction/models/xgboost_trained_model.json"  # Remplacer par le bon chemin de votre modèle
model = load_xgb_model(MODEL_PATH)

# =============================
# 2. CONFIGURATION DE L'EXPLAINER SHAP
# =============================
# L'explainer sera utilisé pour interpréter localement la prédiction
explainer = shap.TreeExplainer(model)

# =============================
# 3. INTERFACE STREAMLIT
# =============================
st.title("Interface de saisie pour le dépistage du cancer du col de l'utérus")

# Barre latérale pour la saisie des informations du patient
st.sidebar.header("Entrez les informations du patient")

# Champs de saisie pour chaque variable
age = st.sidebar.number_input("Âge", min_value=15, max_value=100, value=30)
smokes = st.sidebar.selectbox("Fumeur", ["Non", "Oui"])
num_sexual_partners = st.sidebar.number_input("Nombre de partenaires sexuels", min_value=0, max_value=50, value=1)
first_sexual_intercourse = st.sidebar.number_input("Âge au premier rapport sexuel", min_value=10, max_value=30, value=18)
num_pregnancies = st.sidebar.number_input("Nombre de grossesses", min_value=0, max_value=20, value=0)
smokes_years = st.sidebar.number_input("Durée de tabagisme (années)", min_value=0, max_value=50, value=5)
smokes_packs_per_year = st.sidebar.number_input("Consommation de tabac (paquets/an)", min_value=0, max_value=100, value=1)
hormonal_contraceptives_years = st.sidebar.number_input("Durée d'utilisation de contraceptifs hormonaux (années)", min_value=0, max_value=50, value=0)
iud_years = st.sidebar.number_input("Durée d'utilisation du dispositif intra-utérin (IUD, années)", min_value=0, max_value=50, value=0)
stds = st.sidebar.selectbox("Antécédents de MST (STDs)", options=[0, 1], format_func=lambda x: "Non" if x == 0 else "Oui")
stds_number = st.sidebar.number_input("Nombre de MST diagnostiquées", min_value=0, max_value=10, value=0)
schiller = st.sidebar.selectbox("Test Schiller", options=[0, 1], format_func=lambda x: "Négatif" if x == 0 else "Positif")
hinselmann = st.sidebar.selectbox("Test Hinselmann", options=[0, 1], format_func=lambda x: "Négatif" if x == 0 else "Positif")
citology = st.sidebar.selectbox("Test de cytologie", options=[0, 1], format_func=lambda x: "Négatif" if x == 0 else "Positif")
dx_cancer = st.sidebar.selectbox("Diagnostic Cancer (Dx:Cancer)", options=[0, 1], format_func=lambda x: "Négatif" if x == 0 else "Positif")
dx_cin = st.sidebar.selectbox("Diagnostic CIN (Dx:CIN)", options=[0, 1], format_func=lambda x: "Négatif" if x == 0 else "Positif")


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
if st.sidebar.button("Valider les données"):

    # Créez un DataFrame avec les données saisies
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

    # Créez un DataFrame avec les colonnes que vous avez
    df_input = pd.DataFrame(input_data)

    # Aligner les colonnes de df_input avec celles du modèle
    df_input_aligned = df_input.reindex(columns=model_columns, fill_value=0)  # Remplir les colonnes manquantes avec 0

    st.subheader("Données saisies :")
    st.write(df_input_aligned)

    # =============================
    # 4. FAIRE LA PREDICTION
    # =============================
    # Probabilité d'appartenir à la classe "cancer" (Biopsy = 1)
    proba = model.predict_proba(df_input_aligned)[0, 1]
    # Prédiction binaire
    pred_class = model.predict(df_input_aligned)[0]

    # =============================
    # 5. AFFICHER LE RÉSULTAT
    # =============================
    st.subheader("Résultat de la prédiction :")

    # Affichage de la probabilité sous forme de pourcentage
    pourcentage = round(proba * 100, 2)
    st.write(f"**Probabilité estimée d'avoir un cancer du col de l'utérus : {pourcentage}%**")

    # Affichage de la classe prédite
    if pred_class == 1:
        st.error("Le modèle prédit un risque élevé (Biopsy = 1).")
    else:
        st.success("Le modèle prédit un risque faible (Biopsy = 0).")

    # =============================
    # 6. INTERPRÉTATION LOCALE AVEC SHAP
    # =============================
    # Calcul des valeurs SHAP pour l'observation en cours
    shap_values = explainer.shap_values(df_input_aligned)

    st.subheader("Interprétation locale (SHAP) :")

    # Cas binaire : shap_values est généralement un array 2D
    #   - Si shap_values est une liste (cas multi-classes), prendre shap_values[1] pour la classe positive
    #   - Sinon, directement shap_values

    # Pour la cohérence, on vérifie si shap_values est une liste (multi-class) ou non
    if isinstance(shap_values, list):
        # On récupère la composante liée à la classe 1 (positive)
        sv = shap_values[1]
    else:
        sv = shap_values

    # --- Waterfall Plot ---
    fig, ax = plt.subplots(figsize=(8, 6))
    shap.waterfall_plot(shap.Explanation(values=sv[0],
                                         base_values=explainer.expected_value,
                                         data=df_input_aligned.iloc[0,:],
                                         feature_names=df_input_aligned.columns),
                        max_display=10, show=False)
    st.pyplot(fig)
