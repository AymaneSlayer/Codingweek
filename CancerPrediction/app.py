import streamlit as st
import pandas as pd
import joblib

# Titre de l'application
st.title("Prédiction du risque de cancer du col de l'utérus")

# Charger le modèle entraîné
@st.cache_resource  # Cache le modèle pour éviter de le recharger à chaque interaction
def load_model():
    model = joblib.load("models/trained_model.pkl")  # Charger le modèle sauvegardé
    return model

model = load_model()

# Formulaire pour saisir les informations du patient
st.sidebar.header("Entrez les informations du patient")

# Exemples de champs à remplir (ajustez en fonction de votre modèle)
age = st.sidebar.number_input("Âge", min_value=15, max_value=100, value=30)
num_sexual_partners = st.sidebar.number_input("Nombre de partenaires sexuels", min_value=0, max_value=50, value=1)
first_sexual_intercourse = st.sidebar.number_input("Âge au premier rapport sexuel", min_value=10, max_value=30, value=18)
num_pregnancies = st.sidebar.number_input("Nombre de grossesses", min_value=0, max_value=20, value=0)
smokes = st.sidebar.selectbox("Fumeuse", options=[0, 1], format_func=lambda x: "Non" if x == 0 else "Oui")
hormonal_contraceptives = st.sidebar.selectbox("Utilisation de contraceptifs hormonaux", options=[0, 1], format_func=lambda x: "Non" if x == 0 else "Oui")
iud = st.sidebar.selectbox("Utilisation d'un dispositif intra-utérin (IUD)", options=[0, 1], format_func=lambda x: "Non" if x == 0 else "Oui")
stds = st.sidebar.selectbox("Antécédents de MST", options=[0, 1], format_func=lambda x: "Non" if x == 0 else "Oui")
stds_number = st.sidebar.number_input("Nombre de MST diagnostiquées", min_value=0, max_value=10, value=0)

# Bouton pour lancer la prédiction
if st.sidebar.button("Prédire le risque de cancer"):

    # Créer un DataFrame avec les données saisies
    input_data = pd.DataFrame({
        "Age": [age],
        "Number of sexual partners": [num_sexual_partners],
        "First sexual intercourse": [first_sexual_intercourse],
        "Num of pregnancies": [num_pregnancies],
        "Smokes": [smokes],
        "Hormonal Contraceptives": [hormonal_contraceptives],
        "IUD": [iud],
        "STDs": [stds],
        "STDs (number)": [stds_number]
    })

    # Afficher les données saisies
    st.write("Données saisies :")
    st.write(input_data)

    # Faire la prédiction
    try:
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)

        # Afficher le résultat
        st.subheader("Résultat de la prédiction :")
        if prediction[0] == 1:
            st.error("Risque de cancer du col de l'utérus détecté.")
        else:
            st.success("Pas de risque de cancer du col de l'utérus détecté.")

        # Afficher les probabilités
        st.write(f"Probabilité de risque de cancer : {prediction_proba[0][1]:.2f}")
        st.write(f"Probabilité de non-risque : {prediction_proba[0][0]:.2f}")

    except Exception as e:
        st.error(f"Une erreur s'est produite lors de la prédiction : {e}")
