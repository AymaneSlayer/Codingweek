#Fonctions pour nettoyer et transformer les données.
def clean_data(data):
    # Supprimer les valeurs manquantes ou les remplacer
    data = data.dropna()
    return data
