#Fonctions pour l'extraction et la création de caractéristiques.
def encode_categorical_features(data):
    # Encoder les variables catégorielles
    data = pd.get_dummies(data, drop_first=True)
    return datadef encode_categorical_features(data):
    # Encoder les variables catégorielles
    data = pd.get_dummies(data, drop_first=True)
    return data
