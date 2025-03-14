if __name__ == "__main__":
    file_path = "data/cancer_data.csv"  # file path depends on where the data is located on your pc
    loader = CancerDataLoader(file_path)

    # Charger les données
    X, y = loader.get_features_labels()

    # Afficher la taille des données
    print(f"Features shape: {X.shape}")
    print(f"Labels shape: {y.shape}")

    # Afficher un aperçu des premières lignes
    print("\nAperçu des données :")
    print(pd.read_csv(file_path).head())
