import pandas as pd
class CancerDataLoader:
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)

    def get_features_labels(self):
        X = self.data.iloc[:, :-1]  # All columns except last (features)
        y = self.data.iloc[:, -1]   # Last column (label)
        return X.values, y.values

# Example Usage
file_path = r"CancerPrediction/data/processed_data.csv"
loader = CancerDataLoader(file_path)
X, y = loader.get_features_labels()
print(X.shape, y.shape)

