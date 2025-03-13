import pytest
from src.model import predict_cancer

def test_prediction():
    sample_input = [0.1, 0.5, 0.3]  # Example features
    result = predict_cancer(sample_input)
    assert result in [0, 1]  # Assuming binary classification (0 = benign, 1 = malignant)

