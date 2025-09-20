import numpy as np
from app.utils.network_utils import calculate_metrics

def test_calculate_metrics():
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    metrics = calculate_metrics(y_true, y_pred)
    assert 'Accuracy' in metrics
    assert metrics['Accuracy'] >= 0 and metrics['Accuracy'] <= 1
