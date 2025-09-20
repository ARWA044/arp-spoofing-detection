import numpy as np
from core.ml_models import train_all_models

def test_train_all_models_shapes():
    X_train = np.random.rand(100, 3)
    y_train = np.random.randint(0, 2, 100)
    X_test = np.random.rand(20, 3)
    y_test = np.random.randint(0, 2, 20)
    models, predictions = train_all_models(X_train, y_train, X_test, y_test)
    assert 'Random Forest' in models
    assert predictions['Random Forest']['pred'].shape[0] == 20
    assert 'Autoencoder' in models
    assert predictions['Autoencoder']['pred'].shape[0] == 20
