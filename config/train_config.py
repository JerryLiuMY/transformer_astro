from tensorflow.keras.metrics import TruePositives, FalsePositives, TrueNegatives, FalseNegatives
from tensorflow.keras.metrics import CategoricalAccuracy, TopKCategoricalAccuracy
from tensorflow.keras.metrics import Precision, Recall, AUC
from tensorflow_addons.metrics import F1Score


metrics = [
    CategoricalAccuracy(name='accuracy'),
    TopKCategoricalAccuracy(k=5, name='k_accuracy'),
    Precision(name='precision'),
    Recall(name='recall'),
    AUC(name='auc'),
    # TruePositives(name='tp'),
    # TrueNegatives(name='tn'),
    # FalsePositives(name='fp'),
    # FalseNegatives(name='fn'),
]

train_config = {
    "generator": False,
    "epoch": 10,
    "batch": 128,
    "sample": 40000,
    "metrics": metrics
}

