from tensorflow.keras.metrics import CategoricalAccuracy, Precision, Recall
from tensorflow.keras.metrics import AUC, TruePositives, FalsePositives, TrueNegatives, FalseNegatives
from tensorflow_addons.metrics import F1Score


metrics = [
    CategoricalAccuracy(name='accuracy'),
    Precision(name='precision'),
    Recall(name='recall'),
    # AUC(name='auc'),
    # TruePositives(name='tp'),
    # TrueNegatives(name='tn'),
    # FalsePositives(name='fp'),
    # FalseNegatives(name='fn'),
]

train_config = {
    "use_gen": False,
    "epoch": 100,
    "batch": 128,
    "metrics": metrics,
}

evalu_config = {
    "kfold": 10
}
