from tensorflow.keras.metrics import CategoricalAccuracy, Precision, Recall
from global_settings import FLAG
import tensorflow as tf
import subprocess
import os

tpu_devices = tf.config.list_logical_devices('TPU')
gpu_devices = tf.config.list_physical_devices('GPU')
cpu_devices = tf.config.list_physical_devices('CPU')

if bool(len(tpu_devices)):
    print("Number of TPU devices: ", len(tpu_devices))
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.TPUStrategy(resolver)
elif bool(len(gpu_devices)):
    print("Number of GPU devices: ", len(gpu_devices))
    subprocess.run("nvidia-smi")
    strategy = tf.distribute.MirroredStrategy()
elif bool(len(cpu_devices)):
    print("Number of CPU devices: ", len(gpu_devices))
    if FLAG in ['colab', 'floyd']:
        subprocess.run("cat /proc/cpuinfo  | grep 'name'| uniq")
    else:
        subprocess.run("sysctl -n machdep.cpu.brand_string")
    strategy = tf.distribute.get_strategy()
else:
    raise EnvironmentError("No physical devices or remote devices")


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
