from global_settings import FLAG
import tensorflow as tf
import subprocess
import os

gpu_devices = tf.config.list_physical_devices('GPU')
cpu_devices = tf.config.list_physical_devices('CPU')

try:
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    tpu_devices = tf.config.list_logical_devices('TPU')
    print("Number of TPU devices: ", len(tpu_devices))
    strategy = tf.distribute.TPUStrategy(resolver)
except KeyError:
    if bool(len(gpu_devices)):
        print("Number of GPU devices: ", len(gpu_devices))
        subprocess.run("nvidia-smi", stdout=subprocess.PIPE, shell=True)
        strategy = tf.distribute.MirroredStrategy()
    elif bool(len(cpu_devices)):
        print("Number of CPU devices: ", len(cpu_devices))
        if FLAG in ['colab', 'floyd']:
            subprocess.Popen("cat /proc/cpuinfo | grep name | uniq", stdout=subprocess.PIPE, shell=True)
        else:
            subprocess.Popen("sysctl -n machdep.cpu.brand_string", stdout=subprocess.PIPE, shell=True)
        strategy = tf.distribute.get_strategy()
    else:
        raise EnvironmentError("No physical devices or remote devices")


train_config = {
    "use_gen": False,
    "epoch": 150,
    "metrics": ['CategoricalAccuracy', 'Precision', 'Recall'],
    "implementation": 2
    # [0: encoder only, 1: encoder-decoder freeze, 2: encoder-decoder trainable]
}

evalu_config = {
    "kfold": 10
}
