import os
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from global_settings import DATA_FOLDER
from tensorflow.keras.callbacks import TensorBoard, LambdaCallback
from tensorboard.plugins.hparams import api as hp
from datetime import datetime
from tools.data_tools import data_loader, data_generator, load_one_hot
from tools.model_tools import plot_confusion, plot_to_image
from config.model_config import rnn_nums_hp, rnn_dims_hp, dnn_nums_hp
from config.train_config import train_config


generator, epoch = train_config['generator'], train_config['epoch']
batch, metrics = train_config['batch'], train_config['metrics']
metric_names = ['epoch_loss'] + ['_'.join(['epoch', _.name]) for _ in metrics]


def log_params(dataset_name):
    log_dir = os.path.join(DATA_FOLDER, f'{dataset_name}_log')
    with tf.summary.create_file_writer(log_dir).as_default():
        hp.hparams_config(
            hparams=[rnn_nums_hp, rnn_dims_hp, dnn_nums_hp],
            metrics=[hp.Metric(_) for _ in metric_names]
        )


class Base:

    def __init__(self, dataset_name, hyper_params):
        self.dataset_name = dataset_name
        self.hyper_params = hyper_params
        self._load_name()
        self._load_misc()
        self._load_path()
        self._load_data()
        self.model = None

    def _load_name(self):
        rnn_num = self.hyper_params[rnn_nums_hp]
        rnn_dim = self.hyper_params[rnn_dims_hp]
        dnn_num = self.hyper_params[dnn_nums_hp]
        now = datetime.now().strftime('%Y-%m-%d--%H-%M-%S')
        self.exp_name = '-'.join([f'rnn_num_{rnn_num}', f'rnn_dim_{rnn_dim}', f'dnn_num_{dnn_num}', now])

        print(f'--- Starting trial: {self.exp_name}')
        print({h.name: self.hyper_params[h] for h in self.hyper_params})

    def _load_misc(self):
        encoder = load_one_hot(self.dataset_name)
        self.encoder = encoder
        self.categories = encoder.categories_[0]

    def _load_path(self):
        self.log_dir = os.path.join(DATA_FOLDER, f'{self.dataset_name}_log')
        self.his_dir = os.path.join(self.log_dir, 'scalar')
        self.img_dir = os.path.join(self.log_dir, 'images')
        self.hyp_dir = os.path.join(self.log_dir, 'params')
        if not os.path.isdir(self.log_dir): os.mkdir(self.log_dir)
        if not os.path.isdir(self.his_dir): os.mkdir(self.his_dir)
        if not os.path.isdir(self.img_dir): os.mkdir(self.img_dir)
        if not os.path.isdir(self.hyp_dir): os.mkdir(self.hyp_dir)

        self.his_path = os.path.join(self.his_dir, self.exp_name)
        self.img_path = os.path.join(self.img_dir, self.exp_name)
        self.hyp_path = os.path.join(self.hyp_dir, self.exp_name)

    def _load_data(self):
        self.x_valid, self.y_valid = data_loader(self.dataset_name, 'valid')
        self.x_evalu, self.y_evalu = data_loader(self.dataset_name, 'evalu')

    def _log_confusion(self, epoch, logs):
        y_predi = tf.one_hot(tf.math.argmax(self.model.predict(self.x_evalu), axis=1), depth=len(self.categories))
        y_evalu_spar = self.encoder.inverse_transform(self.y_evalu)
        y_predi_spar = self.encoder.inverse_transform(y_predi)
        matrix = confusion_matrix(y_evalu_spar, y_predi_spar)
        confusion_figure = plot_confusion(matrix, categories=self.categories)
        confusion_image = plot_to_image(confusion_figure)

        with tf.summary.create_file_writer(self.img_path).as_default():
            tf.summary.image('Confusion Matrix', confusion_image, step=epoch)

    def _log_evalu(self):
        performs = self.model.evaluate(x=self.x_evalu, y=self.y_evalu)
        with tf.summary.create_file_writer(self.hyp_path).as_default():
            hp.hparams(self.hyper_params)
            for m, p in list(zip(metric_names, performs)):
                tf.summary.scalar(m, p, step=0)

    def train(self):
        his_callbacks = TensorBoard(log_dir=self.his_path)
        img_callbacks = LambdaCallback(on_epoch_end=self._log_confusion)
        callbacks = [his_callbacks, img_callbacks]

        if generator:
            zip_train = data_generator(self.dataset_name, 'train')
            self.model.fit(
                x=zip_train, epochs=epoch, verbose=1,
                validation_data=(self.x_valid, self.y_valid), callbacks=callbacks,
                max_queue_size=10, workers=5, use_multiprocessing=False
            )
        else:
            x_train, y_train = data_loader(self.dataset_name, 'train')
            self.model.fit(
                x=x_train, y=y_train, batch_size=batch, epochs=epoch, verbose=1,
                validation_data=(self.x_valid, self.y_valid), callbacks=callbacks,
            )

        self._log_evalu()

    def build(self):
        pass


# confusion matrix
# call back data type
# learning rate
# F1 score
# loop 10 times
# k-fold validation
# Stop training after loss stabilize
# attention model
# Phased LSTM

