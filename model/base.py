import os
import numpy as np
import tensorflow as tf
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import TensorBoard, LambdaCallback
from tensorboard.plugins.hparams import api as hp
from datetime import datetime
from tools.data_tools import data_loader, fold_loader
from tools.data_tools import DataGenerator, FoldGenerator
from tools.utils import load_one_hot
from tools.model_tools import plot_confusion, plot_to_image
from config.model_config import rnn_nums_hp, rnn_dims_hp, dnn_nums_hp
from config.exec_config import train_config

use_gen, epoch = train_config['use_gen'], train_config['epoch']
metrics, batch = train_config['metrics'], train_config['batch']
metric_names = ['epoch_loss'] + ['_'.join(['epoch', _.name]) for _ in metrics]


def log_params(exp_dir):
    with tf.summary.create_file_writer(exp_dir).as_default():
        hp.hparams_config(
            hparams=[rnn_nums_hp, rnn_dims_hp, dnn_nums_hp],
            metrics=[hp.Metric(_) for _ in metric_names]
        )


class Base:

    def __init__(self, dataset_name, hyper_param, exp_dir):
        self.dataset_name = dataset_name
        self.hyper_param = hyper_param
        self.exp_dir = exp_dir
        self._load_name()
        self._load_misc()
        self._load_path()
        self._load_data()
        self.model = None

    def _load_name(self):
        rnn_num = self.hyper_param[rnn_nums_hp]
        rnn_dim = self.hyper_param[rnn_dims_hp]
        dnn_num = self.hyper_param[dnn_nums_hp]
        now = datetime.now().strftime('%Y-%m-%d--%H-%M-%S')
        self.exp_name = '-'.join([f'rnn_num_{rnn_num}', f'rnn_dim_{rnn_dim}', f'dnn_num_{dnn_num}', now])

        print(f'--- Starting trial: {self.exp_name}')
        print({h.name: self.hyper_param[h] for h in self.hyper_param})

    def _load_misc(self):
        encoder = load_one_hot(self.dataset_name)
        self.encoder = encoder
        self.categories = encoder.categories_[0]

    def _load_path(self):
        self.his_dir = os.path.join(self.exp_dir, 'scalar')
        self.img_dir = os.path.join(self.exp_dir, 'images')
        self.hyp_dir = os.path.join(self.exp_dir, 'params')
        if not os.path.isdir(self.his_dir): os.mkdir(self.his_dir)
        if not os.path.isdir(self.img_dir): os.mkdir(self.img_dir)
        if not os.path.isdir(self.hyp_dir): os.mkdir(self.hyp_dir)

        self.his_path = os.path.join(self.his_dir, self.exp_name)
        self.img_path = os.path.join(self.img_dir, self.exp_name)
        self.hyp_path = os.path.join(self.hyp_dir, self.exp_name)

    def _load_data(self):
        self.train = data_loader(self.dataset_name, 'train') if not use_gen else DataGenerator(self.dataset_name)
        self.x_valid, self.y_valid = data_loader(self.dataset_name, 'valid')
        self.x_evalu, self.y_evalu = data_loader(self.dataset_name, 'evalu')

    @staticmethod
    def _lnr_schedule(step):
        begin_rate = 0.001
        decay_rate = 0.7
        decay_step = 250

        learn_rate = begin_rate * np.power(decay_rate, np.divmod(step, decay_step)[0])
        tf.summary.scalar('learning Rate', data=learn_rate, step=step)

        return learn_rate

    def _log_confusion(self, step, logs=None):
        max_arg = tf.math.argmax(self.model.predict(self.x_valid), axis=1)
        y_predi = tf.one_hot(max_arg, depth=len(self.categories)).numpy()
        y_valid_spar = self.encoder.inverse_transform(self.y_valid)
        y_predi_spar = self.encoder.inverse_transform(y_predi)
        matrix = confusion_matrix(y_valid_spar, y_predi_spar, labels=self.categories)  # TODO: normalize='true'
        matrix = np.around(matrix, decimals=2)
        confusion_figure = plot_confusion(matrix, categories=self.categories)
        confusion_image = plot_to_image(confusion_figure)

        with tf.summary.create_file_writer(self.img_path).as_default():
            tf.summary.image('Confusion Matrix', confusion_image, step=step)

    def _log_evalu(self, logs=None):
        performances = self.model.evaluate(x=self.x_evalu, y=self.y_evalu)
        with tf.summary.create_file_writer(self.hyp_path).as_default():
            hp.hparams(self.hyper_param)
            for m, p in list(zip(metric_names, performances)):
                tf.summary.scalar(m, p, step=0)

    def build(self):
        model = None
        self.model = model

    def run(self):
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=metrics)

        lnr_callback = LearningRateScheduler(schedule=self._lnr_schedule, verbose=1)
        his_callback = TensorBoard(log_dir=self.his_path, profile_batch=0)
        img_callback = LambdaCallback(on_epoch_end=self._log_confusion)
        eva_callback = LambdaCallback(on_train_end=self._log_evalu)
        callbacks = [lnr_callback, his_callback, img_callback, eva_callback]

        if not use_gen:
            x_train, y_train = self.train
            sample_weight = class_weight.compute_sample_weight('balanced', y_train)
            self.model.fit(
                x=x_train, y=y_train, sample_weight=sample_weight, batch_size=batch, epochs=epoch, verbose=1,
                validation_data=(self.x_valid, self.y_valid), callbacks=callbacks
            )
        else:
            generator = self.train  # (x_train, y_train, sample_weight)
            self.model.fit(
                x=generator, epochs=epoch, verbose=1,
                validation_data=(self.x_valid, self.y_valid), callbacks=callbacks,
                max_queue_size=10, workers=5
            )


class FoldBase(Base):

    def __init__(self, dataset_name, hyper_param, exp_dir, fold):
        self.fold = fold
        super().__init__(dataset_name, hyper_param, exp_dir)

    def _load_data(self):
        if not use_gen:
            self.train = fold_loader(self.dataset_name, 'train', self.fold)
        else:
            self.train = FoldGenerator(self.dataset_name, self.fold)
        self.x_valid, self.y_valid = fold_loader(self.dataset_name, 'valid', self.fold)
        self.x_evalu, self.y_evalu = self.x_valid.copy(), self.y_valid.copy()


# regularization -- bias and variance trade off / terminate training after loss stabilize
# what regularization to use?
# is regularization preferred over dropout

# attention model
# Phased LSTM

