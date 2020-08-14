import os
import re
import numpy as np
import tensorflow as tf
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import TensorBoard, LambdaCallback
from tensorflow.keras.backend import clear_session
from tensorboard.plugins.hparams import api as hp
from datetime import datetime
from tools.exec_tools import plot_confusion, plot_to_image
from tools.misc import check_dataset_name
from tools.data_tools import data_loader, DataGenerator, one_hot_loader
from tools.data_tools import fold_loader, FoldGenerator
from config.model_config import rnn_nums_hp, rnn_dims_hp, dnn_nums_hp
from config.exec_config import train_config, strategy

use_gen, epoch = train_config['use_gen'], train_config['epoch']
metrics, batch = train_config['metrics'], train_config['batch']
metric_names = ['epoch_loss']
for metric in metrics:
    lower = [_.lower() for _ in re.findall('[A-Z][^A-Z]*', metric)]
    metric_names.append('_'.join(['epoch'] + lower))


def log_params(exp_dir):
    with tf.summary.create_file_writer(exp_dir).as_default():
        hp.hparams_config(
            hparams=[rnn_nums_hp, rnn_dims_hp, dnn_nums_hp],
            metrics=[hp.Metric(_) for _ in metric_names]
        )


class _Base:

    def __init__(self, dataset_name, model_name, hyper_param, exp_dir):
        clear_session()
        check_dataset_name(dataset_name)
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.hyper_param = hyper_param
        self.exp_dir = exp_dir
        self._load_name()
        self._load_path()
        self._load_enco()
        self._load_data()
        with strategy.scope():
            self._build()
            self._compile()

    def _load_name(self):
        rnn_num = self.hyper_param[rnn_nums_hp]
        rnn_dim = self.hyper_param[rnn_dims_hp]
        dnn_num = self.hyper_param[dnn_nums_hp]
        now = datetime.now().strftime('%Y-%m-%d--%H-%M-%S')
        self.exp_name = '-'.join([f'rnn_num_{rnn_num}', f'rnn_dim_{rnn_dim}', f'dnn_num_{dnn_num}', now])

        print(f'--- Starting trial: {self.exp_name}')
        print({h.name: self.hyper_param[h] for h in self.hyper_param})

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

    def _load_enco(self):
        encoder = one_hot_loader(self.dataset_name, self.model_name)
        self.encoder = encoder
        self.categories = encoder.categories_[0]

    def _load_data(self):
        if not use_gen:
            self.train = data_loader(self.dataset_name, self.model_name, 'train')
        else:
            self.train = DataGenerator(self.dataset_name, self.model_name)
        self.x_valid, self.y_valid = data_loader(self.dataset_name, self.model_name, 'valid')
        self.x_evalu, self.y_evalu = data_loader(self.dataset_name, self.model_name, 'evalu')

    def _build(self):
        model = None
        self.model = model

    def _compile(self):
        self.model.compile(
            # experimental_steps_per_execution=100,
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=metrics)

    @staticmethod
    def _lnr_schedule(step):
        begin_rate = 0.001
        decay_rate = 0.7
        decay_step = 50

        learn_rate = begin_rate * np.power(decay_rate, np.divmod(step, decay_step)[0])
        tf.summary.scalar('Learning Rate', data=learn_rate, step=step)

        return learn_rate

    def _log_confusion(self, step, logs=None):
        max_arg = tf.math.argmax(self.model.predict(self.x_evalu), axis=1)
        y_predi = tf.one_hot(max_arg, depth=len(self.categories)).numpy()
        y_evalu_spar = self.encoder.inverse_transform(self.y_evalu)
        y_predi_spar = self.encoder.inverse_transform(y_predi)
        matrix = np.around(confusion_matrix(y_evalu_spar, y_predi_spar, labels=self.categories), decimals=2)
        report = classification_report(y_evalu_spar, y_predi_spar, labels=self.categories, zero_division=0)
        confusion_fig = plot_confusion(matrix, report, categories=self.categories)
        confusion_img = plot_to_image(confusion_fig)

        with tf.summary.create_file_writer(self.img_path).as_default():
            tf.summary.image('Confusion Matrix', confusion_img, step=step)

    def _log_evalu(self, logs=None):
        performances = self.model.evaluate(x=self.x_evalu, y=self.y_evalu)
        with tf.summary.create_file_writer(self.hyp_path).as_default():
            hp.hparams(self.hyper_param)
            for m, p in list(zip(metric_names, performances)):
                tf.summary.scalar(m, p, step=0)

    def run(self):
        lnr_callback = LearningRateScheduler(schedule=self._lnr_schedule, verbose=1)
        his_callback = TensorBoard(log_dir=self.his_path, profile_batch=0)
        img_callback = LambdaCallback(on_epoch_end=self._log_confusion)
        eva_callback = LambdaCallback(on_train_end=self._log_evalu)
        callbacks = [lnr_callback, his_callback, img_callback, eva_callback]

        self.model.fit(
            x=self.train, epochs=epoch, batch_size=batch,
            validation_data=(self.x_valid, self.y_valid), verbose=2, callbacks=callbacks,
            max_queue_size=10, workers=5
        )


class _FoldBase(_Base):

    def __init__(self, dataset_name, model_name, hyper_param, exp_dir, fold):
        self.fold = fold
        super().__init__(dataset_name, model_name, hyper_param, exp_dir)

    def _load_data(self):
        if not use_gen:
            self.train = fold_loader(self.dataset_name, self.model_name, 'train', self.fold)
        else:
            self.train = FoldGenerator(self.dataset_name, self.model_name, self.fold)
        self.x_evalu, self.y_evalu = fold_loader(self.dataset_name, self.model_name, 'evalu', self.fold)
        self.x_valid, self.y_valid = self.x_evalu.copy(), self.y_evalu.copy()


# tf.data pipeline
# WISE dataset performance
# test set result
# attention model
# Phased LSTM
