import os
import re
import itertools
import numpy as np
import tensorflow as tf
from datetime import datetime
from tensorflow.keras.backend import clear_session
from tensorboard.plugins.hparams import api as hp
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard, LambdaCallback
from data.loader import data_loader, one_hot_loader, fold_loader
from data.generator import DataGenerator, FoldGenerator
from tools.dir_tools import create_dirs
from tools.log_tools import lnr_schedule, plot_confusion, fig_to_img
from config.model_config import rnn_nums_hp, rnn_dims_hp, dnn_nums_hp
from config.exec_config import train_config

use_gen, epoch = train_config['use_gen'], train_config['epoch']
metrics, metric_names = train_config['metrics'], ['epoch_loss']
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
    def __init__(self, dataset_name, hyper_param, exp_dir):
        clear_session()
        self.dataset_name = dataset_name
        self.hyper_param = hyper_param
        self.exp_dir = exp_dir
        with tf.device('/cpu:0'):
            self._load_name()
            self._load_dir()
            self._load_path()
            self._load_enco()
            self._load_data()

    def _load_name(self):
        now = datetime.now().strftime('%Y-%m-%d--%H-%M-%S')
        self.exp_name = '-'.join([f'{h.name}_{self.hyper_param[h]}' for h in self.hyper_param])
        self.exp_name = '-'.join([self.exp_name, now])

        print(f'--- Starting trial: {self.exp_name}')
        print({h.name: self.hyper_param[h] for h in self.hyper_param})

    def _load_dir(self):
        self.his_dir = os.path.join(self.exp_dir, 'scalar')
        self.img_dir = os.path.join(self.exp_dir, 'images')
        self.hyp_dir = os.path.join(self.exp_dir, 'params')
        self.che_dir = os.path.join(self.exp_dir, 'checks')
        create_dirs(self.his_dir, self.img_dir, self.hyp_dir, self.che_dir)

    def _load_path(self):
        self.his_path = os.path.join(self.his_dir, self.exp_name)
        self.img_path = os.path.join(self.img_dir, self.exp_name)
        self.hyp_path = os.path.join(self.hyp_dir, self.exp_name)
        self.che_path = os.path.join(self.che_dir, self.exp_name)

    def _load_enco(self):
        one_hot = one_hot_loader(self.dataset_name)
        self.one_hot = one_hot
        self.categories = one_hot.categories_[0]

    def _load_data(self):
        if not use_gen:
            self.dataset_train = data_loader(self.dataset_name, 'train')
        else:
            self.dataset_train = DataGenerator(self.dataset_name).get_dataset()
        self.dataset_valid = data_loader(self.dataset_name, 'valid')

    # build, compile and log
    def _build(self):
        model = None
        self.model = model

    def _compile(self):
        self.model.compile()

    def _load_call(self):
        checkpoint = os.path.join(self.che_path, 'epoch_{epoch:02d}-val_acc_{val_categorical_accuracy:.3f}.hdf5')
        lnr_callback = LearningRateScheduler(schedule=lnr_schedule, verbose=1)
        his_callback = TensorBoard(log_dir=self.his_path, profile_batch=0)
        img_callback = LambdaCallback(on_epoch_end=self._log_confusion)
        che_callback = ModelCheckpoint(filepath=checkpoint, save_weights_only=True)
        hyp_callback = LambdaCallback(on_train_end=self._log_hyper_param)
        self.callbacks = [lnr_callback, his_callback, img_callback, che_callback, hyp_callback]

    def _log_confusion(self, step, logs):
        y_evalu = np.array([]).reshape(0, len(self.categories))
        for x_evalu_, y_evalu_ in self.dataset_valid.take(-1):
            y_evalu = np.vstack([y_evalu, y_evalu_.numpy()])
        max_arg = tf.math.argmax(self.model.predict(self.dataset_valid), axis=1)
        y_predi = tf.one_hot(max_arg, depth=len(self.categories)).numpy()

        y_evalu_spar = self.one_hot.inverse_transform(y_evalu)
        y_predi_spar = self.one_hot.inverse_transform(y_predi)
        confusion_fig = plot_confusion(y_evalu_spar, y_predi_spar, categories=self.categories)
        confusion_img = fig_to_img(confusion_fig)

        with tf.summary.create_file_writer(self.img_path).as_default():
            tf.summary.image('Confusion Matrix', confusion_img, step=step)

    def _log_hyper_param(self, logs):
        with tf.summary.create_file_writer(self.hyp_path).as_default():
            hp.hparams(self.hyper_param)
            results = self.model.evaluate(self.dataset_valid)
            for metric_name, result in itertools.product(metric_names, results):
                tf.summary.scalar(metric_name, result, step=0)


class _FoldBase(_Base):

    def __init__(self, dataset_name, hyper_param, exp_dir, fold):
        self.fold = fold
        super().__init__(dataset_name, hyper_param, exp_dir)

    def _load_data(self):
        if not use_gen:
            self.dataset_train = fold_loader(self.dataset_name, 'train', self.fold)
        else:
            self.dataset_train = FoldGenerator(self.dataset_name, self.fold).get_dataset()
        self.dataset_valid = fold_loader(self.dataset_name, 'evalu', self.fold)

# model simple
# model transformer
# visualize attention weights
# visualize encoding
