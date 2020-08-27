import os
import re
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard, LambdaCallback
from tensorflow.keras.backend import clear_session
from tensorboard.plugins.hparams import api as hp
from datetime import datetime
from tools.data_tools import data_loader, DataGenerator, one_hot_loader
from tools.data_tools import fold_loader, FoldGenerator
from tools.log_tools import lnr_schedule, plot_confusion, plot_to_image
from tools.exec_tools import create_dirs, create_paths, check_dataset_name
from config.model_config import rnn_nums_hp, rnn_dims_hp, dnn_nums_hp
from config.exec_config import train_config, strategy

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

    def __init__(self, dataset_name, model_name, hyper_param, exp_dir):
        clear_session()
        check_dataset_name(dataset_name)
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.hyper_param = hyper_param
        self.exp_dir = exp_dir
        self._load_name()
        self._load_dir()
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
        create_paths(self.his_path, self.img_path, self.hyp_path, self.che_path)

    def _load_enco(self):
        encoder = one_hot_loader(self.dataset_name, self.model_name)
        self.encoder = encoder
        self.categories = encoder.categories_[0]

    def _load_data(self):
        if not use_gen:
            self.dataset_train = data_loader(self.dataset_name, self.model_name, 'train')
        else:
            self.dataset_train = DataGenerator(self.dataset_name, self.model_name)
        self.dataset_valid = data_loader(self.dataset_name, self.model_name, 'valid')
        self.dataset_evalu = data_loader(self.dataset_name, self.model_name, 'evalu')

    def _build(self):
        model = None
        self.model = model

    def _compile(self):
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=metrics)

    def run(self):
        checkpoint = os.path.join(self.che_path, 'epoch_{epoch:02d}-val_acc_{val_categorical_accuracy:.3f}.hdf5')
        lnr_callback = LearningRateScheduler(schedule=lnr_schedule, verbose=1)
        his_callback = TensorBoard(log_dir=self.his_path, profile_batch=0)
        img_callback = LambdaCallback(on_epoch_end=self._log_confusion)
        che_callback = ModelCheckpoint(filepath=checkpoint, save_weights_only=True)
        hyp_callback = hp.KerasCallback(self.hyp_path, self.hyper_param)
        callbacks = [lnr_callback, his_callback, img_callback, che_callback, hyp_callback]

        self.model.fit(
            x=self.dataset_train, validation_data=self.dataset_valid, epochs=epoch,
            verbose=1, max_queue_size=10, workers=5, callbacks=callbacks
        )

    def _log_confusion(self, step, logs):
        y_evalu = np.array([]).reshape(0, len(self.categories))
        for x_evalu_, y_evalu_ in self.dataset_valid.take(-1):
            y_evalu = np.vstack([y_evalu, y_evalu_.numpy()])
        max_arg = tf.math.argmax(self.model.predict(self.dataset_valid), axis=1)
        y_predi = tf.one_hot(max_arg, depth=len(self.categories)).numpy()

        y_evalu_spar = self.encoder.inverse_transform(y_evalu)
        y_predi_spar = self.encoder.inverse_transform(y_predi)
        matrix = np.around(confusion_matrix(y_evalu_spar, y_predi_spar, labels=self.categories), decimals=2)
        report = classification_report(y_evalu_spar, y_predi_spar, labels=self.categories, zero_division=0)
        confusion_fig = plot_confusion(matrix, report, categories=self.categories)
        confusion_img = plot_to_image(confusion_fig)

        with tf.summary.create_file_writer(self.img_path).as_default():
            tf.summary.image('Confusion Matrix', confusion_img, step=step)


class _FoldBase(_Base):

    def __init__(self, dataset_name, model_name, hyper_param, exp_dir, fold):
        self.fold = fold
        super().__init__(dataset_name, model_name, hyper_param, exp_dir)

    def _load_data(self):
        if not use_gen:
            self.dataset_train = fold_loader(self.dataset_name, self.model_name, 'train', self.fold)
        else:
            self.dataset_train = FoldGenerator(self.dataset_name, self.model_name, self.fold)
        self.dataset_valid = fold_loader(self.dataset_name, self.model_name, 'evalu', self.fold)


# test set result -- test set only last
# tf.data pipeline -- processing
# test data change

# transformer model
# wait: TPU compatibility
# Transformer presentation & report
# BERT presentation
