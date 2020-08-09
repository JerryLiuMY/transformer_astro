from model._base import _Base, _FoldBase
from config.exec_config import train_config

use_gen = train_config['use_gen']


class Attention(_Base):

    def __init__(self, dataset_name, model_name, hyper_param, exp_dir):
        super().__init__(dataset_name, model_name, hyper_param, exp_dir)


class FoldAttention(Attention, _FoldBase):

    def __init__(self, dataset_name, model_name, hyper_param, exp_dir, fold):
        _FoldBase.__init__(self, dataset_name, model_name, hyper_param, exp_dir, fold)
