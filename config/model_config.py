from tensorboard.plugins.hparams import api as hp

rnn_nums_hp = hp.HParam("rnn_nums", hp.Discrete([1, 2]))
rnn_dims_hp = hp.HParam("rnn_dims", hp.Discrete([5, 10, 20, 40]))
dnn_nums_hp = hp.HParam("dnn_nums", hp.Discrete([1, 2]))
