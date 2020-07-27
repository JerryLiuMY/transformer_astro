from tensorboard.plugins.hparams import api as hp

rnn_nums_hp = hp.HParam("rnn_nums", hp.Discrete([2]))
rnn_dims_hp = hp.HParam("rnn_dims", hp.Discrete([15, 35, 50, 70, 100, 125, 150]))
dnn_nums_hp = hp.HParam("dnn_nums", hp.Discrete([2]))
