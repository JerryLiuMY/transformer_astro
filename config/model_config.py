from tensorboard.plugins.hparams import api as hp

rnn_nums_hp = hp.HParam("rnn_nums", hp.Discrete([2]))
rnn_dims_hp = hp.HParam("rnn_dims", hp.Discrete([70, 100, 125, 150, 200]))
dnn_nums_hp = hp.HParam("dnn_nums", hp.Discrete([2]))


# [0: classifier, 1: encoder freeze, 2: encoder trainable, 3: encoder-classifier, 4: transformer-classifier]
implements_hp = hp.HParam("implements", hp.Discrete([0, 1, 2, 3, 4]))
heads_hp = hp.HParam("heads", hp.Discrete([4, 8]))
emb_dims_hp = hp.HParam("emb_dims", hp.Discrete([32, 64, 128]))
