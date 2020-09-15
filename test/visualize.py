from model.transformer import Transformer
from config.model_config import implements_hp, heads_hp, emb_dims_hp
from tensorflow.python.keras import Model

exp_dir = '/Users/mingyu/Desktop/log/ASAS_tra/experiment_0'
implements, heads, emb_dims = implements_hp.domain.values, heads_hp.domain.values, emb_dims_hp.domain.values
hyper_param = {implements_hp: implements[0], heads_hp: heads[0], emb_dims_hp: emb_dims[0]}
exp = Transformer('ASAS', hyper_param, exp_dir=exp_dir)

que_outputs = exp.model.get_layer('encoder').att.que_linear
key_outputs = exp.model.get_layer('encoder').att.key_linear
val_outputs = exp.model.get_layer('encoder').att.val_linear
model = Model(inputs=exp.model.inputs, outputs=[que_outputs.output, key_outputs.output, val_outputs.output])
model.summary()
