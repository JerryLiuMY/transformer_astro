from global_settings import FLAG
import subprocess

DATASET_NAME, MODEL_NAME = 'ASAS', 'tra'
assert DATASET_NAME in ['ASAS', 'MACHO', 'WISE', 'GAIA', 'OGLE'], 'Invalid dataset name'
assert MODEL_NAME in ['sim', 'tra'], 'Invalid dataset type'

main = f'python main.py -d {DATASET_NAME} -m {MODEL_NAME}'

if FLAG in ['local', 'colab']:
    command = main
    subprocess.run(command, shell=True)
elif FLAG == 'floyd':
    dataset_map = f'{"_".join([DATASET_NAME.lower(), MODEL_NAME])}:{"_".join([DATASET_NAME, MODEL_NAME])}'
    pref = 'floyd run'
    deco = '--cpu --env tensorflow-2.2'
    data = f'--data jerryliumy/datasets/{dataset_map}'
    command = ' '.join([pref, deco, data, f'"{main}"'])
    subprocess.run('floyd login', shell=True)
    subprocess.run('floyd init jerryliumy/self_attention_rnn', shell=True)
    subprocess.run(command, shell=True)
else:
    raise AssertionError('Invalid FLAG')
