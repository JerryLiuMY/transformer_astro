from global_settings import FLAG
import subprocess

DATASET_NAME, MODEL_NAME = 'OGLE', 'sim'
assert DATASET_NAME in ['ASAS', 'MACHO', 'WISE', 'GAIA', 'OGLE'], 'Invalid dataset name'
assert MODEL_NAME in ['sim', 'pha', 'att'], 'Invalid dataset type'


if FLAG == 'local':
    main = f'python main.py'
    para = f'-d {DATASET_NAME}'
    command = ' '.join([main, para])
    subprocess.run(command, shell=True)
elif FLAG == 'floyd':
    dataset_folder_lower = '_'.join([DATASET_NAME.lower(), MODEL_NAME])
    dataset_folder = '_'.join([DATASET_NAME, MODEL_NAME])

    subprocess.run('floyd login', shell=True)
    subprocess.run('floyd init jerryliumy/self_attention_rnn', shell=True)
    pref = 'floyd run'
    deco = '--gpu --env tensorflow-2.2'
    data = f'--data jerryliumy/datasets/{dataset_folder_lower}:{dataset_folder}'
    main = f'"python main.py"'
    para = f'-d {DATASET_NAME} -m {MODEL_NAME}'
    command = ' '.join([pref, deco, data, main, para])
    subprocess.run(command, shell=True)
else:
    raise AssertionError('Invalid FLAG')
