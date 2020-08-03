from config.exec_config import FLAG, DATASET_NAME
import subprocess
file = 'main.py'

if FLAG == 'local':
    pref = 'python'
    command = ' '.join([pref, file])
    subprocess.run(command, shell=True)
elif FLAG == 'floyd':
    subprocess.run('floyd init jerryliumy/self_attention_rnn', shell=True)
    pref = 'floyd run'
    deco = '--gpu --env tensorflow-2.2'
    data = f'--data jerryliumy/datasets/{DATASET_NAME}:lightcurve_data'
    command = ' '.join([pref, deco, data, file])
    subprocess.run(command, shell=True)
else:
    raise AssertionError('Invalid FLAG')
