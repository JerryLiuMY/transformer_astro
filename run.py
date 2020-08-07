from global_settings import DATASET_NAME, FLAG
import subprocess

if FLAG == 'local':
    subprocess.run('python main.py', shell=True)
elif FLAG == 'floyd':
    subprocess.run('floyd login', shell=True)
    subprocess.run('floyd init jerryliumy/self_attention_rnn', shell=True)
    pref = 'floyd run'
    deco = '--gpu --env tensorflow-2.2'
    data = f'--data jerryliumy/datasets/{DATASET_NAME.lower()}:{DATASET_NAME}'
    main = '"python main.py"'
    command = ' '.join([pref, deco, data, main])
    subprocess.run(command, shell=True)
else:
    raise AssertionError('Invalid FLAG')
