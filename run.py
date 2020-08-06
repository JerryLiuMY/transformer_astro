from config.exec_config import FLAG, DATASET_NAME
import subprocess

if FLAG == 'local':
    subprocess.run('python main.py', shell=True)
elif FLAG == 'floyd':
    subprocess.run('floyd login', shell=True)
    subprocess.run('floyd init jerryliumy/self_attention_rnn', shell=True)
    pref = 'floyd run'
    deco = '--cpu2 --env tensorflow-2.2'  # --follow
    data = f'--data jerryliumy/datasets/{DATASET_NAME.lower()}:{DATASET_NAME}'
    main = '"python main.py"'
    command = ' '.join([pref, deco, data, main])
    subprocess.run(command, shell=True)
else:
    raise AssertionError('Invalid FLAG')
