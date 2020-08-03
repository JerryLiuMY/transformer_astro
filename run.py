from global_settings import FLAG
import subprocess
file = 'main.py'

if FLAG == 'local':
    pref = 'python'
    command = ' '.join([pref, file])
    subprocess.run(command, shell=True)

elif FLAG == 'floyd':
    pref = 'floyd run'
    deco = '--gpu --env tensorflow-2.2'
    data = '--data jerryliumy/datasets/lightcurve_data/2:lightcurve_data'
    command = ' '.join([pref, deco, data, file])
    subprocess.run(command, shell=True)
