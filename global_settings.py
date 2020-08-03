import os
cwd = os.getcwd()

if cwd.split('/')[1] == 'Users':
    LOG_FOLDER = '/Users/mingyu/Desktop/log'
    DATA_FOLDER = '/Volumes/Seagate_2T/lightcurve_data'
    SYNTHESIS_FOLDER = '/Volumes/Seagate_2T/synthesis'
elif cwd.split('/')[1] == 'floyd':
    LOG_FOLDER = '/floyd/home/log'
    DATA_FOLDER = '/floyd/input/lightcurve_data'
    SYNTHESIS_FOLDER = None
else:
    raise AssertionError('Invalid path')

os.makedirs(LOG_FOLDER, exist_ok=True)
