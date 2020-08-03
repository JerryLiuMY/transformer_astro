import os

FLAG = 'local'
assert FLAG in ['local', 'floyd'], 'Invalid FLAG'

if FLAG == 'local':
    LOG_FOLDER = '/Users/mingyu/Desktop/log'
    DATA_FOLDER = '/Volumes/Seagate_2T/lightcurve_data'
    SYNTHESIS_FOLDER = '/Volumes/Seagate_2T/synthesis'
elif FLAG == 'floyd':
    LOG_FOLDER = '/output/log'
    DATA_FOLDER = '/floyd/input/lightcurve_data'
    SYNTHESIS_FOLDER = None
else:
    raise AssertionError('Invalid path')

os.makedirs(LOG_FOLDER, exist_ok=True)
