from config.exec_config import FLAG

if FLAG == 'local':
    LOG_FOLDER = '/Users/mingyu/Desktop/log'
    # DATA_FOLDER = '/Volumes/Seagate_2T/lightcurve_data'
    DATA_FOLDER = '/Users/mingyu/Desktop/lightcurve_data'
    SYNTHESIS_FOLDER = '/Volumes/Seagate_2T/synthesis'
elif FLAG == 'floyd':
    LOG_FOLDER = '/output/log'
    DATA_FOLDER = '/floyd/input'
    SYNTHESIS_FOLDER = None
else:
    raise AssertionError('Invalid FLAG')

# os.makedirs(LOG_FOLDER, exist_ok=True)
