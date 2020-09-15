FLAG = 'colab'
assert FLAG in ['local', 'colab', 'floyd'], 'Invalid flag'

if FLAG == 'local':
    LOG_FOLDER = '/Users/mingyu/Desktop/log'
    RAW_FOLDER = '/Users/mingyu/Desktop/lightcurve_data'
    DATA_FOLDER = '/Users/mingyu/Desktop/data'
    SYNTH_FOLDER = '/Volumes/Seagate_2T/synthesis'
elif FLAG == 'colab':
    LOG_FOLDER = '/content/drive/My Drive/log'
    RAW_FOLDER = None
    DATA_FOLDER = '/content/drive/My Drive/data'
    SYNTH_FOLDER = None
elif FLAG == 'floyd':
    LOG_FOLDER = '/output/log'
    RAW_FOLDER = None
    DATA_FOLDER = '/floyd/input'
    SYNTH_FOLDER = None
else:
    raise AssertionError('Invalid FLAG')
