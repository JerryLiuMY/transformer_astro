FLAG = 'local'
assert FLAG in ['local', 'colab', 'floyd'], 'Invalid flag'

if FLAG == 'local':
    LOG_FOLDER = '/Users/mingyu/Desktop/log'
    DATA_FOLDER = '/Users/mingyu/Desktop/lightcurve_data'
    SYNTHESIS_FOLDER = '/Volumes/Seagate_2T/synthesis'
elif FLAG == 'colab':
    LOG_FOLDER = '/content/drive/My Drive/log'
    DATA_FOLDER = '/content/drive/My Drive/lightcurve_data'
    SYNTHESIS_FOLDER = None
elif FLAG == 'floyd':
    LOG_FOLDER = '/output/log'
    DATA_FOLDER = '/floyd/input'
    SYNTHESIS_FOLDER = None
else:
    raise AssertionError('Invalid FLAG')
