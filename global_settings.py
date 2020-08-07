FLAG, DATASET_NAME, DATASET_TYPE = 'local', 'GAIA', 'sim'
assert FLAG in ['local', 'floyd'], 'Invalid flag'
assert DATASET_NAME in ['ASAS', 'MACHO', 'WISE', 'GAIA', 'OGLE'], 'Invalid dataset name'
assert DATASET_TYPE in [None, 'sim', 'pha', 'att'], 'Invalid dataset type'

if FLAG == 'local':
    LOG_FOLDER = '/Users/mingyu/Desktop/log'
    DATA_FOLDER = '/Users/mingyu/Desktop/lightcurve_data'
    SYNTHESIS_FOLDER = '/Volumes/Seagate_2T/synthesis'
elif FLAG == 'floyd':
    LOG_FOLDER = '/output/log'
    DATA_FOLDER = '/floyd/input'
    SYNTHESIS_FOLDER = None
else:
    raise AssertionError('Invalid FLAG')
