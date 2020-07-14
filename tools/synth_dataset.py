import os
import pandas as pd
from shutil import copyfile
from global_settings import SYNTHESIS_FOLDER, DATA_FOLDER
from config.generate_config import dict_funcs, population


def generate_synth(cats):
    synth_dir = new_synth_dir()

    catalog = pd.DataFrame(columns={'Class', 'Param', 'Old_Path'})
    for cat in cats:
        assert cat in ['sin', 'gp', 'car']
        catalog = build_catalog(catalog, cat)
    catalog['Path'] = [os.path.join('LCs', str(_) + '.dat') for _ in list(catalog.index)]
    catalog.to_csv(os.path.join(synth_dir, 'catalog.csv'))

    for old, new in list(zip(catalog['Old_Path'], catalog['Path'])):
        copyfile(os.path.join(SYNTHESIS_FOLDER, old), os.path.join(synth_dir, new))


def build_catalog(catalog, cat):
    folder_names = ['base']
    for key, value in dict_funcs[cat].items():
        if None in [param for (param_key, param) in value.items()]: continue
        params_list = ['='.join([param_key, str(param)]) for (param_key, param) in value.items()]
        folder_name = '_'.join(_ for _ in params_list)
        folder_names.append(folder_name)

    for folder_name in folder_names:
        old_path_ = [os.path.join(cat, folder_name, f'{str(_)}.dat') for _ in range(N)]
        cat_, param_ = [cat] * population, [folder_name] * population
        for o, c, p in list(zip(old_path_, cat_, param_)):
            catalog = catalog.append({'Class': c, 'Param': p, 'Old_Path': o}, ignore_index=True)

    return catalog


def new_synth_dir():
    synth_now = 0
    for _ in next(os.walk(DATA_FOLDER))[1]:
        if len(_.split('_')) == 2 and _.split('_')[0] == 'synthesis':
            synth_now_ = int(_.split('_')[1])
            synth_now = synth_now_ if synth_now_ > synth_now else synth_now

    synth_dir = os.path.join(DATA_FOLDER, '_'.join(['synthesis', str(synth_now + 1)]))
    os.mkdir(synth_dir)
    os.mkdir(os.path.join(synth_dir, 'LCs'))

    return synth_dir

