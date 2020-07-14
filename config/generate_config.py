sin_dict = {'config1': {'num': 1000},  # nums: [1000, 1500, 2000]
            'config2': {'freq': None},  # freqs: [0.1, 0.5, 0.8, 1.0]
            'config3': {'amplitude': None},  # amplitudes: [0.5, 2.0, 5.0]
            'config4': {'noise': 'white-noise', 'sigma': None},  # sigmas: [0.05, 0.1, 0.5]
            'config5': {'noise': 'red-noise', 'std': None, 'tau': None}  # stds: [0.1, 0.2, 0.5], taus: [0.4, 0.8, 1.0]
            }

gp_dict = {'config1': {'num': 1000},  # nums: [1000, 1500, 2000], variances: [0.01, 0.25, 1.0, 4.0, 25.0]
           'config2': {'kernel': 'Exponential', 'variance': None, 'gamma': None},  # gammas: [0.1, 0.5]
           'config3': {'kernel': 'matern', 'variance': None, 'nu': None},  # nus: [1.5]
           'config4': {'kernel': 'RQ', 'variance': None, 'alpha': None}  # alphas: [0.5, 1.0, 2.0]
           }

car_dict = {'config1': {'num': 1000},  # nums: [1000, 1500, 2000]
            'config2': {'sigma': None, 'ar': None}}  # sigmas: [0.1, 0.5, 1.0], ars: [0.3, 0.5, 0.9]

dict_funcs = {"sin": sin_dict, "gp": gp_dict, "car": car_dict}
population = 200
