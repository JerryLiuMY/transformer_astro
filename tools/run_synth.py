from global_settings import SYNTHESIS_FOLDER, CONFIG_FOLDER
import os
import json
import itertools
from tools.synth import sin_series, gp_series, car_series

series_funcs = {"sin": sin_series, "gp": gp_series, "car": car_series}
with open(os.path.join(CONFIG_FOLDER, "synth_config.json"), "rb") as handle:
    params = json.load(handle)


def run_synth(cat, **kwargs):
    """
    :param cat: (str) category of time series to generate
    :param kwargs: (str, float) parameters of the time series
    """
    assert cat in ["sin", "gp", "car"]
    set_dir = os.path.join(SYNTHESIS_FOLDER, cat)
    if not os.path.isdir(set_dir): os.mkdir(set_dir)
    series_func, base_config = series_funcs[cat], params[cat]

    if len(kwargs.items()) == 0:
        base_dir = os.path.join(set_dir, "base"); os.mkdir(base_dir)
        series_func(params[cat], base_dir)

    else:
        new_config = base_config.copy()
        elements = []
        for key, value in kwargs.items():
            new_config[key] = value
            elements.append(f"{key}={str(value)}")
        folder_name = "_".join(_ for _ in elements)
        new_dir = os.path.join(set_dir, folder_name)
        if not os.path.isdir(new_dir): os.mkdir(new_dir)
        series_func(new_config, new_dir)


def run_sin():
    freqs, num_points, amplitudes = [0.1, 0.5, 0.8, 1], [1000, 1500, 2000], [0.5, 2, 5]
    sigmas = [0.05, 0.1, 0.5]; whites = ["white_noise"] * len(sigmas)
    stds, taus = [0.5, 0.5, 0.5, 0.2, 0.1], [0.8, 0.4, 1, 0.8, 0.8]; reds = ["red_noise"] * len(stds)

    run_synth(cat="sin")
    [run_synth(cat="sin", freq=freq) for freq in freqs]
    [run_synth(cat="sin", num_point=num_point) for num_point in num_points]
    [run_synth(cat="sin", amplitude=amplitude) for amplitude in amplitudes]
    [run_synth(cat="sin", noise=white, sigma=sigma) for white, sigma in list(zip(whites, sigmas))]
    [run_synth(cat="sin", noise=red, std=std, tau=tau) for red, std, tau in list(zip(reds, stds, taus))]


def run_gp():
    num_points, sigmas = [1000, 1500, 2000], [0.1, 0.5, 1., 2., 5.]
    gammas, nus, alphas = [0.1, 0.5, 1], [3. / 2], [0.5, 1, 2]

    run_synth(cat="gp")
    [run_synth(cat="gp", num_point=num_point) for num_point in num_points]

    kernels1, variances1 = ['Exponential'] * len(sigmas) * len(gammas), [sigma ** 2 for sigma in sigmas] * len(gammas)
    gammas1 = list(itertools.chain.from_iterable(itertools.repeat(gamma, len(sigmas)) for gamma in gammas))
    for kernel, variance, gamma in list(zip(kernels1, variances1, gammas1)):
        run_synth(cat="gp", kernel=kernel, variance=variance, gamma=gamma)

    kernels2, variances2 = ['Matern'] * len(sigmas) * len(nus), [sigma ** 2 for sigma in sigmas] * len(nus)
    nus2 = list(itertools.chain.from_iterable(itertools.repeat(nu, len(sigmas)) for nu in nus))
    for kernel, variance, nu in list(zip(kernels2, variances2, nus2)):
        run_synth(cat="gp", kernel=kernel, variance=variance, nu=nu)

    kernels3, variances3 = ['RQ'] * len(sigmas) * len(alphas), [sigma ** 2 for sigma in sigmas] * len(alphas)
    alphas3 = list(itertools.chain.from_iterable(itertools.repeat(alpha, len(sigmas)) for alpha in alphas))
    for kernel, variance, alpha in list(zip(kernels3, variances3, alphas3)):
        run_synth(cat="gp", kernel=kernel, variance=variance, alpha=alpha)


def run_car():
    num_points = [1000, 1500, 2000]
    sigmas, ar_params = [1, 0.5, 0.1], [0.9, 0.5, 0.3]
    sigmas1 = sigmas * len(ar_params)
    ar_params1 = list(itertools.chain.from_iterable(itertools.repeat(ar_param, len(sigmas)) for ar_param in ar_params))

    run_synth(cat="car")
    [run_synth(cat="car", num_point=num_point) for num_point in num_points]

    for sigma, ar_param in list(zip(sigmas1, ar_params1)):
        run_synth(cat="car", sigma=sigma, ar_param=ar_param)


if __name__ == "__main__":
    run_sin()
    run_gp()
    run_car()

