from global_settings import SYNTHESIS_FOLDER
import os
import itertools
from data.synth import sin_series, gp_series, car_series
from config.pool_config import pool_dict

series_funcs = {"sin": sin_series, "gp": gp_series, "car": car_series}


def run_synth(cat, **kwargs):
    """
    :param cat: (str) category of time series to generate
    :param kwargs: (str, float) parameters of the time series
    """
    assert cat in ["sin", "gp", "car"]
    set_dir = os.path.join(SYNTHESIS_FOLDER, cat)
    if not os.path.isdir(set_dir): os.mkdir(set_dir)
    series_func, base_config = series_funcs[cat], pool_dict[cat]

    if len(kwargs.items()) == 0:
        base_dir = os.path.join(set_dir, "base")
        if not os.path.isdir(base_dir):
            os.mkdir(base_dir)
            series_func(base_config, base_dir)

    else:
        new_config = base_config.copy()
        elements = []
        for key, value in kwargs.items():
            value = round(value, 2) if type(value) == float else value
            new_config[key] = value
            elements.append(f"{key}={str(value)}")
        folder_name = "_".join(_ for _ in elements)
        new_dir = os.path.join(set_dir, folder_name)

        if not os.path.isdir(new_dir):
            os.mkdir(new_dir)
            series_func(new_config, new_dir)


def run_sin():
    freqs, nums, amplitudes = [0.1, 0.5, 0.8, 1.0], [1000, 1500, 2000], [0.5, 2.0, 5.0]
    sigmas = [0.05, 0.1, 0.5]; whites = ["white-noise"] * len(sigmas)
    stds, taus = [0.1, 0.2, 0.5], [0.4, 0.8, 1.0]

    run_synth(cat="sin")
    [run_synth(cat="sin", freq=freq) for freq in freqs]
    [run_synth(cat="sin", num=num) for num in nums]
    [run_synth(cat="sin", amplitude=amplitude) for amplitude in amplitudes]
    [run_synth(cat="sin", noise=white, sigma=sigma) for white, sigma in list(zip(whites, sigmas))]

    reds1, stds1 = ["red-noise"] * len(stds) * len(taus), stds * len(taus)
    taus1 = list(itertools.chain.from_iterable(itertools.repeat(tau, len(stds)) for tau in taus))
    [run_synth(cat="sin", noise=red, std=std, tau=tau) for red, std, tau in list(zip(reds1, stds1, taus1))]


def run_gp():
    nums, sigmas = [1000, 1500, 2000], [0.1, 0.5, 1.0, 2.0, 5.0]
    gammas, nus, alphas = [0.1, 0.5, 1.0], [1.5], [0.5, 1.0, 2.0]

    run_synth(cat="gp")
    [run_synth(cat="gp", num=num) for num in nums]

    kernels1, variances1 = ["Exponential"] * len(sigmas) * len(gammas), [sigma ** 2 for sigma in sigmas] * len(gammas)
    gammas1 = list(itertools.chain.from_iterable(itertools.repeat(gamma, len(sigmas)) for gamma in gammas))
    for kernel, variance, gamma in list(zip(kernels1, variances1, gammas1)):
        run_synth(cat="gp", kernel=kernel, variance=variance, gamma=gamma)

    kernels2, variances2 = ["Matern"] * len(sigmas) * len(nus), [sigma ** 2 for sigma in sigmas] * len(nus)
    nus2 = list(itertools.chain.from_iterable(itertools.repeat(nu, len(sigmas)) for nu in nus))
    for kernel, variance, nu in list(zip(kernels2, variances2, nus2)):
        run_synth(cat="gp", kernel=kernel, variance=variance, nu=nu)

    kernels3, variances3 = ["RQ"] * len(sigmas) * len(alphas), [sigma ** 2 for sigma in sigmas] * len(alphas)
    alphas3 = list(itertools.chain.from_iterable(itertools.repeat(alpha, len(sigmas)) for alpha in alphas))
    for kernel, variance, alpha in list(zip(kernels3, variances3, alphas3)):
        run_synth(cat="gp", kernel=kernel, variance=variance, alpha=alpha)


def run_car():
    nums = [1000, 1500, 2000]
    sigmas, ars = [1.0, 0.5, 0.1], [0.9, 0.5, 0.3]
    sigmas1 = sigmas * len(ars)
    ars1 = list(itertools.chain.from_iterable(itertools.repeat(ar, len(sigmas)) for ar in ars))

    run_synth(cat="car")
    [run_synth(cat="car", num=num) for num in nums]

    for sigma, ar in list(zip(sigmas1, ars1)):
        run_synth(cat="car", sigma=sigma, ar=ar)


if __name__ == "__main__":
    run_sin()
    run_gp()
    run_car()
