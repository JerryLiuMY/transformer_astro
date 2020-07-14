from timesynth.signals import Sinusoidal, GaussianProcess, CAR
from timesynth.noise import GaussianNoise, RedNoise
from timesynth import TimeSampler, TimeSeries
import os
import json
import pandas as pd
NUM_SERIES = 500


def sin_series(sin_config, save_dir):
    """
    :param sin_config: a dictionary in the same format as the baseConfigDict above, specifies the configurations
    :param save_dir: output the generated data to this directory
    """
    noise = sin_config["noise"]
    assert noise in ["white-noise", "red-noise"], "invalid noise type"

    with open(os.path.join(save_dir, "config.json"), "w", encoding="utf-8") as h:
        json.dump(sin_config, h, ensure_ascii=False, indent=4)

    stop_time = sin_config["stop_time"]
    num, keep_percentage = sin_config["num"], sin_config["keep_percentage"]
    frequency, amplitude = sin_config["frequency"], sin_config["amplitude"]
    std, tau = sin_config["std"], sin_config["tau"]

    for ii in range(NUM_SERIES):
        time_sampler = TimeSampler(stop_time=stop_time)
        mjd = time_sampler.sample_irregular_time(num_points=num, keep_percentage=keep_percentage)

        sinusoid = Sinusoidal(frequency=frequency, amplitude=amplitude)
        noise_generator = GaussianNoise(std=std) if noise == "white-noise" else RedNoise(std=std, tau=tau)

        mag = TimeSeries(sinusoid, noise_generator=noise_generator).sample(mjd)[0]
        pd.DataFrame({"mjd": mjd, "mag": mag}).to_pickle(os.path.join(save_dir, f"{ii}.dat"))


def gp_series(gp_config, save_dir):
    """
    :param gp_config: a dictionary in the same format as the baseConfigDict above, specifies the configurations
    :param save_dir: output the generated data to this directory
    """
    kernel = gp_config["kernel"]
    assert kernel in ["Constant", "Exponential", "Matern", "RQ"], "invalid kernel type"

    with open(os.path.join(save_dir, "config.json"), "w", encoding="utf-8") as h:
        json.dump(gp_config, h, ensure_ascii=False, indent=4)

    stop_time = gp_config["stop_time"]
    num, keep_percentage = gp_config["num"], gp_config["keep_percentage"]
    variance, gamma = gp_config["variance"], gp_config["gamma"]

    for ii in range(NUM_SERIES):
        time_sampler = TimeSampler(stop_time=stop_time)
        mjd = time_sampler.sample_irregular_time(num_points=num, keep_percentage=keep_percentage)

        if kernel == "Constant":
            gp = GaussianProcess(kernel="Constant", variance=variance)
        elif kernel == "Exponential":
            gp = GaussianProcess(kernel="Exponential", variance=variance, gamma=gamma)
        elif kernel == "Matern":
            gp = GaussianProcess(kernel="Matern", variance=variance, nu=gp_config["nu"])
        else:
            gp = GaussianProcess(kernel="RQ", variance=variance, alpha=gp_config["alpha"])

        mag = TimeSeries(signal_generator=gp).sample(mjd)[0]
        pd.DataFrame({"mjd": mjd, "mag": mag}).to_pickle(os.path.join(save_dir, f"{ii}.dat"))


def car_series(car_config, save_dir):
    """
    :param car_config: a dictionary in the same format as the baseConfigDict above, specifies the configurations
    :param save_dir: output the generated data to this directory
    """
    with open(os.path.join(save_dir, "config.json"), "w", encoding="utf-8") as h:
        json.dump(car_config, h, ensure_ascii=False, indent=4)

    stop_time = car_config["stop_time"]
    num, keep_percentage = car_config["num"], car_config["keep_percentage"]
    ar, sigma = car_config["ar"], car_config["sigma"]

    for ii in range(NUM_SERIES):
        time_sampler = TimeSampler(stop_time=stop_time)
        mjd = time_sampler.sample_irregular_time(num_points=num, keep_percentage=keep_percentage)
        car = CAR(ar_param=ar, sigma=sigma)

        mag = TimeSeries(signal_generator=car).sample(mjd)[0]
        pd.DataFrame({"mjd": mjd, "mag": mag}).to_pickle(os.path.join(save_dir, f"{ii}.dat"))
