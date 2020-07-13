from global_settings import SYNTHESIS_FOLDER
from timesynth import TimeSampler, TimeSeries
from timesynth.signals import Sinusoidal
from timesynth.noise import GaussianNoise, RedNoise
import os
import json
import pickle as pkl
import copy
import matplotlib.pyplot as plt
import time
import pandas as pd
NUM_SERIES = 500


def generate_sinusoidal(config_dict, output_dir):
    """
    this function generates the time series according to the dictionary configData,
        and output it to the directory outputDir

    :param config_dict: a dictionary in the same format as the baseConfigDict above, specifies the configurations
    :param output_dir: output the generated data to this directory
    """
    noise = config_dict["noise"]
    assert noise in ["white_noise", "red_noise"], "invalid noise type"
    configJSON = json.dumps(config_dict)

    JSONFile = open(os.path.join(output_dir, "config.json"), "w")
    JSONFile.write(configJSON)
    JSONFile.close()

    std, tau, frequency = config_dict["std"], config_dict["tau"], config_dict["frequency"]
    amplitude, num_points = config_dict["amplitude"], config_dict["num_points"]
    keep_percentage = config_dict["keep_percentage"]

    for ii in range(NUM_SERIES):
        time_sampler = TimeSampler(stop_time=config_dict["stop_time"])
        mjd = time_sampler.sample_irregular_time(num_points=num_points, keep_percentage=keep_percentage)

        sinusoid = Sinusoidal(frequency=frequency, amplitude=amplitude)
        noise_generator = GaussianNoise(std=std) if noise == "white_noise" else RedNoise(std=std, tau=tau)
        mag, _, _ = TimeSeries(sinusoid, noise_generator=noise_generator).sample(mjd)
        pd.DataFrame({"mjd": mjd, "mag": mag}).to_pickle(os.path.join(output_dir, f"{ii}.pkl"))

