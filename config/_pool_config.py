sin = {
    "signal_type": "Sinusoidal",
    "stop_time": 20,
    "num": 500,
    "keep_percentage": 50,
    "frequency": 0.25,
    "amplitude": 1.0,
    "noise": "white-noise",
    "std": 0.3,
    "tau": 0
  }

gp = {
     "signal_type": "GP",
     "stop_time": 20,
     "num": 500,
     "keep_percentage": 50,
     "variance": 1.0,
     "gamma": 1.0,
     "kernel": "Exponential"
   }

car = {
    "signal_type": "CAR",
    "stop_time": 20,
    "num": 500,
    "keep_percentage": 50,
    "ar": 0.9,
    "sigma": 1
   }

pool_dict = {"sin": sin, "gp": gp, "car": car}
