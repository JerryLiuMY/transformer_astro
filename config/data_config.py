sample = {
  "ASAS": 300,
  "MACHO": 300,
  "WISE": 10000,
  "GAIA": 35000,
  "OGLE": 35000,
}

thresh = {
  "ASAS": 0,
  "MACHO": 0,
  "WISE": 500,
  "GAIA": 500,
  "OGLE": 500,
}

window = {
  "ASAS": 200,
  "MACHO": 200,
  "WISE": 20,
  "GAIA": 10,
  "OGLE": 200,
}

stride = {key: int(window[key]/2) for key in window}

ws = {
  "ASAS": (2, 1),
  "MACHO": (2, 1),
  "WISE": (2, 1),
  "GAIA": (2, 1),
  "OGLE": (2, 1),
}

data_config = {
  "thresh": thresh,
  "sample": sample,
  "window": window,
  "stride": stride,
  "ws": ws,
  "batch": 256,
  "kfold": 10
}
