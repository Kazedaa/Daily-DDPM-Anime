from configparser import ConfigParser

config = ConfigParser()

config['DEFAULT'] = {
    "IMG_SIZE" : 64,
    "TIMESTEPS" : 1000,
    "IMG_SHAPE" : (3, 64, 64),

    "BASE_CH" :128,
    "BASE_CH_MULT" : (1, 2, 2, 4),
    "APPLY_ATTENTION" : (False, False, True, True),
    "DROPOUT_RATE" : 0.1,
    "TIME_EMB_MULT" : 4
}

with open("Generator\config.ini" , "w") as file : 
    config.write(file)