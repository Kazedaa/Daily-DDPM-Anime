import os
import torch
from Generator.diffusion import SimpleDiffusion
from Generator.unet import UNet
from Generator.diffusion import reverse_diffusion
import cv2

from configparser import ConfigParser
config = ConfigParser()
config.read(os.path.join("Generator","config.ini"))

DEVICE="cuda" if torch.cuda.is_available() else "cpu"

IMG_SIZE = int(config["DEFAULT"]["img_size"])
TIMESTEPS = int(config["DEFAULT"]["timesteps"])
IMG_SHAPE = eval(config["DEFAULT"]["img_shape"])

BASE_CH = int(config["DEFAULT"]["BASE_CH"])
BASE_CH_MULT = eval(config["DEFAULT"]["BASE_CH_MULT"])
APPLY_ATTENTION = eval(config["DEFAULT"]["APPLY_ATTENTION"])
DROPOUT_RATE = float(config["DEFAULT"]["DROPOUT_RATE"])
TIME_EMB_MULT = int(config["DEFAULT"]["TIME_EMB_MULT"])

MODEL_PATH = os.path.join("Generator","ddpm_lin_model_110.pth")

model = UNet(
    input_channels          = IMG_SHAPE[0],
    output_channels         = IMG_SHAPE[0],
    base_channels           = BASE_CH,
    base_channels_multiples = BASE_CH_MULT,
    apply_attention         = APPLY_ATTENTION,
    dropout_rate            = DROPOUT_RATE,
    time_multiple           = TIME_EMB_MULT,
)

model.load_state_dict(torch.load(MODEL_PATH,map_location=torch.device(DEVICE)),)

sd = SimpleDiffusion(
    num_diffusion_timesteps = TIMESTEPS,
    img_shape               = IMG_SHAPE,
)


def generate(path_to_save):
    reverse_diffusion(
        model = model,
        sd = sd,
        path_to_save = path_to_save
    )
    image = cv2.imread(path_to_save)

    image = cv2.resize(image , (566 , 566) , interpolation=cv2.INTER_CUBIC)

    cv2.imwrite(path_to_save , image)

    return path_to_save
