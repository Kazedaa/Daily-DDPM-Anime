import os
import torch
from torchvision.utils import save_image , make_grid
from tqdm import tqdm
from torch import nn


from configparser import ConfigParser
config = ConfigParser()
config.read(os.path.join("Generator","config.ini"))

DEVICE="cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {DEVICE}...")

IMG_SIZE = int(config["DEFAULT"]["img_size"])
TIMESTEPS = int(config["DEFAULT"]["timesteps"])
IMG_SHAPE = eval(config["DEFAULT"]["img_shape"])

def get(element, t):
    ele = element.gather(-1, t)
    return ele.reshape(-1, 1, 1, 1)

class SimpleDiffusion(nn.Module):
    def __init__(
        self,
        num_diffusion_timesteps=TIMESTEPS,
        img_shape=IMG_SHAPE,
    ):
        super().__init__()
        T = num_diffusion_timesteps
        self.img_shape = img_shape
                
        beta_start = 1e-4
        beta_end = 0.02
        self.beta = torch.linspace(
            beta_start,
            beta_end,
            T,
            dtype=torch.float32,
        )
        self.alpha = 1 - self.beta
        self.sqrt_beta = torch.sqrt(self.beta)
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        self.sqrt_alpha_hat = torch.sqrt(self.alpha_hat)
        self.one_by_sqrt_alpha = 1. / torch.sqrt(self.alpha)
        self.sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat)
    
        
    def forward(self, x0, timesteps ,var_type = 'fixed'):
        def get_mean(x0 , timesteps):
            return get(self.sqrt_alpha_hat, t=timesteps) * x0
        
        def get_std_dev(x0 , timesteps ,var_type):
            return get(self.sqrt_one_minus_alpha_hat, t=timesteps)
            
        eps     = torch.randn_like(x0)  # Noise
        mean    = get_mean(x0 ,timesteps)  # Image scaled
        std_dev = get_std_dev(x0 , timesteps , var_type) # Noise scaled
        sample  = mean + std_dev * eps # scaled inputs * scaled noise

        return sample, eps  # return ... , gt noise --> model predicts this)
    

def get_sample(x , z ,sd , predicted_noise , ts):
    std_dev = get(sd.sqrt_one_minus_alpha_hat.to(DEVICE), t=ts.to(DEVICE))
    
    sqrt_beta_t                = get(sd.sqrt_beta, ts)
    beta_t                     = get(sd.beta, ts)
    one_by_sqrt_alpha_t        = get(sd.one_by_sqrt_alpha, ts)
    sqrt_one_minus_alpha_hat_t = get(sd.sqrt_one_minus_alpha_hat, ts) 

    x = (
        one_by_sqrt_alpha_t
        * (x - (beta_t / sqrt_one_minus_alpha_hat_t) * predicted_noise)
        + sqrt_beta_t * z
    )
    return x

@torch.no_grad()
def reverse_diffusion(model, sd ,path_to_save):

    x = torch.randn((4, *IMG_SHAPE))
    model.eval()

    for time_step in tqdm(iterable=reversed(range(1, TIMESTEPS)), 
                          total=TIMESTEPS-1, dynamic_ncols=False, 
                          desc="Sampling :: ", position=0):
        

        ts = torch.ones(4, dtype=torch.long) * time_step
        z = torch.randn_like(x) if time_step > 1 else torch.zeros_like(x)

        predicted_noise = model(x, ts)
        x = get_sample(x , z, sd , predicted_noise ,ts)

        
    grid = make_grid(x,nrow=2).cpu()
    save_image(grid, path_to_save , nrow=1, normalize=True , pad_value = -1)

if __name__ == "__main__":
    print("Diffusion works")
