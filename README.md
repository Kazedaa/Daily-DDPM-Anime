# Daily DDPM Anime
![DailyDDPMAnime](https://img.shields.io/badge/DailyDDPMAnime-v1.0-lightred)

This repository contains the code for an Instagram AI bot that generates and posts anime pictures. The bot utilizes a deep learning model, specifically a Denoising Diffusion Probabilistic Model (DDPM), implemented with PyTorch. The generated images are then automatically posted to [Instagram: Dailly Anime DDPM](https://www.instagram.com/daily.ddpm.anime/).

![image](https://github.com/Kazedaa/Daily-DDPM-Anime/assets/120291477/b0d1fb90-dceb-4db8-9e57-29f30720f125) ![image](https://github.com/Kazedaa/Daily-DDPM-Anime/assets/120291477/4b11d751-8ad9-498b-8dcd-97438344e930)
![image](https://github.com/Kazedaa/Daily-DDPM-Anime/assets/120291477/1e069377-0a89-4485-a728-78782699d126) ![image](https://github.com/Kazedaa/Daily-DDPM-Anime/assets/120291477/2e481b0d-8aa4-4603-b7c1-b795f00784b4)


## Features

- Generate high-quality anime images using a DDPM.
- Automatically post generated images to an Instagram account.
- Customizable posting schedule.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Denoising Diffusion Probabilistic Model](#Denoising-Diffusion-Probabilistic-Model-(DDPM))

## Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/Kazedaa/Daily-DDPM-Anime
    cd Daily-DDPM-Anime
    ```

2. **Create and activate a virtual environment:**

    ```bash
    conda create -n anime_ddpm python=3.10
    conda activate anime_ddpm
    ```

3. **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Set up Instagram credentials:**

    Create a `.env` file in the root directory of the project and add your Instagram credentials:

    ```plaintext
    INSTAGRAM_USERNAME=your_instagram_username
    INSTAGRAM_PASSWORD=your_instagram_password
    ```

## Usage

1. **Generate and post an image:**

    ```bash
    python post.py
    ```

2. **Schedule automatic posting:**

    Use a task scheduler (like cron jobs on Unix systems or Task Scheduler on Windows) to run the `post_image.py` script at your desired intervals.


## Denoising Diffusion Probabilistic Model (DDPM)

DDPMs are generative models that use a diffusion process to generate high-quality images. The process involves gradually adding noise to training data and then learning to reverse this process to create new data samples. The model was trained on several Anime face images from the [Dataset: Anime Face Dataset](https://www.kaggle.com/datasets/splcher/animefacedataset).

The model is built using PyTorch and follows the standard DDPM architecture:

1. **Forward Diffusion Process:** Adds Gaussian noise to the images over a series of timesteps.
2. **Reverse Diffusion Process:** Trains a neural network to denoise the images, step by step, to generate new samples.

### Model Architecture

- **U-Net Backbone:** A simple Unet with Alternating Residual and Attention Block along with Sinusoidal Positional Embedding.
- **Noise Schedule:** Controls the amount of noise added at each time step during the forward process a Linear Noise Schedule was used.
- **Loss Function:** Mean Squared Error (MSE) between the denoised output and the original image.

### Resources
- [Paper: Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- [Dataset: Anime Face Dataset](https://www.kaggle.com/datasets/splcher/animefacedataset)

Note: The Instagram account is no longer active since Instagram banned the proxy. If you have any solutions please feel free to post an Issue or contact me (the Author).
