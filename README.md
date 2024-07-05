# Daily DDPM Anime
![DailyDDPMAnime](https://img.shields.io/badge/DailyDDPMAnime-v1.0-lightred)
## Overview

This repository contains the code for an Instagram AI bot that generates and posts anime pictures. The bot utilizes a deep learning model, specifically a Denoising Diffusion Probabilistic Model (DDPM), implemented with PyTorch. The generated images are then automatically posted to Instagram.

## Features

- Generate high-quality anime images using a DDPM.
- Automatically post generated images to an Instagram account.
- Customizable posting schedule.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Deep Learning Model](#deep-learning-model)
- [Contributing](#contributing)

## Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/Kazedaa/Daily-DDPM-Anime
    cd Daily-DDPM-Anime
    ```

2. **Create and activate a virtual environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate
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
    python post_image.py
    ```

2. **Schedule automatic posting:**

    Use a task scheduler (like cron jobs on Unix systems or Task Scheduler on Windows) to run the `post_image.py` script at your desired intervals.

## Deep Learning Model

### Denoising Diffusion Probabilistic Model (DDPM)

DDPMs are generative models that use a diffusion process to generate high-quality images. The process involves gradually adding noise to training data and then learning to reverse this process to create new data samples.

Our model is built using PyTorch and follows the standard DDPM architecture:

1. **Forward Diffusion Process:** Adds Gaussian noise to the images over a series of timesteps.
2. **Reverse Diffusion Process:** Trains a neural network to denoise the images, step by step, to generate new samples.

### Model Architecture

- **U-Net Backbone:** A simple Unet with Alternating Residual and Attention Block along with Sinusoidal Positional Embedding.
- **Noise Schedule:** Controls the amount of noise added at each time step during the forward process.
- **Loss Function:** A combination of mean squared error (MSE) between the denoised output and the original image.

