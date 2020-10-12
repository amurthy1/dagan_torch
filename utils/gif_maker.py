import os
import torch
import imageio
import numpy as np
from PIL import Image
import PIL


def convert_to_pil(arr, scale=1.0):
    arr = arr.reshape(-1, arr.shape[-1])
    arr = (arr * 0.5) + 0.5
    arr = np.uint8(arr * 255)
    h, w = arr.shape
    return (
        Image.fromarray(arr, mode="L")
        .resize((int(w * scale), int(h * scale)))
        .transpose(PIL.Image.TRANSPOSE)
    )


def create_gif(
    checkpoint_path,
    gif_path,
    fps=15,
    scale=1.0,
    sampling_rate=1,
    tracking_images_path=None,
):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))

    # Sample generations and append tracking images on top
    gens = checkpoint["tracking_images_gens"][0::sampling_rate]
    pil_images = [convert_to_pil(tensor, scale=scale) for tensor in gens]

    # Arbitrarily freeze last frame for half the length of existing gif
    pil_images += [pil_images[-1]] * (len(pil_images) // 2)

    imageio.mimsave(gif_path, [np.array(img) for img in pil_images], fps=fps)

    if tracking_images_path is not None:
        tracking_images = checkpoint["tracking_images"]
        imageio.imsave(tracking_images_path, np.array(convert_to_pil(tracking_images)))