#!/usr/bin/env python

from __future__ import annotations

import argparse
import functools
import os
import pickle
import sys

import gradio as gr
import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

sys.path.insert(0, 'projected_gan')

TITLE = 'autonomousvision/projected_gan'
DESCRIPTION = '''This is a demo for https://github.com/autonomousvision/projected_gan.

Expected execution time on Hugging Face Spaces: 1s
'''
SAMPLE_IMAGE_DIR = 'https://huggingface.co/spaces/hysts/projected_gan/resolve/main/samples'
ARTICLE = f'''## Generated images
- truncation: 0.7
- size: 256x256
- seed: 0-99
### Art painting
![Art painting samples]({SAMPLE_IMAGE_DIR}/art_painting.jpg)
### Bedroom
![Bedroom samples]({SAMPLE_IMAGE_DIR}/bedroom.jpg)
### Church
![Church samples]({SAMPLE_IMAGE_DIR}/church.jpg)
### Cityscapes
![Cityscapes samples]({SAMPLE_IMAGE_DIR}/cityscapes.jpg)
### CLEVR
![CLEVR samples]({SAMPLE_IMAGE_DIR}/clevr.jpg)
### FFHQ
![FFHQ samples]({SAMPLE_IMAGE_DIR}/ffhq.jpg)
### Flowers
![Flowers samples]({SAMPLE_IMAGE_DIR}/flowers.jpg)
### Landscape
![Landscape samples]({SAMPLE_IMAGE_DIR}/landscape.jpg)
### Pokemon
![Pokemon samples]({SAMPLE_IMAGE_DIR}/pokemon.jpg)
'''

TOKEN = os.environ['TOKEN']


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--theme', type=str)
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--share', action='store_true')
    parser.add_argument('--port', type=int)
    parser.add_argument('--disable-queue',
                        dest='enable_queue',
                        action='store_false')
    parser.add_argument('--allow-flagging', type=str, default='never')
    parser.add_argument('--allow-screenshot', action='store_true')
    return parser.parse_args()


def generate_z(z_dim: int, seed: int, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(
        np.random.RandomState(seed).randn(1,
                                          z_dim).astype(np.float32)).to(device)


@torch.inference_mode()
def generate_image(model_name: str, seed: int, truncation_psi: float,
                   model_dict: dict[str, nn.Module],
                   device: torch.device) -> np.ndarray:
    model = model_dict[model_name]
    seed = int(np.clip(seed, 0, np.iinfo(np.uint32).max))

    z = generate_z(model.z_dim, seed, device)
    label = torch.zeros([1, model.c_dim], device=device)

    out = model(z, label, truncation_psi=truncation_psi)
    out = (out.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    return out[0].cpu().numpy()


def load_model(model_name: str, device: torch.device) -> nn.Module:
    path = hf_hub_download('hysts/projected_gan',
                           f'models/{model_name}.pkl',
                           use_auth_token=TOKEN)
    with open(path, 'rb') as f:
        model = pickle.load(f)['G_ema']
    model.eval()
    model.to(device)
    with torch.inference_mode():
        z = torch.zeros((1, model.z_dim)).to(device)
        label = torch.zeros([1, model.c_dim], device=device)
        model(z, label)
    return model


def main():
    args = parse_args()
    device = torch.device(args.device)

    model_names = [
        'art_painting',
        'church',
        'bedroom',
        'cityscapes',
        'clevr',
        'ffhq',
        'flowers',
        'landscape',
        'pokemon',
    ]

    model_dict = {name: load_model(name, device) for name in model_names}

    func = functools.partial(generate_image,
                             model_dict=model_dict,
                             device=device)
    func = functools.update_wrapper(func, generate_image)

    gr.Interface(
        func,
        [
            gr.inputs.Radio(
                model_names, type='value', default='pokemon', label='Model'),
            gr.inputs.Number(default=0, label='Seed'),
            gr.inputs.Slider(
                0, 2, step=0.05, default=0.7, label='Truncation psi'),
        ],
        gr.outputs.Image(type='numpy', label='Output'),
        title=TITLE,
        description=DESCRIPTION,
        article=ARTICLE,
        theme=args.theme,
        allow_screenshot=args.allow_screenshot,
        allow_flagging=args.allow_flagging,
        live=args.live,
    ).launch(
        enable_queue=args.enable_queue,
        server_port=args.port,
        share=args.share,
    )


if __name__ == '__main__':
    main()
