from __future__ import annotations

import os
import pathlib
import pickle
import sys

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

current_dir = pathlib.Path(__file__).parent
submodule_dir = current_dir / 'projected_gan'
sys.path.insert(0, submodule_dir.as_posix())

HF_TOKEN = os.getenv('HF_TOKEN')


class Model:

    MODEL_NAMES = [
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

    def __init__(self):
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        self._download_all_models()
        self.model_name = self.MODEL_NAMES[3]
        self.model = self._load_model(self.model_name)

    def _load_model(self, model_name: str) -> nn.Module:
        path = hf_hub_download('hysts/projected_gan',
                               f'models/{model_name}.pkl',
                               use_auth_token=HF_TOKEN)
        with open(path, 'rb') as f:
            model = pickle.load(f)['G_ema']
        model.eval()
        model.to(self.device)
        return model

    def set_model(self, model_name: str) -> None:
        if model_name == self.model_name:
            return
        self.model_name = model_name
        self.model = self._load_model(model_name)

    def _download_all_models(self):
        for name in self.MODEL_NAMES:
            self._load_model(name)

    def generate_z(self, seed: int) -> torch.Tensor:
        seed = int(np.clip(seed, 0, np.iinfo(np.uint32).max))
        z = np.random.RandomState(seed).randn(1, self.model.z_dim)
        return torch.from_numpy(z).float().to(self.device)

    def postprocess(self, tensor: torch.Tensor) -> np.ndarray:
        tensor = (tensor.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(
            torch.uint8)
        return tensor.cpu().numpy()

    @torch.inference_mode()
    def generate(self, z: torch.Tensor, label: torch.Tensor,
                 truncation_psi: float) -> torch.Tensor:
        return self.model(z, label, truncation_psi=truncation_psi)

    def generate_image(self, seed: int, truncation_psi: float) -> np.ndarray:
        z = self.generate_z(seed)
        label = torch.zeros([1, self.model.c_dim], device=self.device)

        out = self.generate(z, label, truncation_psi)
        out = self.postprocess(out)
        return out[0]

    def set_model_and_generate_image(self, model_name: str, seed: int,
                                     truncation_psi: float) -> np.ndarray:
        self.set_model(model_name)
        return self.generate_image(seed, truncation_psi)
