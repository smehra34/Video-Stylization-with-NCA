import os
from functools import partial

import torch
import gdown

import progressbar

pbar = None


def show_progress(block_num, block_size, total_size):
    global pbar
    if pbar is None:
        pbar = progressbar.ProgressBar(maxval=total_size)
        pbar.start()

    downloaded = block_num * block_size
    if downloaded < total_size:
        pbar.update(downloaded)
    else:
        pbar.finish()
        pbar = None


def _load_MSOEmultiscale_model(model_name, models_path, download=False):
    model_directory = os.path.join(models_path, 'two_stream')
    model_path = os.path.join(model_directory, f'{model_name}.pth')

    # Ensure the model directory exists
    os.makedirs(model_directory, exist_ok=True)

    if not os.path.exists(model_path):
        download = True

    if download:
        # Define the Google Drive URL and output path
        url = 'https://drive.google.com/uc?id=10qoSx0P3TJzf17bUN42x1ZAFNjr-J69f'
        gdown.download(url, model_path, quiet=False)

    from models.MSOEmultiscale import MSOEmultiscale
    model = MSOEmultiscale()
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model = model.eval()
    return model


_model_factories = {}
_model_factories['two_stream_dynamic'] = partial(_load_MSOEmultiscale_model, model_name='two_stream_dynamic_model')


def get_available_models():
    return _model_factories.keys()


def get_model(name, *args, **kwargs):
    return _model_factories[name](*args, **kwargs)
