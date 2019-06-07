
from pretraining.autoencoder_dataset import UnsupDataset

def create(cfg):
    data = UnsupDataset(cfg)
    return data
