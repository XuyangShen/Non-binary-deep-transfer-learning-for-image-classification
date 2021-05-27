import os

from .dataset import IterableImageDataset, ImageDataset
from .sdata.aircraft import Aircraft
from .sdata.dtd import DTD
from .sdata.air_dtd import AirDtd
from .sdata.caltech import Caltech256
from .sdata.cars import Cars

def _search_split(root, split):
    # look for sub-folder with name of split in root and use that if it exists
    split_name = split.split('[')[0]
    try_root = os.path.join(root, split_name)
    if os.path.exists(try_root):
        return try_root
    if split_name == 'validation':
        try_root = os.path.join(root, 'val')
        if os.path.exists(try_root):
            return try_root
    return root

def create_dataset(name, root, split='validation', search_split=True, is_training=False, batch_size=None, seed=None, **kwargs):
    name = name.lower()
    if name.startswith('tfds'):
        ds = IterableImageDataset(
            root, parser=name, split=split, is_training=is_training, batch_size=batch_size, **kwargs)
    else:
        if name == 'aircraft':
            ds = Aircraft(root, transform=None, train=is_training, download=False, test=False)
        elif name == 'dtd':
            ds = DTD(root, transform=None, train=is_training, download=False, test=False)
        elif name == 'airdtd':
            ds = AirDtd(root, transform=None, train=is_training, test=False, num_class=47, num_per_class=33, seed=seed)
        elif name == 'caltech':
            ds = Caltech256(root, transform=None, download=False, train=is_training, test=False)
        elif name == 'car':
            ds = Cars(root, transform=None, download=False, train=is_training, test=False, size=10)
        elif name == "imagenet":
            if search_split and os.path.isdir(root):
                root = _search_split(root, split)
            ds = ImageDataset(root, parser=name, **kwargs)
    return ds
