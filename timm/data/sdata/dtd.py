import os.path
import numpy as np
import torch.utils.data as data

from PIL import Image

"""
The data is available from https://www.robots.ox.ac.uk/~vgg/data/dtd/
"""


class DTD(data.Dataset):
    def __init__(self, root, transform=None, train=True, download=False, test=False, num_class=47, num_per_class=None, seed=None):
        self.seed = seed
        self.root = root
        self.train = train
        self.transform = transform
        self.num_class= num_class
        self.num_per_class = num_per_class
        self.mean = [0.5329876098715876, 0.474260843249454, 0.42627281899380676]
        self.std = [0.26549755708788914, 0.25473554309855373, 0.2631728035662832]

        split = '1'
        if train:
            if test:
                self.txtnames = [os.path.join(root, 'labels/train' + split + '.txt'),
                                 os.path.join(root, 'labels/val' + split + '.txt')]
            else:
                self.txtnames = [os.path.join(root, 'labels/train' + split + '.txt')]
        else:
            if test:
                self.txtnames = [os.path.join(root, 'labels/test' + split + '.txt')]
            else:
                self.txtnames = [os.path.join(root, 'labels/val' + split + '.txt')]

        self.classes = None
        self.class_to_idx = None
        self.find_classes()
        assert self.class_to_idx is not None
        self.images, self.labels = self.make_dataset()

        assert (len(self.images) == len(self.labels))

    def __getitem__(self, index):
        _img = Image.open(self.images[index]).convert('RGB')
        _label = self.labels[index]
        if self.transform is not None:
            _img = self.transform(_img)

        return _img, _label

    def __len__(self):
        return len(self.images)

    def find_classes(self):
        dir = os.path.join(self.root, 'images')
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        self.classes = classes
        self.class_to_idx = class_to_idx

    def filter_size_per_class(self,
                              images_in_class: list, labels_in_class: list) -> [list, list]:
        if self.num_per_class is None:
            return list(images_in_class), list(labels_in_class)
        assert len(images_in_class) == len(labels_in_class)
        msk = np.arange(len(images_in_class))
        if self.seed is None:
            np.random.shuffle(msk)
        msk = msk[: self.num_per_class]
        sorted(msk)
        images_in_class = np.array(images_in_class)[msk]
        labels_in_class = np.array(labels_in_class)[msk]
        return list(images_in_class), list(labels_in_class)

    def make_dataset(self):
        images = []
        labels = []
        for txtname in self.txtnames:
            with open(txtname, 'r') as lines:
                all_lines = lines.readlines()
                pre_class = all_lines[0].split('/')[0]
                # image id and label inside each class
                images_in_class = []
                labels_in_class = []
                for line in all_lines:
                    classname = line.split('/')[0]
                    _img = os.path.join(self.root, 'images', line.strip())
                    if classname != pre_class:
                        pre_class = classname
                        # push class's label and data into main ist
                        images_in_class, labels_in_class = self.filter_size_per_class(images_in_class, labels_in_class)
                        images.extend(images_in_class)
                        labels.extend(labels_in_class)
                        images_in_class = []
                        labels_in_class = []
                    assert os.path.isfile(_img)
                    images_in_class.append(_img)
                    labels_in_class.append(self.class_to_idx[classname])
                images_in_class, labels_in_class = self.filter_size_per_class(images_in_class, labels_in_class)
                images.extend(images_in_class)
                labels.extend(labels_in_class)
        return images, labels