import numpy as np
import os
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torchvision.datasets.utils import extract_archive

class Aircraft(VisionDataset):
    """`Dataset_Description <http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/>`_ Dataset.

    Args:
        root (string): Root directory of the dataset.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        class_type (string, optional): choose from ('variant', 'family', 'manufacturer').
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """
    url = 'http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz'
    class_types = ('variant', 'family', 'manufacturer')
    splits = ('train', 'val', 'trainval', 'test')
    img_folder = os.path.join('fgvc-aircraft-2013b', 'data', 'images')

    def __init__(self, root, train=True, class_type='variant', transform=None, target_transform=None, download=False,
                 load_bytes=False, test=True, num_class=100, num_per_class=None, seed=None):

        super().__init__(root, transform, target_transform)
        self.num_class = num_class
        self.num_per_class = num_per_class
        self.seed = seed
        self.mean = [0.48933587508932375, 0.5183537408957618, 0.5387914411673883]
        self.std = [0.22388883112804625, 0.21641635409388751, 0.24615605842636115]

        if test:
            split = 'trainval' if train else 'test'
        else:
            split = 'train' if train else 'val'
        if split not in self.splits:
            raise ValueError('Split "{}" not found. Valid splits are: {}'.format(
                split, ', '.join(self.splits),
            ))
        if class_type not in self.class_types:
            raise ValueError('Class type "{}" not found. Valid class types are: {}'.format(
                class_type, ', '.join(self.class_types),
            ))

        self.class_type = class_type
        self.split = split
        self.root = root
        self.classes_file = os.path.join(self.root, 'fgvc-aircraft-2013b', 'data',
                                         'images_%s_%s.txt' % (self.class_type, self.split))

        if download:
            self.download()

        (image_ids, targets, classes, class_to_idx) = self.find_classes()

        samples = self.make_dataset(image_ids, targets)

        self.loader = default_loader

        self.samples = samples
        self.imgs = self.samples  # torchvision ImageFolder compat
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.load_bytes = load_bytes
        self.transform = transform
        self.target_transform = target_transform


    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def __len__(self):
        return len(self.samples)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.img_folder)) and \
               os.path.exists(self.classes_file)

    def download(self):
        if self._check_exists():
            return

        # prepare to download data to PARENT_DIR/fgvc-aircraft-2013.tar.gz
        print('Downloading %s...' % self.url)
        tar_name = self.url.rpartition('/')[-1]
        download_url(self.url, root=self.root, filename=tar_name)
        tar_path = os.path.join(self.root, tar_name)
        print('Extracting %s...' % tar_path)
        extract_archive(tar_path)
        print('Done!')

    def filter_size_per_class(self,
                              images_in_class: list) -> list:
        if self.num_per_class is None:
            # do not need to filter
            return images_in_class
        msk = np.arange(len(images_in_class))
        if self.seed is None:
            np.random.shuffle(msk)
        msk = msk[: self.num_per_class]
        sorted(msk)
        images_in_class = np.array(images_in_class)[msk]
        return list(images_in_class)

    def filter_class(self,
                     sample_with_class: list) -> list:
        if self.num_class == len(sample_with_class):
            return sample_with_class
        msk = np.arange(len(sample_with_class))
        if self.seed is None:
            np.random.shuffle(msk)
        msk = msk[: self.num_class]
        sorted(msk)
        sample_with_class = np.array(sample_with_class)[msk]
        return list(sample_with_class)

    def find_classes(self):
        # read classes file, separating out image IDs and class names

        samples_with_class = dict()

        with open(self.classes_file, 'r') as f:
            all_lines = f.readlines()
            pre_class = ' '.join(all_lines[0][:-1].split(' ')[1:])
            class_sample = []
            for line in all_lines:
                line = line[:-1]
                split_line = line.split(' ')
                _id = split_line[0]
                _class = ' '.join(split_line[1:])
                # print(line, _class, pre_class)
                if _class != pre_class:
                    # print(pre_class, samples_with_class.keys())
                    if pre_class not in samples_with_class:
                        samples_with_class[pre_class] = self.filter_size_per_class(class_sample)
                    else:
                        samples_with_class[pre_class].extend(self.filter_size_per_class(class_sample))
                    class_sample = []
                    pre_class = _class
                class_sample.append(_id)

            if pre_class not in samples_with_class:
                samples_with_class[pre_class] = self.filter_size_per_class(class_sample)
            else:
                samples_with_class[pre_class].extend(self.filter_size_per_class(class_sample))

        classes = self.filter_class(list(samples_with_class.keys()))
        image_ids = []
        targets = []
        for k in classes:
            ims = samples_with_class[k]
            image_ids.extend(ims)
            targets.extend([k] * len(ims))
        # index class names
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        targets = [class_to_idx[c] for c in targets]

        return image_ids, targets, classes, class_to_idx

    def make_dataset(self, image_ids, targets):
        assert (len(image_ids) == len(targets))
        images = []
        for i in range(len(image_ids)):
            item = (os.path.join(self.root, self.img_folder,
                                 '%s.jpg' % image_ids[i]), targets[i])
            images.append(item)
        return images


if __name__ == '__main__':
    train_dataset = Aircraft('./aircraft', train=True, download=False)
    test_dataset = Aircraft('./aircraft', train=False, download=False)
