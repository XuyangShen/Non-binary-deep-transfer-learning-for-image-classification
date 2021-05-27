import os
import numpy as np
import scipy.io as sio
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torchvision.datasets.utils import extract_archive


class Cars(VisionDataset):
    """`Stanford Cars <https://ai.stanford.edu/~jkrause/cars/car_dataset.html>`_ Dataset.

    Args:
        root (string): Root directory of the dataset.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """
    file_list = {
        'imgs': ('http://imagenet.stanford.edu/internal/car196/car_ims.tgz', 'car_ims.tgz'),
        'annos': ('http://imagenet.stanford.edu/internal/car196/cars_annos.mat', 'cars_annos.mat')
    }

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, load_bytes=False, test=False, size=42):
        
        self.mean = [0.4105214534294453, 0.38574356611082533, 0.3959628699849632]
        self.std = [0.28611458811352136, 0.2801378084154138, 0.28520087594365295]

                
        split = 'trainval' if train else 'test'
        self.split = split    
        self.loader = default_loader
        self.train = train
        self.root = root
        self.test = test
        
        
        if self._check_exists():
            print('Files already downloaded and verified.')
        elif download:
            self._download()
        else:
            raise RuntimeError(
                'Dataset not found. You can use download=True to download it.')
        (image_ids, targets, classes, class_to_idx) = self.find_classes()
        samples = self.make_dataset(image_ids, targets, train, test, classes, size)
        
        print('num class: ' + str(len(classes)))
         
        self.samples = samples              
        self.imgs = self.samples  # torchvision ImageFolder compat
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.load_bytes = load_bytes
        self.transform = transform
        self.target_transform = target_transform 

    def __getitem__(self, index):
        path, target = self.samples[index]
        path = os.path.join(self.root, path)

        image = self.loader(path)
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target

    def __len__(self):
        return len(self.samples)

    def _check_exists(self):
        return (os.path.exists(os.path.join(self.root, self.file_list['imgs'][1]))
                and os.path.exists(os.path.join(self.root, self.file_list['annos'][1])))

    def _download(self):
        print('Downloading...')
        for url, filename in self.file_list.values():
            download_url(url, root=self.root, filename=filename)
        print('Extracting...')
        archive = os.path.join(self.root, self.file_list['imgs'][1])
        extract_archive(archive)

    def find_classes(self):
        # read classes file, separating out image IDs and class names
        image_ids = []
        targets = []
        loaded_mat = sio.loadmat(os.path.join(self.root, self.file_list['annos'][1]))
        loaded_mat = loaded_mat['annotations'][0]
        self.samples = []
        for item in loaded_mat:
            if self.train != bool(item[-1][0]):
                path = str(item[0][0])
                label = int(item[-2][0]) - 1
                self.samples.append((path, label))
                targets.append(label)
                image_ids.append(path)
          

                

        # index class names
        classes = np.unique(targets)
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        targets = [class_to_idx[c] for c in targets]

        return image_ids, targets, classes, class_to_idx
        
    def make_dataset(self, image_ids, targets, train, test, classes, size):
        assert (len(image_ids) == len(targets))
        images = []
        if test:
            for i in range(len(image_ids)):
                item = (image_ids[i], targets[i])
                images.append(item)
            return images
        else:
            num_per_class = np.zeros((len(classes)))
            if size > 40:
                size == 100
            for i in range(len(image_ids)):
                if (num_per_class[targets[i]]>10)==train:
                    if train:
                        if num_per_class[targets[i]]<(size + 11):
                            item = (image_ids[i], targets[i])
                            images.append(item)
                    else:
                        item = (image_ids[i], targets[i])
                        images.append(item)
                    num_per_class[targets[i]]+=1
                else:
                    num_per_class[targets[i]]+=1
                    continue
            print('len: ' + str(len(images)))
        return images
                


if __name__ == '__main__':
    train_dataset = Cars('./cars', train=True, download=False)
    test_dataset = Cars('./cars', train=False, download=False)
