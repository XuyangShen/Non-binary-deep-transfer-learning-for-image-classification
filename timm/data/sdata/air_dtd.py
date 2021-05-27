import numpy as np
import os
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader


class AirDtd(VisionDataset):
    # pre-defined variables
    SPLITS = ('train', 'val', 'trainval', 'test')
    AIR_PATH = os.path.join('fgvc-aircraft-2013b', 'data', 'images')
    DTD_PATH = os.path.join("dtd")
    DTD_SPLIT = '1'

    def __init__(self, root, train, test,
                 transform=None,
                 num_class=47, num_per_class=33,
                 seed=None):
        super().__init__(root, transform)
        self.transform = transform
        self.loader = default_loader

        self.num_class = num_class
        self.num_per_class = num_per_class
        self.seed = seed

        self.mean = [(0.48933587508932375 + 0.5329876098715876) / 2, (0.5183537408957618 + 0.474260843249454) / 2,
                     (0.5387914411673883 + 42627281899380676) / 2]
        self.std = [(0.22388883112804625 + 0.26549755708788914) / 2, (0.21641635409388751 + 0.25473554309855373) / 2,
                    (0.24615605842636115 + 0.2631728035662832) / 2]

        if test:
            self.split = 'trainval' if train else 'test'
        else:
            self.split = 'train' if train else 'val'

        if self.split not in self.SPLITS:
            raise ValueError('Split "{}" not found. Valid splits are: {}'.format(
                self.split, ', '.join(self.SPLITS),
            ))

        self.root = root  # common = data
        self.air_classes_file = os.path.join(self.root, 'fgvc-aircraft-2013b', 'data',
                                             'images_%s_%s.txt' % ("variant", self.split))
        _set = []
        _set.extend(self.air_make_dataset())
        _set.extend(self.dtd_make_dataset())

        self.samples = _set
        self.imgs = self.samples  # torchvision ImageFolder compat


    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target

    def __len__(self):
        return len(self.samples)

    def air_filter_size_per_class(self,
                                  images_in_class: list) -> list:
        if self.num_per_class is None:
            # do not need to filter
            return images_in_class
        elif self.num_per_class == len(images_in_class):
            return images_in_class

        assert self.num_per_class < len(
            images_in_class), "filter sample amounts in dataset's class is larger, actual %i, requried %i" % (
        len(images_in_class), self.num_per_class)
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
        assert self.num_class < len(
            sample_with_class), "filter class amounts in dataset is larger, actual %i, requried %i" % (
        len(sample_with_class), self.num_class)
        msk = np.arange(len(sample_with_class))
        if self.seed is None:
            np.random.shuffle(msk)
        msk = msk[: self.num_class]
        sorted(msk)
        sample_with_class = np.array(sample_with_class)[msk]
        return list(sample_with_class)

    def air_make_dataset(self) -> list:
        # read classes file, separating out image IDs and class names

        samples_with_class = dict()

        with open(self.air_classes_file, 'r') as f:
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
                        samples_with_class[pre_class] = self.air_filter_size_per_class(class_sample)
                    else:
                        samples_with_class[pre_class].extend(self.air_filter_size_per_class(class_sample))
                    class_sample = []
                    pre_class = _class
                class_sample.append(_id)
            if pre_class not in samples_with_class:
                samples_with_class[pre_class] = self.air_filter_size_per_class(class_sample)
            else:
                samples_with_class[pre_class].extend(self.air_filter_size_per_class(class_sample))

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

        # return image_ids, targets, classes, class_to_idx

        assert (len(image_ids) == len(targets)), "air image amounts is not equal to label amounts"
        dataset = []
        for i in range(len(image_ids)):
            item = (os.path.join(self.root, self.AIR_PATH,
                                 '%s.jpg' % image_ids[i]), targets[i])
            dataset.append(item)
        return dataset  # List[(img path, label), ... ]

    def dtd_filter_size_per_class(self,
                                  images_in_class: list, labels_in_class: list) -> [list, list]:
        assert len(images_in_class) == len(labels_in_class)
        assert self.num_per_class < len(
            images_in_class), "filter sample amounts in dataset's class is larger, actual %i, requried %i" % (
            len(images_in_class), self.num_per_class)

        msk = np.arange(len(images_in_class))
        if self.seed is None:
            np.random.shuffle(msk)
        msk = msk[: self.num_per_class]
        sorted(msk)
        images_in_class = np.array(images_in_class)[msk]
        labels_in_class = np.array(labels_in_class)[msk]
        return list(images_in_class), list(labels_in_class)

    def dtd_make_dataset(self) -> list:
        dir = os.path.join(self.root, self.DTD_PATH, 'images')
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        del classes  # clear memory

        if self.split != "trainval":
            txtnames = [os.path.join(self.root, self.DTD_PATH, "labels", self.split + self.DTD_SPLIT + '.txt')]
        else:
            txtnames = [os.path.join(self.root, self.DTD_PATH, "labels", "train" + self.DTD_SPLIT + '.txt'),
                        os.path.join(self.root, self.DTD_PATH, "labels", "val" + self.DTD_SPLIT + '.txt')]

        images = []
        labels = []
        for txtname in txtnames:
            with open(txtname, 'r') as lines:
                all_lines = lines.readlines()
                pre_class = all_lines[0].split('/')[0]
                # image id and label inside each class
                images_in_class = []
                labels_in_class = []
                for line in all_lines:
                    classname = line.split('/')[0]
                    _img = os.path.join(self.root, self.DTD_PATH, 'images', line.strip())
                    if classname != pre_class:
                        pre_class = classname
                        # push class's label and data into main ist
                        images_in_class, labels_in_class = self.dtd_filter_size_per_class(images_in_class, labels_in_class)
                        images.extend(images_in_class)
                        labels.extend(labels_in_class)
                        images_in_class = []
                        labels_in_class = []
                    assert os.path.isfile(_img)
                    images_in_class.append(_img)
                    labels_in_class.append(class_to_idx[classname] + self.num_class)
                images_in_class, labels_in_class = self.dtd_filter_size_per_class(images_in_class, labels_in_class)
                images.extend(images_in_class)
                labels.extend(labels_in_class)

        dataset = list(zip(images, labels))

        return dataset
