import os
import torch
import numpy as np
from glob import glob
from torch.utils.data import Dataset
import h5py
import itertools
from torch.utils.data.sampler import Sampler
from torchvision.transforms import Compose

class BraTS2019(Dataset):
    """ BraTS2019 Dataset """

    def __init__(self, data_dir, list_dir, split, aug_times=1):
        self.data_dir = data_dir
        self.list_dir = list_dir
        self.split = split
        self.aug_times = aug_times
        # self._base_dir = base_dir
        # self.transform = transform
        # self.sample_list = []
        tr_transform = Compose([
                             RandomRotFlip(),
                             RandomCrop((96, 96, 96)),
                             ToTensor(),
                             ])
        test_transform = Compose([
                                CenterCrop((96, 96, 96)),
                                ToTensor(),
                                ])
        if split == 'lab':
            data_path = os.path.join(list_dir,'train_lab.txt')
            self.transform = tr_transform
        elif split == 'unlab':
            data_path = os.path.join(list_dir,'train_unlab.txt')
            self.transform = test_transform 
        elif split == 'train':
            data_path = os.path.join(list_dir,'train.txt')
            self.transform = tr_transform
        elif split == 'val':
            data_path = os.path.join(list_dir,'val.txt')
            self.transform = test_transform
        else:
            data_path = os.path.join(list_dir,'test.txt')
            self.transform = test_transform

        with open(data_path, 'r') as f:
            self.image_list = f.readlines()

        self.image_list = [self.data_dir+ "/{}".format(item.strip()) + '.h5' for item in self.image_list]
        print("Split : {}, total {} samples".format(split, len(self.image_list)))
        
    def __len__(self):
        if self.split != 'test':
            return len(self.image_list) * self.aug_times
        else:
            return len(self.image_list)

    def __getitem__(self, idx):
        image_path = self.image_list[idx % len(self.image_list)]
        h5f = h5py.File(image_path, 'r')
        image, label = h5f['image'][:], h5f['label'][:].astype(np.float32)
        # image_norm = (image-np.min(image))/(np.max(image)-np.min(image))

        samples = {'image': image, 'label': label.astype(np.uint8)}
        if self.transform:
            tr_samples = self.transform(samples)
        image_, label_= tr_samples['image'], tr_samples['label']
        return {'image':image_.float(), 'label':label_.long(), 'name':image_path}


class CenterCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)],
                           mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)],
                           mode='constant', constant_values=0)

        (w, h, d) = image.shape

        w1 = int(round((w - self.output_size[0]) / 2.))
        h1 = int(round((h - self.output_size[1]) / 2.))
        d1 = int(round((d - self.output_size[2]) / 2.))

        label = label[w1:w1 + self.output_size[0], h1:h1 +
                      self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 +
                      self.output_size[1], d1:d1 + self.output_size[2]]

        return {'image': image, 'label': label}


class RandomCrop(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size, with_sdf=False):
        self.output_size = output_size
        self.with_sdf = with_sdf

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if self.with_sdf:
            sdf = sample['sdf']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)],
                           mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)],
                           mode='constant', constant_values=0)
            if self.with_sdf:
                sdf = np.pad(sdf, [(pw, pw), (ph, ph), (pd, pd)],
                             mode='constant', constant_values=0)

        (w, h, d) = image.shape
        # if np.random.uniform() > 0.33:
        #     w1 = np.random.randint((w - self.output_size[0])//4, 3*(w - self.output_size[0])//4)
        #     h1 = np.random.randint((h - self.output_size[1])//4, 3*(h - self.output_size[1])//4)
        # else:
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        label = label[w1:w1 + self.output_size[0], h1:h1 +
                      self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 +
                      self.output_size[1], d1:d1 + self.output_size[2]]
        if self.with_sdf:
            sdf = sdf[w1:w1 + self.output_size[0], h1:h1 +
                      self.output_size[1], d1:d1 + self.output_size[2]]
            return {'image': image, 'label': label, 'sdf': sdf}
        else:
            return {'image': image, 'label': label}


class RandomRotFlip(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        k = np.random.randint(0, 4)
        image = np.rot90(image, k)
        label = np.rot90(label, k)
        axis = np.random.randint(0, 2)
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()

        return {'image': image, 'label': label}


class RandomNoise(object):
    def __init__(self, mu=0, sigma=0.1):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        noise = np.clip(self.sigma * np.random.randn(
            image.shape[0], image.shape[1], image.shape[2]), -2*self.sigma, 2*self.sigma)
        noise = noise + self.mu
        image = image + noise
        return {'image': image, 'label': label}


class CreateOnehotLabel(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        onehot_label = np.zeros(
            (self.num_classes, label.shape[0], label.shape[1], label.shape[2]), dtype=np.float32)
        for i in range(self.num_classes):
            onehot_label[i, :, :, :] = (label == i).astype(np.float32)
        return {'image': image, 'label': label, 'onehot_label': onehot_label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['image']
        image = image.reshape(
            1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
        if 'onehot_label' in sample:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long(),
                    'onehot_label': torch.from_numpy(sample['onehot_label']).long()}
        else:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long()}


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                   grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)


if __name__ == '__main__':
    data_dir = '../Data/BraTS19'
    list_dir = '../datalist/BraTS19_5'

    labset = BraTS2019(data_dir, list_dir,split='lab')
    unlabset = BraTS2019(data_dir,list_dir,split='unlab')
    trainset = BraTS2019(data_dir,list_dir,split='train')
    testset = BraTS2019(data_dir, list_dir,split='test')

    lab_sample = labset[0]
    unlab_sample = unlabset[0]
    train_sample = trainset[0] 
    test_sample = testset[0]

    print(torch.max(lab_sample['image']), torch.min(lab_sample['image']))

    print(len(labset), lab_sample['image'].shape, lab_sample['label'].shape)  # 12 torch.Size([1, 96, 96, 96]) torch.Size([96, 96, 96])
    print(len(unlabset), unlab_sample['image'].shape, unlab_sample['label'].shape) # 50 torch.Size([1, 96, 96, 96]) torch.Size([96, 96, 96])
    print(len(trainset), train_sample['image'].shape, train_sample['label'].shape) # 62 torch.Size([1, 96, 96, 96]) torch.Size([96, 96, 96])
    print(len(testset), test_sample['image'].shape, test_sample['label'].shape) # 18 torch.Size([1, 96, 96, 96]) torch.Size([96, 96, 96])


    labset = BraTS2019(data_dir, list_dir,split='lab', aug_times=5)
    unlabset = BraTS2019(data_dir,list_dir,split='unlab', aug_times=5)
    trainset = BraTS2019(data_dir,list_dir,split='train', aug_times=5)
    testset = BraTS2019(data_dir, list_dir,split='test', aug_times=5)

    lab_sample = labset[0]
    unlab_sample = unlabset[0]
    train_sample = trainset[0] 
    test_sample = testset[0]

    print(len(labset), lab_sample['image'].shape, lab_sample['label'].shape)  # 60 torch.Size([1, 96, 96, 96]) torch.Size([96, 96, 96])
    print(len(unlabset), unlab_sample['image'].shape, unlab_sample['label'].shape) # 250 torch.Size([1, 96, 96, 96]) torch.Size([96, 96, 96])
    print(len(trainset), train_sample['image'].shape, train_sample['label'].shape) # 310 torch.Size([1, 96, 96, 96]) torch.Size([96, 96, 96])
    print(len(testset), test_sample['image'].shape, test_sample['label'].shape) # 18 torch.Size([1, 96, 96, 96]) torch.Size([96, 96, 96])
