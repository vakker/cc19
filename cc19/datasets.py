import asyncio
import concurrent
import math
import os
from os import path as osp

import cv2
import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
from imblearn.over_sampling import RandomOverSampler
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class NPDataset(Dataset):
    def __init__(self, labels, data, transform=None, balance=True):
        self.labels = labels
        self.data = data
        self.data_idx = np.arange(len(labels))

        # self.print_stats()

        if balance:
            ros = RandomOverSampler(random_state=0)
            self.data_idx, self.labels = ros.fit_resample(self.data_idx.reshape(-1, 1),
                                                          self.labels)
            self.data_idx = self.data_idx.squeeze()
            # self.print_stats()

        self.transform = transform

        # import IPython; IPython.embed(colors="neutral")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.transform:
            sample = self.transform(resize(self.data[self.data_idx[idx]], 224))
        label = self.labels[idx]

        return sample, label

    def print_stats(self):
        unique, counts = np.unique(self.labels, return_counts=True)
        total = len(self.labels)
        print(total, unique, 100 * counts / total)


class CXRDataset(Dataset):
    def __init__(self, metadata, data_dir, train, transform=None, subset=None):
        md = pd.read_csv(metadata)
        md_filtered = md[md['view'].isin(['PA', 'AP', 'AP Supine', 'AP semi erect'])]

        md_filtered = md_filtered.sample(frac=1, random_state=0)
        if train:
            self.md_filtered = md_filtered.iloc[:int(0.8 * len(md_filtered))]
        else:
            self.md_filtered = md_filtered.iloc[int(0.8 * len(md_filtered)):]

        self.md_filtered = self.md_filtered.iloc[:30]
        self.labels = (self.md_filtered['finding'] == "COVID-19").values.astype(int)
        unique_elements, counts_elements = np.unique(self.labels, return_counts=True)
        print(train, unique_elements, 100 * counts_elements / len(self.labels))

        imgs = load_images(data_dir, self.md_filtered['filename'])
        self.imgs = resize(imgs, size=512)

        self.data_dir = data_dir
        self.transform = transform
        # import ipdb; ipdb.set_trace()

    def __len__(self):
        return len(self.md_filtered)

    def __getitem__(self, idx):
        if self.transform:
            sample = self.transform(self.imgs[idx])
        label = self.labels[idx]

        return sample, label


def setup_datasets_np(md_file, data_file, batch_size=128, use_cuda=False, subset=None):
    metadata = pd.read_csv(md_file)
    # metadata = metadata.sample(frac=1, random_state=0)
    labels = metadata['Target'].values.astype(int)
    data = np.load(data_file)
    if subset:
        subset_idx = int(subset * len(data))
        data = data[:subset_idx]
        labels = labels[:subset_idx]

    last_train_idx = int(0.8 * len(labels))

    trans = transforms.ToTensor()
    tng_ds = NPDataset(labels=labels[:last_train_idx], data=data[:last_train_idx], transform=trans)
    val_ds = NPDataset(labels=labels[last_train_idx:], data=data[last_train_idx:], transform=trans)

    tng_loader = torch.utils.data.DataLoader(dataset=tng_ds, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_ds, batch_size=batch_size, shuffle=False)
    return tng_loader, val_loader


def setup_data_loaders(metadata, data_dir, batch_size=128, use_cuda=False):
    trans = transforms.ToTensor()
    train_set = CXRDataset(metadata=metadata, data_dir=data_dir, train=True, transform=trans)
    test_set = CXRDataset(metadata=metadata, data_dir=data_dir, train=False, transform=trans)

    kwargs = {'num_workers': 0, 'pin_memory': use_cuda}
    train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               **kwargs)
    test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              **kwargs)
    return train_loader, test_loader


# async def load_image(img_path, event_loop, executor):
#     return await event_loop.run_in_executor(executor, cv2.imread, img_path,
#                                             cv2.IMREAD_GRAYSCALE)


def load_image(img_path, size=None):
    if img_path.endswith('.dcm'):
        img = pydicom.dcmread(img_path).pixel_array
    else:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # img = img / 255.0

    if size is None:
        return img
    return resize(img, size)


def load_images_(directory, img_names):
    loop = asyncio.get_event_loop()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=min(60, len(img_names)))
    imgs = asyncio.gather(
        *[load_image(osp.join(directory, im), loop, executor) for im in img_names])
    return loop.run_until_complete(imgs)


# async def load_images_(directory, img_names):
#     loop = asyncio.get_event_loop()
#     executor = concurrent.futures.ThreadPoolExecutor(
#         max_workers=min(60, len(img_names)))
#     coros = [
#         load_image(osp.join(directory, im), loop, executor) for im in img_names
#     ]
#     res = []
#     for first_completed in tqdm(asyncio.as_completed(coros), total=len(coros)):
#         res.append(await first_completed)
#     return res

# def load_images(directory, img_names):
#     return load_images_(directory, img_names)
# return asyncio.run(load_images_(directory, img_names))


def load_images(directory, img_names, size=None):
    return [load_image(osp.join(directory, i), size) for i in tqdm(img_names)]


def batch_resize(imgs, size=None):
    if size is None:
        size = int(np.average([i.shape for i in imgs]))

    imgs_resized = [resize(i, size) for i in imgs]
    return imgs_resized


def resize(img, size):
    return square(img, size)


def square(img, size):
    factor = max(size / img.shape[0], size / img.shape[1])
    img_resized = cv2.resize(img, None, fx=factor, fy=factor)
    assert size in img_resized.shape
    img_resized = center_crop(img_resized, [size, size])
    assert img_resized.shape[0] == size and img_resized.shape[1] == size
    return img_resized
    # return center_crop(img_resized, [size, size])


def center_crop(img, output_size):
    h, w = img.shape[:2]

    oh, ow = output_size

    if h < oh or w < ow:
        raise ValueError('Input image size is too small for desired output [%d %d] -> [%d %d]' %
                         (h, w, oh, ow))

    h_diff = h - oh
    w_diff = w - ow
    h_off = h_diff // 2
    w_off = w_diff // 2

    return img[h_off:h_off + oh, w_off:w_off + ow]
