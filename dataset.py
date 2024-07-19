import copy
import os
import collections
import numpy as np
import torch
import util
import random
import mlconfig
import pandas
from util import onehot, rand_bbox
from torch.utils.data.dataset import Dataset
from functools import partial
from PIL import Image, ImageFilter
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision
from fast_autoaugment.FastAutoAugment.archive import fa_reduced_cifar10
from fast_autoaugment.FastAutoAugment.augmentations import apply_augment
import random
import matplotlib.image as img
from typing import Any, Callable, Optional, Tuple
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
from glob import glob

# Datasets
transform_options = {
    "CIFAR10": {
        "train_transform": [transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor()],
        "test_transform": [transforms.ToTensor()]},
    "CIFAR100": {
         "train_transform": [transforms.RandomCrop(32, padding=4),
                             transforms.RandomHorizontalFlip(),
                             # transforms.RandomRotation(20),
                             transforms.ToTensor()],
         "test_transform": [transforms.ToTensor()]},
    "SVHN": {
        "train_transform": [transforms.ToTensor()],
        "test_transform": [transforms.ToTensor()]},
    "ImageNet": {
        "train_transform": [transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            # transforms.ColorJitter(brightness=0.4,
                            #                        contrast=0.4,
                            #                        saturation=0.4,
                            #                        hue=0.2),
                            transforms.ToTensor()],
        "test_transform": [transforms.Resize(256),
                           transforms.CenterCrop(224),
                           transforms.ToTensor()]},
    "TinyImageNet": {
        "train_transform": [transforms.CenterCrop(256),
                            transforms.Resize((32, 32)),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor()],
        "test_transform": [transforms.Resize((32, 32)),
                           transforms.ToTensor()]},
    'CatDog': {
        "train_transform": [transforms.Resize((32, 32)),
                            transforms.ToTensor()],
        "test_transform": [transforms.Resize((32, 32)),
                           transforms.ToTensor()]},
    'CelebA': {
        "train_transform": [transforms.CenterCrop((128, 128)),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor()],
        "test_transform": [transforms.CenterCrop((128, 128)),
                           transforms.ToTensor()]},
    'FaceScrub': {
        "train_transform": [transforms.RandomHorizontalFlip(),
                            transforms.ToTensor()],
        "test_transform": [transforms.Resize((128, 128)),
                           transforms.ToTensor()]},
    'WebFace': {
        "train_transform": [transforms.RandomHorizontalFlip(),
                            transforms.ToTensor()],
        "test_transform": [transforms.ToTensor()]},
    'FairFace': {
        "train_transform": [transforms.RandomHorizontalFlip()],
        "test_transform": []},
    'PoisonFairFace': {
        "train_transform": [transforms.RandomHorizontalFlip()],
        "test_transform": []},
    # 'FairFace': {
    #     "train_transform": [],
    #     "test_transform": []},
    # 'PoisonFairFace': {
    #     "train_transform": [],
    #     "test_transform": []},
    "CIFAR10C": {
        "train_transform": [transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip()],
        "test_transform": []},
}
transform_options['PoisonCIFAR10'] = transform_options['CIFAR10']
transform_options['PoisonCIFAR100'] = transform_options['CIFAR100']
transform_options['AdversarialPoisonCIFAR10'] = transform_options['CIFAR10']
transform_options['NTGACIFAR10'] = transform_options['CIFAR10']
transform_options['AdversarialPoisonCIFAR100'] = transform_options['CIFAR100']
transform_options['PoisonCIFAR101'] = transform_options['CIFAR100']
transform_options['PoisonSVHN'] = transform_options['SVHN']
transform_options['ImageNetMini'] = transform_options['ImageNet']
transform_options['PoisonImageNetMini'] = transform_options['ImageNet']
transform_options['ImageNetMini_for_detection'] = transform_options['ImageNet']
transform_options['ImageNetMini_partial'] = transform_options['ImageNet']
transform_options['CelebAMini'] = transform_options['CelebA']
transform_options['PoisonCIFAR10C'] = transform_options['CIFAR10C']
transform_options['PoisonCIFAR10_v2'] = transform_options['CIFAR10']
transform_options['PoisonCIFAR10_v3'] = transform_options['CIFAR10']
transform_options['PoisonCIFAR10_v4'] = transform_options['CIFAR10']
transform_options['PoisonCIFAR10_partial_defense'] = transform_options['CIFAR10']
transform_options['PoisonCIFAR100_partial_defense'] = transform_options['CIFAR100']
transform_options['CIFAR10_numpy'] = transform_options['CIFAR10']
transform_options['CIFAR100_numpy'] = transform_options['CIFAR100']
transform_options['CIFAR10_numpy_partial_diffpure'] = transform_options['CIFAR10']
transform_options['CIFAR100_numpy_partial_diffpure'] = transform_options['CIFAR100']
transform_options['CIFAR10_NTGA'] = transform_options['CIFAR10']

from io import BytesIO
def JPEGcompression(image):
    qf = 10
    outputIoStream = BytesIO()
    image.save(outputIoStream, "JPEG", quality=qf, optimice=True)
    outputIoStream.seek(0)
    return Image.open(outputIoStream)

def BDR(x):
    depth = 2
    scale = 2 ** depth
    x = ((scale * x).int() / scale).float()
    return x

@mlconfig.register
class DatasetGenerator():
    def __init__(self, train_batch_size=128, eval_batch_size=256, num_of_workers=4,
                 train_data_path='../datasets/', train_data_type='CIFAR10', seed=0,
                 test_data_path='../datasets/', test_data_type='CIFAR10', fa=False,
                 no_train_augments=False, poison_rate=1.0, perturb_type='classwise',
                 perturb_tensor_filepath=None, patch_location='center', img_denoise=False,
                 add_uniform_noise=False, poison_classwise=False, poison_classwise_idx=None,
                 use_cutout=None, use_cutmix=False, use_mixup=False, identity = 'race', load_to_memory = False, grayscale = False, jpeg = False, bdr = False, poison_samples_idx = None):

        np.random.seed(seed)
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_of_workers = num_of_workers
        self.seed = seed
        self.train_data_type = train_data_type
        self.test_data_type = test_data_type
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path

        train_transform = transform_options[train_data_type]['train_transform']
        test_transform = transform_options[test_data_type]['test_transform']

        if jpeg:
            train_transform.insert(0, transforms.Lambda(JPEGcompression))
            # test_transform.insert(0, transforms.Lambda(JPEGcompression))
        if grayscale:
            train_transform.append(transforms.Grayscale(num_output_channels=3))
            # test_transform.append(transforms.Grayscale(num_output_channels=3))
        if bdr:
            train_transform.append(transforms.Lambda(BDR))
            # test_transform.append(transforms.Lambda(BDR))

        train_transform = transforms.Compose(train_transform)
        test_transform = transforms.Compose(test_transform)
        if no_train_augments:
            train_transform = test_transform

        if fa:
            # FastAutoAugment
            train_transform.transforms.insert(0, Augmentation(fa_reduced_cifar10()))
        elif use_cutout is not None:
            print('Using Cutout')
            train_transform.transforms.append(Cutout(16))

        # Training Datasets
        if train_data_type == 'CIFAR10':
            num_of_classes = 10
            train_dataset = datasets.CIFAR10(root=train_data_path, train=True,
                                             download=True, transform=train_transform)
        elif train_data_type == 'PoisonCIFAR10':
            num_of_classes = 10
            train_dataset = PoisonCIFAR10(root=train_data_path, transform=train_transform,
                                          poison_rate=poison_rate, perturb_type=perturb_type,
                                          patch_location=patch_location, seed=seed, img_denoise=img_denoise,
                                          perturb_tensor_filepath=perturb_tensor_filepath,
                                          add_uniform_noise=add_uniform_noise,
                                          poison_classwise=poison_classwise,
                                          poison_classwise_idx=poison_classwise_idx, poison_samples_idx = poison_samples_idx)
        elif train_data_type == 'PoisonCIFAR10_partial_defense':
            num_of_classes = 10
            train_dataset = PoisonCIFAR10_partial_defense(root=train_data_path, transform=train_transform,
                                          poison_rate=poison_rate, perturb_type=perturb_type,
                                          patch_location=patch_location, seed=seed, img_denoise=img_denoise,
                                          perturb_tensor_filepath=perturb_tensor_filepath,
                                          add_uniform_noise=add_uniform_noise,
                                          poison_classwise=poison_classwise,
                                          poison_classwise_idx=poison_classwise_idx, poison_samples_idx = poison_samples_idx)
        elif train_data_type == 'AdversarialPoisonCIFAR10':
            base_train_dataset = datasets.CIFAR10(root='../datasets', train=True,
                                             download=True, transform=transforms.Compose(transform_options['CIFAR10']["train_transform"]))
            num_of_classes = 10
            train_dataset = AdversarialPoison(root=train_data_path, baseset = base_train_dataset, poison_rate=poison_rate, poison_classwise=poison_classwise, poison_classwise_idx=poison_classwise_idx)
        elif train_data_type == 'NTGACIFAR10':
            base_train_dataset = datasets.CIFAR10(root='../datasets', train=True,
                                             download=True, transform=transforms.Compose(transform_options['CIFAR10']["train_transform"]))
            num_of_classes = 10
            train_dataset = NTGACIFAR10(root=train_data_path, baseset = base_train_dataset)
        elif train_data_type == 'CIFAR10_NTGA':
            base_train_dataset = datasets.CIFAR10(root='../datasets', train=True,
                                             download=True, transform=transforms.Compose(transform_options['CIFAR10']["train_transform"]))
            num_of_classes = 10
            train_dataset = CIFAR10_NTGA(root=train_data_path, baseset = base_train_dataset, perturb_tensor_filepath=perturb_tensor_filepath, poison_samples_idx = poison_samples_idx)
        elif train_data_type == 'CIFAR10_numpy':
            base_train_dataset = datasets.CIFAR10(root='../datasets', train=True,
                                             download=True, transform=transforms.Compose(transform_options['CIFAR10']["train_transform"]))
            num_of_classes = 10
            train_dataset = CIFAR10_numpy(root=train_data_path, baseset = base_train_dataset, perturb_tensor_filepath=perturb_tensor_filepath, poison_samples_idx = poison_samples_idx)
        elif train_data_type == 'CIFAR100_numpy':
            base_train_dataset = datasets.CIFAR100(root='../datasets', train=True,
                                             download=True, transform=transforms.Compose(transform_options['CIFAR100']["train_transform"]))
            num_of_classes = 10
            train_dataset = CIFAR100_numpy(root=train_data_path, baseset = base_train_dataset, perturb_tensor_filepath=perturb_tensor_filepath, poison_samples_idx = poison_samples_idx)
        elif train_data_type == 'CIFAR10_numpy_partial_diffpure':
            base_train_dataset = datasets.CIFAR10(root='../datasets', train=True,
                                             download=True, transform=transforms.Compose(transform_options['CIFAR10']["train_transform"]))
            num_of_classes = 10
            train_dataset = CIFAR10_numpy_partial_diffpure(root=train_data_path, baseset = base_train_dataset, perturb_tensor_filepath=perturb_tensor_filepath, poison_samples_idx = poison_samples_idx)
        elif train_data_type == 'CIFAR100_numpy_partial_diffpure':
            base_train_dataset = datasets.CIFAR100(root='../datasets', train=True,
                                             download=True, transform=transforms.Compose(transform_options['CIFAR100']["train_transform"]))
            num_of_classes = 10
            train_dataset = CIFAR100_numpy_partial_diffpure(root=train_data_path, baseset = base_train_dataset, perturb_tensor_filepath=perturb_tensor_filepath, poison_samples_idx = poison_samples_idx)
        elif train_data_type == 'PoisonCIFAR10_v2':
            num_of_classes = 11
            train_dataset = PoisonCIFAR10_v2(root=train_data_path, transform=train_transform,
                                          poison_rate=poison_rate, perturb_type=perturb_type,
                                          patch_location=patch_location, seed=seed, img_denoise=img_denoise,
                                          perturb_tensor_filepath=perturb_tensor_filepath,
                                          add_uniform_noise=add_uniform_noise,
                                          poison_classwise=poison_classwise,
                                          poison_classwise_idx=poison_classwise_idx)
        elif train_data_type == 'PoisonCIFAR10_v3':
            num_of_classes = 20
            train_dataset = PoisonCIFAR10_v3(root=train_data_path, transform=train_transform,
                                          poison_rate=poison_rate, perturb_type=perturb_type,
                                          patch_location=patch_location, seed=seed, img_denoise=img_denoise,
                                          perturb_tensor_filepath=perturb_tensor_filepath,
                                          add_uniform_noise=add_uniform_noise,
                                          poison_classwise=poison_classwise,
                                          poison_classwise_idx=poison_classwise_idx)
        elif train_data_type == 'PoisonCIFAR10_v4':
            num_of_classes = 10
            train_dataset = PoisonCIFAR10_v4(root=train_data_path, transform=train_transform,
                                          poison_rate=poison_rate, perturb_type=perturb_type,
                                          patch_location=patch_location, seed=seed, img_denoise=img_denoise,
                                          perturb_tensor_filepath=perturb_tensor_filepath,
                                          add_uniform_noise=add_uniform_noise,
                                          poison_classwise=poison_classwise,
                                          poison_classwise_idx=poison_classwise_idx)
        elif train_data_type == 'CIFAR100':
            num_of_classes = 100
            train_dataset = datasets.CIFAR100(root=train_data_path, train=True,
                                              download=True, transform=train_transform)
        elif train_data_type == 'PoisonCIFAR100':
            num_of_classes = 100
            train_dataset = PoisonCIFAR100(root=train_data_path, transform=train_transform,
                                           poison_rate=poison_rate, perturb_type=perturb_type,
                                           patch_location=patch_location, seed=seed, img_denoise=img_denoise,
                                           perturb_tensor_filepath=perturb_tensor_filepath,
                                           add_uniform_noise=add_uniform_noise,
                                           poison_classwise=poison_classwise, poison_samples_idx = poison_samples_idx)
        elif train_data_type == 'PoisonCIFAR100_partial_defense':
            num_of_classes = 100
            train_dataset = PoisonCIFAR100_partial_defense(root=train_data_path, transform=train_transform,
                                          poison_rate=poison_rate, perturb_type=perturb_type,
                                          patch_location=patch_location, seed=seed, img_denoise=img_denoise,
                                          perturb_tensor_filepath=perturb_tensor_filepath,
                                          add_uniform_noise=add_uniform_noise,
                                          poison_classwise=poison_classwise,
                                          poison_classwise_idx=poison_classwise_idx, poison_samples_idx = poison_samples_idx)
        elif train_data_type == 'AdversarialPoisonCIFAR100':
            base_train_dataset = datasets.CIFAR100(root='../datasets', train=True,
                                             download=True, transform=transforms.Compose(transform_options['CIFAR100']["train_transform"]))
            num_of_classes = 100
            train_dataset = AdversarialPoison(root=train_data_path, baseset = base_train_dataset, poison_rate=poison_rate, poison_classwise=poison_classwise, poison_classwise_idx=poison_classwise_idx)
        elif train_data_type == 'PoisonCIFAR101':
            num_of_classes = 101
            poison_cifar10 = PoisonCIFAR10(root=train_data_path, transform=train_transform,
                                           poison_rate=poison_rate, perturb_type=perturb_type,
                                           patch_location=patch_location, seed=seed, img_denoise=img_denoise,
                                           perturb_tensor_filepath=perturb_tensor_filepath,
                                           add_uniform_noise=add_uniform_noise,
                                           poison_classwise=poison_classwise,
                                           poison_classwise_idx=poison_classwise_idx, poison_samples_idx = poison_samples_idx)
            train_dataset = PoisonCIFAR101(train_data_path, split='poison_train',
                                           transform=train_transform, seed=0,
                                           poisn_cifar10_data=poison_cifar10)
        elif train_data_type == 'SVHN':
            num_of_classes = 10
            train_dataset = datasets.SVHN(root=train_data_path, split='train',
                                          download=True, transform=train_transform)
        elif train_data_type == 'PoisonSVHN':
            num_of_classes = 10
            train_dataset = PoisonSVHN(root=train_data_path, split='train', transform=train_transform,
                                       poison_rate=poison_rate, perturb_type=perturb_type,
                                       patch_location=patch_location, seed=seed, img_denoise=img_denoise,
                                       perturb_tensor_filepath=perturb_tensor_filepath,
                                       add_uniform_noise=add_uniform_noise,
                                       poison_classwise=poison_classwise)
        elif train_data_type == 'TinyImageNet':
            num_of_classes = 1000
            train_dataset = datasets.ImageNet(root=train_data_path, split='train',
                                              transform=train_transform)
        elif train_data_type == 'ImageNetMini':
            num_of_classes = 100
            train_dataset = ImageNetMini(root=train_data_path, split='train',
                                         transform=train_transform, load_to_memory = load_to_memory)
        elif train_data_type == 'ImageNetMini_partial':
            num_of_classes = 100
            train_dataset = ImageNetMini_partial(root=train_data_path, split='train',
                                         transform=train_transform, load_to_memory = load_to_memory)
        elif train_data_type == 'PoisonImageNetMini':
            num_of_classes = 100
            train_dataset = PoisonImageNetMini(root=train_data_path, split='train', seed=seed,
                                               transform=train_transform, poison_rate=poison_rate,
                                               perturb_tensor_filepath=perturb_tensor_filepath, load_to_memory = load_to_memory)
        elif train_data_type == 'ImageNetMini_for_detection':
            num_of_classes = 100
            train_dataset = ImageNetMini_for_detection(root=train_data_path, split='train',
                                         transform=train_transform, load_to_memory = load_to_memory)
        elif train_data_type == 'CatDog':
            train_dataset = CatDogDataset(root=train_data_path, split='train',
                                          transform=train_transform)
        elif train_data_type == 'CelebAMini':
            train_dataset = CelebAMini(root=train_data_path, split="all",
                                       target_type="identity", transform=train_transform)
            test_dataset = CelebAMini(root=train_data_path, split="all",
                                      target_type="identity", transform=test_transform)
        elif train_data_type == 'WebFace':
            train_dataset = datasets.ImageFolder(root=train_data_path, transform=train_transform)
            test_dataset = datasets.ImageFolder(root=test_data_path, transform=test_transform)
        elif train_data_type == 'CelebA':
            train_dataset = datasets.CelebA(root=train_data_path, split="all",
                                            target_type="identity", transform=train_transform)
            test_dataset = datasets.CelebA(root=train_data_path, split="all",
                                           target_type="identity", transform=test_transform)
        elif train_data_type == 'FairFace':
            train_dataset = FairFaceDataset(root=train_data_path, split = 'train', identity = identity, transform=train_transform)
        elif train_data_type == 'PoisonFairFace':
            train_dataset = PoisonFairFaceDataset(root=train_data_path, split = 'train', identity = identity, transform=train_transform,
                                           poison_rate=poison_rate, perturb_type=perturb_type,
                                           patch_location=patch_location, seed=seed, img_denoise=img_denoise,
                                           perturb_tensor_filepath=perturb_tensor_filepath,
                                           add_uniform_noise=add_uniform_noise,
                                           poison_classwise=poison_classwise,
                                           poison_classwise_idx=poison_classwise_idx)
        elif train_data_type == 'CIFAR10C':
            train_dataset = CIFAR10CDataset(root=train_data_path, split = 'train', identity = identity, transform=train_transform)
        elif train_data_type == 'PoisonCIFAR10C':
            train_dataset = PoisonCIFAR10CDataset(root=train_data_path, split = 'train', identity = identity, transform=train_transform,
                                           poison_rate=poison_rate, perturb_type=perturb_type,
                                           patch_location=patch_location, seed=seed, img_denoise=img_denoise,
                                           perturb_tensor_filepath=perturb_tensor_filepath,
                                           add_uniform_noise=add_uniform_noise,
                                           poison_classwise=poison_classwise,
                                           poison_classwise_idx=poison_classwise_idx)
        else:
            raise('Training Dataset type %s not implemented' % train_data_type)

        # Test Datset
        if test_data_type == 'CIFAR10':
            test_dataset = datasets.CIFAR10(root=test_data_path, train=False,
                                            download=True, transform=test_transform)
        elif test_data_type == 'PoisonCIFAR10':
            test_dataset = PoisonCIFAR10(root=test_data_path, train=False, transform=test_transform,
                                         poison_rate=poison_rate, perturb_type=perturb_type,
                                         patch_location=patch_location, seed=seed, img_denoise=img_denoise,
                                         perturb_tensor_filepath=perturb_tensor_filepath,
                                         add_uniform_noise=add_uniform_noise,
                                         poison_classwise=poison_classwise,
                                         poison_classwise_idx=poison_classwise_idx, poison_samples_idx = poison_samples_idx)

        elif test_data_type == 'CIFAR100':
            test_dataset = datasets.CIFAR100(root=test_data_path, train=False,
                                             download=True, transform=test_transform)
        elif test_data_type == 'PoisonCIFAR100':
            test_dataset = PoisonCIFAR100(root=test_data_path, train=False, transform=test_transform,
                                          poison_rate=poison_rate, perturb_type=perturb_type,
                                          patch_location=patch_location, seed=seed, img_denoise=img_denoise,
                                          perturb_tensor_filepath=perturb_tensor_filepath,
                                          add_uniform_noise=add_uniform_noise,
                                          poison_classwise=poison_classwise, poison_samples_idx = poison_samples_idx)
        elif test_data_type == 'PoisonCIFAR101':
            test_dataset = PoisonCIFAR101(test_data_path, split='test',
                                          transform=test_transform, seed=0,
                                          poisn_cifar10_data=poison_cifar10)
        elif test_data_type == 'SVHN':
            test_dataset = datasets.SVHN(root=test_data_path, split='test',
                                         download=True, transform=test_transform)
        elif test_data_type == 'PoisonSVHN':
            test_dataset = PoisonSVHN(root=test_data_path, split='test', transform=test_transform,
                                       poison_rate=poison_rate, perturb_type=perturb_type,
                                       patch_location=patch_location, seed=seed, img_denoise=img_denoise,
                                       perturb_tensor_filepath=perturb_tensor_filepath,
                                       add_uniform_noise=add_uniform_noise,
                                       poison_classwise=poison_classwise)
        elif test_data_type == 'ImageNetMini':
            test_dataset = ImageNetMini(root=test_data_path, split='val',
                                        transform=test_transform, load_to_memory = load_to_memory)
        elif test_data_type == 'TinyImageNet':
            test_dataset = datasets.ImageNet(root=test_data_path, split='val',
                                             transform=test_transform)
        elif test_data_type == 'PoisonImageNetMini':
            test_dataset = PoisonImageNetMini(root=test_data_path, split='val', seed=0,
                                              transform=test_transform, poison_rate=poison_rate,
                                              perturb_tensor_filepath=perturb_tensor_filepath, load_to_memory = load_to_memory)
        elif test_data_type == 'CatDog':
            # Cat Dog only used for transfer exp, no test dataset
            test_dataset = CatDogDataset(root=train_data_path, split='train',
                                         transform=train_transform)
        elif test_data_type == 'FairFace':
            test_dataset = FairFaceDataset(root=test_data_path, split = 'test', identity = identity, transform=test_transform)
        elif test_data_type == 'CIFAR10C':
            test_dataset = CIFAR10CDataset(root=test_data_path, split = 'test', identity = identity, transform=test_transform)
        elif test_data_type == 'CelebAMini' or 'CelebA':
            pass
        elif test_data_type == 'FaceScrub' or test_data_type == 'WebFace':
            pass
        else:
            raise('Test Dataset type %s not implemented' % test_data_type)

        if use_cutmix:
            train_dataset = CutMix(dataset=train_dataset, num_class=num_of_classes)
        elif use_mixup:
            train_dataset = MixUp(dataset=train_dataset, num_class=num_of_classes)

        self.datasets = {
            'train_dataset': train_dataset,
            'test_dataset': test_dataset,
        }
        return

    def getDataLoader(self, train_shuffle=True, train_drop_last=True):
        data_loaders = {}

        data_loaders['train_dataset'] = DataLoader(dataset=self.datasets['train_dataset'],
                                                   batch_size=self.train_batch_size,
                                                   shuffle=train_shuffle, pin_memory=True,
                                                   drop_last=train_drop_last, num_workers=self.num_of_workers)

        data_loaders['test_dataset'] = DataLoader(dataset=self.datasets['test_dataset'],
                                                  batch_size=self.eval_batch_size,
                                                  shuffle=False, pin_memory=True,
                                                  drop_last=False, num_workers=self.num_of_workers)

        return data_loaders

    def _split_validation_set(self, train_portion, train_shuffle=True, train_drop_last=True):
        np.random.seed(self.seed)
        train_subset = copy.deepcopy(self.datasets['train_dataset'])
        valid_subset = copy.deepcopy(self.datasets['train_dataset'])

        if self.train_data_type == 'ImageNet' or self.train_data_type == 'ImageNetMini' or self.train_data_type == 'TinyImageNet' or self.train_data_type == 'PoisonImageNetMini':
            data, targets = list(zip(*self.datasets['train_dataset'].samples))
            datasplit = train_test_split(data, targets, test_size=1-train_portion,
                                         train_size=train_portion, shuffle=True, stratify=targets)
            train_D, valid_D, train_L, valid_L = datasplit
            print('Train Labels: ', np.array(train_L))
            print('Valid Labels: ', np.array(valid_L))
            train_subset.samples = list(zip(train_D, train_L))
            valid_subset.samples = list(zip(valid_D, valid_L))
        elif self.train_data_type == 'SVHN':
            data, targets = self.datasets['train_dataset'].data, self.datasets['train_dataset'].labels
            datasplit = train_test_split(data, targets, test_size=1-train_portion,
                                         train_size=train_portion, shuffle=True, stratify=targets)
            train_D, valid_D, train_L, valid_L = datasplit
            print('Train Labels: ', np.array(train_L))
            print('Valid Labels: ', np.array(valid_L))
            train_subset.data = np.array(train_D)
            valid_subset.data = np.array(valid_D)
            train_subset.labels = train_L
            valid_subset.labels = valid_L
        else:
            datasplit = train_test_split(self.datasets['train_dataset'].data,
                                         self.datasets['train_dataset'].targets,
                                         test_size=1-train_portion, train_size=train_portion,
                                         shuffle=True, stratify=self.datasets['train_dataset'].targets)
            train_D, valid_D, train_L, valid_L = datasplit
            print('Train Labels: ', np.array(train_L))
            print('Valid Labels: ', np.array(valid_L))
            train_subset.data = np.array(train_D)
            valid_subset.data = np.array(valid_D)
            train_subset.targets = train_L
            valid_subset.targets = valid_L

        self.datasets['train_subset'] = train_subset
        self.datasets['valid_subset'] = valid_subset
        print(self.datasets)

        data_loaders = {}

        data_loaders['train_dataset'] = DataLoader(dataset=self.datasets['train_dataset'],
                                                   batch_size=self.train_batch_size,
                                                   shuffle=train_shuffle, pin_memory=True,
                                                   drop_last=train_drop_last, num_workers=self.num_of_workers)

        data_loaders['test_dataset'] = DataLoader(dataset=self.datasets['test_dataset'],
                                                  batch_size=self.eval_batch_size,
                                                  shuffle=False, pin_memory=True,
                                                  drop_last=False, num_workers=self.num_of_workers)

        data_loaders['train_subset'] = DataLoader(dataset=self.datasets['train_subset'],
                                                  batch_size=self.train_batch_size,
                                                  shuffle=train_shuffle, pin_memory=True,
                                                  drop_last=train_drop_last, num_workers=self.num_of_workers)

        data_loaders['valid_subset'] = DataLoader(dataset=self.datasets['valid_subset'],
                                                  batch_size=self.eval_batch_size,
                                                  shuffle=False, pin_memory=True,
                                                  drop_last=False, num_workers=self.num_of_workers)
        return data_loaders


def patch_noise_extend_to_img(noise, image_size=[32, 32, 3], patch_location='center'):
    h, w, c = image_size[0], image_size[1], image_size[2]
    mask = np.zeros((h, w, c), np.float32)
    x_len, y_len = noise.shape[0], noise.shape[1]

    if patch_location == 'center' or (h == w == x_len == y_len):
        x = h // 2
        y = w // 2
    elif patch_location == 'random':
        x = np.random.randint(x_len // 2, w - x_len // 2)
        y = np.random.randint(y_len // 2, h - y_len // 2)
    else:
        raise('Invalid patch location')

    x1 = np.clip(x - x_len // 2, 0, h)
    x2 = np.clip(x + x_len // 2, 0, h)
    y1 = np.clip(y - y_len // 2, 0, w)
    y2 = np.clip(y + y_len // 2, 0, w)
    mask[x1: x2, y1: y2, :] = noise
    return mask


class CIFAR10CDataset(Dataset):
    def __init__(self, root, split, identity = 'class', transform=None, seed = 0):
        super(CIFAR10CDataset, self).__init__()
        self.transform = transform
        self.root = root
        self.identity = identity
        self.seed = seed
        random.seed(seed)
        torch.manual_seed(seed)
        print (random.random())
        print (torch.randn(3,2))

        if split=='train':
            self.align = glob(os.path.join(root, 'align',"*","*"))
            self.conflict = glob(os.path.join(root, 'conflict',"*","*"))
            self.data = self.align + self.conflict

        elif split=='valid':
            self.data = glob(os.path.join(root,split,"*", "*"))

        elif split=='test':
            self.data = glob(os.path.join(root, '../test',"*","*"))

        print (len(self.data))
        self.image = {}
        self.label = {}
        for index in range(len(self.data)):
            attr = [int(self.data[index].split('_')[-2]), int(self.data[index].split('_')[-1].split('.')[0])]
            self.label[index] = torch.tensor(attr[0]) if self.identity == 'class' else torch.tensor(attr[1])
            self.image[index] = Image.open(self.data[index]).convert('RGB')

            if self.transform is not None:
                self.image[index] = self.transform(self.image[index])


    def __len__(self):
        return len(self.data)

    def __getitem__(self,index):
        image = self.image[index]
        label = self.label[index]
        return transforms.ToTensor()(image), label


class PoisonCIFAR10CDataset(Dataset):
    def __init__(self, root, split, identity = 'class', transform=None,
                 poison_rate=1.0, perturb_tensor_filepath=None,
                 seed=0, perturb_type='classwise', patch_location='center', img_denoise=False,
                 add_uniform_noise=False, poison_classwise=False, poison_classwise_idx=None):
        super(PoisonCIFAR10CDataset, self).__init__()
        self.transform = transform
        self.root = root
        self.identity = identity
        self.seed = seed
        random.seed(seed)
        torch.manual_seed(seed)
        print (random.random())
        print (torch.randn(3,2))

        if split=='train':
            self.align = glob(os.path.join(root, 'align',"*","*"))
            self.conflict = glob(os.path.join(root, 'conflict',"*","*"))
            self.data = self.align + self.conflict

        elif split=='valid':
            self.data = glob(os.path.join(root,split,"*", "*"))

        elif split=='test':
            self.data = glob(os.path.join(root, '../test',"*","*"))

        print (len(self.data))
        self.image = {}
        self.label = {}
        for index in range(len(self.data)):
            attr = [int(self.data[index].split('_')[-2]), int(self.data[index].split('_')[-1].split('.')[0])]
            self.label[index] = torch.tensor(attr[0]) if self.identity == 'class' else torch.tensor(attr[1])
            self.image[index] = Image.open(self.data[index]).convert('RGB')

            if self.transform is not None:
                self.image[index] = self.transform(self.image[index])


        self.perturb_tensor = torch.load(perturb_tensor_filepath, map_location=torch.device('cpu'))
        print(self.perturb_tensor)
        if len(self.perturb_tensor.shape) == 4:
            self.perturb_tensor = self.perturb_tensor.mul(255).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu').numpy()
        else:
            self.perturb_tensor = self.perturb_tensor.mul(255).clamp_(0, 255).permute(0, 1, 3, 4, 2).to('cpu').numpy()
        self.patch_location = patch_location
        self.img_denoise = img_denoise
        # Check Shape
        target_dim = self.perturb_tensor.shape[0] if len(self.perturb_tensor.shape) == 4 else self.perturb_tensor.shape[1]
        if perturb_type == 'samplewise' and target_dim != len(self):
            raise('Poison Perturb Tensor size not match for samplewise')
        elif perturb_type == 'classwise' and target_dim != 10:
            raise('Poison Perturb Tensor size not match for classwise')

        # Random Select Poison Targets
        self.poison_samples = collections.defaultdict(lambda: False)
        self.poison_class = []
        if poison_classwise:
            targets = list(range(0, len(self.map2label[identity])))
            if poison_classwise_idx is None:
                self.poison_class = sorted(np.random.choice(targets, int(len(targets) * poison_rate), replace=False).tolist())
            else:
                self.poison_class = poison_classwise_idx
            self.poison_samples_idx = []
            for i, label in enumerate(self.targets):
                if label in self.poison_class:
                    self.poison_samples_idx.append(i)
        else:
            targets = list(range(0, len(self)))
            self.poison_samples_idx = sorted(np.random.choice(targets, int(len(targets) * poison_rate), replace=False).tolist())

        for idx in self.poison_samples_idx:
            self.poison_samples[idx] = True
            if len(self.perturb_tensor.shape) == 5:
                perturb_id = random.choice(range(self.perturb_tensor.shape[0]))
                perturb_tensor = self.perturb_tensor[perturb_id]
            else:
                perturb_tensor = self.perturb_tensor
            if perturb_type == 'samplewise':
                # Sample Wise poison
                noise = perturb_tensor[idx]
                noise = patch_noise_extend_to_img(noise, [32, 32, 3], patch_location=self.patch_location)
            elif perturb_type == 'classwise':
                # Class Wise Poison
                noise = perturb_tensor[self.targets[idx]]
                noise = patch_noise_extend_to_img(noise, [32, 32, 3], patch_location=self.patch_location)
            if add_uniform_noise:
                noise += np.random.uniform(0, 8, (32, 32, 3))

            self.image[idx] = np.array(self.image[idx]).astype(np.float32)
            self.image[idx] = self.image[idx] + noise
            self.image[idx] = np.clip(self.image[idx], a_min=0, a_max=255)
            self.image[idx] = self.image[idx].astype(np.uint8)
            self.image[idx] = Image.fromarray(self.image[idx]).convert('RGB')

        print('add_uniform_noise: ', add_uniform_noise)
        print(self.perturb_tensor.shape)
        print('Poison samples: %d/%d' % (len(self.poison_samples), len(self)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self,index):
        image = self.image[index]
        label = self.label[index]
        if self.transform is not None:
            image = self.transform(image)
        return transforms.ToTensor()(image), label


class FairFaceDataset(Dataset):
    def __init__(self, root, split = 'train', identity = 'race', transform = None, seed = 0):
        super().__init__()
        print('Identity: ', identity)
        train_labels = pandas.read_csv(os.path.join(root, 'fairface_label_train.csv'))
        test_labels = pandas.read_csv(os.path.join(root, 'fairface_label_val.csv'))
        # train_labels = pandas.read_csv(r'./fairface-img-margin025-trainval/fairface_label_train.csv')
        # test_labels = pandas.read_csv(r'./fairface-img-margin025-trainval/fairface_label_val.csv')
        # data_path = r'./fairface-img-margin025-trainval/'
        if split == 'train':
            self.data = train_labels.values
        else:
            self.data = test_labels.values
        self.path = root
        self.transform = transform
        self.identity = identity
        self.seed = seed
        random.seed(seed)
        torch.manual_seed(seed)
        print (random.random())
        print (torch.randn(3,2))

        # dict to map given label to a number
        self.map2label = {}
        self.map2label['race'] = {'White' : 0, 
                               'Black': 1, 
                               'Latino_Hispanic': 2, 
                               'East Asian' : 3, 
                               'Southeast Asian' : 4, 
                               'Indian' : 5, 
                               'Middle Eastern' : 6}
        self.map2label['age'] = {'0-2' : 0,
                       '3-9' : 1, 
                       '10-19': 2, 
                       '20-29': 3, 
                       '30-39' : 4, 
                       '40-49' : 5, 
                       '50-59' : 6, 
                       '60-69' : 7,
                       'more than 70':8}
        self.map2label['gender'] = {'Male' : 0, 
                               'Female': 1}
        self.map2index = {'race':3, 'age': 1, 'gender': 2}

        self.image = {}
        self.label = {}
        for index in range(len(self.data)):
            img_name = self.data[index][0]
            label = self.map2label[self.identity][self.data[index][self.map2index[self.identity]]]   # index 3 for race, need as tensor -> convert to number from str first
            label = torch.tensor(label)
            img_path = os.path.join(self.path, img_name)
            # print (img_path)
            image = transforms.Resize((128, 128))(Image.open(img_path))
            if self.transform is not None:
                image = self.transform(image)
            self.image[index] = image
            self.label[index] = label
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,index):
        image = self.image[index]
        label = self.label[index]
        return transforms.ToTensor()(image), label


class PoisonFairFaceDataset(Dataset):
    def __init__(self, root, split = 'train', identity = 'race', transform=None,
                 poison_rate=1.0, perturb_tensor_filepath=None,
                 seed=0, perturb_type='classwise', patch_location='center', img_denoise=False,
                 add_uniform_noise=False, poison_classwise=False, poison_classwise_idx=None):
        super().__init__()
        print('Identity: ', identity)
        train_labels = pandas.read_csv(os.path.join(root, 'fairface_label_train.csv'))
        test_labels = pandas.read_csv(os.path.join(root, 'fairface_label_val.csv'))
        # train_labels = pandas.read_csv(r'./fairface-img-margin025-trainval/fairface_label_train.csv')
        # test_labels = pandas.read_csv(r'./fairface-img-margin025-trainval/fairface_label_val.csv')
        # data_path = r'./fairface-img-margin025-trainval/'
        if split == 'train':
            self.data = train_labels.values
        else:
            self.data = test_labels.values
        self.path = root
        self.transform = transform
        self.identity = identity
        self.seed = seed
        random.seed(seed)
        torch.manual_seed(seed)
        print (random.random())
        print (torch.randn(3,2))

        # dict to map given label to a number
        self.map2label = {}
        self.map2label['race'] = {'White' : 0, 
                               'Black': 1, 
                               'Latino_Hispanic': 2, 
                               'East Asian' : 3, 
                               'Southeast Asian' : 4, 
                               'Indian' : 5, 
                               'Middle Eastern' : 6}
        self.map2label['age'] = {'0-2' : 0,
                       '3-9' : 1, 
                       '10-19': 2, 
                       '20-29': 3, 
                       '30-39' : 4, 
                       '40-49' : 5, 
                       '50-59' : 6, 
                       '60-69' : 7,
                       'more than 70':8}
        self.map2label['gender'] = {'Male' : 0, 
                               'Female': 1}
        self.map2index = {'race':3, 'age': 1, 'gender': 2}

        self.image = {}
        self.label = {}
        for index in range(len(self.data)):
            img_name = self.data[index][0]
            label = self.map2label[self.identity][self.data[index][self.map2index[self.identity]]]   # index 3 for race, need as tensor -> convert to number from str first
            label = torch.tensor(label)
            img_path = os.path.join(self.path, img_name)
            image = transforms.Resize((128, 128))(Image.open(img_path))
            if self.transform is not None:
                image = self.transform(image)
            self.image[index] = transforms.Resize((128, 128))(Image.open(img_path))
            self.label[index] = label

        self.perturb_tensor = torch.load(perturb_tensor_filepath, map_location=torch.device('cpu'))
        print(self.perturb_tensor)
        if len(self.perturb_tensor.shape) == 4:
            self.perturb_tensor = self.perturb_tensor.mul(255).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu').numpy()
        else:
            self.perturb_tensor = self.perturb_tensor.mul(255).clamp_(0, 255).permute(0, 1, 3, 4, 2).to('cpu').numpy()
        self.patch_location = patch_location
        self.img_denoise = img_denoise
        # Check Shape
        target_dim = self.perturb_tensor.shape[0] if len(self.perturb_tensor.shape) == 4 else self.perturb_tensor.shape[1]
        if perturb_type == 'samplewise' and target_dim != len(self):
            raise('Poison Perturb Tensor size not match for samplewise')
        elif perturb_type == 'classwise' and target_dim != len(self.map2label[identity]):
            raise('Poison Perturb Tensor size not match for classwise')

        # Random Select Poison Targets
        self.poison_samples = collections.defaultdict(lambda: False)
        self.poison_class = []
        if poison_classwise:
            targets = list(range(0, len(self.map2label[identity])))
            if poison_classwise_idx is None:
                self.poison_class = sorted(np.random.choice(targets, int(len(targets) * poison_rate), replace=False).tolist())
            else:
                self.poison_class = poison_classwise_idx
            self.poison_samples_idx = []
            for i, label in enumerate(self.targets):
                if label in self.poison_class:
                    self.poison_samples_idx.append(i)
        else:
            targets = list(range(0, len(self)))
            self.poison_samples_idx = sorted(np.random.choice(targets, int(len(targets) * poison_rate), replace=False).tolist())

        for idx in self.poison_samples_idx:
            self.poison_samples[idx] = True
            if len(self.perturb_tensor.shape) == 5:
                perturb_id = random.choice(range(self.perturb_tensor.shape[0]))
                perturb_tensor = self.perturb_tensor[perturb_id]
            else:
                perturb_tensor = self.perturb_tensor
            if perturb_type == 'samplewise':
                # Sample Wise poison
                noise = perturb_tensor[idx]
                noise = patch_noise_extend_to_img(noise, [128, 128, 3], patch_location=self.patch_location)
            elif perturb_type == 'classwise':
                # Class Wise Poison
                noise = perturb_tensor[self.targets[idx]]
                noise = patch_noise_extend_to_img(noise, [128, 128, 3], patch_location=self.patch_location)
            if add_uniform_noise:
                noise += np.random.uniform(0, 8, (128, 128, 3))

            self.image[idx] = np.array(self.image[idx]).astype(np.float32)
            self.image[idx] = self.image[idx] + noise
            self.image[idx] = np.clip(self.image[idx], a_min=0, a_max=255)
            self.image[idx] = self.image[idx].astype(np.uint8)
            self.image[idx] = Image.fromarray(self.image[idx]).convert('RGB')

        print('add_uniform_noise: ', add_uniform_noise)
        print(self.perturb_tensor.shape)
        print('Poison samples: %d/%d' % (len(self.poison_samples), len(self)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self,index):
        image = self.image[index]
        label = self.label[index]
        if self.transform is not None:
            image = self.transform(image)
        return transforms.ToTensor()(image), label


class AdversarialPoison(torch.utils.data.Dataset):
    def __init__(self, root, baseset, poison_rate=1.0, poison_classwise=False, poison_classwise_idx=None):
        self.baseset = baseset
        self.transform = self.baseset.transform
        self.samples = os.listdir(os.path.join(root, 'data'))
        self.root = root
        self.poison_class = []
        self.poison_samples_idx = []

        if poison_classwise:
            targets = list(range(0, 10))
            if poison_classwise_idx is None:
                self.poison_class = sorted(np.random.choice(targets, int(len(targets) * poison_rate), replace=False).tolist())
            else:
                self.poison_class = poison_classwise_idx
            self.poison_samples_idx = []
            for i, label in enumerate(self.targets):
                if label in self.poison_class:
                    self.poison_samples_idx.append(i)
        else:
            targets = list(range(0, len(self.baseset)))
            self.poison_samples_idx = sorted(np.random.choice(targets, int(len(targets) * poison_rate), replace=False).tolist())

        # Load images into memory to prevent IO from disk
        self.data, self.targets = self._load_images()
        # print (self.transform)
        self.array = False


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if not self.array:
            return self.transform(self.data[idx]), self.targets[idx]
        else:
            img, target = self.data[idx], self.targets[idx]
            img = Image.fromarray(img)
            if self.transform is not None:
                img = self.transform(img)
            return img, target

    def _load_images(self):
        data = []
        targets = []
        num_data_to_load = len(self.baseset)
        # print (len(self.samples))
        for i in range(num_data_to_load):
            true_index = int(self.samples[i].split('.')[0])
            if true_index in self.poison_samples_idx:
                data.append(Image.open(os.path.join(self.root, 'data', self.samples[i])).copy())
            else:
                data.append(Image.fromarray(self.baseset.data[true_index]).convert('RGB'))
            label = self.baseset.targets[true_index]
            targets.append(label)
        return data, targets

    # def _load_images(self):
    #     data = dict()
    #     targets = []
    #     num_data_to_load = len(self.baseset)
    #     # print (len(self.samples))
    #     for i in range(num_data_to_load):
    #         # print (i)
    #         true_index = int(self.samples[i].split('.')[0])
    #         if true_index in self.poison_samples_idx:
    #             # data.append(Image.open(os.path.join(self.root, 'data', self.samples[i])).copy())
    #             data[true_index] = Image.open(os.path.join(self.root, 'data', self.samples[i])).copy()
    #         else:
    #             data.append(Image.fromarray(self.baseset.data[true_index]).convert('RGB'))
    #         label = self.baseset.targets[true_index]
    #         targets.append(label)
    #     return data, targets

class NTGACIFAR10(torch.utils.data.Dataset):
    def __init__(self, root, baseset):
        self.baseset = baseset
        self.transform = self.baseset.transform
        self.samples = os.path.join(root, 'x_train_cifar10_ntga_cnn_best.npy')
        self.labels = os.path.join(root, 'y_train_cifar10.npy')
        self.root = root

        # Load images into memory to prevent IO from disk
        self.data, self.targets = self._load_images()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def _load_images(self):
        labels = np.load(self.labels)
        targets = np.argmax(labels, 1)
        data = (255 * np.load(self.samples)).astype(np.uint8)
        return data, targets

class CIFAR10_NTGA(torch.utils.data.Dataset):
    def __init__(self, root, baseset, perturb_tensor_filepath, poison_samples_idx = None):
        self.baseset = baseset
        self.transform = self.baseset.transform
        self.root = root

        # Load images into memory to prevent IO from disk
        poison_idxs = np.load('./experiments/cifar10/ntga8/poison_idxs.npy')
        poison_list = np.load('./experiments/cifar10/ntga8/poison_list.npy')
        if perturb_tensor_filepath is not None:
            perturb_tensor = torch.load(perturb_tensor_filepath, map_location=device)
            perturb_tensor = perturb_tensor.mul(255).clamp_(-255, 255).permute(0, 2, 3, 1).to('cpu').numpy()
        print (perturb_tensor.shape)
        self.data = []
        self.targets = []
        for i in range(len(poison_idxs)):
            noise = perturb_tensor[poison_list[i]]
            noise = patch_noise_extend_to_img(noise, [32, 32, 3], patch_location='center')
            image = baseset.data[poison_idxs[i]].astype(np.float32)
            image = image + noise
            image = np.clip(image, a_min=0, a_max=255)
            self.data.append(image)
            self.targets.append(baseset.targets[poison_idxs[i]])
        self.data = np.stack(self.data, axis=0)
        self.data = self.data.astype(np.uint8)
        print (self.data.shape)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target


class CIFAR10_numpy(torch.utils.data.Dataset):
    def __init__(self, root, baseset, perturb_tensor_filepath, poison_samples_idx = None):
        self.baseset = baseset
        self.transform = self.baseset.transform
        self.samples = os.path.join(root, 'cifar10_data.npy')
        self.labels = os.path.join(root, 'cifar10_target.npy')
        self.root = root

        # Load images into memory to prevent IO from disk
        self.data, self.targets = self._load_images()
        if perturb_tensor_filepath is not None:
            self.add_additional_perturb_tensor(perturb_tensor_filepath, poison_samples_idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def _load_images(self):
        labels = np.load(self.labels)
        targets = labels
        data = np.load(self.samples).astype(np.uint8)
        return data, targets

    def add_additional_perturb_tensor(self, perturb_tensor_filepath, poison_samples_idx):
        perturb_tensor = torch.load(perturb_tensor_filepath, map_location=device)
        perturb_tensor = perturb_tensor.mul(255).clamp_(-255, 255).permute(0, 2, 3, 1).to('cpu').numpy()
        self.data = self.data.astype(np.float32)
        for idx in poison_samples_idx:
            noise = perturb_tensor[idx]
            noise = patch_noise_extend_to_img(noise, [32, 32, 3], patch_location='center')
            self.data[idx] = self.data[idx] + noise
            self.data[idx] = np.clip(self.data[idx], a_min=0, a_max=255)
        self.data = self.data.astype(np.uint8)


class CIFAR10_numpy_partial_diffpure(torch.utils.data.Dataset):
    def __init__(self, root, baseset, perturb_tensor_filepath, poison_samples_idx = None):
        self.baseset = baseset
        self.transform = self.baseset.transform
        self.samples = os.path.join(root, 'cifar10_data.npy')
        self.labels = os.path.join(root, 'cifar10_target.npy')
        self.root = root

        # Load images into memory to prevent IO from disk
        self.data = baseset.data
        self.targets = baseset.targets
        if perturb_tensor_filepath is not None:
            self.add_additional_perturb_tensor(perturb_tensor_filepath, poison_samples_idx)
        self.data_diffpure, self.targets_diffpure = self._load_images()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def _load_images(self):
        labels = np.load(self.labels)
        targets = labels
        data = np.load(self.samples).astype(np.uint8)
        return data, targets

    def add_additional_perturb_tensor(self, perturb_tensor_filepath, poison_samples_idx):
        perturb_tensor = torch.load(perturb_tensor_filepath, map_location=device)
        perturb_tensor = perturb_tensor.mul(255).clamp_(-255, 255).permute(0, 2, 3, 1).to('cpu').numpy()
        self.data = self.data.astype(np.float32)
        for idx in poison_samples_idx:
            noise = perturb_tensor[idx]
            noise = patch_noise_extend_to_img(noise, [32, 32, 3], patch_location='center')
            self.data[idx] = self.data[idx] + noise
            self.data[idx] = np.clip(self.data[idx], a_min=0, a_max=255)
        self.data = self.data.astype(np.uint8)

    def replace_diffpure_data(self, defense_samples_idx):
        for idx in defense_samples_idx:
            # print (idx)
            self.data[idx] = self.data_diffpure[idx]
            self.targets[idx] = self.targets_diffpure[idx]
        print ("Replace diffpure data!")


class CIFAR100_numpy(torch.utils.data.Dataset):
    def __init__(self, root, baseset, perturb_tensor_filepath, poison_samples_idx = None):
        self.baseset = baseset
        self.transform = self.baseset.transform
        self.samples = os.path.join(root, 'cifar100_data.npy')
        self.labels = os.path.join(root, 'cifar100_target.npy')
        self.root = root

        # Load images into memory to prevent IO from disk
        self.data, self.targets = self._load_images()
        if perturb_tensor_filepath is not None:
            self.add_additional_perturb_tensor(perturb_tensor_filepath, poison_samples_idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def _load_images(self):
        labels = np.load(self.labels)
        targets = labels
        data = np.load(self.samples).astype(np.uint8)
        return data, targets

    def add_additional_perturb_tensor(self, perturb_tensor_filepath, poison_samples_idx):
        perturb_tensor = torch.load(perturb_tensor_filepath, map_location=device)
        perturb_tensor = perturb_tensor.mul(255).clamp_(-255, 255).permute(0, 2, 3, 1).to('cpu').numpy()
        self.data = self.data.astype(np.float32)
        for idx in poison_samples_idx:
            noise = perturb_tensor[idx]
            noise = patch_noise_extend_to_img(noise, [32, 32, 3], patch_location='center')
            self.data[idx] = self.data[idx] + noise
            self.data[idx] = np.clip(self.data[idx], a_min=0, a_max=255)
        self.data = self.data.astype(np.uint8)


class CIFAR100_numpy_partial_diffpure(torch.utils.data.Dataset):
    def __init__(self, root, baseset, perturb_tensor_filepath, poison_samples_idx = None):
        self.baseset = baseset
        self.transform = self.baseset.transform
        self.samples = os.path.join(root, 'cifar100_data.npy')
        self.labels = os.path.join(root, 'cifar100_target.npy')
        self.root = root

        # Load images into memory to prevent IO from disk
        self.data = baseset.data
        self.targets = baseset.targets
        if perturb_tensor_filepath is not None:
            self.add_additional_perturb_tensor(perturb_tensor_filepath, poison_samples_idx)
        self.data_diffpure, self.targets_diffpure = self._load_images()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def _load_images(self):
        labels = np.load(self.labels)
        targets = labels
        data = np.load(self.samples).astype(np.uint8)
        return data, targets

    def add_additional_perturb_tensor(self, perturb_tensor_filepath, poison_samples_idx):
        perturb_tensor = torch.load(perturb_tensor_filepath, map_location=device)
        perturb_tensor = perturb_tensor.mul(255).clamp_(-255, 255).permute(0, 2, 3, 1).to('cpu').numpy()
        self.data = self.data.astype(np.float32)
        for idx in poison_samples_idx:
            noise = perturb_tensor[idx]
            noise = patch_noise_extend_to_img(noise, [32, 32, 3], patch_location='center')
            self.data[idx] = self.data[idx] + noise
            self.data[idx] = np.clip(self.data[idx], a_min=0, a_max=255)
        self.data = self.data.astype(np.uint8)

    def replace_diffpure_data(self, defense_samples_idx):
        for idx in defense_samples_idx:
            self.data[idx] = self.data_diffpure[idx]
            self.targets[idx] = self.targets_diffpure[idx]
        print ("Replace diffpure data!")


class PoisonCIFAR10(datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, poison_rate=1.0, perturb_tensor_filepath=None,
                 seed=0, perturb_type='classwise', patch_location='center', img_denoise=False,
                 add_uniform_noise=False, poison_classwise=False, poison_classwise_idx=None, poison_samples_idx = None):
        super(PoisonCIFAR10, self).__init__(root=root, train=train, download=download, transform=transform, target_transform=target_transform)
        self.perturb_tensor = torch.load(perturb_tensor_filepath, map_location=device)
        print(self.perturb_tensor)
        if len(self.perturb_tensor.shape) == 4:
            # self.perturb_tensor = self.perturb_tensor.mul(255).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu').numpy()
            self.perturb_tensor = self.perturb_tensor.mul(255).clamp_(-255, 255).permute(0, 2, 3, 1).to('cpu').numpy()
        else:
            # self.perturb_tensor = self.perturb_tensor.mul(255).clamp_(0, 255).permute(0, 1, 3, 4, 2).to('cpu').numpy()
            self.perturb_tensor = self.perturb_tensor.mul(255).clamp_(-255, 255).permute(0, 1, 3, 4, 2).to('cpu').numpy()
        self.patch_location = patch_location
        self.img_denoise = img_denoise
        self.data = self.data.astype(np.float32)
        # Check Shape
        target_dim = self.perturb_tensor.shape[0] if len(self.perturb_tensor.shape) == 4 else self.perturb_tensor.shape[1]
        if perturb_type == 'samplewise' and target_dim != len(self):
            raise('Poison Perturb Tensor size not match for samplewise')
        elif perturb_type == 'classwise' and target_dim != 10:
            raise('Poison Perturb Tensor size not match for classwise')

        # Random Select Poison Targets
        self.poison_samples = collections.defaultdict(lambda: False)
        self.poison_class = []
        if poison_classwise:
            targets = list(range(0, 10))
            if poison_classwise_idx is None:
                self.poison_class = sorted(np.random.choice(targets, int(len(targets) * poison_rate), replace=False).tolist())
            else:
                self.poison_class = poison_classwise_idx
            self.poison_samples_idx = []
            for i, label in enumerate(self.targets):
                if label in self.poison_class:
                    self.poison_samples_idx.append(i)
        else:
            if poison_samples_idx is not None:
                self.poison_samples_idx = poison_samples_idx
            else:
                targets = list(range(0, len(self)))
                self.poison_samples_idx = sorted(np.random.choice(targets, int(len(targets) * poison_rate), replace=False).tolist())

        for idx in self.poison_samples_idx:
            self.poison_samples[idx] = True
            if len(self.perturb_tensor.shape) == 5:
                perturb_id = random.choice(range(self.perturb_tensor.shape[0]))
                perturb_tensor = self.perturb_tensor[perturb_id]
            else:
                perturb_tensor = self.perturb_tensor
            if perturb_type == 'samplewise':
                # Sample Wise poison
                noise = perturb_tensor[idx]
                noise = patch_noise_extend_to_img(noise, [32, 32, 3], patch_location=self.patch_location)
            elif perturb_type == 'classwise':
                # Class Wise Poison
                noise = perturb_tensor[self.targets[idx]]
                noise = patch_noise_extend_to_img(noise, [32, 32, 3], patch_location=self.patch_location)
            if add_uniform_noise:
                noise += np.random.uniform(0, 8, (32, 32, 3))

            self.data[idx] = self.data[idx] + noise
            self.data[idx] = np.clip(self.data[idx], a_min=0, a_max=255)
        self.data = self.data.astype(np.uint8)
        print('add_uniform_noise: ', add_uniform_noise)
        print(self.perturb_tensor.shape)
        print('Poison samples: %d/%d' % (len(self.poison_samples), len(self)))

    def add_additional_perturb_tensor(self, perturb_tensor_filepath):
        perturb_tensor = torch.load(perturb_tensor_filepath, map_location=device)
        perturb_tensor = perturb_tensor.mul(255).clamp_(-255, 255).permute(0, 2, 3, 1).to('cpu').numpy()
        self.data = self.data.astype(np.float32)
        for idx in range(len(self.data)):
            noise = perturb_tensor[idx]
            noise = patch_noise_extend_to_img(noise, [32, 32, 3], patch_location=self.patch_location)
            self.data[idx] = self.data[idx] + noise
            self.data[idx] = np.clip(self.data[idx], a_min=0, a_max=255)
        self.data = self.data.astype(np.uint8)


class PoisonCIFAR10_shuffle(datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, poison_rate=1.0, perturb_tensor_filepath=None,
                 seed=0, perturb_type='classwise', patch_location='center', img_denoise=False,
                 add_uniform_noise=False, poison_classwise=False, poison_classwise_idx=None, poison_samples_idx = None, shuffle = 0):
        super(PoisonCIFAR10_shuffle, self).__init__(root=root, train=train, download=download, transform=transform, target_transform=target_transform)
        self.perturb_tensor = torch.load(perturb_tensor_filepath, map_location=device)
        print(self.perturb_tensor)
        if len(self.perturb_tensor.shape) == 4:
            # self.perturb_tensor = self.perturb_tensor.mul(255).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu').numpy()
            self.perturb_tensor = self.perturb_tensor.mul(255).clamp_(-255, 255).permute(0, 2, 3, 1).to('cpu').numpy()
        else:
            # self.perturb_tensor = self.perturb_tensor.mul(255).clamp_(0, 255).permute(0, 1, 3, 4, 2).to('cpu').numpy()
            self.perturb_tensor = self.perturb_tensor.mul(255).clamp_(-255, 255).permute(0, 1, 3, 4, 2).to('cpu').numpy()
        self.patch_location = patch_location
        self.img_denoise = img_denoise
        self.data = self.data.astype(np.float32)
        # Check Shape
        target_dim = self.perturb_tensor.shape[0] if len(self.perturb_tensor.shape) == 4 else self.perturb_tensor.shape[1]
        if perturb_type == 'samplewise' and target_dim != len(self):
            raise('Poison Perturb Tensor size not match for samplewise')
        elif perturb_type == 'classwise' and target_dim != 10:
            raise('Poison Perturb Tensor size not match for classwise')

        # Random Select Poison Targets
        self.poison_samples = collections.defaultdict(lambda: False)
        self.poison_class = []
        self.newdata = np.zeros_like(self.data)
        for i in range(10):
            self.newdata[np.where(np.array(self.targets)==i)] = self.data[np.where(np.array(self.targets)==(i+shuffle)%10)]
        self.data = self.newdata
        if poison_classwise:
            targets = list(range(0, 10))
            if poison_classwise_idx is None:
                self.poison_class = sorted(np.random.choice(targets, int(len(targets) * poison_rate), replace=False).tolist())
            else:
                self.poison_class = poison_classwise_idx
            self.poison_samples_idx = []
            for i, label in enumerate(self.targets):
                if label in self.poison_class:
                    self.poison_samples_idx.append(i)
        else:
            if poison_samples_idx is not None:
                self.poison_samples_idx = poison_samples_idx
            else:
                targets = list(range(0, len(self)))
                self.poison_samples_idx = sorted(np.random.choice(targets, int(len(targets) * poison_rate), replace=False).tolist())

        for idx in self.poison_samples_idx:
            self.poison_samples[idx] = True
            if len(self.perturb_tensor.shape) == 5:
                perturb_id = random.choice(range(self.perturb_tensor.shape[0]))
                perturb_tensor = self.perturb_tensor[perturb_id]
            else:
                perturb_tensor = self.perturb_tensor
            if perturb_type == 'samplewise':
                # Sample Wise poison
                noise = perturb_tensor[idx]
                noise = patch_noise_extend_to_img(noise, [32, 32, 3], patch_location=self.patch_location)
            elif perturb_type == 'classwise':
                # Class Wise Poison
                noise = perturb_tensor[self.targets[idx]]
                noise = patch_noise_extend_to_img(noise, [32, 32, 3], patch_location=self.patch_location)
            if add_uniform_noise:
                noise += np.random.uniform(0, 8, (32, 32, 3))

            self.data[idx] = self.data[idx] + noise
            self.data[idx] = np.clip(self.data[idx], a_min=0, a_max=255)
        self.data = self.data.astype(np.uint8)
        print('add_uniform_noise: ', add_uniform_noise)
        print(self.perturb_tensor.shape)
        print('Poison samples: %d/%d' % (len(self.poison_samples), len(self)))


class PoisonCIFAR10_perturbation(datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, poison_rate=1.0, perturb_tensor_filepath=None,
                 seed=0, perturb_type='classwise', patch_location='center', img_denoise=False,
                 add_uniform_noise=False, poison_classwise=False, poison_classwise_idx=None, poison_samples_idx = None, shuffle = 0):
        super(PoisonCIFAR10_perturbation, self).__init__(root=root, train=train, download=download, transform=transform, target_transform=target_transform)
        self.perturb_tensor = torch.load(perturb_tensor_filepath, map_location=device)
        print(self.perturb_tensor)
        if len(self.perturb_tensor.shape) == 4:
            # self.perturb_tensor = self.perturb_tensor.mul(255).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu').numpy()
            self.perturb_tensor = self.perturb_tensor.mul(255).clamp_(-255, 255).permute(0, 2, 3, 1).to('cpu').numpy()
        else:
            # self.perturb_tensor = self.perturb_tensor.mul(255).clamp_(0, 255).permute(0, 1, 3, 4, 2).to('cpu').numpy()
            self.perturb_tensor = self.perturb_tensor.mul(255).clamp_(-255, 255).permute(0, 1, 3, 4, 2).to('cpu').numpy()
        self.patch_location = patch_location
        self.img_denoise = img_denoise
        self.data = self.data.astype(np.float32)
        # Check Shape
        target_dim = self.perturb_tensor.shape[0] if len(self.perturb_tensor.shape) == 4 else self.perturb_tensor.shape[1]
        if perturb_type == 'samplewise' and target_dim != len(self):
            raise('Poison Perturb Tensor size not match for samplewise')
        elif perturb_type == 'classwise' and target_dim != 10:
            raise('Poison Perturb Tensor size not match for classwise')

        # Random Select Poison Targets
        self.poison_samples = collections.defaultdict(lambda: False)
        self.poison_class = []
        self.newdata = np.zeros_like(self.data)
        for i in range(10):
            self.newdata[np.where(np.array(self.targets)==i)] = self.data[np.where(np.array(self.targets)==(i+shuffle)%10)]
        self.data = self.newdata
        if poison_classwise:
            targets = list(range(0, 10))
            if poison_classwise_idx is None:
                self.poison_class = sorted(np.random.choice(targets, int(len(targets) * poison_rate), replace=False).tolist())
            else:
                self.poison_class = poison_classwise_idx
            self.poison_samples_idx = []
            for i, label in enumerate(self.targets):
                if label in self.poison_class:
                    self.poison_samples_idx.append(i)
        else:
            if poison_samples_idx is not None:
                self.poison_samples_idx = poison_samples_idx
            else:
                targets = list(range(0, len(self)))
                self.poison_samples_idx = sorted(np.random.choice(targets, int(len(targets) * poison_rate), replace=False).tolist())

        for idx in self.poison_samples_idx:
            self.poison_samples[idx] = True
            if len(self.perturb_tensor.shape) == 5:
                perturb_id = random.choice(range(self.perturb_tensor.shape[0]))
                perturb_tensor = self.perturb_tensor[perturb_id]
            else:
                perturb_tensor = self.perturb_tensor
            if perturb_type == 'samplewise':
                # Sample Wise poison
                noise = perturb_tensor[idx]
                noise = patch_noise_extend_to_img(noise, [32, 32, 3], patch_location=self.patch_location)
            elif perturb_type == 'classwise':
                # Class Wise Poison
                noise = perturb_tensor[self.targets[idx]]
                noise = patch_noise_extend_to_img(noise, [32, 32, 3], patch_location=self.patch_location)
            if add_uniform_noise:
                noise += np.random.uniform(0, 8, (32, 32, 3))

            self.data[idx] = 128*np.ones_like(self.data[idx]) + noise
            self.data[idx] = np.clip(self.data[idx], a_min=0, a_max=255)
        self.data = self.data.astype(np.uint8)
        print('add_uniform_noise: ', add_uniform_noise)
        print(self.perturb_tensor.shape)
        print('Poison samples: %d/%d' % (len(self.poison_samples), len(self)))

class PoisonCIFAR10_v2(datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, poison_rate=1.0, perturb_tensor_filepath=None,
                 seed=0, perturb_type='classwise', patch_location='center', img_denoise=False,
                 add_uniform_noise=False, poison_classwise=False, poison_classwise_idx=None):
        super(PoisonCIFAR10_v2, self).__init__(root=root, train=train, download=download, transform=transform, target_transform=target_transform)
        self.perturb_tensor = torch.load(perturb_tensor_filepath, map_location=device)
        print(self.perturb_tensor)
        if len(self.perturb_tensor.shape) == 4:
            self.perturb_tensor = self.perturb_tensor.mul(255).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu').numpy()
        else:
            self.perturb_tensor = self.perturb_tensor.mul(255).clamp_(0, 255).permute(0, 1, 3, 4, 2).to('cpu').numpy()
        self.patch_location = patch_location
        self.img_denoise = img_denoise
        self.data = self.data.astype(np.float32)
        # Check Shape
        target_dim = self.perturb_tensor.shape[0] if len(self.perturb_tensor.shape) == 4 else self.perturb_tensor.shape[1]
        if perturb_type == 'samplewise' and target_dim != len(self):
            raise('Poison Perturb Tensor size not match for samplewise')
        elif perturb_type == 'classwise' and target_dim != 10:
            raise('Poison Perturb Tensor size not match for classwise')

        # Random Select Poison Targets
        self.poison_samples = collections.defaultdict(lambda: False)
        self.poison_class = []
        if poison_classwise:
            targets = list(range(0, 10))
            if poison_classwise_idx is None:
                self.poison_class = sorted(np.random.choice(targets, int(len(targets) * poison_rate), replace=False).tolist())
            else:
                self.poison_class = poison_classwise_idx
            self.poison_samples_idx = []
            for i, label in enumerate(self.targets):
                if label in self.poison_class:
                    self.poison_samples_idx.append(i)
        else:
            targets = list(range(0, len(self)))
            self.poison_samples_idx = sorted(np.random.choice(targets, int(len(targets) * poison_rate), replace=False).tolist())

        for idx in range(len(self.data)):
            if idx in self.poison_samples_idx:
                self.poison_samples[idx] = True
                if len(self.perturb_tensor.shape) == 5:
                    perturb_id = random.choice(range(self.perturb_tensor.shape[0]))
                    perturb_tensor = self.perturb_tensor[perturb_id]
                else:
                    perturb_tensor = self.perturb_tensor
                if perturb_type == 'samplewise':
                    # Sample Wise poison
                    noise = perturb_tensor[idx]
                    noise = patch_noise_extend_to_img(noise, [32, 32, 3], patch_location=self.patch_location)
                elif perturb_type == 'classwise':
                    # Class Wise Poison
                    noise = perturb_tensor[self.targets[idx]]
                    noise = patch_noise_extend_to_img(noise, [32, 32, 3], patch_location=self.patch_location)
                if add_uniform_noise:
                    noise += np.random.uniform(0, 8, (32, 32, 3))

                self.data[idx] = self.data[idx] + noise
                self.data[idx] = np.clip(self.data[idx], a_min=0, a_max=255)
            else:
                self.targets[idx] = 10
        self.data = self.data.astype(np.uint8)
        print('add_uniform_noise: ', add_uniform_noise)
        print(self.perturb_tensor.shape)
        print('Poison samples: %d/%d' % (len(self.poison_samples), len(self)))


class PoisonCIFAR10_v3(datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, poison_rate=1.0, perturb_tensor_filepath=None,
                 seed=0, perturb_type='classwise', patch_location='center', img_denoise=False,
                 add_uniform_noise=False, poison_classwise=False, poison_classwise_idx=None):
        super(PoisonCIFAR10_v3, self).__init__(root=root, train=train, download=download, transform=transform, target_transform=target_transform)
        self.perturb_tensor = torch.load(perturb_tensor_filepath, map_location=device)
        print(self.perturb_tensor)
        if len(self.perturb_tensor.shape) == 4:
            self.perturb_tensor = self.perturb_tensor.mul(255).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu').numpy()
        else:
            self.perturb_tensor = self.perturb_tensor.mul(255).clamp_(0, 255).permute(0, 1, 3, 4, 2).to('cpu').numpy()
        self.patch_location = patch_location
        self.img_denoise = img_denoise
        self.data = self.data.astype(np.float32)
        # Check Shape
        target_dim = self.perturb_tensor.shape[0] if len(self.perturb_tensor.shape) == 4 else self.perturb_tensor.shape[1]
        if perturb_type == 'samplewise' and target_dim != len(self):
            raise('Poison Perturb Tensor size not match for samplewise')
        elif perturb_type == 'classwise' and target_dim != 10:
            raise('Poison Perturb Tensor size not match for classwise')

        # Random Select Poison Targets
        self.poison_samples = collections.defaultdict(lambda: False)
        self.poison_class = []
        if poison_classwise:
            targets = list(range(0, 10))
            if poison_classwise_idx is None:
                self.poison_class = sorted(np.random.choice(targets, int(len(targets) * poison_rate), replace=False).tolist())
            else:
                self.poison_class = poison_classwise_idx
            self.poison_samples_idx = []
            for i, label in enumerate(self.targets):
                if label in self.poison_class:
                    self.poison_samples_idx.append(i)
        else:
            targets = list(range(0, len(self)))
            self.poison_samples_idx = sorted(np.random.choice(targets, int(len(targets) * poison_rate), replace=False).tolist())

        k = 0
        for idx in range(len(self.data)):
            if idx in self.poison_samples_idx:
                self.poison_samples[idx] = True
                if len(self.perturb_tensor.shape) == 5:
                    perturb_id = random.choice(range(self.perturb_tensor.shape[0]))
                    perturb_tensor = self.perturb_tensor[perturb_id]
                else:
                    perturb_tensor = self.perturb_tensor
                if perturb_type == 'samplewise':
                    # Sample Wise poison
                    noise = perturb_tensor[idx]
                    noise = patch_noise_extend_to_img(noise, [32, 32, 3], patch_location=self.patch_location)
                elif perturb_type == 'classwise':
                    # Class Wise Poison
                    noise = perturb_tensor[self.targets[idx]]
                    noise = patch_noise_extend_to_img(noise, [32, 32, 3], patch_location=self.patch_location)
                if add_uniform_noise:
                    noise += np.random.uniform(0, 8, (32, 32, 3))

                self.data[idx] = self.data[idx] + noise
                self.data[idx] = np.clip(self.data[idx], a_min=0, a_max=255)
            else:
                k+=1
                if k % 2 == 0:
                    self.targets[idx] = self.targets[idx] + 10

        self.data = self.data.astype(np.uint8)
        print('add_uniform_noise: ', add_uniform_noise)
        print(self.perturb_tensor.shape)
        print('Poison samples: %d/%d' % (len(self.poison_samples), len(self)))


class PoisonCIFAR10_v4(datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, poison_rate=1.0, perturb_tensor_filepath=None,
                 seed=0, perturb_type='classwise', patch_location='center', img_denoise=False,
                 add_uniform_noise=False, poison_classwise=False, poison_classwise_idx=None):
        super(PoisonCIFAR10_v4, self).__init__(root=root, train=train, download=download, transform=transform, target_transform=target_transform)
        self.perturb_tensor = torch.load(perturb_tensor_filepath, map_location=device)
        print(self.perturb_tensor)
        if len(self.perturb_tensor.shape) == 4:
            self.perturb_tensor = self.perturb_tensor.mul(255).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu').numpy()
        else:
            self.perturb_tensor = self.perturb_tensor.mul(255).clamp_(0, 255).permute(0, 1, 3, 4, 2).to('cpu').numpy()
        self.patch_location = patch_location
        self.img_denoise = img_denoise
        self.data = self.data.astype(np.float32)
        # Check Shape
        target_dim = self.perturb_tensor.shape[0] if len(self.perturb_tensor.shape) == 4 else self.perturb_tensor.shape[1]
        if perturb_type == 'samplewise' and target_dim != len(self):
            raise('Poison Perturb Tensor size not match for samplewise')
        elif perturb_type == 'classwise' and target_dim != 10:
            raise('Poison Perturb Tensor size not match for classwise')

        # Random Select Poison Targets
        self.poison_samples = collections.defaultdict(lambda: False)
        self.poison_class = []
        if poison_classwise:
            targets = list(range(0, 10))
            if poison_classwise_idx is None:
                self.poison_class = sorted(np.random.choice(targets, int(len(targets) * poison_rate), replace=False).tolist())
            else:
                self.poison_class = poison_classwise_idx
            self.poison_samples_idx = []
            for i, label in enumerate(self.targets):
                if label in self.poison_class:
                    self.poison_samples_idx.append(i)
        else:
            targets = list(range(0, len(self)))
            self.poison_samples_idx = sorted(np.random.choice(targets, int(len(targets) * poison_rate), replace=False).tolist())

        for idx in self.poison_samples_idx:
            self.poison_samples[idx] = True
            if len(self.perturb_tensor.shape) == 5:
                perturb_id = random.choice(range(self.perturb_tensor.shape[0]))
                perturb_tensor = self.perturb_tensor[perturb_id]
            else:
                perturb_tensor = self.perturb_tensor
            if perturb_type == 'samplewise':
                # Sample Wise poison
                noise = perturb_tensor[idx]
                noise = patch_noise_extend_to_img(noise, [32, 32, 3], patch_location=self.patch_location)
            elif perturb_type == 'classwise':
                # Class Wise Poison
                noise = perturb_tensor[self.targets[idx]]
                noise = patch_noise_extend_to_img(noise, [32, 32, 3], patch_location=self.patch_location)
            if add_uniform_noise:
                noise += np.random.uniform(0, 8, (32, 32, 3))

            self.data[idx] = self.data[idx] + noise
            self.data[idx] = np.clip(self.data[idx], a_min=0, a_max=255)
        self.data = self.data.astype(np.uint8)
        print('add_uniform_noise: ', add_uniform_noise)
        print(self.perturb_tensor.shape)
        print('Poison samples: %d/%d' % (len(self.poison_samples), len(self)))



class PoisonCIFAR10_v5(datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, poison_rate=1.0, perturb_tensor_filepath=None,
                 seed=0, perturb_type='classwise', patch_location='center', img_denoise=False,
                 add_uniform_noise=False, poison_classwise=False, poison_classwise_idx=None, scale = 1):
        super(PoisonCIFAR10_v5, self).__init__(root=root, train=train, download=download, transform=transform, target_transform=target_transform)
        self.perturb_tensor = torch.load(perturb_tensor_filepath, map_location=device)
        print(self.perturb_tensor)
        if len(self.perturb_tensor.shape) == 4:
            # self.perturb_tensor = self.perturb_tensor.mul(255).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu').numpy()
            self.perturb_tensor = self.perturb_tensor.mul(255).clamp_(-255, 255).permute(0, 2, 3, 1).to('cpu').numpy()
        else:
            # self.perturb_tensor = self.perturb_tensor.mul(255).clamp_(0, 255).permute(0, 1, 3, 4, 2).to('cpu').numpy()
            self.perturb_tensor = self.perturb_tensor.mul(255).clamp_(-255, 255).permute(0, 1, 3, 4, 2).to('cpu').numpy()
        self.patch_location = patch_location
        self.img_denoise = img_denoise
        self.data = self.data.astype(np.float32)
        # Check Shape
        target_dim = self.perturb_tensor.shape[0] if len(self.perturb_tensor.shape) == 4 else self.perturb_tensor.shape[1]
        if perturb_type == 'samplewise' and target_dim != len(self):
            raise('Poison Perturb Tensor size not match for samplewise')
        elif perturb_type == 'classwise' and target_dim != 10:
            raise('Poison Perturb Tensor size not match for classwise')

        # Random Select Poison Targets
        self.poison_samples = collections.defaultdict(lambda: False)
        self.poison_class = []
        if poison_classwise:
            targets = list(range(0, 10))
            if poison_classwise_idx is None:
                self.poison_class = sorted(np.random.choice(targets, int(len(targets) * poison_rate), replace=False).tolist())
            else:
                self.poison_class = poison_classwise_idx
            self.poison_samples_idx = []
            for i, label in enumerate(self.targets):
                if label in self.poison_class:
                    self.poison_samples_idx.append(i)
        else:
            targets = list(range(0, len(self)))
            self.poison_samples_idx = sorted(np.random.choice(targets, int(len(targets) * poison_rate), replace=False).tolist())

        for idx in self.poison_samples_idx:
            self.poison_samples[idx] = True
            if len(self.perturb_tensor.shape) == 5:
                perturb_id = random.choice(range(self.perturb_tensor.shape[0]))
                perturb_tensor = self.perturb_tensor[perturb_id]
            else:
                perturb_tensor = self.perturb_tensor
            if perturb_type == 'samplewise':
                # Sample Wise poison
                noise = perturb_tensor[idx]
                noise = patch_noise_extend_to_img(noise, [32, 32, 3], patch_location=self.patch_location)
            elif perturb_type == 'classwise':
                # Class Wise Poison
                noise = perturb_tensor[self.targets[idx]]
                noise = patch_noise_extend_to_img(noise, [32, 32, 3], patch_location=self.patch_location)
            if add_uniform_noise:
                noise += np.random.uniform(0, 8, (32, 32, 3))

            self.data[idx] = self.data[idx] + noise * scale
            self.data[idx] = np.clip(self.data[idx], a_min=0, a_max=255)
        self.data = self.data.astype(np.uint8)
        print('add_uniform_noise: ', add_uniform_noise)
        print(self.perturb_tensor.shape)
        print('Poison samples: %d/%d' % (len(self.poison_samples), len(self)))



class PoisonCIFAR10_v6(datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, poison_rate=1.0, perturb_tensor_filepath=None,
                 seed=0, perturb_type='classwise', patch_location='center', img_denoise=False,
                 add_uniform_noise=False, poison_classwise=False, poison_classwise_idx=None, scale = 1):
        super(PoisonCIFAR10_v6, self).__init__(root=root, train=train, download=download, transform=transform, target_transform=target_transform)
        self.perturb_tensor = torch.load(perturb_tensor_filepath, map_location=device)
        print(self.perturb_tensor)
        if len(self.perturb_tensor.shape) == 4:
            # self.perturb_tensor = self.perturb_tensor.mul(255).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu').numpy()
            self.perturb_tensor = self.perturb_tensor.mul(255).clamp_(-255, 255).permute(0, 2, 3, 1).to('cpu').numpy()
        else:
            # self.perturb_tensor = self.perturb_tensor.mul(255).clamp_(0, 255).permute(0, 1, 3, 4, 2).to('cpu').numpy()
            self.perturb_tensor = self.perturb_tensor.mul(255).clamp_(-255, 255).permute(0, 1, 3, 4, 2).to('cpu').numpy()
        self.patch_location = patch_location
        self.img_denoise = img_denoise
        self.data = self.data.astype(np.float32)
        # Check Shape
        target_dim = self.perturb_tensor.shape[0] if len(self.perturb_tensor.shape) == 4 else self.perturb_tensor.shape[1]
        if perturb_type == 'samplewise' and target_dim != len(self):
            raise('Poison Perturb Tensor size not match for samplewise')
        elif perturb_type == 'classwise' and target_dim != 10:
            raise('Poison Perturb Tensor size not match for classwise')

        # Random Select Poison Targets
        self.poison_samples = collections.defaultdict(lambda: False)
        self.poison_class = []
        if poison_classwise:
            targets = list(range(0, 10))
            if poison_classwise_idx is None:
                self.poison_class = sorted(np.random.choice(targets, int(len(targets) * poison_rate), replace=False).tolist())
            else:
                self.poison_class = poison_classwise_idx
            self.poison_samples_idx = []
            for i, label in enumerate(self.targets):
                if label in self.poison_class:
                    self.poison_samples_idx.append(i)
        else:
            targets = list(range(0, len(self)))
            self.poison_samples_idx = sorted(np.random.choice(targets, int(len(targets) * poison_rate), replace=False).tolist())

        for idx in self.poison_samples_idx:
            self.poison_samples[idx] = True
            if len(self.perturb_tensor.shape) == 5:
                perturb_id = random.choice(range(self.perturb_tensor.shape[0]))
                perturb_tensor = self.perturb_tensor[perturb_id]
            else:
                perturb_tensor = self.perturb_tensor
            if perturb_type == 'samplewise':
                # Sample Wise poison
                noise = perturb_tensor[idx]
                noise = patch_noise_extend_to_img(noise, [32, 32, 3], patch_location=self.patch_location)
            elif perturb_type == 'classwise':
                # Class Wise Poison
                noise = perturb_tensor[self.targets[idx]]
                noise = patch_noise_extend_to_img(noise, [32, 32, 3], patch_location=self.patch_location)
            if add_uniform_noise:
                noise += np.random.uniform(0, 8, (32, 32, 3))

            self.data[idx] = (self.data[idx] + noise) * scale
            self.data[idx] = np.clip(self.data[idx], a_min=0, a_max=255)
        self.data = self.data.astype(np.uint8)
        print('add_uniform_noise: ', add_uniform_noise)
        print(self.perturb_tensor.shape)
        print('Poison samples: %d/%d' % (len(self.poison_samples), len(self)))


class PoisonCIFAR10_partial_defense(datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, poison_rate=1.0, perturb_tensor_filepath=None,
                 seed=0, perturb_type='classwise', patch_location='center', img_denoise=False,
                 add_uniform_noise=False, poison_classwise=False, poison_classwise_idx=None, poison_samples_idx = None):
        super(PoisonCIFAR10_partial_defense, self).__init__(root=root, train=train, download=download, transform=transform, target_transform=target_transform)
        self.perturb_tensor = torch.load(perturb_tensor_filepath, map_location=device)
        print(self.perturb_tensor)
        if len(self.perturb_tensor.shape) == 4:
            # self.perturb_tensor = self.perturb_tensor.mul(255).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu').numpy()
            self.perturb_tensor = self.perturb_tensor.mul(255).clamp_(-255, 255).permute(0, 2, 3, 1).to('cpu').numpy()
        else:
            # self.perturb_tensor = self.perturb_tensor.mul(255).clamp_(0, 255).permute(0, 1, 3, 4, 2).to('cpu').numpy()
            self.perturb_tensor = self.perturb_tensor.mul(255).clamp_(-255, 255).permute(0, 1, 3, 4, 2).to('cpu').numpy()
        self.patch_location = patch_location
        self.img_denoise = img_denoise
        self.data = self.data.astype(np.float32)
        # Check Shape
        target_dim = self.perturb_tensor.shape[0] if len(self.perturb_tensor.shape) == 4 else self.perturb_tensor.shape[1]
        if perturb_type == 'samplewise' and target_dim != len(self):
            raise('Poison Perturb Tensor size not match for samplewise')
        elif perturb_type == 'classwise' and target_dim != 10:
            raise('Poison Perturb Tensor size not match for classwise')

        # Random Select Poison Targets
        self.poison_samples = collections.defaultdict(lambda: False)
        self.poison_class = []
        if poison_classwise:
            targets = list(range(0, 10))
            if poison_classwise_idx is None:
                self.poison_class = sorted(np.random.choice(targets, int(len(targets) * poison_rate), replace=False).tolist())
            else:
                self.poison_class = poison_classwise_idx
            self.poison_samples_idx = []
            for i, label in enumerate(self.targets):
                if label in self.poison_class:
                    self.poison_samples_idx.append(i)
        else:
            if poison_samples_idx is not None:
                self.poison_samples_idx = poison_samples_idx
            else:
                targets = list(range(0, len(self)))
                self.poison_samples_idx = sorted(np.random.choice(targets, int(len(targets) * poison_rate), replace=False).tolist())

        for idx in self.poison_samples_idx:
            self.poison_samples[idx] = True
            if len(self.perturb_tensor.shape) == 5:
                perturb_id = random.choice(range(self.perturb_tensor.shape[0]))
                perturb_tensor = self.perturb_tensor[perturb_id]
            else:
                perturb_tensor = self.perturb_tensor
            if perturb_type == 'samplewise':
                # Sample Wise poison
                noise = perturb_tensor[idx]
                noise = patch_noise_extend_to_img(noise, [32, 32, 3], patch_location=self.patch_location)
            elif perturb_type == 'classwise':
                # Class Wise Poison
                noise = perturb_tensor[self.targets[idx]]
                noise = patch_noise_extend_to_img(noise, [32, 32, 3], patch_location=self.patch_location)
            if add_uniform_noise:
                noise += np.random.uniform(0, 8, (32, 32, 3))

            self.data[idx] = self.data[idx] + noise
            self.data[idx] = np.clip(self.data[idx], a_min=0, a_max=255)
        self.data = self.data.astype(np.uint8)
        print('add_uniform_noise: ', add_uniform_noise)
        print(self.perturb_tensor.shape)
        print('Poison samples: %d/%d' % (len(self.poison_samples), len(self)))
        self.original_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor()])
        self.defense_samples_idx = None

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        img = Image.fromarray(img)
        if self.transform is not None:
            if (self.defense_samples_idx is not None) and (idx in self.defense_samples_idx):
                img = self.transform(img)
            else:
                img = self.original_transform(img)
        return img, target

    def add_additional_perturb_tensor(self, perturb_tensor_filepath):
        perturb_tensor = torch.load(perturb_tensor_filepath, map_location=device)
        perturb_tensor = perturb_tensor.mul(255).clamp_(-255, 255).permute(0, 2, 3, 1).to('cpu').numpy()
        self.data = self.data.astype(np.float32)
        for idx in range(len(self.data)):
            noise = perturb_tensor[idx]
            noise = patch_noise_extend_to_img(noise, [32, 32, 3], patch_location=self.patch_location)
            self.data[idx] = self.data[idx] + noise
            self.data[idx] = np.clip(self.data[idx], a_min=0, a_max=255)
        self.data = self.data.astype(np.uint8)



class PoisonCIFAR100(datasets.CIFAR100):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, poison_rate=1.0, perturb_tensor_filepath=None,
                 seed=0, perturb_type='classwise', patch_location='center', img_denoise=False,
                 add_uniform_noise=False, poison_classwise=False, poison_samples_idx = None):
        super(PoisonCIFAR100, self).__init__(root=root, train=train, download=download, transform=transform, target_transform=target_transform)
        self.perturb_tensor = torch.load(perturb_tensor_filepath, map_location=device)
        self.perturb_tensor = self.perturb_tensor.mul(255).clamp_(-255, 255).permute(0, 2, 3, 1).to('cpu').numpy()
        self.patch_location = patch_location
        self.img_denoise = img_denoise
        self.data = self.data.astype(np.float32)

        # Check Shape
        if perturb_type == 'samplewise' and self.perturb_tensor.shape[0] != len(self):
            raise('Poison Perturb Tensor size not match for samplewise')
        elif perturb_type == 'classwise' and self.perturb_tensor.shape[0] != 100:
            raise('Poison Perturb Tensor size not match for classwise')

        # Random Select Poison Targets
        self.poison_samples = collections.defaultdict(lambda: False)
        self.poison_class = []
        if poison_classwise:
            targets = list(range(0, 100))
            self.poison_class = sorted(np.random.choice(targets, int(len(targets) * poison_rate), replace=False).tolist())
            self.poison_samples_idx = []
            for i, label in enumerate(self.targets):
                if label in self.poison_class:
                    self.poison_samples_idx.append(i)
        else:
            if poison_samples_idx is not None:
                self.poison_samples_idx = poison_samples_idx
            else:
                targets = list(range(0, len(self)))
                self.poison_samples_idx = sorted(np.random.choice(targets, int(len(targets) * poison_rate), replace=False).tolist())

        for idx in self.poison_samples_idx:
            self.poison_samples[idx] = True
            if perturb_type == 'samplewise':
                # Sample Wise poison
                noise = self.perturb_tensor[idx]
                noise = patch_noise_extend_to_img(noise, [32, 32, 3], patch_location=self.patch_location)
            elif perturb_type == 'classwise':
                # Class Wise Poison
                noise = self.perturb_tensor[self.targets[idx]]
                noise = patch_noise_extend_to_img(noise, [32, 32, 3], patch_location=self.patch_location)

            if add_uniform_noise:
                noise = np.random.uniform(0, 8, (32, 32, 3))

            self.data[idx] += noise
            self.data[idx] = np.clip(self.data[idx], 0, 255)

        self.data = self.data.astype(np.uint8)
        print('add_uniform_noise: ', add_uniform_noise)
        print(self.perturb_tensor.shape)
        print('Poison samples: %d/%d' % (len(self.poison_samples), len(self)))

    def add_additional_perturb_tensor(self, perturb_tensor_filepath):
        perturb_tensor = torch.load(perturb_tensor_filepath, map_location=device)
        perturb_tensor = perturb_tensor.mul(255).clamp_(-255, 255).permute(0, 2, 3, 1).to('cpu').numpy()
        self.data = self.data.astype(np.float32)
        for idx in range(len(self.data)):
            noise = perturb_tensor[idx]
            noise = patch_noise_extend_to_img(noise, [32, 32, 3], patch_location=self.patch_location)
            self.data[idx] = self.data[idx] + noise
            self.data[idx] = np.clip(self.data[idx], a_min=0, a_max=255)
        self.data = self.data.astype(np.uint8)


class PoisonCIFAR100_partial_defense(datasets.CIFAR100):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, poison_rate=1.0, perturb_tensor_filepath=None,
                 seed=0, perturb_type='classwise', patch_location='center', img_denoise=False,
                 add_uniform_noise=False, poison_classwise=False, poison_classwise_idx=None, poison_samples_idx = None):
        super(PoisonCIFAR100_partial_defense, self).__init__(root=root, train=train, download=download, transform=transform, target_transform=target_transform)
        self.perturb_tensor = torch.load(perturb_tensor_filepath, map_location=device)
        print(self.perturb_tensor)
        if len(self.perturb_tensor.shape) == 4:
            # self.perturb_tensor = self.perturb_tensor.mul(255).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu').numpy()
            self.perturb_tensor = self.perturb_tensor.mul(255).clamp_(-255, 255).permute(0, 2, 3, 1).to('cpu').numpy()
        else:
            # self.perturb_tensor = self.perturb_tensor.mul(255).clamp_(0, 255).permute(0, 1, 3, 4, 2).to('cpu').numpy()
            self.perturb_tensor = self.perturb_tensor.mul(255).clamp_(-255, 255).permute(0, 1, 3, 4, 2).to('cpu').numpy()
        self.patch_location = patch_location
        self.img_denoise = img_denoise
        self.data = self.data.astype(np.float32)
        # Check Shape
        target_dim = self.perturb_tensor.shape[0] if len(self.perturb_tensor.shape) == 4 else self.perturb_tensor.shape[1]
        if perturb_type == 'samplewise' and target_dim != len(self):
            raise('Poison Perturb Tensor size not match for samplewise')
        elif perturb_type == 'classwise' and target_dim != 10:
            raise('Poison Perturb Tensor size not match for classwise')

        # Random Select Poison Targets
        self.poison_samples = collections.defaultdict(lambda: False)
        self.poison_class = []
        if poison_classwise:
            targets = list(range(0, 10))
            if poison_classwise_idx is None:
                self.poison_class = sorted(np.random.choice(targets, int(len(targets) * poison_rate), replace=False).tolist())
            else:
                self.poison_class = poison_classwise_idx
            self.poison_samples_idx = []
            for i, label in enumerate(self.targets):
                if label in self.poison_class:
                    self.poison_samples_idx.append(i)
        else:
            if poison_samples_idx is not None:
                self.poison_samples_idx = poison_samples_idx
            else:
                targets = list(range(0, len(self)))
                self.poison_samples_idx = sorted(np.random.choice(targets, int(len(targets) * poison_rate), replace=False).tolist())

        for idx in self.poison_samples_idx:
            self.poison_samples[idx] = True
            if len(self.perturb_tensor.shape) == 5:
                perturb_id = random.choice(range(self.perturb_tensor.shape[0]))
                perturb_tensor = self.perturb_tensor[perturb_id]
            else:
                perturb_tensor = self.perturb_tensor
            if perturb_type == 'samplewise':
                # Sample Wise poison
                noise = perturb_tensor[idx]
                noise = patch_noise_extend_to_img(noise, [32, 32, 3], patch_location=self.patch_location)
            elif perturb_type == 'classwise':
                # Class Wise Poison
                noise = perturb_tensor[self.targets[idx]]
                noise = patch_noise_extend_to_img(noise, [32, 32, 3], patch_location=self.patch_location)
            if add_uniform_noise:
                noise += np.random.uniform(0, 8, (32, 32, 3))

            self.data[idx] = self.data[idx] + noise
            self.data[idx] = np.clip(self.data[idx], a_min=0, a_max=255)
        self.data = self.data.astype(np.uint8)
        print('add_uniform_noise: ', add_uniform_noise)
        print(self.perturb_tensor.shape)
        print('Poison samples: %d/%d' % (len(self.poison_samples), len(self)))
        self.original_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor()])
        self.defense_samples_idx = None

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        img = Image.fromarray(img)
        if self.transform is not None:
            if (self.defense_samples_idx is not None) and (idx in self.defense_samples_idx):
                img = self.transform(img)
            else:
                img = self.original_transform(img)
        return img, target

    def add_additional_perturb_tensor(self, perturb_tensor_filepath):
        perturb_tensor = torch.load(perturb_tensor_filepath, map_location=device)
        perturb_tensor = perturb_tensor.mul(255).clamp_(-255, 255).permute(0, 2, 3, 1).to('cpu').numpy()
        self.data = self.data.astype(np.float32)
        for idx in range(len(self.data)):
            noise = perturb_tensor[idx]
            noise = patch_noise_extend_to_img(noise, [32, 32, 3], patch_location=self.patch_location)
            self.data[idx] = self.data[idx] + noise
            self.data[idx] = np.clip(self.data[idx], a_min=0, a_max=255)
        self.data = self.data.astype(np.uint8)


class PoisonCIFAR101(datasets.VisionDataset):
    def __init__(self, root, split='poison_train', transform=None, target_transform=None,
                 poisn_cifar10_data=None, seed=0):
        np.random.seed(seed)
        self.transform = transform
        self.root = root
        if split == 'poison_train':
            self.clean_cifar100 = datasets.CIFAR100(root=root, train=True, download=True, transform=None)
            cifar10 = poisn_cifar10_data
            cifar10_sample_count = 500
        elif split == 'test':
            self.clean_cifar100 = datasets.CIFAR100(root=root, train=False, download=True, transform=None)
            cifar10 = datasets.CIFAR10(root=root, train=False, download=True, transform=None)
            cifar10_sample_count = 100

        self.data, self.targets = self.clean_cifar100.data, self.clean_cifar100.targets
        print(self.clean_cifar100.class_to_idx)
        # Add Ship samples of CIFAR10
        ship_idx = np.where(np.array(cifar10.targets) == 8)[0]
        selected_idx = np.random.choice(ship_idx, cifar10_sample_count, replace=False)
        extra_samples, extra_targets = [], []
        for idx in selected_idx:
            extra_samples.append(cifar10.data[idx])
            extra_targets.append(100)
        self.data = np.concatenate((self.data, np.array(extra_samples)))
        self.targets = self.targets + extra_targets
        self.poison_samples_idx = np.array(range(len(self.clean_cifar100), len(self)))
        self.poison_class = [100]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target


class PoisonSVHN(datasets.SVHN):
    def __init__(self, root, split='train', transform=None, target_transform=None,
                 download=False, poison_rate=1.0, perturb_tensor_filepath=None,
                 seed=0, perturb_type='classwise', patch_location='center', img_denoise=False,
                 add_uniform_noise=False, poison_classwise=False):
        super(PoisonSVHN, self).__init__(root=root, split=split, download=download, transform=transform, target_transform=target_transform)
        self.perturb_tensor = torch.load(perturb_tensor_filepath, map_location=device)
        self.perturb_tensor = self.perturb_tensor.mul(255).clamp_(0, 255).to('cpu').numpy()
        self.patch_location = patch_location
        self.img_denoise = img_denoise
        # Check Shape
        if perturb_type == 'samplewise' and self.perturb_tensor.shape[0] != len(self):
            raise('Poison Perturb Tensor size not match for samplewise')
        elif perturb_type == 'classwise' and self.perturb_tensor.shape[0] != 10:
            raise('Poison Perturb Tensor size not match for classwise')

        self.data = self.data.astype(np.float32)

        # Random Select Poison Targets
        self.poison_samples = collections.defaultdict(lambda: False)
        self.poison_class = []
        if poison_classwise:
            targets = list(range(0, 10))
            self.poison_class = sorted(np.random.choice(targets, int(len(targets) * poison_rate), replace=False).tolist())
            self.poison_samples_idx = []
            for i, label in enumerate(self.labels):
                if label in self.poison_class:
                    self.poison_samples_idx.append(i)
        else:
            targets = list(range(0, len(self)))
            self.poison_samples_idx = sorted(np.random.choice(targets, int(len(targets) * poison_rate), replace=False).tolist())

        for idx in self.poison_samples_idx:
            self.poison_samples[idx] = True
            if perturb_type == 'samplewise':
                # Sample Wise poison
                noise = self.perturb_tensor[idx]
                # noise = patch_noise_extend_to_img(noise, [32, 32, 3], patch_location=self.patch_location)
            elif perturb_type == 'classwise':
                # Class Wise Poison
                noise = self.perturb_tensor[self.labels[idx]]
                # noise = patch_noise_extend_to_img(noise, [32, 32, 3], patch_location=self.patch_location)

            if add_uniform_noise:
                noise = np.random.uniform(0, 8, (32, 32, 3))

            self.data[idx] += noise
            self.data[idx] = np.clip(self.data[idx], 0, 255)

        self.data = self.data.astype(np.uint8)
        print('add_uniform_noise: ', add_uniform_noise)
        print(self.perturb_tensor.shape)
        print('Poison samples: %d/%d' % (len(self.poison_samples), len(self)))


def datasetImageNet(root='./data', train=True, transform=None):
    if train: root = os.path.join(root, 'train')
    else: root = os.path.join(root, 'val')
    return torchvision.datasets.ImageFolder(root=root, transform=transform)


def datasetImageNetMini(root='./data', train=True, transform=None):
    dataset = datasetImageNet(root=root, train=train, transform=transform)
    ''' imagenet-mini is a subset of the first 100 classes of ImageNet '''
    idx = np.where( np.array(dataset.targets) < 100 )[0]
    dataset.samples = [ dataset.samples[ii] for ii in idx ]
    dataset.targets = [ dataset.targets[ii] for ii in idx ]
    return dataset


class ImageNetMini(torchvision.datasets.ImageFolder):
    def __init__(self, root, split='train', transform = None, load_to_memory = False):
        root = os.path.join(root, split)
        super(ImageNetMini, self).__init__(root, transform = transform)
        idx = np.where( np.array(self.targets) < 100 )[0]
        self.samples = [ self.samples[ii] for ii in idx ]
        self.targets = [ self.targets[ii] for ii in idx ]
        self.load_to_memory = load_to_memory
        if self.load_to_memory:
            self.new_samples = []
            for ii in range(len(self.samples)):
                if ii % 20000 == 0:
                    print (ii)
                self.new_samples.append((self.loader(self.samples[ii][0]), self.samples[ii][1]))
            self.samples = self.new_samples
        print(len(self.samples))
        print(len(self.targets))
        self.return_path = False

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        if self.load_to_memory:
            sample, target = self.samples[index]
        else:
            path, target = self.samples[index]
            sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.return_path:
            return sample, target, path
        return sample, target

    def make_purified_dir(self, name0, name):
        for path, _ in self.samples:
            d = os.path.dirname(path).replace('imagenet' + '_' + name0 + "_it0", 'imagenet' + '_' + name)
            # print (d)
            os.makedirs(d, exist_ok=True)

class PoisonImageNetMini(ImageNetMini):
    def __init__(self, root, split, poison_rate=1.0, transform = None, seed=0,
                 perturb_tensor_filepath=None, load_to_memory = False):
        super(PoisonImageNetMini, self).__init__(root=root, split=split, transform = transform, load_to_memory = load_to_memory)
        np.random.seed(seed)
        self.poison_rate = poison_rate
        self.perturb_tensor = torch.load(perturb_tensor_filepath)
        self.noise_h, self.noise_w = self.perturb_tensor.shape[2], self.perturb_tensor.shape[3]
        self.perturb_tensor = self.perturb_tensor.permute(0, 2, 3, 1).to('cpu').numpy()

        # Random Select Poison Targets
        targets = list(range(0, len(self)))
        self.poison_samples_idx = sorted(np.random.choice(targets, int(len(targets) * poison_rate), replace=False).tolist())
        self.poison_samples = collections.defaultdict(lambda: False)
        self.poison_class = []
        for idx in self.poison_samples_idx:
            self.poison_samples[idx] = True

        print(self.perturb_tensor.shape)
        print('Poison samples: %d/%d' % (len(self.poison_samples), len(self)))
        self.return_path = False
        self.purified = False

    def __getitem__(self, index):
        if self.load_to_memory:
            sample, target = self.samples[index]
        else:
            path, target = self.samples[index]
            sample = self.loader(path)
        # sample = np.array(transforms.RandomResizedCrop(224)(sample)).astype(np.float32)
        if not self.purified:
            sample = np.array(transforms.Resize((self.noise_h, self.noise_w))(sample)).astype(np.float32)
            if self.poison_samples[index]:
                noise = self.perturb_tensor[index].astype(np.float32)
                sample = sample + noise
                sample = np.clip(sample, 0, 255)
            sample = sample.astype(np.uint8)
            sample = Image.fromarray(sample).convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.return_path:
            return sample, target, path
        return sample, target

    def make_purified_dir(self, name):
        for path, _ in self.samples:
            d = os.path.dirname(path).replace('imagenet', 'imagenet' + '_' + name)
            os.makedirs(d, exist_ok=True)


class ImageNetMini_for_detection(torchvision.datasets.ImageFolder):
    def __init__(self, root, split='train', transform = None, load_to_memory = False):
        root = os.path.join(root, split)
        super(ImageNetMini_for_detection, self).__init__(root, transform = transform)
        idx = np.where( np.array(self.targets) < 100 )[0]
        self.samples = [ self.samples[ii] for ii in idx ]
        self.targets = [ self.targets[ii] for ii in idx ]
        self.load_to_memory = load_to_memory
        if self.load_to_memory:
            self.new_samples = []
            for ii in range(len(self.samples)):
                if ii % 20000 == 0:
                    print (ii)
                self.new_samples.append((self.loader(self.samples[ii][0]), self.samples[ii][1]))
            self.samples = self.new_samples
        print(len(self.samples))
        print(len(self.targets))
        self.return_path = False
        self.original = False

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        if self.load_to_memory:
            sample, target = self.samples[index]
        else:
            path, target = self.samples[index]
            sample = self.loader(path)
            sample_clean = self.loader(path.replace("imagenet_clean_VAE_it0","imagenet"))
        if self.transform is not None:
            sample = self.transform(sample)
            sample_clean = transforms.Resize((224,224))(sample_clean)
            sample_clean = self.transform(sample_clean)
            if not self.original:
                sample = 2 * sample_clean - sample
                sample = sample.clamp(0,1)
            else:
                sample = sample_clean
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.return_path:
            return sample, target, path
        return sample, target

    def make_purified_dir(self, name0, name):
        for path, _ in self.samples:
            d = os.path.dirname(path).replace('imagenet' + '_' + name0 + "_it0", 'imagenet' + '_' + name)
            # print (d)
            os.makedirs(d, exist_ok=True)


class ImageNetMini_partial(torchvision.datasets.ImageFolder):
    def __init__(self, root, split='train', transform = None, load_to_memory = False):
        root = os.path.join(root, split)
        super(ImageNetMini_partial, self).__init__(root, transform = transform)
        self.new_samples = []
        self.new_targets = []
        for i in range(100):
            idx = np.where( np.array(self.targets) == i )[0]
            selected_idx = idx[:int(0.2 * len(idx))]
            self.new_samples.extend([self.samples[ii] for ii in selected_idx ])
            self.new_targets.extend([self.targets[ii] for ii in selected_idx ]) 
        self.samples = self.new_samples
        self.targets = self.new_targets
        self.load_to_memory = load_to_memory
        if self.load_to_memory:
            self.new_samples = []
            for ii in range(len(self.samples)):
                if ii % 20000 == 0:
                    print (ii)
                self.new_samples.append((self.loader(self.samples[ii][0]), self.samples[ii][1]))
            self.samples = self.new_samples
        print(len(self.samples))
        print(len(self.targets))
        self.return_path = False

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        if self.load_to_memory:
            sample, target = self.samples[index]
        else:
            path, target = self.samples[index]
            sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.return_path:
            return sample, target, path
        return sample, target

    def make_purified_dir(self, name0, name):
        for path, _ in self.samples:
            d = os.path.dirname(path).replace('imagenet' + '_' + name0 + "_it0", 'imagenet' + '_' + name)
            # print (d)
            os.makedirs(d, exist_ok=True)

class Augmentation(object):
    def __init__(self, policies):
        self.policies = policies

    def __call__(self, img):
        for _ in range(1):
            policy = random.choice(self.policies)
            for name, pr, level in policy:
                if random.random() > pr:
                    continue
                img = apply_augment(img, name, level)
        return img


class CatDogDataset(datasets.VisionDataset):
    def __init__(self, root, split='train', transform=None, target_transform=None):
        self.root = root
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.img_file_names = os.listdir(os.path.join(root, split))

    def __len__(self):
        return len(self.img_file_names)

    def __getitem__(self, index):
        filename = self.img_file_names[index]
        label = filename[:3]
        if label == 'cat':
            label = 0
        elif label == 'dog':
            label = 1
        else:
            print(filename)
            raise('Unknown label')

        with open(os.path.join(self.root, self.split, filename), 'rb') as f:
            img = Image.open(f).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label


class CelebAMini(datasets.CelebA):
    def __init__(self, root, split="train", target_type="attr", transform=None,
                 target_transform=None, download=False, num_of_classes=1000):
        super(CelebAMini, self).__init__(root=root, split=split, target_type=target_type,
                                         transform=transform, target_transform=target_transform,
                                         download=False)

        split_map = {
            "train": 0,
            "valid": 1,
            "test": 2,
            "all": None,
        }
        split_ = split_map[datasets.utils.verify_str_arg(split.lower(), "split", ("train", "valid", "test", "all"))]

        fn = partial(os.path.join, self.root, self.base_folder)
        splits = pandas.read_csv(fn("list_eval_partition.txt"), delim_whitespace=True, header=None, index_col=0)
        identity = pandas.read_csv(fn("identity_CelebA.txt"), delim_whitespace=True, header=None, index_col=0)

        mask = slice(None) if split_ is None else (splits[1] == split_)
        identity = identity[mask]
        identity = identity[identity[1] < num_of_classes]
        self.filename = identity.index.values
        self.identity = identity.values
        print(self.identity)

    def __len__(self):
        return len(self.identity)

    def __getitem__(self, index):
        filename = self.filename[index]
        target = self.identity[index][0]
        X = Image.open(os.path.join(self.root, self.base_folder, "img_align_celeba", filename))
        if self.transform is not None:
            X = self.transform(X)
        return X, target


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


class CutMix(Dataset):
    def __init__(self, dataset, num_class, num_mix=2, beta=1.0, prob=0.5):
        self.dataset = dataset
        self.num_class = num_class
        self.num_mix = num_mix
        self.beta = beta
        self.prob = prob

    def __getitem__(self, index):
        img, lb = self.dataset[index]
        lb_onehot = onehot(self.num_class, lb)

        for _ in range(self.num_mix):
            r = np.random.rand(1)
            if self.beta <= 0 or r > self.prob:
                continue

            # generate mixed sample
            lam = np.random.beta(self.beta, self.beta)
            rand_index = random.choice(range(len(self)))

            img2, lb2 = self.dataset[rand_index]
            lb2_onehot = onehot(self.num_class, lb2)

            bbx1, bby1, bbx2, bby2 = rand_bbox(img.size(), lam)
            img[:, bbx1:bbx2, bby1:bby2] = img2[:, bbx1:bbx2, bby1:bby2]
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (img.size()[-1] * img.size()[-2]))
            lb_onehot = lb_onehot * lam + lb2_onehot * (1. - lam)

        return img, lb_onehot

    def __len__(self):
        return len(self.dataset)


class MixUp(Dataset):
    def __init__(self, dataset, num_class, num_mix=2, beta=1.0, prob=0.5):
        self.dataset = dataset
        self.num_class = num_class
        self.num_mix = num_mix
        self.beta = beta
        self.prob = prob

    def __getitem__(self, index):
        img, lb = self.dataset[index]
        lb_onehot = onehot(self.num_class, lb)

        for _ in range(self.num_mix):
            r = np.random.rand(1)
            if self.beta <= 0 or r > self.prob:
                continue

            # generate mixed sample
            lam = np.random.beta(self.beta, self.beta)
            rand_index = random.choice(range(len(self)))

            img2, lb2 = self.dataset[rand_index]
            lb2_onehot = onehot(self.num_class, lb2)

            img = img * lam + img2 * (1-lam)
            lb_onehot = lb_onehot * lam + lb2_onehot * (1. - lam)

        return img, lb_onehot

    def __len__(self):
        return len(self.dataset)
