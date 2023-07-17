import torch
import torchvision
import torchvision.transforms as tf
import models.trans as mtf
import random
import numpy as np


torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
batch_size = 128

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

normalize = tf.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])
rparams = {
    'ORG': {'center': (0, 0), 'radius': 32, 'value': (1.0, 1.0)},
    'R32_LU_4X': {'center': (0, 0), 'radius': 32, 'value': (4.0, 1.0)},
    'R16_LU_4X': {'center': (0, 0), 'radius': 16, 'value': (4.0, 1.0)},
    'R16_CE_4X': {'center': (8, 8), 'radius': 16, 'value': (4.0, 1.0)},
    'R32_LU_2X': {'center': (0, 0), 'radius': 32, 'value': (2.0, 1.0)},
    'R16_LU_2X': {'center': (0, 0), 'radius': 16, 'value': (2.0, 1.0)},
    'R16_CE_2X': {'center': (8, 8), 'radius': 16, 'value': (2.0, 1.0)},
    'R32_LU_0X': {'center': (0, 0), 'radius': 32, 'value': (0, 1.0)},
    'R16_LU_0X': {'center': (0, 0), 'radius': 16, 'value': (0, 1.0)},
    'R16_CE_0X': {'center': (8, 8), 'radius': 16, 'value': (0, 1.0)}
}

Names = list(rparams.keys())
rtrs = {k: mtf.RadicalGradation(dist=lambda x: x**2, **p) for k, p in rparams.items()}

train_ds_sets = {}
train_pds_sets = {}
train_ld_sets = {}
train_pld_sets = {}
test_ds_sets = {}
test_pds_sets = {}
test_ld_sets = {}
test_pld_sets = {}
test_ds_im_sets = {}
test_ld_im_sets = {}
train_dn = True
test_dn = True
train_trf = {}
def toNumpy(im):
    return np.array(im, dtype=np.uint8)

def get_loader(ds, seed=0, num_workers=2, shuffle=False, batch_size=batch_size):
    gen = torch.Generator()
    gen.manual_seed(seed)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, generator=gen, worker_init_fn=seed_worker)


for k in rtrs:
    train_ds_sets[k] = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=train_dn,
        transform=tf.Compose([
            rtrs[k],
            tf.RandomCrop(32, padding=4),
            tf.RandomHorizontalFlip(),
            tf.ToTensor(),
            normalize
        ]))
    train_ld_sets[k] = get_loader(train_ds_sets[k], seed=0)
    train_dn &= False

    train_pds_sets[k] = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=train_dn,
        transform=tf.Compose([
            tf.ToTensor()
        ]))
    train_pld_sets[k] = get_loader(train_pds_sets[k], seed=0)

    train_trf[k] = tf.Compose([rtrs[k], tf.RandomCrop(32, padding=4), tf.RandomHorizontalFlip(), normalize])


    test_ds_sets[k] = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=test_dn,
        transform=tf.Compose([
            rtrs[k],
            tf.ToTensor(),
            normalize
        ]))
    test_ld_sets[k] = get_loader(test_ds_sets[k], seed=1)
    test_dn &= False

    test_pds_sets[k] = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=test_dn,
        transform=tf.Compose([
            rtrs[k],
            tf.ToTensor(),
        ]))
    test_pld_sets[k] = get_loader(test_pds_sets[k], seed=1)

    gen_test = torch.Generator()
    gen_test.manual_seed(1)
    test_ds_im_sets[k] = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=False,
        transform=tf.Compose([
            rtrs[k],
            toNumpy
        ]))
    test_ld_im_sets[k] = get_loader(test_ds_im_sets[k], seed=1)


def merge(ds, *kargs, seed=1):
    gen = torch.Generator()
    gen.manual_seed(seed)
    mds = torch.utils.data.ConcatDataset([ds[x] for x in kargs])
    return get_loader(mds, seed=seed, shuffle=True), mds
