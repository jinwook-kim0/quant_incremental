import numpy as np
import math
import torch
from PIL import Image

class ScaleClipping:
    def __init__(self, scale, min=0, max=255):
        self.scale = scale
        self.min = min
        self.max = max

    def set_scale(self, scale):
        self.scale = scale

    def __call__(self, img):
        imnp = np.clip(np.round(np.array(img) * self.scale), self.min, self.max)
        return np.array(imnp, dtype=np.uint8)


class RadicalGradation:
    def __init__(self, imsize=(3, 32, 32), center=(0, 0), radius=1, value=(1.0, 0), mi = 0, ma = 255, dist=lambda x: x, randtr=False):
        self.imsize = imsize
        self.center = center
        self.sd = min(self.imsize[1:])
        self.radius = radius / self.sd
        self.value = value
        self.min = mi
        self.max = ma

        self.dist = dist
        self.map = self.rotate(self.scaling_map())
        self.randtr = randtr

    def set_param(self, center, radius, value=(1.0, 0)):
        self.center = center
        self.radius = radius
        self.value = value
        self.map = self.rotate(self.scaling_map())

    def set_dist(self, dist):
        self.dist = dist
        self.map = self.roatate(self.scaling_map())

    def get_coef(self, i, j, verbose=False):
        d = self.dist(math.sqrt((i - self.center[0]) ** 2 + (j - self.center[1]) ** 2) / self.sd / self.radius)
        if verbose: print(d)
        return min(max((self.value[1] - self.value[0]) * d + self.value[0], min(self.value)), max(self.value))

    def rotate(self, x):
        ret = []
        ret.append(x)
        shape = x.shape
        ret.append(np.flip(x, 0))
        ret.append(np.flip(x, 1))
        ret.append(np.flip(np.flip(x, 0), 1))
        return ret


    def scaling_map(self):
        mu = np.zeros(self.imsize[1:])
        for i in range(self.imsize[1]):
            for j in range(self.imsize[2]):
                mu[i, j] = self.get_coef(i, j)
        return np.stack([mu for x in range(self.imsize[0])], axis=2)

    def __call__(self, img):
        mapsel = self.map[0] if not self.randtr else self.map[np.random.randint(0, 4)]

        if isinstance(img, torch.Tensor):
            mapt = torch.Tensor(np.moveaxis(mapsel, -1, 0)).cuda()
            return img.mul(mapt).clip(min=0, max=1.0)
        else:
            return Image.fromarray(np.uint8(np.clip(np.round(np.array(img) * self.map[0]), self.min, self.max)))

    def __str__(self):
        return f'Center: {self.center}, R: {self.radius * self.sd}, V: {self.value}'

