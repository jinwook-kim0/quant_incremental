import torch

class TensorQ(torch.Tensor):
    def __init__(self, *kargs, amp=1.0, bw=4, **kwargs):
        super().__init__(*kargs, **kwargs)

        self._amp = amp
        self._bw = bw
