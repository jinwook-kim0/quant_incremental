
class TestBench:
    def __init__(self, name, func, *kargs, **kwargs):
        self.name = name
        self.func = func
        self.kargs = kargs
        self.kwargs = kwargs.copy()


    def run(self, *kargs, **kwargs):
        return self.func(*kargs, *self.kargs, **kwargs, **self.kwargs)
