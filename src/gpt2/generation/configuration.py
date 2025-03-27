class GenerateConfig(object):
    def __init__(self, context_len: int, nucleus_prob: float, use_gpu: bool):
        self.context_len = context_len
        self.nucleus_prob = nucleus_prob
        self.use_gpu = use_gpu
