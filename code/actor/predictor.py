import os
import torch

class TorchPredictor(object):
    def __init__(self, net):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = net.to(self.device)
        self.model_path = None

    def load_model(self, model_path):
        self.model_path = model_path
        ckpt = torch.load(self.model_path, map_location=self.device)
        self.net.load_state_dict(ckpt["network_state_dict"])

    def inference(self):
        self.net.eval()
        # TODO:
        pass