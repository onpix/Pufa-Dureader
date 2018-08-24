import torch 
import numpy 
import torch.nn as nn
import my_rc_model

class FC(nn.Module):
    def __init__(self):
        super(FC, self).__init__()
        self.layer1 = nn.Linear()
