import torch
import torch.nn as nn

class PolicyNetwork(nn.Module):
    """
    Policy Network over continuous action set
    Archictecture: 3-64-64 backbone + mean & log_std output heads
    """
    def __init__(self, ):
        super().__init__()

        # backbone feature extractor
        self.backbone = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        # output mean and log_std heads
        self.mean_head = nn.Linear(64, 1)
        self.log_std_head = nn.Linear(64, 1)
    
    def forward(self, state):
        # features
        x = self.backbone(state)

        # output mean and clamped log_std (clamp log_std so that it doesnt explode)
        mean = self.mean_head(x)
        log_std = self.log_std_head(x).clamp(-20, 2) # this gives between ~0 and 7.5 (actual action range -2, 2)

        return mean, log_std
    