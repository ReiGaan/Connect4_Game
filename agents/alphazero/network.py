import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset



class InitialConvLayer(nn.Module):
    def __init__(self, in_ch=3, out_ch=128):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn   = nn.BatchNorm2d(out_ch)
    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class ResidualLayer(nn.Module):
    def __init__(self, channels=128):
        super().__init__()
        self.c1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(channels)
        self.c2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(channels)
    def forward(self, x):
        r = x
        x = F.relu(self.b1(self.c1(x)))
        x = self.b2(self.c2(x))
        return F.relu(x + r)


class OutputLayer(nn.Module):
    def __init__(self, board_h=6, board_w=7, res_channels=128):
        super().__init__()
        # Value head
        self.v_conv = nn.Conv2d(res_channels, 3, kernel_size=1)
        self.v_bn   = nn.BatchNorm2d(3)
        self.v_fc1  = nn.Linear(3 * board_h * board_w, 32)
        self.v_fc2  = nn.Linear(32, 1)
        # Policy head (global-avg → 7 logits)
        self.p_conv = nn.Conv2d(res_channels, 32, kernel_size=1)
        self.p_bn   = nn.BatchNorm2d(32)
        self.p_fc   = nn.Linear(32, board_w)

    def forward(self, x):
        # Value
        v = F.relu(self.v_bn(self.v_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.v_fc1(v))
        v = torch.tanh(self.v_fc2(v))
        # Policy
        p = F.relu(self.p_bn(self.p_conv(x)))          # [B,32,6,7]
        p = F.adaptive_avg_pool2d(p, 1).view(-1, 32)    # → [B,32]
        p = self.p_fc(p)                                # → [B,7]
        p = F.softmax(p, dim=1)
        return p, v


class Connect4Net(nn.Module):
    def __init__(self, num_residual_layers=4):
        super().__init__()
        self.init = InitialConvLayer()
        self.res_layers = nn.ModuleList([ResidualLayer() for _ in range(num_residual_layers)])
        self.out = OutputLayer()

    def forward(self, x):
        x = self.init(x)
        for layer in self.res_layers:
            x = layer(x)
        return self.out(x)
    
class CustomLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, target_value, pred_value, target_policy, pred_policy):
        # value: [B,1], policy: [B,7]
        value_loss = F.mse_loss(pred_value, target_value)
        policy_loss = torch.sum(-target_policy * torch.log(pred_policy + 1e-8), dim=1).mean()
        return value_loss + policy_loss


