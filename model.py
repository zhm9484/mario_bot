import os
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F


def _init_w(ps):
    for p in ps:
        nn.init.kaiming_normal_(p.weight, nonlinearity="relu")


def _init_b(ps):
    for p in ps:
        nn.init.zeros_(p.bias)


class MarioModel(nn.Module):
    def __init__(self):
        super(MarioModel, self).__init__()
        # public params
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(6 * 6 * 32, 512)
        # actor params
        self.actor_fc2 = nn.Linear(512, 12)
        # critic params
        self.critic_fc2 = nn.Linear(512, 1)

        _init_w([self.conv1, self.conv2, self.conv3, self.conv4, self.fc1])
        nn.init.xavier_normal_(self.actor_fc2.weight)
        nn.init.xavier_normal_(self.critic_fc2.weight)
        _init_b([
            self.conv1, self.conv2, self.conv3, self.conv4, self.fc1,
            self.actor_fc2, self.critic_fc2
        ])

    def forward(self, x):
        # public forward
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)
        o = F.relu(self.fc1(x))
        # actor forward
        actor_o = self.actor_fc2(o)
        # critic forward
        critic_o = self.critic_fc2(o)

        return {"logits": actor_o, "value": critic_o.view(-1)}
