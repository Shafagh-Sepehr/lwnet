import torch.nn as nn


def initialize_fr_weights(module):
    """Apply the shared FR-family initialization contract."""
    for child in module.modules():
        if isinstance(child, nn.Conv2d):
            nn.init.kaiming_normal_(
                child.weight, mode="fan_out", nonlinearity="relu"
            )
        elif isinstance(child, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(
                child.weight, mode="fan_out", nonlinearity="relu"
            )
        elif isinstance(child, nn.BatchNorm2d):
            nn.init.constant_(child.weight, 1)
            nn.init.constant_(child.bias, 0)
