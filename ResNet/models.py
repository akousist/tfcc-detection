import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ResNet18(nn.Module):
    """
    Args:
        num_labels (int): Number of classification labels.
    """
    def __init__(self, num_labels, pretrained):
        super().__init__()
        self.compress = nn.Conv2d(in_channels=8, out_channels=3, kernel_size=3, stride=(1, 1), padding=(1, 1))
        self.resnet = nn.Sequential(*list(models.resnet18(pretrained=pretrained).children())[:-1])
        self.classifier = nn.Linear(in_features=512, out_features=num_labels)
        
    def forward(self, x):
        x = self.compress(x)
        x = self.resnet(x)
        x = torch.flatten(x, start_dim=1)
        logits = self.classifier(x) 
        log_p = F.log_softmax(logits, dim=-1)

        return log_p

class ResNet18_compress1(nn.Module):
    """
    Args:
        num_labels (int): Number of classification labels.
    """
    def __init__(self, num_labels, pretrained):
        super().__init__()
        self.compress = nn.Conv2d(in_channels=8, out_channels=3, kernel_size=1, padding=0)
        self.resnet = nn.Sequential(*list(models.resnet18(pretrained=pretrained).children())[:-1])
        self.classifier = nn.Linear(in_features=512, out_features=num_labels)
        
    def forward(self, x):
        x = self.compress(x)
        x = self.resnet(x)
        x = torch.flatten(x, start_dim=1)
        logits = self.classifier(x) 
        log_p = F.log_softmax(logits, dim=-1)

        return log_p