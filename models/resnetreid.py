import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights

class ResNetReID(nn.Module):
    def __init__(self, num_classes, dropout_p=0.5):
        super(ResNetReID, self).__init__()
        resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.base = nn.Sequential(*list(resnet.children())[:-2])
        self.gap = nn.AdaptiveAvgPool2d(1)
        feature_dim = resnet.fc.in_features
        
        self.bottleneck = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        
        self.dropout = nn.Dropout(p=dropout_p)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.base(x)
        x = self.gap(x)

        global_feat = x.view(x.shape[0], -1)
        bn_feat = self.bottleneck(global_feat)
        bn_feat = self.dropout(bn_feat)

        cls_score = self.classifier(bn_feat)
        
        if self.training:
            return bn_feat, cls_score
        else:
            return bn_feat
        
class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes, epsilon=0.1):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss