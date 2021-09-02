import torch
import torch.nn as nn
import torchvision
from .Densenet import densenet121


class Head(nn.Module):
    def __init__(self, head_elements, num_classes=1):
        super().__init__()
        self.fc1 = nn.Linear(head_elements, 256)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class Multihead(nn.Module):
    def __init__(self, num_classes, head_elements=28, base_model='densenet121', pretrained=True):
        super().__init__()
        self.num_classes = num_classes
        self.head_elements = head_elements
        self.base_model_out_size = num_classes * head_elements
        if base_model == 'densenet121':
            if pretrained:
                self.base_model = torchvision.models.densenet121(pretrained=True)
                num_ftrs = self.base_model.classifier.in_features
                self.base_model.classifier = nn.Linear(num_ftrs, self.base_model_out_size)
            else:
                self.base_model = densenet121(num_classes=self.base_model_out_size)

        self.relu = nn.ReLU(inplace=True)
        self.heads = nn.ModuleList([Head(head_elements=head_elements) for i in range(num_classes)])
        

    def forward(self, x):
        x = self.base_model(x)
        x = self.relu(x)
        a = []
        for i in range(self.num_classes):
            a.append(self.heads[i](x[:, i*self.head_elements:(i+1)*self.head_elements]).squeeze())
        
        return torch.stack(a, dim=1)

