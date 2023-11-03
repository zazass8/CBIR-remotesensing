import torch.nn as nn
import torch
import torch.nn.functional as F


# Define a custom model that combines both feature extraction and class prediction
class CustomModel(nn.Module):
    def __init__(self, feature_extractor, class_predictor, classes = 3):
        super(CustomModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.class_predictor = class_predictor
        self.classes = classes

    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        class_output = self.class_predictor(features)


        # multi-class activation
        if self.classes == 3:
            class_output = F.softmax(class_output, dim=1)

        # binary activation
        else:
            class_output = torch.sigmoid(class_output)

        return features, class_output


# Define a custom model that combines both feature extraction and class prediction. Model from SCRATCH
class Scratch(nn.Module):
    def __init__(self, classes = 3):
        super(Scratch, self).__init__()
        self.classes = classes

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.class_predictor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 32 * 32, 128 * 32 * 16),  
            nn.ReLU(),
            # nn.Linear(256 * 32 * 16, 256 * 32 * 8),  
            # nn.ReLU(),
            # nn.Linear(128 * 64 * 16, 128 * 64 * 8),  
            # nn.ReLU(),
            # nn.Linear(128 * 64 * 8, 128 * 64 * 2),  
            # nn.ReLU(),
            nn.Linear(128 * 32 * 16, classes)
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        class_output = self.class_predictor(features)

        # multi-class activation
        if self.classes == 3:
            class_output = F.softmax(class_output, dim=1)

        # binary activation
        else:
            class_output = torch.sigmoid(class_output)

        return features, class_output
    