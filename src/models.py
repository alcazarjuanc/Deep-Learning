import torch
import torch.nn as nn


class MetadataMLP(nn.Module):
    """
    Modelo MLP para clasificación usando únicamente metadatos.
    """

    def __init__(self, input_dim, hidden_dim=64, num_classes=2):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.3),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, metadata):
        return self.network(metadata)


class ImageCNN(nn.Module):
    """
    Modelo CNN para clasificación usando únicamente imágenes.
    """

    def __init__(self, num_classes=2):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, image):
        x = self.features(image)
        x = self.classifier(x)
        return x


class HybridCNNMetadata(nn.Module):
    """
    Modelo híbrido que combina imágenes y metadatos para clasificación binaria.
    """

    def __init__(self, metadata_input_dim, num_classes=2):
        super().__init__()

        self.image_features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.image_projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=0.3)
        )

        self.metadata_projection = nn.Sequential(
            nn.Linear(metadata_input_dim, 64),
            nn.ReLU(),
            nn.Dropout(p=0.3),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(64 + 32, 64),
            nn.ReLU(),
            nn.Dropout(p=0.4),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, num_classes)
        )

    def forward(self, image, metadata):
        image_features = self.image_features(image)
        image_features = self.image_projection(image_features)

        metadata_features = self.metadata_projection(metadata)

        combined_features = torch.cat(
            [image_features, metadata_features],
            dim=1
        )

        output = self.classifier(combined_features)

        return output