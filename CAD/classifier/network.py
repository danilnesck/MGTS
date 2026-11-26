from torch import nn
from torchvision import transforms
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


def create_classifier(num_classes):
    model = mobilenet_v3_small(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model


def image_transform():
    weights = MobileNet_V3_Small_Weights.DEFAULT.transforms()
    return transforms.Compose([
        transforms.Resize((224, 224), interpolation=weights.interpolation, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(weights.mean, weights.std),
    ])
