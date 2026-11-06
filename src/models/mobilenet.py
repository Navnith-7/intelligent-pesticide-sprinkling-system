
import torch
from torch import nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


class PlantDiseaseMobileNetV2(nn.Module):
    def __init__(self, num_classes=38, pretrained=True):
        super().__init__()

        weights = (
            MobileNet_V2_Weights.DEFAULT
            if pretrained
            else None
        )

        self.model = mobilenet_v2(weights=weights)

        # Initially freeze the pretrained feature extractor.
        for parameter in self.model.features.parameters():
            parameter.requires_grad = False

        # Replace the original 1000-class ImageNet classifier.
        input_features = self.model.classifier[1].in_features

        self.model.classifier[1] = nn.Linear(
            input_features,
            num_classes,
        )

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = PlantDiseaseMobileNetV2(num_classes=38).to(device)

    dummy_input = torch.randn(2, 3, 224, 224, device=device)

    model.eval()
    with torch.inference_mode():
        output = model(dummy_input)

    total_parameters = sum(
        parameter.numel() for parameter in model.parameters()
    )
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(f"Device: {device}")
    print(f"Output shape: {tuple(output.shape)}")
    print(f"Total parameters: {total_parameters:,}")
    print(f"Trainable parameters: {trainable_parameters:,}")
    print("SUCCESS: MobileNetV2 model check passed.")
