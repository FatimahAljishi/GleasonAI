import torch
import segmentation_models_pytorch as smp

# Build the model exactly as during training
model = smp.Unet(
    encoder_name="efficientnet-b4",
    encoder_weights=None,
    in_channels=3,
    classes=6,
    activation=None,
)

# Load your weights
weights = torch.load(
    "models/unet_epoch8_weights.pt",
    map_location="cpu",
    weights_only=True,
)

model.load_state_dict(weights)
model.eval()

# Dummy input
dummy = torch.randn(1, 3, 512, 512)

# Export
torch.onnx.export(
    model,
    dummy,
    "unet_epoch8.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch"},
        "output": {0: "batch"},
    },
    opset_version=17,
)

print("Done!")
