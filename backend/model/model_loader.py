import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np

# Load ResNet18 model (pretrained) and remove final layer to get embeddings
def load_resnet18():
    model = models.resnet18(pretrained=True)
    model.eval()
    model = torch.nn.Sequential(*list(model.children())[:-1])
    return model

# Transform for images: resize, grayscale to 3 channels, normalize
def get_transform():
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

# Convert a patch image to embedding
def patch_to_embedding(patch_path, model, transform):
    from PIL import Image
    import torch

    try:
        im = Image.open(patch_path).convert('RGB')
    except Exception as e:
        raise RuntimeError(f"Failed to open patch {patch_path}: {e}")

    tensor = transform(im).unsqueeze(0)
    with torch.no_grad():
        emb = model(tensor)
    return emb.squeeze().reshape(-1).cpu().numpy()
