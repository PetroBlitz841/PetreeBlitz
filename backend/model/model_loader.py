import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import torch.nn.functional as F

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

def load_classifier():
    # Load ResNet18 with the classification head intact
    model = models.resnet18(pretrained=True)
    model.eval()
    return model

def get_top_predictions(output, top_k=3):
    # Convert raw model output (logits) to probabilities
    probabilities = F.softmax(output, dim=1)
    top_prob, top_catid = torch.topk(probabilities, top_k)
    
    # In a real app, you'd map top_catid to actual class names
    results = []
    for i in range(top_prob.size(1)):
        results.append({
            "label": f"Class_{top_catid[0][i].item()}", 
            "confidence": float(top_prob[0][i].item())
        })
    return results