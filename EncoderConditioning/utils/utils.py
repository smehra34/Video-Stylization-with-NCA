import torch
from PIL import Image
import torchvision.transforms as transforms

def load_image(img_path, size=64):
    # Load the image using PIL
    img = Image.open(img_path).convert('RGB')

    # Get the largest square possible by cropping
    width, height = img.size
    new_size = min(width, height)
    left = (width - new_size) // 2
    top = (height - new_size) // 2
    right = (width + new_size) // 2
    bottom = (height + new_size) // 2
    img = img.crop((left, top, right, bottom))

    # Resize the image to the specified size
    img = img.resize((size, size), Image.LANCZOS)

    # Convert PIL Image to Torch Tensor
    transform = transforms.ToTensor()
    img_tensor = transform(img)

    return img_tensor
