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


# taken from https://github.com/shyamsn97/controllable-ncas/blob/master/controllable_nca/utils.py
def create_2d_circular_mask(h, w, center=None, radius=3):

    if center is None:  # use the middle of the image
        # center = (int(w / 2), int(h / 2))
        center = (
            np.random.randint(radius + 2, w - (radius + 2)),
            np.random.randint(radius + 2, h - (radius + 2)),
        )
        # center = (int(w/2), int(h/2))
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w - center[0], h - center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

    mask = dist_from_center <= radius
    return mask
