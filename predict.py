"""Inference script for MRI image denoising.

Usage:
    # Single image
    python predict.py --input noisy.png --output denoised.png --weights model_best.pth

    # Directory of images
    python predict.py --input ./data/test/DATA_noisy5 --output ./data/result --weights model_best.pth
"""

import os
import argparse
import torch
from PIL import Image
from torchvision.utils import save_image
import torchvision.transforms.functional as ttf

from models import mymodel


def predict_single(model, image_path, device):
    """Run inference on a single image, return output tensor."""
    img = Image.open(image_path)
    tensor = ttf.to_tensor(img).unsqueeze(0).to(device)  # (1, C, H, W)
    with torch.no_grad():
        output = model(tensor)
    return output


def main():
    parser = argparse.ArgumentParser(description="MRI Image Denoising Inference")
    parser.add_argument("--input", type=str, required=True, help="Path to input image or directory")
    parser.add_argument("--output", type=str, required=True, help="Path to output image or directory")
    parser.add_argument("--weights", type=str, default="./model_best.pth", help="Path to model weights")
    parser.add_argument("--device", type=str, default=None, help="Device: cuda:0 or cpu")
    args = parser.parse_args()

    # Device
    if args.device:
        device = args.device
    else:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Load model
    model = mymodel(in_channels=1).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()
    print(f"Model loaded from {args.weights}, device: {device}")

    input_path = args.input
    output_path = args.output

    if os.path.isfile(input_path):
        # Single image inference
        output = predict_single(model, input_path, device)
        save_image(output, output_path)
        print(f"Saved denoised image to {output_path}")

    elif os.path.isdir(input_path):
        # Batch inference on directory
        os.makedirs(output_path, exist_ok=True)
        image_files = sorted([f for f in os.listdir(input_path)
                              if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'))])

        print(f"Processing {len(image_files)} images...")
        for filename in image_files:
            img_path = os.path.join(input_path, filename)
            output = predict_single(model, img_path, device)
            out_file = os.path.join(output_path, os.path.splitext(filename)[0] + '.png')
            save_image(output, out_file)

        print(f"Saved {len(image_files)} denoised images to {output_path}")

    else:
        print(f"Error: {input_path} is not a valid file or directory")
        return


if __name__ == '__main__':
    main()
