"""Data augmentation utility: rotate all images in a directory 90 degrees clockwise."""

import os
import argparse
from PIL import Image
from concurrent.futures import ThreadPoolExecutor


def rotate_and_save_image(input_dir, filename):
    input_path = os.path.join(input_dir, filename)
    try:
        img = Image.open(input_path)
        img_rotated = img.transpose(Image.ROTATE_270)  # 90 degrees clockwise
        output_filename = f"{os.path.splitext(filename)[0]}_90L.png"
        output_path = os.path.join(input_dir, output_filename)
        img_rotated.save(output_path)
        print(f"Processed {filename} -> {output_filename}")
    except Exception as e:
        print(f"Error processing {filename}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Rotate all PNG images in a directory 90 degrees clockwise")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing images to rotate")
    args = parser.parse_args()

    input_dir = args.input_dir
    filenames = [f for f in os.listdir(input_dir) if f.endswith('.png')]

    with ThreadPoolExecutor() as executor:
        executor.map(lambda f: rotate_and_save_image(input_dir, f), filenames)


if __name__ == '__main__':
    main()
