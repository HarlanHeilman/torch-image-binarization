from argparse import ArgumentParser
from os.path import splitext

from PIL import Image
import torch
from torchvision.transforms.functional import rgb_to_grayscale, to_tensor

from torch_image_binarization.thresholding import su


def run_binarize(img_path, output_path=None, device="cuda"):
    file_path, file_ext = splitext(img_path)
    if not output_path:
        output_path = f"{file_path}-binarized{file_ext}"
    # Read image using PIL instead of torchvision to support more formats
    img = Image.open(img_path)
    with torch.inference_mode():
        img = to_tensor(img)
        img = rgb_to_grayscale(img)
        img = img.to(device)
        result = su(img)
        result_img = Image.fromarray(
            (1 - result.cpu().squeeze(0).numpy().astype("uint8")) * 255
        )
    result_img.save(output_path)
    return output_path


def main():
    parser = ArgumentParser()
    parser.add_argument("img", metavar="FILE")
    args = parser.parse_args()
    run_binarize(args.img)


if __name__ == "__main__":
    main()
