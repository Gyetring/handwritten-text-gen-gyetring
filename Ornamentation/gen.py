import os
import argparse
import torch
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm

from ornamentation_network import OrnamentationUNet


# =========================
# Device
# =========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# Main
# =========================
def main(args):
    input_dir = args.input_dir
    output_dir = args.output_dir
    checkpoint_path = args.checkpoint

    os.makedirs(output_dir, exist_ok=True)

    # Load model
    model = OrnamentationUNet().to(DEVICE)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Transform (must match training)
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),  # [1,64,64], [0,1]
    ])

    filenames = sorted([
        f for f in os.listdir(input_dir)
        if f.lower().endswith(".png")
    ])

    with torch.no_grad():
        for filename in tqdm(filenames, ncols=100):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            # Load image
            img = Image.open(input_path)
            x = transform(img).unsqueeze(0).to(DEVICE)  # [1,1,64,64]

            # Forward
            pred = model(x)

            # Tensor -> Image
            pred_img = pred.squeeze(0).squeeze(0).cpu().numpy()
            pred_img = (pred_img * 255).astype("uint8")

            Image.fromarray(pred_img).save(output_path)

    print(f"✅ Results saved to {output_dir}")


# =========================
# Entry
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ornamentation UNet inference script"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing input skeletonized images"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save reconstructed images"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pth",
        help="Path to model checkpoint"
    )

    args = parser.parse_args()
    main(args)
