import Ornamentation.preprocess
import SDT
import Ornamentation
from PIL import Image
import os
import subprocess
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run generation for a specific user")
    parser.add_argument(
        "--user_id",
        type=int,
        required=True,
        help="User ID, e.g. 1, 2, 3"
    )
    args = parser.parse_args()

    user_id = args.user_id
    print(f"Using user_id = {user_id}")

    # 构造 style_samples/user_{user_id}
    style_dir = os.path.join("style_samples", f"user_{user_id}")

    if not os.path.exists(style_dir):
        raise FileNotFoundError(f"{style_dir} does not exist")

    print(f"Style directory: {style_dir}")

    input_dir = f"style_samples/user_{user_id}"
    skeletons_dir = f"skeletons/user_{user_id}"

    os.makedirs(skeletons_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".png"):
            input_path = os.path.join(input_dir, filename)
            skeleton_img_dir = os.path.join(skeletons_dir, filename)

            img = Image.open(input_path)

            skeleton_img = Ornamentation.preprocess.my_skeletonize(img)

            skeleton_img_pil = Image.fromarray(skeleton_img)

            skeleton_img_pil.save(skeleton_img_dir)

    # print("✅ Skeletonization 完成")

    ROOT = os.path.dirname(os.path.abspath(__file__))

    # SDT 目录（user_generate.py 默认假设的 cwd）
    SDT_DIR = os.path.join(ROOT, 'SDT')

    os.chdir(SDT_DIR)

    style_path = f'../skeletons/user_{user_id}'
    sdt_gen_dir   = f'../sdt_gen/user_{user_id}'

    cmd = [
        sys.executable,
        'user_generate.py',
        '--pretrained_model', 'fine_trained.pth',
        '--style_path', style_path,
        '--dir', sdt_gen_dir,
    ]

    subprocess.run(cmd, check=True)

    ORNAMENTATION_DIR = os.path.join(ROOT,'Ornamentation')

    os.chdir(ORNAMENTATION_DIR)

    ornamented_dir = f'../ornamented/user_{user_id}'

    cmd = [
        sys.executable,
        'gen.py',
        '--input_dir', sdt_gen_dir,
        '--output_dir', ornamented_dir,
    ]

    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()