import os
from PIL import Image
import pandas as pd
from tqdm import tqdm

def data_preprocess(images_dir='data/Trees', patches_dir='data/patches'):
    """Split images into 4x4 patches and save to patches_dir."""
    os.makedirs(patches_dir, exist_ok=True)
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    image_files.sort()

    records = []
    for img_file in tqdm(image_files, desc="Processing images"):
        src_path = os.path.join(images_dir, img_file)
        try:
            im = Image.open(src_path).convert('RGB')
        except Exception as e:
            print(f"Skipping {img_file}: cannot open ({e})")
            continue

        w, h = im.size
        if w < 896 or h < 896:
            print(f"Skipping {img_file}: smaller than 896x896 ({w}x{h})")
            continue
        im_cropped = im.crop((0, 0, 896, 896))

        pw = ph = 896 // 4
        base = os.path.splitext(img_file)[0]
        idx = 0

        for r in range(4):
            for c in range(4):
                patch = im_cropped.crop((c*pw, r*ph, (c+1)*pw, (r+1)*ph))
                patch_name = f"{base}_patch{idx}.png"
                patch_path = os.path.join(patches_dir, patch_name)
                try:
                    patch.save(patch_path)
                except Exception as e:
                    print(f"Failed to save patch {patch_path}: {e}")
                    continue
                records.append({'patch_path': patch_path, 'original_name': img_file})
                idx += 1

    df = pd.DataFrame(records)
    print(f"Created {len(df)} patches in {patches_dir}")
    return df
