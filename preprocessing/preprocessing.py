import os
import cv2
import numpy as np
import face_alignment
from skimage import transform as trans
from tqdm import tqdm
import albumentations as A

# ==== CONFIG ====
INPUT_ROOT = './data'
OUTPUT_ROOT = './output_aligned_augmented'
MEAN_FACE_PATH = 'mean_face.npy'
IMAGE_SIZE = 112

# ==== LOAD mean_face & face_alignment ====
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device='cpu', flip_input=False)
mean_face = np.load(MEAN_FACE_PATH)

# ==== FACE ALIGNMENT ====
def align_face(img, landmarks, mean_shape, output_size=112):
    src = mean_shape.astype(np.float32)
    dst = landmarks.astype(np.float32)
    tform = trans.SimilarityTransform()
    tform.estimate(dst, src)
    M = tform.params[0:2, :]
    aligned = cv2.warpAffine(img, M, (output_size, output_size), borderValue=0.0)
    return aligned

# ==== AUGMENTATION PIPELINE ====
augmentation = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomResizedCrop(height=IMAGE_SIZE, width=IMAGE_SIZE, scale=(0.8, 1.0), p=0.5),
    A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.3),
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.3),
    A.ImageCompression(quality_lower=30, quality_upper=90, p=0.5),
    A.MotionBlur(blur_limit=3, p=0.3),
    A.Downscale(scale_min=0.7, scale_max=0.95, p=0.3),
])

# ==== PROCESSING ====
for root, dirs, files in tqdm(list(os.walk(INPUT_ROOT))):
    for file in files:
        if not file.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue

        img_path = os.path.join(root, file)
        image = cv2.imread(img_path)
        if image is None:
            continue

        preds = fa.get_landmarks(image)
        if preds is None:
            print(f"[!] No face found in {img_path}")
            continue

        landmarks = preds[0]
        aligned = align_face(image, landmarks, mean_face, output_size=IMAGE_SIZE)
        # augmented = augmentation(image=aligned)['image']

        # === Reconstruct output path ===
        rel_path = os.path.relpath(root, INPUT_ROOT)
        output_dir = os.path.join(OUTPUT_ROOT, rel_path)
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, file)

        cv2.imwrite(out_path, aligned)
        print(f"[✓] Saved: {out_path}")
