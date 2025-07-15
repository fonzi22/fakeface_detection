import os
import cv2
import dlib
import numpy as np
from tqdm import tqdm
import albumentations as A

# ==== CONFIG ====
INPUT_ROOT = "input_images"
OUTPUT_ROOT = "output_aligned_augmented"
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
IMAGE_SIZE = 112
FACE_MARGIN = 20

# ==== KHỞI TẠO ====
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

# ==== AUGMENTATION PIPELINE ====
augment = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomResizedCrop(height=IMAGE_SIZE, width=IMAGE_SIZE, scale=(0.8, 1.0), p=0.5),
    A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.3),
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.3),
    A.ImageCompression(quality_lower=30, quality_upper=90, p=0.5),
    A.MotionBlur(blur_limit=3, p=0.3),
    A.Downscale(scale_min=0.7, scale_max=0.95, p=0.3),
])

# ==== HÀM ALIGNMENT DỰA TRÊN 2 MẮT ====
def align_by_eyes(image, landmarks, output_size=112):
    left_eye_pts = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
    right_eye_pts = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])

    left_eye_center = left_eye_pts.mean(axis=0)
    right_eye_center = right_eye_pts.mean(axis=0)

    # Góc nghiêng giữa 2 mắt
    dx = right_eye_center[0] - left_eye_center[0]
    dy = right_eye_center[1] - left_eye_center[1]
    angle = np.degrees(np.arctan2(dy, dx))

    # Tâm khuôn mặt
    eyes_center = tuple(((left_eye_center + right_eye_center) / 2).astype(int))

    # Affine rotation
    M = cv2.getRotationMatrix2D(eyes_center, angle, scale=1.0)
    rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_CUBIC)

    # Cắt vùng quanh mặt (với margin)
    x, y, w, h = landmarks.rect.left(), landmarks.rect.top(), landmarks.rect.width(), landmarks.rect.height()
    x = max(x - FACE_MARGIN, 0)
    y = max(y - FACE_MARGIN, 0)
    w = min(w + 2 * FACE_MARGIN, rotated.shape[1] - x)
    h = min(h + 2 * FACE_MARGIN, rotated.shape[0] - y)
    face_crop = rotated[y:y + h, x:x + w]
    resized = cv2.resize(face_crop, (output_size, output_size))
    return resized

# ==== XỬ LÝ TẬP ẢNH ====
for root, _, files in tqdm(os.walk(INPUT_ROOT)):
    for fname in files:
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        img_path = os.path.join(root, fname)
        image = cv2.imread(img_path)
        if image is None:
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        if len(faces) == 0:
            print(f"[!] No face: {img_path}")
            continue

        face = faces[0]
        landmarks = predictor(gray, face)
        aligned = align_by_eyes(image, landmarks, output_size=IMAGE_SIZE)
        # augmented = augment(image=aligned)['image']

        # === Tạo thư mục đầu ra giữ cấu trúc ===
        rel_dir = os.path.relpath(root, INPUT_ROOT)
        save_dir = os.path.join(OUTPUT_ROOT, rel_dir)
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, fname)
        cv2.imwrite(out_path, aligned)
        print(f"[✓] Saved: {out_path}")
