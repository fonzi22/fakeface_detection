import cv2
import dlib
import os
import numpy as np
import argparse
from tqdm import tqdm

# Argument parser for configurable behavior
parser = argparse.ArgumentParser(description="Detect facial attributes and apply black masks, either individually or combined")
parser.add_argument('--input_root', type=str, default='data', help='Folder containing input images')
parser.add_argument('--output_root', type=str, default='data', help='Root folder for outputs')
parser.add_argument('--att_margin', type=int, default=5, help='Margin around attributes for cropping/masking')
parser.add_argument('--face_margin', type=int, default=10, help='Margin around face for cropping/masking')
args = parser.parse_args()

# Paths and parameters
input_root = args.input_root
output_root = args.output_root
att_margin = args.att_margin
face_margin = args.face_margin

# Initialize detectors
print("Loading face detector and landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Helper: compute bounding box
def get_bounding_box(points, img_shape, margin=0):
    x_coords = [p.x for p in points]
    y_coords = [p.y for p in points]
    x_min = max(min(x_coords) - margin, 0)
    x_max = min(max(x_coords) + margin, img_shape[1])
    y_min = max(min(y_coords) - margin, 0)
    y_max = min(max(y_coords) + margin, img_shape[0])
    return x_min, y_min, x_max, y_max

# Facial landmark index ranges
coords = {
    'eyes': list(range(36, 48)),
    'nose': list(range(27, 36)),
    'mouth': list(range(48, 68))
}

# Iterate through dataset
to_iterate = list(os.walk(input_root))
for root, dirs, files in tqdm(to_iterate, total=len(to_iterate)):
    for file in files:
        if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        # Read image
        img_path = os.path.join(root, file)
        image = cv2.imread(img_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rel_dir = os.path.relpath(root, input_root)

        # Detect faces
        faces = detector(gray)
        if not faces:
            continue
        face = faces[0]

        # Get landmarks
        landmarks = predictor(gray, face)
        masked_img = np.zeros_like(image)
        
        for coord in coords.values():
            pts = [landmarks.part(i) for i in coord]
            bbox = get_bounding_box(pts, image.shape, margin=att_margin)
            x1, y1, x2, y2 = bbox
            masked_img[y1:y2, x1:x2] = image[y1:y2, x1:x2]
            

        x1 = face.left() - face_margin
        y1 = face.top() - face_margin + 10
        x2 = face.right() + face_margin
        y2 = face.bottom() + face_margin
        face_crop = image[max(y1,0):min(y2,image.shape[0]), max(x1,0):min(x2,image.shape[1])]
        masked_face = masked_img[max(y1,0):min(y2,image.shape[0]), max(x1,0):min(x2,image.shape[1])]
        
        # Save cropped face
        out_dir = os.path.join(output_root, 'face', rel_dir)
        os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(os.path.join(out_dir, file), face_crop)
        
        # Save masked image
        out_dir = os.path.join(output_root, f'masked_face', rel_dir)
        os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(os.path.join(out_dir, file), masked_face)

print("Processing complete.")
