import cv2
import os
import glob
import numpy as np
from pathlib import Path
import random

def apply_motion_blur(image, kernel_size=15, angle=None):
    """Applies a directional motion blur to the given image patch."""
    if angle is None:
        angle = random.uniform(0, 360)
        
    # Create a motion blur kernel
    kernel = np.zeros((kernel_size, kernel_size))
    center = kernel_size // 2
    
    # Draw a line in the matrix to represent the blur direction
    cv2.ellipse(kernel, (center, center), (center, 0), angle, 0, 360, 1, thickness=1)
    kernel = kernel / np.sum(kernel)
    
    # Apply the kernel to the image
    blurred = cv2.filter2D(image, -1, kernel)
    return blurred

def process_dataset(images_dir, labels_dir, output_images_dir, output_labels_dir):
    """
    Reads YOLO dataset, applies motion blur specifically to tennis ball bounding boxes,
    and saves the augmented images.
    """
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)
    
    image_files = glob.glob(os.path.join(images_dir, "*.jpg")) + glob.glob(os.path.join(images_dir, "*.png"))
    
    for img_path in image_files:
        filename = os.path.basename(img_path)
        name, ext = os.path.splitext(filename)
        label_path = os.path.join(labels_dir, f"{name}.txt")
        
        # We only augment images that have a label
        if not os.path.exists(label_path):
            continue
            
        # 30% chance to augment an image to ensure dataset variety
        if random.random() > 0.3:
            continue
            
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        h, w = img.shape[:2]
        
        with open(label_path, 'r') as f:
            lines = f.readlines()
            
        augmented = False
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
                
            class_id = int(parts[0])
            # Assuming class 0 is tennis ball
            if class_id == 0:
                cx, cy, bw, bh = map(float, parts[1:5])
                
                # Convert normalized coordinates to pixel coordinates
                x_center, y_center = int(cx * w), int(cy * h)
                box_w, box_h = int(bw * w), int(bh * h)
                
                # Expand the box slightly to blur the edges as well
                pad = max(5, int(box_w * 0.5))
                x1 = max(0, x_center - box_w // 2 - pad)
                y1 = max(0, y_center - box_h // 2 - pad)
                x2 = min(w, x_center + box_w // 2 + pad)
                y2 = min(h, y_center + box_h // 2 + pad)
                
                if x2 <= x1 or y2 <= y1:
                    continue
                    
                # Extract the patch, apply motion blur, and paste it back
                patch = img[y1:y2, x1:x2]
                
                # Blur intensity based on box size (larger box = closer ball = more blur)
                kernel_size = max(5, min(31, int(box_w * 0.8)))
                # Make kernel size odd
                if kernel_size % 2 == 0: kernel_size += 1
                
                blurred_patch = apply_motion_blur(patch, kernel_size=kernel_size)
                
                # Blend the edges to avoid harsh square boundaries
                mask = np.zeros_like(patch, dtype=np.float32)
                cv2.rectangle(mask, (pad//2, pad//2), (patch.shape[1]-pad//2, patch.shape[0]-pad//2), (1, 1, 1), -1)
                mask = cv2.GaussianBlur(mask, (pad|1, pad|1), 0)
                
                img[y1:y2, x1:x2] = (blurred_patch * mask + patch * (1 - mask)).astype(np.uint8)
                augmented = True
                
        if augmented:
            out_img_name = f"{name}_aug_blur{ext}"
            out_lbl_name = f"{name}_aug_blur.txt"
            
            cv2.imwrite(os.path.join(output_images_dir, out_img_name), img)
            
            # Copy the label file for the augmented image
            with open(os.path.join(output_labels_dir, out_lbl_name), 'w') as f_out:
                f_out.writelines(lines)
                
if __name__ == "__main__":
    print("This script applies targeted motion blur to tennis ball bounding boxes in your YOLO dataset.")
    print("It generates new augmented images to help YOLO recognize blurry balls.")
    print("Usage example:")
    print("process_dataset('datasets/tennis/images/train', 'datasets/tennis/labels/train', 'datasets/tennis/images/train_aug', 'datasets/tennis/labels/train_aug')")
