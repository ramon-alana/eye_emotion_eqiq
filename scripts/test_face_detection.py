#!/usr/bin/env python3
"""测试人脸检测"""
import cv2
import mediapipe as mp
import sys

image_path = sys.argv[1] if len(sys.argv) > 1 else "data/demo_images/ben.jpg"

img = cv2.imread(image_path)
if img is None:
    print(f"无法加载图片: {image_path}")
    sys.exit(1)

print(f"图片尺寸: {img.shape}")

# 尝试不同的检测阈值
for threshold in [0.3, 0.4, 0.5, 0.6]:
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=threshold,
    )
    
    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)
    
    if results.multi_face_landmarks:
        print(f"✓ 检测到人脸 (阈值={threshold})")
        break
    else:
        print(f"✗ 未检测到人脸 (阈值={threshold})")

