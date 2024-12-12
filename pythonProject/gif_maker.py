import os
import cv2
import imageio

images = []

directory_path = os.path.join(os.getcwd(), 'new_video')

for filename in os.listdir(directory_path):
    full_path = os.path.join(directory_path, filename)
    try:
        img = imageio.read(full_path)
        if img is not None:
            images.append(img)
    except IOError:
        print(f"Could not read file {full_path}")

if len(images) > 0:
    imageio.mimsave(os.path.join(os.getcwd(), 'drone_detection.gif'), images, duration=0.04)
else:
    print("No images found in directory.")