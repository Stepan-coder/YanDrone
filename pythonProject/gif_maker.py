import os
import cv2
import imageio

images = []

for i in range(len(os.listdir(os.path.join(os.getcwd(), 'new_video'))) - 1):
    print(f"filename={i}-", os.path.join(os.getcwd(), 'new_video', f"video_{i}.jpg"))
    images.append(imageio.imread(os.path.join(os.getcwd(), 'new_video', f"video_{i}.jpg")))

imageio.mimsave(os.path.join(os.getcwd(), 'drone_detecting_yolov5m-set.gif'), images, duration = 0.04)
#
# import os
# import cv2
# import numpy as np
# from PIL import Image
#
# # Set the paths
# ars_img_dir = os.path.join(os.getcwd(), 'Ars_img')
# output_dir = os.path.join(os.getcwd(), 'ars_final_small')
#
# # Load templates
# background_template = Image.open("background_tg.png").convert("RGBA")
# ars_template = Image.open("ars_template.png").convert("RGBA")
#
# # Process images
# for i in range(69, len(os.listdir(ars_img_dir))):
#     face_path = os.path.join(ars_img_dir, f"wtf_{i}.png")
#     print(i)
#     if os.path.exists(face_path):
#         background = background_template.copy()  # Use a copy of the background template
#         face = Image.open(face_path).convert("RGBA")  # Convert to RGBA
#
#         # Resize face image
#         face = face.resize((807, 807))
#
#         # Paste images
#         background.paste(face, (120, 240), face)
#         background.paste(ars_template, (0, 0), ars_template)
#
#         # Save the result
#         output_path = os.path.join(output_dir, f"wtf_{i}.png")
#         background.save(output_path)