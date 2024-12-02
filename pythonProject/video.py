import os
import cv2
import torch
import warnings
import numpy as np

warnings.filterwarnings('ignore')

torch.device("mps")
model = torch.hub.load('yolov5', 'custom', path='best.pt', source='local', force_reload=True)
model.conf = 0.7
cap = cv2.VideoCapture('IMG_2037.MOV')

def model_drone_detecting(frame, treshhold: float):
    detected = []
    output = model(frame)
    for i in range(len(output.pandas().xyxy[0]['name'])):
        label = output.pandas().xyxy[0]['name'][i]
        confidence = output.pandas().xyxy[0]['confidence'][i]
        if label == 'drone' and confidence >= treshhold:
            xmin = int(output.pandas().xyxy[0]['xmin'][i])
            ymin = int(output.pandas().xyxy[0]['ymin'][i])
            xmax = int(output.pandas().xyxy[0]['xmax'][i])
            ymax = int(output.pandas().xyxy[0]['ymax'][i])
            print([xmin, ymin, xmax, ymax])
            detected.append([xmin, ymin, xmax, ymax])
    return detected


counter = 0
while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.convertScaleAbs(frame, alpha=1.1, beta=0)

    # Определение ядра для увеличения резкости
    sharpening_kernel = np.array([[0, -1, 0],
                                  [-1, 5, -1],
                                  [0, -1, 0]])

    # Применение фильтра к изображению
    frame = cv2.filter2D(frame, -1, sharpening_kernel)

    height = frame.shape[0]
    width = frame.shape[1]
    more_then_treshhold = False
    for h in range(int(height / 320)):
        for w in range(int(width / 320)):
            cropped_image = frame[h * 320: (h + 1) * 320, w * 320: (w + 1) * 320]
            # cv2.imshow(f'frame_{h}_{w}', cropped_image) # view of the small frames
            detected = model_drone_detecting(cropped_image, 0.4)
            if len(detected) > 0:
                more_then_treshhold = True
            for drone in detected:
                cv2.rectangle(frame,
                              (w * 320 + drone[0], h * 320 + drone[1]),
                              (w * 320 + drone[2], h * 320 + drone[3]), (0, 255, 0), 2)
    cv2.imshow(f'frame', frame)
    # if more_then_treshhold:
    cv2.imwrite(os.path.join(os.getcwd(), 'new_video', f"video_8280_{counter}>85.jpg"), frame)
    print(counter)
    counter += 1
    # img[80:280, 150:330]
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()