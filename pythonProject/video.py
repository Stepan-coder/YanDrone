import os
import cv2
import torch
import warnings

warnings.filterwarnings('ignore')

model = torch.hub.load('yolov5', 'custom', path='best.pt', source='local', force_reload=True)
model.conf = 0.5
cap = cv2.VideoCapture('Гоночки.mp4')

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
    # height = frame.shape[0]
    # width = frame.shape[1]
    # for h in range(int(height / 320)):
    #     for w in range(int(width / 320)):
    #         cropped_image = frame[h * 320: (h + 1) * 320, w * 320: (w + 1) * 320]
    #         # cv2.imshow(f'frame_{h}_{w}', cropped_image) # view of the small frames
    #         detected = model_drone_detecting(cropped_image, 0.1)
    #         for drone in detected:
    #             cv2.rectangle(frame,
    #                           (w * 320 + drone[0], h * 320 + drone[1]),
    #                           (w * 320 + drone[2], h * 320 + drone[3]), (0, 255, 0), 2)
    cv2.imshow(f'frame', frame)
    cv2.imwrite(os.path.join(os.getcwd(), 'video', f"video_{counter}.jpg"), frame)
    print(counter)
    counter += 1
    # img[80:280, 150:330]
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()