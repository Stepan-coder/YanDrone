import os
import cv2
import torch


number_of_drones = 0


model = torch.hub.load('yolov5', 'custom', path='models/best.pt', source='local', force_reload=True)
model.conf = 0.1

for file in os.listdir('drones'):
    output = model(os.path.join(os.getcwd(), 'drones', file))
    print(">>>", output.pandas().xyxy[0], len(output.pandas().xyxy[0]['xmin']))
    drone_positive = output.pandas().xyxy[0]['name']
    confidence_sent = output.pandas().xyxy[0]['confidence']
    number_of_drones = output.pandas().xyxy[0].value_counts('name')[0]
    drone_positive = drone_positive[0]
    if drone_positive == "drone":
       # Получение координат рамок
        bbox = output.pandas().xyxy[0][['xmin', 'ymin', 'xmax', 'ymax']].values
        # Загружаем изображение с помощью OpenCV
        img = cv2.imread(os.path.join(os.getcwd(), 'drones', file))
        # Рисуем квадраты вокруг найденных объектов
        for box in bbox:
            xmin, ymin, xmax, ymax = map(int, box)
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)  # Рисуем зеленый квадрат
        # Сохранение или отображение результата
        cv2.imshow('Detected Drones', img)  # Отображаем изображение
        # print(os.path.join('recognized_drones', file))
        cv2.imwrite(os.path.join(os.getcwd(), 'recognized_drones', file), img)

cv2.waitKey(0)  # Ждем нажатия клавиши
cv2.destroyAllWindows()  # Закрываем окна

# python train.py --img 925 --batch 8 --epochs 5 --data /Users/stepanborodin/Desktop/Projects/Yan/YanDrone/pythonProject/YOLOv5_dataset/data.yaml --weights yolov5l.pt  --nosave --cache