import os
import cv2
import tqdm
import torch
import warnings
import numpy as np

warnings.filterwarnings('ignore')

torch.device("mps")
model = torch.hub.load('yolov5', 'yolov5m', source='local')
model.conf = 0.3
cap = cv2.VideoCapture('videos/Гоночки.mp4')

class Position:
    """
    A class to represent a rectangular position defined
    by its minimum and maximum x and y coordinates.
    """

    def __init__(self, xmin: int, ymin: int, xmax: int, ymax: int):
        """
        Initializes the Position with the given coordinates.

        Parameters:
        xmin (int): The minimum x-coordinate (must be >= 0).
        ymin (int): The minimum y-coordinate (must be >= 0).
        xmax (int): The maximum x-coordinate (must be >= xmin).
        ymax (int): The maximum y-coordinate (must be >= ymin).
        """
        self._xmin = xmin
        self._ymin = ymin
        self._xmax = xmax
        self._ymax = ymax

    @property
    def center_x(self) -> int:
        """Calculates the x-coordinate of the center of the position."""
        return int((self.xmin + self.xmax) / 2)

    @property
    def center_y(self) -> int:
        """Calculates the y-coordinate of the center of the position."""
        return int((self.ymin + self.ymax) / 2)

    @property
    def xmin(self) -> int:
        """Gets the minimum x-coordinate."""
        return self._xmin

    @xmin.setter
    def xmin(self, xmin: int) -> None:
        """
        Sets the minimum x-coordinate.

        Parameters:
        xmin (int): The minimum x-coordinate (must be >= 0 and < xmax).
        """
        if not isinstance(xmin, int):
            raise TypeError(f"'xmin' must be 'int', but got {type(xmin).__name__}")
        if xmin < 0:
            raise ValueError(f"'xmin' must be greater than or equal to zero")
        if xmin >= self.xmax:
            raise ValueError(f"'xmin' must be less than 'xmax'")
        self._xmin = xmin

    @property
    def ymin(self) -> int:
        """Gets the minimum y-coordinate."""
        return self._ymin

    @ymin.setter
    def ymin(self, ymin: int) -> None:
        """
        Sets the minimum y-coordinate.

        Parameters:
        ymin (int): The minimum y-coordinate (must be >= 0 and < ymax).
        """
        if not isinstance(ymin, int):
            raise TypeError(f"'ymin' must be 'int', but got {type(ymin).__name__}")
        if ymin < 0:
            raise ValueError(f"'ymin' must be greater than or equal to zero")
        if ymin >= self.ymax:
            raise ValueError(f"'ymin' must be less than 'ymax'")
        self._ymin = ymin

    @property
    def xmax(self) -> int:
        """Gets the maximum x-coordinate."""
        return self._xmax

    @xmax.setter
    def xmax(self, xmax: int) -> None:
        """
        Sets the maximum x-coordinate.

        Parameters:
        xmax (int): The maximum x-coordinate (must be >= 0 and > xmin).
        """
        if not isinstance(xmax, int):
            raise TypeError(f"'xmax' must be 'int', but got {type(xmax).__name__}")
        if xmax < 0:
            raise ValueError(f"'xmax' must be greater than or equal to zero")
        if xmax <= self.xmin:
            raise ValueError(f"'xmax' must be greater than 'xmin'")
        self._xmax = xmax

    @property
    def ymax(self) -> int:
        """Gets the maximum y-coordinate."""
        return self._ymax

    @ymax.setter
    def ymax(self, ymax: int) -> None:
        """
        Sets the maximum y-coordinate.

        Parameters:
        ymax (int): The maximum y-coordinate (must be >= 0 and > ymin).
        """
        if not isinstance(ymax, int):
            raise TypeError(f"'ymax' must be 'int', but got {type(ymax).__name__}")
        if ymax < 0:
            raise ValueError(f"'ymax' must be greater than or equal to zero")
        if ymax <= self.ymin:
            raise ValueError(f"'ymax' must be greater than 'ymin'")
        self._ymax = ymax


def model_drone_detecting(frame, treshhold: float):
    detected = []
    output = model(frame)
    for i in range(len(output.pandas().xyxy[0]['name'])):
        label = output.pandas().xyxy[0]['name'][i]
        confidence = output.pandas().xyxy[0]['confidence'][i]
        if confidence >= treshhold:
            drone_pos = Position(xmin=int(output.pandas().xyxy[0]['xmin'][i]),
                                 ymin=int(output.pandas().xyxy[0]['ymin'][i]),
                                 xmax=int(output.pandas().xyxy[0]['xmax'][i]),
                                 ymax=int(output.pandas().xyxy[0]['ymax'][i]))
            detected.append(drone_pos)
    return detected

def draw_drone_area(frame, pos: Position, color):
    min_size = min(pos.xmax - pos.xmin, pos.ymax - pos.ymin)
    frame = cv2.rectangle(frame, (pos.xmin, pos.ymin), (pos.xmax, pos.ymax), color, 1)
    frame = cv2.line(frame,
             (int(position.x * k), pos.center_y),
             (int(position.x * k + position.width * k), pos.center_y),
             color, thickness)

    frame = cv2.circle(frame, (pos.center_x, pos.center_y), int(min_size * 0.1), color, 1)


is_crop = True
frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
with tqdm.tqdm(total=frames_count) as pbar:
    while cap.isOpened():
        ret, frame = cap.read()

        height = frame.shape[0]
        width = frame.shape[1]
        more_then_treshhold = False
        drones = []
        for h in range(int(height / 320)):
            for w in range(int(width / 320)):
                cropped_image = frame[h * 320: (h + 1) * 320, w * 320: (w + 1) * 320]
                detected = model_drone_detecting(cropped_image, 0.15)
                if len(detected) > 0:
                    more_then_treshhold = True
                for drone in detected:
                    drones.append(Position(xmin=w * 320 + drone.xmin,
                                           ymin=h * 320 + drone.ymin,
                                           xmax=w * 320 + drone.xmax,
                                           ymax=h * 320 + drone.ymax))
        for drone in drones:
            draw_drone_area(frame=frame, pos=drone, color=(0, 255, 0))

        cv2.imshow(f'frame', frame)
        # if more_then_treshhold:
        if is_crop:
            frame = cv2.resize(frame, None, fx=.5, fy=.5, interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(os.path.join(os.getcwd(), 'new_video', f"video_{pbar.n}.jpg"), frame)
        pbar.update(1)
        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()