import os
from datetime import datetime

import cv2

IP_WEBCAM_IP = "https://192.168.178.122:8080/video"
IMAGES_DIRECTORY = "images"

if __name__ == "__main__":
    os.makedirs(IMAGES_DIRECTORY, exist_ok=True)

    cap = cv2.VideoCapture(IP_WEBCAM_IP)
    ret, frame = cap.read()
    filename = f"{datetime.now().isoformat().replace(':', '.')}.png"
    path = os.path.join(IMAGES_DIRECTORY, filename)
    print(path)
    cv2.imwrite(path, frame)
    cap.release()
    cv2.destroyAllWindows()
