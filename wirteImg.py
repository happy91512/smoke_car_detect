import cv2
import numpy as np
from smoke import adjustContrast
from smoke import binarytransfrom

if __name__ == "__main__":
    frameCount = 0
    while True:
        try:
            frame = cv2.imread(f"video_src/srcImg{frameCount}.jpg")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
            frame = adjustContrast(frame)
            img = binarytransfrom(frame)
            cv2.putText(img, f"{frameCount}", (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow("img", img)
            cv2.waitKey(1)
            cv2.imwrite(f"video_binary/binaryIMG{frameCount}.jpg", img)
            frameCount += 1        
        except: break
