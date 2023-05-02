import cv2
import numpy as np

def adjustContrast(frame):
    contrast = 150
    brightness = 0
    frame = frame * (contrast/127 + 1) - contrast + brightness 
    frame = np.clip(frame, 0, 255)
    frame = np.uint8(frame)
    return frame

def binarytransfrom(src_img:np.array):  
    src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2HLS)
    src_img = adjustContrast(src_img)
    lower = np.array([100, 160 , 0])  
    upper = np.array([180, 255 , 5]) 
    mask = cv2.inRange(src_img, lower, upper)
    kernel = np.ones((7, 7), np.uint8)
    output = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations = 2)
    return output
    
if __name__ == "__main__":
    counter = 0
    frame = cv2.imread("video_src/srcImg10.jpg") 
    #frame = cv2.resize(frame, (640,360), interpolation=cv2.INTER_NEAREST)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    frame = adjustContrast(frame)
    img = binarytransfrom(frame)
    cv2.imshow("img", img)       
    cv2.imshow("src", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
           