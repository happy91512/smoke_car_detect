import cv2
import numpy as np
import pickle
import cv2
import time
import copy
from typing import List
from Tracker_structure import Tracker



 
def transferTracker(trackers : list, frameNumber : int):
    trackers2 = copy.deepcopy(trackers)
    for obj in trackers2:
        remainFrame = obj.lframe - obj.fframe
        if remainFrame < 90:#移除出現不超過3秒的物件    
            trackers2.remove(obj)

    FrameBbox = []
    for i in range(frameNumber):
        FrameBbox.append([]) # *創一個跟影片總幀數大小的list,用這個來裝每個frame的資訊
        
    for obj in trackers2: #先挑一個物件
        classID = obj.cls
        allBboxList = obj.bboxs #讀取這個物件所有的bounding boxs
        for oneBbox in allBboxList: #挑一個bounding boxs放再對應的FrameBbox[index]
            showFrame = oneBbox[0] #獲得該bounding boxs要顯示的frame
            oneBbox[0] = obj.tid #把list第一個元素改成track ID 
            oneBbox = np.append(oneBbox, classID)     
            FrameBbox[showFrame-1].append(list(oneBbox)) #把資料存入FrameBbox對應的frame的index中
            
    return FrameBbox #這樣就能獲得每一frame有哪些bounding box，跟這些bounding box的tid and 座標 and cls


def adjustContrast(frame : np.ndarray):#調整影片對比度，比較好篩選出煙霧
    contrast = 150
    brightness = 0
    frame = frame * (contrast/127 + 1) - contrast + brightness 
    frame = np.clip(frame, 0, 255)
    frame = np.uint8(frame)
    return frame


def binarytransfrom(src_img : np.ndarray): #篩選顏色 獲得二值化後的結果 
    src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2HLS)
    src_img = adjustContrast(src_img)
    lower = np.array([100, 160 , 0])  
    upper = np.array([180, 255 , 5]) 
    mask = cv2.inRange(src_img, lower, upper)
    kernel = np.ones((7, 7), np.uint8)
    output = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations = 2)
    return output


def determineSmog(recImg : np.ndarray): #用面積佔比判斷是否為烏賊車
    smogArea = np.sum(recImg == 255)
    bboxArea = np.uint(recImg.shape[0] * recImg.shape[1])
    if smogArea / bboxArea >= 0.065: return True
    else: return False


def main(video_addr : str, trackers : List[Tracker]):
    start = time.time()
    copyTrackers = copy.deepcopy(trackers)
    vioDict = {0:0} #紀錄違規次數的dictionary
    cap = cv2.VideoCapture(video_addr)   
    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    videoHight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    videoWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    videoSavePath = "./output.avi"
    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*'XVID') #設定影片的格式
    out = cv2.VideoWriter(videoSavePath, fourcc, fps, (videoWidth, videoHight))
    #vioCut = cv2.VideoWriter(videoSavePath, fourcc, fps, (videoWidth, videoHight)) #*這邊是存結果影片用的，要存的話記得把註解用掉
    frameBbox = transferTracker(trackers, total_frame) #轉換Trackers list
    
    frameCounter = 0
    for i in range(total_frame):# *每一個frame去做
        ret, frame = cap.read()
        bboxInfo = frameBbox[frameCounter]# *已經把trackers的資訊轉換成以每個frame個別資訊，所以只要把frame數放入到index裡就行了
        binaryFrame = binarytransfrom(frame)# 產生白煙二值化結果
        for bbox in bboxInfo:
            trackID = bbox[0] #有事先把tid與cls放入
            classID = bbox[-1]
            if  classID == 3 or classID == 2: #只對機車與汽車做處理
                (x1, y1, x2, y2) = bbox[1:5]
                bboxArea = (x2 - x1) * (y2 - y1) #bounding box面積
                if bboxArea < 10000 or x1 > videoWidth or x2 > videoWidth or y1 > videoHight or y2 >videoHight: continue #面積過小或資料不合理都跳過
                (x1, y1, x2, y2) = (int(x1 * 1.05), int(y1 * 1.05), int(x2 * 1.05), int(y2 * 1.05)) #把bounding box向右下角移動，比較能涵蓋更多煙霧
                isSomgCar = determineSmog(binaryFrame[y1 : y2, x1 : x2]) #利用面積判斷是否為烏賊車
                if isSomgCar:             
                    if vioDict.get(trackID) == None: vioDict[trackID] = 1 #沒有對應的key的話就新增一組key and value
                    else: vioDict[trackID] += 1 #如果是烏賊車就在對應的value上+1 
                    cv2.rectangle(frame, (x1, y1), (x1+190, y1-65), (0, 0, 255), -1, cv2.FONT_HERSHEY_SIMPLEX)
                    cv2.putText(frame, f"Car{trackID} is bad", (x1,y1-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4, cv2.FONT_HERSHEY_SIMPLEX)# 烏賊車放上紅色bounding box 與資訊
                    continue

                cv2.rectangle(frame, (x1, y1), (x1+190, y1-65), (0, 255, 0), -1, cv2.FONT_HERSHEY_SIMPLEX)
                cv2.putText(frame, f"Car{trackID} is OK", (x1,y1-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4, cv2.FONT_HERSHEY_SIMPLEX) #一般車輛放綠色bounding box
            else: pass
            
        frameCounter += 1 #計算當前frame 
        cv2.putText(frame, f"Smog frames {str(vioDict)[7:-1]}", (30,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA) #放上當下的烏賊車累積紀錄
        out.write(frame) #寫入目前的frame到輸出影片
        vioCarID = max(vioDict, key=lambda key: vioDict[key]) #獲得目前累積最多value的key
        if vioDict[vioCarID] == 30: #設定value=30時是擷取影片起點
            startFrame = frameCounter
            endFrame = frameCounter + 240 #八秒後為擷取影片終點
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Spent {time.time()-start} seconds.")
    vioCarID = max(vioDict, key=lambda key: vioDict[key]) #這邊用最大值來判斷違規車輛，但這樣一個影片只能抓到一個烏賊車
    vioTrackers = []
    for t in copyTrackers:
        if t.tid == vioCarID:      
            vioTrackers.append(t) #只加入剛剛判斷為烏賊車的車輛資料
            t.smog_fframe = startFrame
            t.smog_lframe = endFrame #新增attributes到要傳出的trackers list中
    #print(vioTrackers[0].bboxs)
    return(vioTrackers)

if __name__ == "__main__":
    with open("sample.pickle", "rb") as f:
        trackers : List[Tracker] = pickle.load(f) 
    vio = main("sample.mp4", trackers)
    #print(vio[0].bboxs)
    