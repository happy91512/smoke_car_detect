import numpy as np


class Tracker:
    # tid = 0

    def __init__(self, key: str, bboxs: np.ndarray) -> None:
        tid, cls, fframe = key.split('_')[1:]

        self.tid: int = int(tid.split('_')[-1])
        self.cls: int = cls #物件class
        self.fframe: int = int(fframe[1:]) #start frame
        self.lframe: int = self.fframe + bboxs.shape[0] #end frame
        self.bboxs: np.ndarray = bboxs #每一frame的bounding box
        self.vio:int = 0

    def __repr__(self) -> str:
        return f'OT-{self.tid}'
