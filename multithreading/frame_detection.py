import sys
import pytz
from datetime import datetime
import threading
import logging

import torch
import cv2

sys.path.append("../")
sys.path.append("../pytorchYOLOv4")
from pytorchYOLOv4.models import Yolov4
from pytorchYOLOv4.tool.torch_utils import do_detect


class FrameDetectionWorker(threading.Thread):
    def __init__(self, cfg, frameQ, detectionQ, finishedCaptureEvent, finishedDetectionEvent, logFile):
        threading.Thread.__init__(self)
        self.WEIGHTFILE = cfg['weightfile']
        self.N_CLASSES = cfg['n_classes']
        self.USE_CUDA = cfg['useCuda']
        self.TIMEZONE = pytz.timezone(cfg['timezone'])
        logging.basicConfig(filename=logFile,level=logging.DEBUG,format='%(asctime)s.%(msecs)03d %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
        self.finishedCaptureEvent = finishedCaptureEvent
        self.finishedDetectionEvent = finishedDetectionEvent
        self.frameQ = frameQ
        self.detectionQ = detectionQ
        logging.info(f'[LOG] FrameDetectionWorker has been initialized')

    def run(self):
        # Load model from Yolov4
        t5 = datetime.now(self.TIMEZONE)
        model = Yolov4(yolov4conv137weight=None, n_classes=self.N_CLASSES, inference=True)
        pretrained_dict = torch.load(self.WEIGHTFILE, map_location=torch.device('cuda'))
        model.load_state_dict(pretrained_dict)
        if self.USE_CUDA:
            model.cuda()
        t6 = datetime.now(self.TIMEZONE)
        logging.info(f'[LOG] FrameDetectionWorker model has been loaded {t6-t5}')

        detection_total_time = None
        start_time = datetime.now(self.TIMEZONE)
        while not self.finishedCaptureEvent.isSet() or not self.frameQ.empty():
            if self.frameQ.empty():
                continue
            else:
                # Receive message from frameQ
                t1 = datetime.now(self.TIMEZONE)
                rev_msg = self.frameQ.get_nowait()
                self.frameQ.task_done()
                t2 = datetime.now(self.TIMEZONE)
                id = rev_msg['id']
                img = rev_msg['img']
                logging.info(f'[LOG] FrameDetectionWorker id:{id} self.frameQ.get() {t2-t1}')
                # Resize image
                t3 = datetime.now(self.TIMEZONE)
                resized_img = cv2.resize(img, (320, 320))
                resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
                t4 = datetime.now(self.TIMEZONE)
                logging.info(f'[LOG] FrameDetectionWorker id:{id} resize image {t4-t3}')
                # Inference
                t5 = datetime.now(self.TIMEZONE)
                boxes = do_detect(model, resized_img, 0.4, 0.6, self.USE_CUDA)
                t6 = datetime.now(self.TIMEZONE)
                if id == 1:
                    detection_total_time = (t6-t5)
                else:
                    detection_total_time += (t6-t5)
                logging.info(f'[LOG] FrameDetectionWorker id:{id} do_detect() {t6-t5}')
                # Send id & boxes to detectionQ
                msg = {'id': id, 'img': img, 'boxes': boxes}
                t7 = datetime.now(self.TIMEZONE)
                self.detectionQ.put(msg)
                t8 = datetime.now(self.TIMEZONE)
                logging.info(f'[LOG] FrameDetectionWorker id:{id} self.detectionQ.put(msg) {t8-t7}')

        end_time = datetime.now(self.TIMEZONE)
        logging.info(f'[LOG] FrameDetectionWorker has finished {end_time-start_time}')
        logging.info(f'[LOG] FrameDetectionWorker detection_total_time: {detection_total_time}')
        self.finishedDetectionEvent.set()
        