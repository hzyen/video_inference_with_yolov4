import os
import pytz
from datetime import datetime
import time
from multiprocessing import Process
import logging

import cv2

class CaptureVideoWorker(Process):
    def __init__(self, cfg, frameQ, finishedCaptureEvent, logFile):
        Process.__init__(self)
        self.INPUT_FOLDER_DIR = cfg['input_folder_dir']
        self.INPUT_VIDEO_FILENAME = cfg['input_video_filename']
        self.INGEST_PICTURE_FPS = cfg['ingest_picture_fps']
        self.TIMEZONE = pytz.timezone(cfg['timezone'])
        logging.basicConfig(filename=logFile,level=logging.DEBUG,format='%(asctime)s.%(msecs)03d %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
        self.finishedCaptureEvent = finishedCaptureEvent
        self.frameQ = frameQ
        logging.info('[LOG] CaptureVideoWorker has been initialized')
    
    def run(self):
        if not os.path.isfile(f'{self.INPUT_FOLDER_DIR}{self.INPUT_VIDEO_FILENAME}'):
            logging.error(f'[ERROR] CaptureVideoWorker: The video file {self.INPUT_FOLDER_DIR}{self.INPUT_VIDEO_FILENAME} does not exist')
        else:
            vidCap = cv2.VideoCapture(f'{self.INPUT_FOLDER_DIR}{self.INPUT_VIDEO_FILENAME}')
            count = 1
            hasFrames = True
            start_time = datetime.now(self.TIMEZONE)
            while hasFrames:
                # Sleep for ingest picture to specified Hz
                if count > 1:
                    time.sleep(1/self.INGEST_PICTURE_FPS)

                t1 = datetime.now(self.TIMEZONE)
                hasFrames, img = vidCap.read()
                t2 = datetime.now(self.TIMEZONE)

                if hasFrames:
                    logging.info(f'[LOG] CaptureVideoWorker id:{count} vidCap.read() {t2-t1}')
                    # to-do send id & img to queue, id == count
                    msg = {'id': count, 'img': img}
                    t3 = datetime.now(self.TIMEZONE)
                    self.frameQ.put(msg)
                    t4 = datetime.now(self.TIMEZONE)
                    logging.info(f'[LOG] CaptureVideoWorker self.frameQ.put(msg) {t4-t3}')
                    count += 1
                else:
                    end_time = datetime.now(self.TIMEZONE)
                    self.finishedCaptureEvent.set()
                    logging.info(f'[LOG] CaptureVideoWorker has finished {end_time-start_time}')
                    vidCap.release()
                    
