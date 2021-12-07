import os
import sys
import pytz
from datetime import datetime
import threading
import logging

import cv2

sys.path.append("../")
sys.path.append("../pytorchYOLOv4")
from pytorchYOLOv4.tool.utils import load_class_names, plot_boxes_cv2

class WriteVideoWorker(threading.Thread):
    def __init__(self, cfg, frameQ, detectionQ, finishedDetectionEvent, finishedWriteEvent, logFile):
        threading.Thread.__init__(self)
        self.OUTPUT_FOLDER_DIR = cfg['output_folder_dir']
        self.OUTPUT_FRAMES_FOLDERNAME = cfg['output_frames_foldername']
        self.OUTPUT_VIDEO_FILENAME = cfg['output_video_filename']
        self.OUTPUT_PICTURE_RESOLUTION_WIDTH = cfg['output_picture_resolution_width']
        self.OUTPUT_PICTURE_RESOLUTION_HEIGHT = cfg['output_picture_resolution_height']
        self.TRANSMISSION_RATE = cfg['transmission_rate']
        self.NAMESFILE = cfg['namesfile']
        self.SAVE_IMAGE = cfg['save_image']
        self.FRAME_SIZE = (self.OUTPUT_PICTURE_RESOLUTION_WIDTH, self.OUTPUT_PICTURE_RESOLUTION_HEIGHT)
        self.TIMEZONE = pytz.timezone(cfg['timezone'])
        os.makedirs(self.OUTPUT_FOLDER_DIR, exist_ok=True)
        logging.basicConfig(filename=logFile,level=logging.DEBUG,format='%(asctime)s.%(msecs)03d %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
        self.finishedDetectionEvent = finishedDetectionEvent
        self.finishedWriteEvent = finishedWriteEvent
        self.frameQ = frameQ
        self.detectionQ = detectionQ
        logging.info(f'[LOG] WriteVideoWorker has been initialized')

    def run(self):
        vidWrit = cv2.VideoWriter(f'{self.OUTPUT_FOLDER_DIR}{self.OUTPUT_VIDEO_FILENAME}',cv2.VideoWriter_fourcc(*'mp4v'), self.TRANSMISSION_RATE, self.FRAME_SIZE)
        start_time = datetime.now(self.TIMEZONE)
        while not self.finishedDetectionEvent.isSet() or not self.detectionQ.empty():
            if self.detectionQ.empty():
                continue
            else:
                # Recieve message from detectionQ
                t9 = datetime.now(self.TIMEZONE)
                rev_msg = self.detectionQ.get_nowait()
                self.detectionQ.task_done()
                t10 = datetime.now(self.TIMEZONE)
                id = rev_msg['id']
                img = rev_msg['img']
                boxes = rev_msg['boxes']
                logging.info(f'[LOG] WriteVideoWorker id:{id} self.detectionQ.get() {t10-t9}')
                # Load class names
                t1 = datetime.now(self.TIMEZONE)
                class_names = load_class_names(self.NAMESFILE)
                t2 = datetime.now(self.TIMEZONE)
                logging.info(f'[LOG] WriteVideoWorker id:{id} load_class_name() {t2-t1}')
                # Generate the plotted box picture
                t3 = datetime.now(self.TIMEZONE)
                if self.SAVE_IMAGE:
                    plotted_img = plot_boxes_cv2(img, boxes[0], f'{self.OUTPUT_FOLDER_DIR}{self.OUTPUT_FRAMES_FOLDERNAME}image{id:04d}.jpg',class_names)
                else:
                    plotted_img = plot_boxes_cv2(img, boxes[0], None,class_names)
                t4 = datetime.now(self.TIMEZONE)
                logging.info(f'[LOG] WriteVideoWorker id:{id} plot_boxes_cv2() {t4-t3}')
                # Write to MP4
                t5 = datetime.now(self.TIMEZONE)
                vidWrit.write(plotted_img)
                t6 = datetime.now(self.TIMEZONE) 
                logging.info(f'[LOG] WriteVideoWorker id:{id} frame_out.write(plotted_img) {t6-t5}')
        end_time = datetime.now(self.TIMEZONE)
        logging.info(f'[LOG] WriteVideoWorker has finished {end_time-start_time}')
        self.finishedWriteEvent.set()