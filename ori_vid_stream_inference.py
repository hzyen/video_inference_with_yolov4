import os
from os.path import isfile, join
from datetime import datetime
import sys
import argparse
import math

import json
import pytz
import cv2
import numpy as np
import torch

sys.path.append("./pytorchYOLOv4")
from pytorchYOLOv4.models import Yolov4
from pytorchYOLOv4.tool.torch_utils import do_detect
from pytorchYOLOv4.tool.utils import load_class_names, plot_boxes_cv2

class VideoInference():

    def __init__(self, config):
        self.input_folder_dir = config['input_folder_dir']
        self.input_video_filename = config['input_video_filename']
        self.output_folder_dir = config['output_folder_dir']
        self.output_frames_foldername = config['output_frames_foldername']
        self.output_video_filename = config['output_video_filename']
        self.frame_width = config['frame_width']
        self.frame_height = config['frame_height']
        self.fps = config['fps']
        self.weightfile = config['weightfile']
        self.n_classes = config['n_classes']
        self.namesfile = config['namesfile']
        self.save_image = config['save_image']
        self.timezone = pytz.timezone(config['timezone'])

        print('Video Inference init.')

    def run(self):
        start_time = datetime.now(self.timezone)
        # Load model from Yolov4
        model = Yolov4(yolov4conv137weight=None, n_classes=self.n_classes, inference=True)
        pretrained_dict = torch.load(self.weightfile, map_location=torch.device('cuda'))
        model.load_state_dict(pretrained_dict)
        print('Loaded model')

        # Use Cuda
        use_cuda = True
        if use_cuda:
            model.cuda()

        # Configuration with VideoCapture and VideoWriter
        os.makedirs(self.output_folder_dir, exist_ok=True)
        os.makedirs(f'{self.output_folder_dir}{self.output_frames_foldername}', exist_ok=True)
        count = 1
        frame_size = (self.frame_width, self.frame_height)
        frame_in = cv2.VideoCapture(f'{self.input_folder_dir}{self.input_video_filename}')
        frame_out = cv2.VideoWriter(f'{self.output_folder_dir}{self.output_video_filename}',cv2.VideoWriter_fourcc(*'mp4v'), self.fps, frame_size)

        while True:
            t11 = datetime.now(self.timezone)
            hasFrames,img = frame_in.read()
            t12 = datetime.now(self.timezone)
            if hasFrames:
                # Resized img
                t1 = datetime.now(self.timezone)
                resized_img = cv2.resize(img, (320, 320))
                resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
                t2 = datetime.now(self.timezone)
                #Inference img
                t3 = datetime.now(self.timezone)
                boxes = do_detect(model, resized_img, 0.4, 0.6, use_cuda)
                t4 = datetime.now(self.timezone)
                # Load class names
                t5 = datetime.now(self.timezone)
                class_names = load_class_names(self.namesfile)
                t6 = datetime.now(self.timezone)
                # Generate the plotted box picture
                t7 = datetime.now(self.timezone)
                if self.save_image:
                    plotted_img = plot_boxes_cv2(img, boxes[0], f'{self.output_folder_dir}{self.output_frames_foldername}image{count:04d}.jpg',class_names)
                else:
                    plotted_img = plot_boxes_cv2(img, boxes[0], None, class_names)
                t8 = datetime.now(self.timezone)
                # Write to MP4
                t9 = datetime.now(self.timezone)
                frame_out.write(plotted_img)
                t10 = datetime.now(self.timezone)

                total_time = datetime.now(self.timezone)
                print(f'---------------------------------')
                print(f'{t12-t11} [LOG] frame_in.read()')
                print(f'{t2-t1} [LOG] resize image')
                print(f'{t4-t3} [LOG] do_detect()')
                print(f'{t6-t5} [LOG] load_class_name()')
                print(f'{t8-t7} [LOG] plot_boxes_cv2()')
                print(f'{t10-t9} [LOG] frame_out.write(plotted_img)')
                print(f'{total_time} [LOG] total time, count: {count}')
                count += 1
            else:
                end_time = datetime.now(self.timezone)
                print(f'{end_time-start_time} finished ')
                frame_in.release()
                frame_out.release()
                break

def _get_parser():
    parser = argparse.ArgumentParser(description='video inference')
    parser.add_argument('--config', help='Config file path')
    return parser

if __name__ == '__main__':
    parser = _get_parser()
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    vid_infer = VideoInference(config)
    vid_infer.run()