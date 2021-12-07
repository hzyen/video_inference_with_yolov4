import argparse
import json
from multiprocessing import Process, Queue, Event
from datetime import datetime
import pytz
import logging

from capture_video import CaptureVideoWorker
from frame_detection import FrameDetectionWorker
from write_video import WriteVideoWorker

def main(cfg):
    inputVideoFilename = cfg['input_video_filename']
    logFolderDir = cfg['log_folder_dir']
    start_time = datetime.now(pytz.timezone(cfg['timezone']))
    logFile = f'{logFolderDir}{start_time}_{inputVideoFilename}.log'
    logging.basicConfig(filename=logFile,level=logging.DEBUG,format='%(asctime)s.%(msecs)03d %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
    # Define 2 queues, which are for storing frame info and detection info, respectively.
    frameQ = Queue()
    detectionQ = Queue()
    # Define 2 events
    finishedCaptureEvent = Event()
    finishedDetectionEvent = Event()
    finishedWriteEvent = Event()

    # Construct all process workers
    captureVideoWorker = CaptureVideoWorker(cfg, frameQ, finishedCaptureEvent, logFile)
    frameDetectionWorker = FrameDetectionWorker(cfg, frameQ, detectionQ, finishedCaptureEvent, finishedDetectionEvent, logFile)
    writeVideoWorker = WriteVideoWorker(cfg, frameQ, detectionQ, finishedDetectionEvent, finishedWriteEvent, logFile)
    #checkQueueWorker = CheckQueueWorker(cfg, frameQ, detectionQ, finishedDetectionEvent, logFile)
    # Start all process workers
    captureVideoWorker.start()
    frameDetectionWorker.start()
    writeVideoWorker.start()
    # Waiting finish capture all video frames, and then alert other workers to quit
    finishedCaptureEvent.wait()
    finishedDetectionEvent.wait()
    finishedWriteEvent.wait()
    # Terminate all workers 
    captureVideoWorker.terminate()
    frameDetectionWorker.terminate()
    writeVideoWorker.terminate()
    # Close all workers
    #captureVideoWorker.close()
    #captureVideoWorker.close()
    #captureVideoWorker.close()

    end_time = datetime.now(pytz.timezone(cfg['timezone']))
    logging.info(f'[LOG] Finished! {end_time-start_time}')

def _get_parser():
    parser = argparse.ArgumentParser(description='video stream inference')
    parser.add_argument('--config', help='Config file path')
    return parser

if __name__ == '__main__':
    parser = _get_parser()
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    main(cfg)

