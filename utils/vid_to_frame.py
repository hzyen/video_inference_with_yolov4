import cv2
import numpy as np
import os
from os.path import isfile, join
from multiprocessing import Process, Queue

#Opens the video file


sec = 0
fps = 25
video_file = './data/animals_5sec.mp4'
recombined_vid_file = './output/pred_animal_2.mp4'
frame_path = './output/animals_1/'
frame_size = (1280, 720)

def vidToFrame(video_file, frame_path, sec, fps):
    vidcap = cv2.VideoCapture(video_file)
    count=1
    frameRate = fps

    hasFrames,image = vidcap.read()
    if hasFrames:
        cv2.imwrite(f'{frame_path}image{count:04d}.jpg', image)
    while hasFrames:
        print(f'count: {count}')
        count = count + 1
        sec = sec + frameRate
        sec = round(sec, 2)
        hasFrames,image = vidcap.read()
        if hasFrames:
            cv2.imwrite(f'{frame_path}image{count:04d}.jpg', image)
    print('finished')


def frameToVid(pathIn, pathOut, fps, frame_size=(1280, 720)):
    out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'mp4v'), 1/fps, frame_size)

    frame_array = []
    files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]

    #for sorting the file names properly
    files.sort(key = lambda x: x[5:-4])

    for i in range(len(files)):
        print(f'count: {i}')
        filename=pathIn + files[i]
        #reading each files
        img = cv2.imread(filename)
        height, width, layers = img.shape
        #frame_size = (width,height)

        #inserting the frames into an image array
        frame_array.append(img)
        # writing to a image array
        out.write(frame_array[i])
    
    out.release()


if __name__ == '__main__':
    #vidToFrame(video_file, frame_path, sec, fps)
    frameToVid(frame_path, recombined_vid_file, fps, frame_size)