import os
import cv2
import numpy
from utils import visualizar_video

def segmentar_dados(frame):
    pass

if __name__ == '__main__':
    videos = range(1,5)
    for num in videos:
        video = visualizar_video(f'tirada_{num}.mp4')