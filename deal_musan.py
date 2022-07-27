import glob
import os

audio_dir = "/home3/jicheng/source-data/musan/noise/"
data_dir = "/home3/jicheng/lirui/Accent-Recognition-Multitask-System-with-ASRInit/data/musan_noise/"
audio_list = glob.glob(audio_dir+"**/*wav")
with open(data_dir+"musan_noise_file_list.txt","w") as f:
    for line in audio_list:
        f.write(line+'\n')
