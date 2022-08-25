import glob
import os
import sys
import re
audio_dir = sys.argv[1]
data_dir = sys.argv[2]
if not os.path.exists(data_dir):
    os.mkdir(data_dir)
audio_list = glob.glob(audio_dir+"/**/*wav")
with open(data_dir+"/wav.scp","w") as f, open(data_dir+"/utt2spk","w") as F:
    for audio in audio_list:
        uttid = re.split('[/.]',audio)[-2]
        f.write(uttid + " " + audio + "\n")
        F.write(uttid + " " + uttid + "\n")
            
