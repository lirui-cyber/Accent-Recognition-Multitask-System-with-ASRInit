# Accent Recognition Multi-task System with ASRInit

# Data preparation scripts and training pipeline for the Accented English Recognition.

# Environment dependent
  1. Kaldi (Data preparation related function script) [Github link](https://github.com/kaldi-asr/kaldi)
  2. Espnet-0.10.4
  4. Modify the installation address of espnet in the path.sh file
## Installation
### Set up kaldi environment
```
git clone -b 5.4 https://github.com/kaldi-asr/kaldi.git kaldi
cd kaldi/tools/; make; cd ../src; ./configure; make
```
### Set up espnet environment
```
git clone -b v.0.10.4 https://github.com/espnet/espnet.git
cd espnet/tools/        # change to tools folder
ln -s {kaldi_root}      # Create link to Kaldi. e.g. ln -s home/theanhtran/kaldi/
```
### Set up Conda environment
```
./setup_anaconda.sh anaconda espnet 3.7.9   # Create a anaconda environmetn - espnet with Python 3.7.9
make TH_VERSION=1.8.0 CUDA_VERSION=10.2     # Install Pytorch and CUDA
. ./activate_python.sh; python3 check_install.py  # Check the installation
conda install torchvision==0.9.0 torchaudio==0.8.0 -c pytorch
```
### Set your own execution environment
Open path.sh file, change $MAIN_ROOT$ to your espnet directory, 
```
e.g. MAIN_ROOT=/home/jicheng/espnet
```
# Instructions for use
## Data preparation
  1. All the data used in the experiment are stored in the `data` directory, in which train is used for training, valid is the verification set, 
    cv_all and test are used for testing respectively.
  2. In order to better reproduce my experimental results, you can download the data set first, and then directly change the path in `wav.scp` in different sets in `data` directory. <br>
  You can also use the `sed` command to replace the path in the wav.scp file with your path.
```
egs: 
origin path: /home/zhb502/raw_data/2020AESRC/American_English_Speech_Data/G00473/G00473S1002.wav
your path: /home/jicheng/ASR-data/American_English_Speech_Data/G00473/G00473S1002.wav
sed -i "s#/home/zhb502/raw_data/2020AESRC/#/home/jicheng/ASR-data/#g" data/train/wav.scp
```
3. Other files can remain unchanged, you can use it directly (eg, utt2IntLabel, text, utt2spk...).

## Add noise to the test set 
To test the performance in the noise background, we added musan noise to the test set.
At the same time, different SNR(5,10,15,20) are used for noise addition. <br>
### Generate format file
```python
# The first parameter is the path to the musan noise audio
python deal_musan.py  /home3/jicheng/source-data/musan/noise data/musan_noise
```
### Add noise
```sh
cd Add-Noise
bash add-noise.sh --steps 2 --src-train ../data/test --noise_dir ../data/musan_noise
```

## Accent recognition system
  1. Model file preparation
    `run_asr_multitask_accent_recognition_16k.sh` and `run_asr_multitask_accent_recognition_8k.sh` are both used to train the multi-task model.<br>
    Before running, you need to first move the corresponding files of espnet to the corresponding directory of your espnet directory. 
```
eg: 
  move `espnet/nets/pytorch_backend/e2e_asr_transformer_multitask_accent.py` to ` your espnet localtion/espnet/nets/pytorch_backend` 
```
  2. pretrained asr model
  In order to get better results, we first use librispeech data set(960 hours) to train an ASR system, and then use it to initialize our multi task system. <br>
  You can train yourself to train the ASR model using the following script:
  ```
  # You can also use the `sed` command to replace the path in the wav.scp file with your path
  sed -i "s#/data/users/zpz505/LibriSpeech#/data/LibriSpeech#g" data-librispeech/train_960/wav.scp

  bash run_librispeech_asr.sh  --nj 20 --steps 1-8
  ```
  You can also find the pretrained model in folder `pretrained-librispeech-model`, contains 16K and 8K model. <br>
  The pretrained ASR model can be downloaded from the link below: https://drive.google.com/drive/folders/1SLJ54dPTk8EUCqurnl3xPSSZEhMuPP5E <br>
  3. step by step
    The overall code is divided into four parts, including feature extraction, JSON file generation, model training and decoding. <br>
    You can control the steps by changing the value of the step variable. 
```
egs: 
  ### for 16k data
  bash run_asr_multitask_accent_recognition_16k.sh --nj 20 --steps 1
  bash run_asr_multitask_accent_recognition_16k.sh --nj 20 --steps 3
  bash run_asr_multitask_accent_recognition_16k.sh --nj 20 --steps 4
  bash run_asr_multitask_accent_recognition_16k.sh --nj 20 --steps 5
  bash run_asr_multitask_accent_recognition_16k.sh --nj 20 --steps 6
  ### for 8k data
  bash run_asr_multitask_accent_recognition_8k.sh --nj 20 --steps 1
  bash run_asr_multitask_accent_recognition_8k.sh --nj 20 --steps 2
  bash run_asr_multitask_accent_recognition_8k.sh --nj 20 --steps 3
  bash run_asr_multitask_accent_recognition_8k.sh --nj 20 --steps 4
  bash run_asr_multitask_accent_recognition_8k.sh --nj 20 --steps 5
  bash run_asr_multitask_accent_recognition_8k.sh --nj 20 --steps 6
```


  4. In addition, in order to better reproduce and avoid you training asr system again, I uploaded an ASR model trained use 16k accent160 data.<br>
     For pretrained model, you can download from this link: https://drive.google.com/file/d/1mP81esvRycnzqpvxpm7HjNPHT8SFfdig/view?usp=sharing <br>
     You can run the following two commands to directly reproduce our results.
```
  # 16k data
  bash run_asr_multitask_accent_recognition_16k.sh --nj 20 --steps 7 
```

## notice
```
  All scripts have one inputs: steps
  steps: Control execution parameters
```  

## Add codec (simulation narrow-band data)
  In reality, it is hard to obtain sufficient domain specific real telephony data to train acoustic models due to data privacy consideration. So we employ diversified audio codecs simulation based data augmentation method to train telephony speech recognition system.<br>
  In this study, we use AESRC accent data as wide-band data, we first down-sample the 16 kHz accent data to the 8 kH. For simulate narrow-band data, we select randomly from the full list of codecs, and using FFMPEG tools convert it to narrow-band data.<br>
  For specific implementation, you can refer to `add-codec/add-codec.sh` script, but before you run it, you must change the value `"/home4/hhx502/w2019/ffmpeg_source/bin/ffmpeg"` in add-codec/scripts/add-codec-with-ffmpeg.pl to you ffmpeg path. Then you should modify the value of `data_set` and `source_dir` variable in the `add-codec/add-codec.sh` script. After the first two steps, you can run it directly<br>
```
egs:
  bash add-codec.sh
```
