#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

cuda_cmd="slurm.pl --quiet --exclude=node0[4-9]"
decode_cmd="slurm.pl --quiet --exclude=node0[3-5]"
cmd="slurm.pl --quiet --exclude=node0[3-5]"
# general configuration
backend=pytorch
steps=1
ngpu=1         # number of gpus ("0" uses cpu, otherwise use gpu)
nj=20
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume=        # Resume the training from snapshot
vocab_size=2000
bpemode=bpe
# feature configuration
do_delta=false

lm_config=conf/espnet_lm.yaml
decode_config=conf/espnet_decode.yaml

# rnnlm related
lm_resume=         # specify a snapshot file to resume LM training
lmtag=0             # tag for managing LMs

# decoding parameter
recog_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'
n_average=5

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -o pipefail

. utils/parse_options.sh || exit 1;
. path.sh

steps=$(echo $steps | perl -e '$steps=<STDIN>;  $has_format = 0;
  if($steps =~ m:(\d+)\-$:g){$start = $1; $end = $start + 10; $has_format ++;}
        elsif($steps =~ m:(\d+)\-(\d+):g) { $start = $1; $end = $2; if($start == $end){}
        elsif($start < $end){ $end = $2 +1;}else{die;} $has_format ++; }
      if($has_format > 0){$steps=$start;  for($i=$start+1; $i < $end; $i++){$steps .=":$i"; }}
      print $steps;' 2>/dev/null)  || exit 1

if [ ! -z "$steps" ]; then
  for x in $(echo $steps|sed 's/[,:]/ /g'); do
     index=$(printf "%02d" $x);
     declare step$index=1
  done
fi

vocab_size=5000
bpemode=unigram
data=data-librispeech
exp=exp-librispeech

train_set="train_960"
valid_set="dev_other"
recog_set="test_clean test_other dev_other dev_clean"

if [ ! -z $step01 ]; then
   echo "extracting filter-bank features and cmvn"
   for i in ${train_set} ;do #${valid_set}  $recog_set;do 
      utils/fix_data_dir.sh $data/$i
      steps/make_fbank_pitch.sh --cmd "$cmd" --nj $nj --write_utt2num_frames true \
          $data/$i $data/$i/feats/log $data/$i/feats/ark
      utils/fix_data_dir.sh $data/$i
   done

   compute-cmvn-stats scp:$data/${train_set}/feats.scp $data/${train_set}/cmvn.ark
   echo "step01 Extracting filter-bank features and cmvn Done"
fi

if [ ! -z $step02 ]; then
   echo "generate label file and dump features for track2:E2E"

   for x in ${train_set} ;do
      dump.sh --cmd "$cmd" --nj $nj  --do_delta false \
          $data/$x/feats.scp $data/${train_set}/cmvn.ark $data/$x/dump/log $data/$x/dump # for track2 e2e training
   done

   for x in ${valid_set} $recog_set;do 
       dump.sh --cmd "$cmd" --nj $nj  --do_delta false \
          $data/$x/feats.scp $data/${train_set}/cmvn.ark $data/$x/dump_${train_set}/log $data/$x/dump_${train_set} # for track2 e2e training
   done
   echo "step02 Generate label file and dump features for track2:E2E Done"   
fi

bpe_set=$train_set
bpe_model=$data/lang/$train_set/${train_set}_${bpemode}_${vocab_size}
dict=$data/lang/$train_set/${train_set}_${bpemode}_${vocab_size}_units.txt
if [ ! -z $step03 ]; then
   echo "stage 03: Dictionary Preparation" 

   [ -d $data/lang/$train_set ] || mkdir -p $data/lang/$train_set || exit;
   echo "<unk> 1" > ${dict}

   awk '{$1=""; print}' $data/$bpe_set/text | sed -r 's#^ ##g' > $data/lang/$train_set/${train_set}_input.txt

   spm_train --input=$data/lang/$train_set/${train_set}_input.txt --vocab_size=${vocab_size} --model_type=${bpemode} --model_prefix=${bpe_model} --input_sentence_size=100000000
   spm_encode --model=${bpe_model}.model --output_format=piece < $data/lang/$train_set/${train_set}_input.txt | tr ' ' '\n' | sort | uniq | awk '{print $0 " " NR+1}' >> ${dict}
   echo "stage 03: Dictionary Preparation Done"
fi

if [ ! -z $step04 ]; then
    # make json labels
    data2json.sh --nj $nj --cmd "${cmd}" --feat $data/${train_set}/dump/feats.scp --bpecode ${bpe_model}.model \
       $data/${train_set} ${dict} > ${data}/${train_set}/${train_set}_${bpemode}_${vocab_size}.json

    for i in $recog_set $valid_set;do 
       data2json.sh --nj 10 --cmd "${cmd}" --feat $data/$i/dump_${train_set}/feats.scp --bpecode ${bpe_model}.model \
           $data/$i ${dict} > ${data}/$i/${train_set}_${bpemode}_${vocab_size}.json
    done
    echo "stage 04: Make Json Labels Done"
fi

lmexpdir=${exp}/${train_set}_rmmlm_${bpemode}
# train rnnlm 
if [ ! -z $step06 ]; then
    lmdatadir=$exp/local/lm_${train_set}_${bpemode}
    [ -d $lmdatadir ] ||  mkdir -p $lmdatadir
    cut -f 2- -d" " $data/${train_set}/text | spm_encode --model=${bpe_model}.model --output_format=piece \
        > ${lmdatadir}/${train_set}.txt
    cut -f 2- -d" " $data/${train_valid}/text | spm_encode --model=${bpe_model}.model --output_format=piece \
        > ${lmdatadir}/${train_valid}.txt
 
    ${cuda_cmd} --gpu ${ngpu} ${lmexpdir}/train.log \
        lm_train.py \
        --config ${lm_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --verbose 1 \
        --outdir ${lmexpdir} \
        --tensorboard-dir tensorboard/${lmexpname} \
        --train-label ${lmdatadir}/${train_set}.txt \
        --valid-label ${lmdatadir}/${train_valid}.txt \
        --resume ${lm_resume} \
        --dict ${dict}
fi

if [ ! -z $step07 ]; then
    train_set=${train_set}
    transformer_lr=10
    batch_size=32
    epochs=120
    intermediate_ctc_weight=0
    intermediate_ctc_layer=9
    expname=${train_set}_conformer_large_${bpemode}_12enc_6dec_withSpecAug_lr_${transformer_lr}_batch_size_${batch_size}_${backend}
    # expname=${train_set}_conformer_12enc_6dec_noSpecAug_lr_${transformer_lr}_batch_size_${batch_size}_${backend}
    expdir=$exp/${expname}
    epoch_stage=0
    mkdir -p ${expdir}
    echo "stage 2: Network Training"
    ngpu=4
    if  [ ${epoch_stage} -gt 0 ]; then
        echo "stage 6: Resume network from epoch ${epoch_stage}"
        resume=${exp}/${expname}/results/snapshot.ep.${epoch_stage}
    fi  
    # train_config=conf/train_pytorch_conformer.yaml
    train_config=conf/train_pytorch_conformer_large.yaml
    preprocess_config=conf/specaug.yaml
#                --preprocess-conf ${preprocess_config} \
    ${cuda_cmd} --gpu $ngpu ${expdir}/train.log \
         asr_train.py \
                --config ${train_config} \
                --preprocess-conf ${preprocess_config} \
                --ngpu $ngpu \
                --backend ${backend} \
                --outdir ${expdir}/results \
                --tensorboard-dir tensorboard/${expname} \
                --debugmode ${debugmode} \
                --dict ${dict} \
                --debugdir ${expdir} \
                --minibatches ${N} \
                --verbose ${verbose} \
                --resume ${resume} \
                --batch-size ${batch_size} \
                --epochs ${epochs} \
                --transformer-lr ${transformer_lr} \
                --train-json $data/${train_set}/${train_set}_${bpemode}_${vocab_size}.json \
                --valid-json $data/${valid_set}/${train_set}_${bpemode}_${vocab_size}.json
fi

if [ ! -z $step08 ]; then    
    train_set=${train_set}
    transformer_lr=10
    batch_size=32
    epochs=120
    expname=${train_set}_conformer_12enc_6dec_withSpecAug_lr_${transformer_lr}_batch_size_${batch_size}_${backend}
    expname=${train_set}_conformer_12enc_6dec_noSpecAug_lr_${transformer_lr}_batch_size_${batch_size}_${backend}
    expname=test
    expdir=$exp/${expname}    
    epoch_stage=0
    [ -d $expdir ] || mkdir -p ${expdir}    
    echo "stage 8: Network Training"    
    ngpu=1    
    if  [ ${epoch_stage} -gt 0 ]; then
        echo "stage 6: Resume network from epoch ${epoch_stage}"
        resume=${exp}/${expname}/results/snapshot.ep.${epoch_stage}
    fi
    train_config=conf/train_pytorch_conformer.yaml
    train_config=conf/train_pytorch_transformer_withUttName.yaml
    preprocess_config=conf/specaug.yaml
#                --preprocess-conf ${preprocess_config} \
    ${cuda_cmd} --gpu $ngpu ${expdir}/train.log \
         asr_train_withUttName.py \
                --config ${train_config} \
                --ngpu $ngpu \
                --backend ${backend} \
                --outdir ${expdir}/results \
                --tensorboard-dir tensorboard/${expname} \
                --debugmode ${debugmode} \
                --dict ${dict} \
                --debugdir ${expdir} \
                --minibatches ${N} \
                --verbose ${verbose} \
                --resume ${resume} \
                --transformer-lr ${transformer_lr} \
                --batch-size ${batch_size} \
                --epochs ${epochs} \
                --train-json $data/${valid_set}/${train_set}_${bpemode}_${vocab_size}.json \
                --valid-json $data/${valid_set}/${train_set}_${bpemode}_${vocab_size}.json
fi
use_valbest_average=true
max_epoch=120

if [ ! -z $step16 ]; then
    echo "stage 3: Decoding"
    # train_config=conf/train_pytorch_transformer.yaml
    train_config=conf/train_pytorch_conformer.yaml
    train_config=conf/train_pytorch_conformer_large.yaml
    nj=200
    for expname in train_960_conformer_large_12enc_6dec_withSpecAug_lr_10_batch_size_32_pytorch;do
    expdir=$exp/${expname}
    for test in $recog_set $valid_set; do
    if [[ $(get_yaml.py ${train_config} model-module) = *conformer* ]]; then
        # Average ASR models
        if ${use_valbest_average}; then
            [ -f ${expdir}/results/model.val5.avg.best ] && rm ${expdir}/results/model.val5.avg.best
            recog_model=model.val${n_average}.avg.best
            opt="--log ${expdir}/results/log"
        else
            [ -f ${expdir}/results/model.last5.avg.best ] && rm ${expdir}/results/model.last5.avg.best
            recog_model=model.last${n_average}.avg.best
            opt="--log"
        fi
        scripts/average_max_epoch_checkpoints.py \
            ${opt} \
            --backend ${backend} \
            --snapshots ${expdir}/results/snapshot.ep.* \
            --out ${expdir}/results/${recog_model} \
            --num ${n_average} \
            --max-epoch ${max_epoch}
    fi

    pids=() # initialize pids
    for rtask in ${test}; do
    (
        decode_dir=decode_${rtask}_$(basename ${decode_config%.*})_${use_valbest_average}_max_epoch_${max_epoch}
        feat_recog_dir=$data/$rtask
	echo $feat_recog_dir 
        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/${train_set}_${bpemode}_${vocab_size}.json
        #### use CPU for decoding
        ngpu=0

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --config ${decode_config} \
            --beam-size 60 \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --batchsize 0 \
            --recog-json ${feat_recog_dir}/split${nj}utt/${train_set}_${bpemode}_${vocab_size}.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${recog_model} #\
            # --rnnlm ${lmexpdir}/rnnlm.model.best
        score_sclite.sh --bpe ${vocab_size} --bpemodel ${bpe_model}.model --wer true ${expdir}/${decode_dir} ${dict} 
    )&
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    done
    done
    echo "Finished"
fi
use_valbest_average=false
if [ ! -z $step17 ]; then
    echo "stage 3: Decoding"    
    train_config=conf/train_pytorch_conformer.yaml
    nj=100
    for expname in train_clean_100_conformer_12enc_6dec_noSpecAug_lr_10_batch_size_32_pytorch;do
    expdir=$exp/${expname}
    for test in $recog_set; do
    if [[ $(get_yaml.py ${train_config} model-module) = *transformer* ]]; then
        # Average ASR models
        if ${use_valbest_average}; then
            [ -f ${expdir}/results/model.val5.avg.best ] && rm ${expdir}/results/model.val5.avg.best
            recog_model=model.val${n_average}.avg.best
            opt="--log ${expdir}/results/log"
        else
            [ -f ${expdir}/results/model.last5.avg.best ] && rm ${expdir}/results/model.last5.avg.best
            recog_model=model.last${n_average}.avg.best
            opt="--log"
        fi
        average_checkpoints.py \
            ${opt} \
            --backend ${backend} \
            --snapshots ${expdir}/results/snapshot.ep.* \
            --out ${expdir}/results/${recog_model} \
            --num ${n_average}
    fi

    pids=() # initialize pids
    for rtask in ${test}; do
    (
        decode_dir=decode_${rtask}_$(basename ${decode_config%.*})_${use_valbest_average}
        feat_recog_dir=$data/$rtask
        echo $feat_recog_dir 
        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/${train_set}_${bpemode}_${vocab_size}.json
        #### use CPU for decoding
        ngpu=0

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --config ${decode_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --batchsize 0 \
            --recog-json ${feat_recog_dir}/split${nj}utt/${train_set}_${bpemode}_${vocab_size}.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${recog_model} #\
            # --rnnlm ${lmexpdir}/rnnlm.model.best
        score_sclite.sh --bpe ${vocab_size} --bpemodel ${bpe_model}.model --wer true ${expdir}/${decode_dir} ${dict}
    )&
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    done
    done
    echo "Finished"
fi
