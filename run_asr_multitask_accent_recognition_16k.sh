#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;


cuda_cmd="slurm.pl --quiet --exclude=node0[3-8]"
decode_cmd="slurm.pl --quiet --exclude=node0[3-8]"
cmd="slurm.pl --quiet --exclude=node0[3-8]"

#cuda_cmd="run.pl"
#decode_cmd="run.pl"
#cmd="run.pl"
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
log=100
vocab_size=2000
bpemode=bpe
# feature configuration
do_delta=false

train_multitask_config=conf/e2e_asr_transformer_multitask_accent.yaml
lm_config=conf/espnet_lm.yaml
decode_config=conf/espnet_decode.yaml
preprocess_config=conf/espnet_specaug.yaml

# rnnlm related
lm_resume=         # specify a snapshot file to resume LM training
lmtag=0             # tag for managing LMs

# decoding parameter
recog_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'
n_average=5

# others
accum_grad=2
n_iter_processes=2
lsm_weight=0.0
epochs=30
elayers=6
batch_size=32
recog_mode="accent"
use_valbest_average=true

#Select which pre-trained model features to extract, [wavlm | wav2vec2(xlsr53)]
model_type=wavlm
feat_layer=16 # which layer to extract

# exp tag
tag="base" # tag for managing experiments.

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
# set -u
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
#  echo $steps
  for x in $(echo $steps|sed 's/[,:]/ /g'); do
     index=$(printf "%02d" $x);
    # echo $index
     declare step$index=1
  done
fi

data=data
exp=exp
dump_features=data-${model_type}

train_set="train"
valid_set="cv_all"
recog_set="cv_all test"


if [ ! -z $step01 ]; then
   echo "extracting pretrain features and cmvn for 16k data"
   for i in $train_set $recog_set;do
      utils/fix_data_dir.sh $data/$i
      cp -r $data $dump_features
      ${cuda_cmd} --gpu 1 extract_feature.log python extract_feature.py $dump_features $feat_layer
      ./utils/fix_data_dir.sh $dump_features/$i
   done

   compute-cmvn-stats scp:$dump_features/${train_set}/feats.scp $dump_features/${train_set}/cmvn.ark
   echo "step01 Extracting pretrain features and cmvn Done"
fi
data=$dump_features
if [ ! -z $step02 ]; then
   echo "generate label file and dump features for track2:E2E"

   for x in ${train_set} ;do
       dump.sh --cmd "$cmd" --nj $nj  --do_delta false \
          $data/$x/feats.scp $data/${train_set}/cmvn.ark $data/$x/dump/log $data/$x/dump
   done

   for x in $train_set $recog_set;do 
       dump.sh --cmd "$cmd" --nj $nj  --do_delta false \
          $data/$x/feats.scp $data/${train_set}/cmvn.ark $data/$x/dump_${train_set}/log $data/$x/dump_${train_set}
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
    for i in $train_set $recog_set $valid_set;do
       data2json.sh --nj 10 --cmd "${cmd}" --feat $data/$i/dump_${train_set}/feats.scp --bpecode ${bpe_model}.model \
           $data/$i ${dict} > ${data}/$i/${train_set}_${bpemode}_${vocab_size}.json
    done
    echo "stage 04: Make Json Labels Done"
fi

epochs=120
pretrained_model=pretrained-model/16k/model.val5.avg.best
if [ ! -z $step05 ]; then
    epoch_stage=0
    accentWeight=0.3
    asrWeight=1
    intermediate_ctc_weight=0
    intermediate_ctc_layer="12"
    transformer_lr=5
    batch_size=32
    expname=${train_set}_transformer_12_enc_6_dec_asrWeight_${asrWeight}_accentWeight_${accentWeight}_intermediate_ctc_layer_${intermediate_ctc_layer}_withSpecAug_lr_${transformer_lr}_batch_size_${batch_size}_${backend} 
    expdir=$exp/${expname}
    mkdir -p ${expdir}
    echo "stage 05: Network Training"
    ngpu=1
    if  [ ${epoch_stage} -gt 0 ]; then
        echo "stage 05: Resume network from epoch ${epoch_stage}"
        resume=${exp}/${expname}/results/snapshot.ep.${epoch_stage}
    fi  
    train_multitask_config=conf/e2e_asr_transformer_multitask_accent.yaml
    preprocess_config=conf/specaug.yaml
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        asr_train_multitask_accent.py \
        --config ${train_multitask_config} \
        --preprocess-conf ${preprocess_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --outdir ${expdir}/results \
        --debugmode ${debugmode} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --verbose ${verbose} \
        --resume ${resume} \
        --epochs ${epochs} \
        --batch-size ${batch_size} \
        --dict ${dict} \
        --num-save-ctc 0 \
        --NumClass 8 \
        --asrWeight ${asrWeight} \
        --accentWeight ${accentWeight} \
        --batch-size ${batch_size} \
        --transformer-lr ${transformer_lr} \
        --utt2LabelTrain $data/${train_set}/utt2IntLabel \
        --utt2LabelValid $data/${valid_set}/utt2IntLabel \
        --intermediate-ctc-weight ${intermediate_ctc_weight} \
        --intermediate-ctc-layer ${intermediate_ctc_layer} \
        --train-json $data/${train_set}/${train_set}_${bpemode}_${vocab_size}.json \
        --valid-json $data/${valid_set}/${train_set}_${bpemode}_${vocab_size}.json \
        --enc-init ${pretrained_model} \
        --enc-init-mods='encoder.' \
        --dec-init ${pretrained_model} \
        --dec-init-mods='decoder.decoders'
fi


if [ ! -z $step06 ]; then
    train_multitask_config=conf/e2e_asr_transformer_multitask_accent.yaml
    for expname in train_transformer_12_enc_6_dec_asrWeight_1_accentWeight_0.1_intermediate_ctc_layer_12_withSpecAug_lr_5_batch_size_32_pytorch;do
    expdir=$exp/${expname}
    for test in $recog_set;do
    nj=30
    if [[ $(get_yaml.py ${train_multitask_config} model-module) = *transformer* ]]; then
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
        echo "$opt"
        average_checkpoints.py \
            ${opt} \
            --backend ${backend} \
            --snapshots ${expdir}/results/snapshot.ep.* \                                                                                                                           │·················································································
            --out ${expdir}/results/${recog_model} \                                                                                                                                │·················································································
            --num ${n_average} 

    if [[ "${recog_mode}" == "asr" ]];then
         decode_dir=asr_decode_${test}_max_epoch_$n
    else
         decode_dir=accent_decode_${test}_max_epoch_$n
    fi
    echo "decoder mode: ${recog_mode}, decode_dir=${decode_dir}"
    # split data
    dev_root=$data/${test}
    splitjson.py --parts ${nj} ${dev_root}/${train_set}_${bpemode}_${vocab_size}.json
    #### use CPU for decoding
    ngpu=0
    ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
        asr_recog_for_multitask_accent.py \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --batchsize 0 \
        --recog-json ${dev_root}/split${nj}utt/${train_set}_${bpemode}_${vocab_size}.JOB.json \
        --result-label ${expdir}/${decode_dir}/data.JOB.json \
        --model ${expdir}/results/${recog_model} \
        --recog-mode ${recog_mode} 

    concatjson.py ${expdir}/${decode_dir}/data.*.json >  ${expdir}/${decode_dir}/${train_set}_${bpemode}_${vocab_size}.json
    python scripts/parse_track1_jsons.py  ${expdir}/${decode_dir}/${train_set}_${bpemode}_${vocab_size}.json ${expdir}/${decode_dir}/result.txt
    python scripts/parse_track1_jsons.py  ${expdir}/${decode_dir}/${train_set}_${bpemode}_${vocab_size}.json ${expdir}/${decode_dir}/result.txt > ${expdir}/${decode_dir}/result_acc.txt
    fi
    echo "Decoding finished"
  done
  done
  done
fi


#### Test for 16k data
if [ ! -z $step07 ]; then
    train_multitask_config=conf/e2e_asr_transformer_multitask_accent.yaml
    expdir=pretrained_model/16k_model/
    for test in $recog_set;do
    nj=30
    recog_model=model.val5.avg.best
    if [[ "${recog_mode}" == "asr" ]];then
         decode_dir=asr_decode_${test}_max_epoch_$n
    else
         decode_dir=accent_decode_${test}_max_epoch_$n
    fi
    echo "decoder mode: ${recog_mode}, decode_dir=${decode_dir}"
    # split data
    dev_root=$data/${test}
    splitjson.py --parts ${nj} ${dev_root}/${train_set}_${bpemode}_${vocab_size}.json
    #### use CPU for decoding
    ngpu=0
    ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
        asr_recog_for_multitask_accent.py \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --batchsize 0 \
        --recog-json ${dev_root}/split${nj}utt/${train_set}_${bpemode}_${vocab_size}.JOB.json \
        --result-label ${expdir}/${decode_dir}/data.JOB.json \
        --model ${expdir}/${recog_model} \
        --recog-mode ${recog_mode} 


    concatjson.py ${expdir}/${decode_dir}/data.*.json >  ${expdir}/${decode_dir}/${train_set}_${bpemode}_${vocab_size}.json
    python scripts/parse_track1_jsons.py  ${expdir}/${decode_dir}/${train_set}_${bpemode}_${vocab_size}.json ${expdir}/${decode_dir}/result.txt
    python scripts/parse_track1_jsons.py  ${expdir}/${decode_dir}/${train_set}_${bpemode}_${vocab_size}.json ${expdir}/${decode_dir}/result.txt > ${expdir}/${decode_dir}/result_acc.txt
    fi
    echo "Decoding finished"
  done
fi
