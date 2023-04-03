seed=$1
dataset_name=pdbbind:set_name=general-set-2019-coreset-2016,cutoffs=5-5-5,seed=$seed
data_path=~/dataset
rm -rf ~/dataset
batch_size=$2
lr=1e-5
ckpt_path=$3
save_dir=$4

# modify this line to your machine, numer of gpus
python -m torch.distributed.launch --nproc_per_node=8 \
  $(which fairseq-train) \
  --user-dir "$(realpath ./dynaformer)" \
  --num-workers 16 --ddp-backend=legacy_ddp \
  --finetune-from-model $ckpt_path \
  --dataset-name "$dataset_name" \
  --dataset-source pyg --data-path "$data_path" \
  --batch-size $batch_size --data-buffer-size 40 \
  --task graph_prediction_with_flag --criterion l1_loss_with_flag --arch graphormer_base --num-classes 1 \
  --lr $lr --end-learning-rate 1e-9 --lr-scheduler polynomial_decay --power 1 \
  --warmup-updates 10280 --total-num-update 50900 --max-update 50900 --update-freq 1 --patience 50 \
  --encoder-layers 4 --encoder-attention-heads 32 \
  --encoder-embed-dim 512 --encoder-ffn-embed-dim 512 \
  --attention-dropout 0.1 --act-dropout 0.1 --dropout 0.1 --weight-decay 1e-5 \
  --optimizer adam --adam-betas "(0.9,0.999)" --adam-eps 1e-8 --flag-m 1 --flag-step-size 0.0001 --flag-mag 0.001 --clip-norm 5 \
  --fp16 --save-dir "$save_dir" --seed $seed --fingerprint \
  --max-nodes 600 --dist-head gbf3d \
  --num-dist-head-kernel 256 --num-edge-types 16384
