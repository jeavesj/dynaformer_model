seed=2022
cv=0
dataset_name=pdbbind:set_name=general-set-2019-coreset-2016,cutoffs=5-5-5,seed=$seed,cv=$cv
data_path=~/dataset
batch_size=8
lr=1e-4
ckpt_path="path/to/ckpt"
save_dir=~/finetune-layer4-l1loss-seed$seed-fold$cv-final

rm -rf ~/dataset

python -m torch.distributed.launch --nproc_per_node=8 \
  $(which fairseq-train) \
  --user-dir "$(realpath ./dynaformer)" \
  --num-workers 16 --ddp-backend=legacy_ddp \
  --finetune-from-model $ckpt_path \
  --dataset-name "$dataset_name" \
  --dataset-source pyg --data-path "$data_path" \
  --batch-size $batch_size --data-buffer-size 20 \
  --task graph_prediction_with_flag --criterion l1_loss_with_flag --arch graphormer_base --num-classes 1 \
  --lr $lr --end-learning-rate 1e-9 --lr-scheduler polynomial_decay --power 1 \
  --warmup-updates 1000 --total-num-update 10000 --max-update 10000 --update-freq 1 --patience 50 \
  --encoder-layers 4 --encoder-attention-heads 32 \
  --encoder-embed-dim 512 --encoder-ffn-embed-dim 512 \
  --attention-dropout 0.1 --act-dropout 0.1 --dropout 0.1 --weight-decay 1e-5 \
  --optimizer adam --adam-betas "(0.9,0.999)" --adam-eps 1e-8 --flag-m 3 --flag-step-size 0.001 --flag-mag 0.01 --clip-norm 5 \
  --fp16 --save-dir "$save_dir" --seed $seed --fingerprint \
  --max-nodes 512 --droppath-prob 0.0 --dist-head gbf3d \
  --num-dist-head-kernel 256 --num-edge-types 16384

python dynaformer/evaluate/evaluate.py \
  --split "test" --suffix "_pdbbind2016" \
  --user-dir "$(realpath ./dynaformer)" \
  --num-workers 16 --ddp-backend=legacy_ddp \
  --dataset-name "pdbbind:set_name=refined-set-2019-coreset-2016,cutoffs=5-5-5,seed=2022" \
  --dataset-source pyg --data-path "$data_path" \
  --batch-size $batch_size --data-buffer-size 20 \
  --task graph_prediction_with_flag --criterion l1_loss_with_flag --arch graphormer_base --num-classes 1 \
  --lr $lr --end-learning-rate 1e-9 --lr-scheduler polynomial_decay --power 1 \
  --warmup-updates 1000 --total-num-update 10000 --max-update 10000 --update-freq 1 --patience 50 \
  --encoder-layers 4 --encoder-attention-heads 32 \
  --encoder-embed-dim 512 --encoder-ffn-embed-dim 512 \
  --attention-dropout 0.1 --act-dropout 0.1 --dropout 0.1 --weight-decay 1e-5 \
  --optimizer adam --adam-betas "(0.9,0.999)" --adam-eps 1e-8 --flag-m 3 --flag-step-size 0.001 --flag-mag 0.01 --clip-norm 5 \
  --fp16 --save-dir "$save_dir" --seed $seed --fingerprint \
  --max-nodes 512 --droppath-prob 0.0 --dist-head gbf3d \
  --num-dist-head-kernel 256 --num-edge-types 16384


python dynaformer/evaluate/evaluate.py \
  --split "test" --suffix "_pdbbind2013" \
  --user-dir "$(realpath ./dynaformer)" \
  --num-workers 16 --ddp-backend=legacy_ddp \
  --dataset-name "pdbbind:set_name=refined-set-2019-coreset-2013,cutoffs=5-5-5,seed=2022" \
  --dataset-source pyg --data-path "$data_path" \
  --batch-size $batch_size --data-buffer-size 20 \
  --task graph_prediction_with_flag --criterion l1_loss_with_flag --arch graphormer_base --num-classes 1 \
  --lr $lr --end-learning-rate 1e-9 --lr-scheduler polynomial_decay --power 1 \
  --warmup-updates 1000 --total-num-update 10000 --max-update 10000 --update-freq 1 --patience 50 \
  --encoder-layers 4 --encoder-attention-heads 32 \
  --encoder-embed-dim 512 --encoder-ffn-embed-dim 512 \
  --attention-dropout 0.1 --act-dropout 0.1 --dropout 0.1 --weight-decay 1e-5 \
  --optimizer adam --adam-betas "(0.9,0.999)" --adam-eps 1e-8 --flag-m 3 --flag-step-size 0.001 --flag-mag 0.01 --clip-norm 5 \
  --fp16 --save-dir "$save_dir" --seed $seed --fingerprint \
  --max-nodes 512 --droppath-prob 0.0 --dist-head gbf3d \
  --num-dist-head-kernel 256 --num-edge-types 16384


python dynaformer/evaluate/evaluate.py \
  --split "test" --suffix "_md" \
  --user-dir "$(realpath ./dynaformer)" \
  --num-workers 16 --ddp-backend=legacy_ddp \
  --dataset-name "mddata:set_name=md-refined2019-5-5-5,seed=2022" \
  --dataset-source pyg --data-path "$data_path" \
  --batch-size $batch_size --data-buffer-size 20 \
  --task graph_prediction_with_flag --criterion l1_loss_with_flag --arch graphormer_base --num-classes 1 \
  --lr $lr --end-learning-rate 1e-9 --lr-scheduler polynomial_decay --power 1 \
  --warmup-updates 1000 --total-num-update 10000 --max-update 10000 --update-freq 1 --patience 50 \
  --encoder-layers 4 --encoder-attention-heads 32 \
  --encoder-embed-dim 512 --encoder-ffn-embed-dim 512 \
  --attention-dropout 0.1 --act-dropout 0.1 --dropout 0.1 --weight-decay 1e-5 \
  --optimizer adam --adam-betas "(0.9,0.999)" --adam-eps 1e-8 --flag-m 3 --flag-step-size 0.001 --flag-mag 0.01 --clip-norm 5 \
  --fp16 --save-dir "$save_dir" --seed $seed --fingerprint \
  --max-nodes 512 --droppath-prob 0.0 --dist-head gbf3d \
  --num-dist-head-kernel 256 --num-edge-types 16384


python dynaformer/evaluate/evaluate.py \
  --split "test" --suffix "_exp" \
  --user-dir "$(realpath ./dynaformer)" \
  --num-workers 16 --ddp-backend=legacy_ddp \
  --dataset-name "exp:set_name=5-5-5" \
  --dataset-source pyg --data-path "$data_path" \
  --batch-size $batch_size --data-buffer-size 20 \
  --task graph_prediction_with_flag --criterion l1_loss_with_flag --arch graphormer_base --num-classes 1 \
  --lr $lr --end-learning-rate 1e-9 --lr-scheduler polynomial_decay --power 1 \
  --warmup-updates 1000 --total-num-update 10000 --max-update 10000 --update-freq 1 --patience 50 \
  --encoder-layers 4 --encoder-attention-heads 32 \
  --encoder-embed-dim 512 --encoder-ffn-embed-dim 512 \
  --attention-dropout 0.1 --act-dropout 0.1 --dropout 0.1 --weight-decay 1e-5 \
  --optimizer adam --adam-betas "(0.9,0.999)" --adam-eps 1e-8 --flag-m 3 --flag-step-size 0.001 --flag-mag 0.01 --clip-norm 5 \
  --fp16 --save-dir "$save_dir" --seed $seed --fingerprint \
  --max-nodes 512 --droppath-prob 0.0 --dist-head gbf3d \
  --num-dist-head-kernel 256 --num-edge-types 16384
