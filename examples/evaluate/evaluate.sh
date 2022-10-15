save_dir=$1
echo "# On $save_dir"

layers=4
num_head=32
hidden_size=512
ffn_size=512
seed=2022
dist_head=gbf3d
num_dist_head_kernel=256
num_edge_types=16384
loss="l1_loss"

python dynaformer/evaluate/evaluate.py \
  --split "test" --suffix "_pdbbind2016" \
  --user-dir "$(realpath ./dynaformer)" \
  --num-workers 16 --ddp-backend=legacy_ddp \
  --dataset-name "pdbbind:set_name=refined-set-2019-coreset-2016,cutoffs=5-5-5,seed=2022" \
  --dataset-source pyg --data-path ~ \
  --batch-size 1 --data-buffer-size 20 \
  --task graph_prediction_with_flag --criterion ${loss}_with_flag --arch graphormer_base --num-classes 1 \
  --encoder-layers $layers --encoder-attention-heads $num_head \
  --encoder-embed-dim $hidden_size --encoder-ffn-embed-dim $ffn_size \
  --fp16 --save-dir "$save_dir" \
  --max-nodes 600 --dist-head $dist_head  \
  --num-dist-head-kernel $num_dist_head_kernel --num-edge-types $num_edge_types --fingerprint



