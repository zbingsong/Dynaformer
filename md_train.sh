# set -o xtrace
# set -x
ulimit -c unlimited
# [ -z "${n_gpu}" ] && n_gpu=$(nvidia-smi -L | wc -l)
n_gpu=1
lr=1e-4
end_lr=1e-9
max_epoch=100
layers=4
hidden_size=512
ffn_size=512
num_head=32
batch_size=20
clip_norm=5

update_freq=1
total_steps=$((320000*(max_epoch+1)/batch_size/n_gpu/update_freq))
warmup_steps=$((total_steps*10/100))
seed=2022

dataset_name="hybrid:set_name=md-refined2019-5-5-5+general-set-2019-coreset-2016,cutoffs=5-5-5,seed=2022"
data_path=$(realpath /data/bzeng/Dynaformer/dataset)
save_path=$(realpath /data/bzeng/Dynaformer)
dropout=0.1
act_dropout=0.1
attn_dropout=0.1
weight_decay=0.00001
sandwich_ln="false"

adam_betas="(0.9,0.999)"
adam_eps=1e-8

save_prefix="MD"
flag=true
flag_m=3
flag_step_size=0.001
flag_mag=0.01

dist_head="gbf3d"
num_dist_head_kernel=256
num_edge_types=$((512*32))
task="graph_prediction"
loss="l2_loss"
patience="50"
fingerprint="true"

test_set="pdbbind:set_name=refined-set-2019-coreset-2016,cutoffs=5-5-5,seed=2022"
ddp_options=""

if [ "$flag" = 'true' ]; then
  task="${task}_with_flag"
  loss="${loss}_with_flag"
fi

OLDIFS=$IFS
IFS=':' #setting comma as delimiter
read -a strarr <<< "$dataset_name" #reading str as an array as tokens separated by IFS
IFS=$OLDIFS

hyperparams="${save_prefix}-dataset${strarr[0]}_${strarr[1]}-lr$lr-end_lr$end_lr-epoch${max_epoch}-task${task}-loss${loss}"
hyperparams+="-L${layers}D${hidden_size}F${ffn_size}H${num_head}BS$((batch_size*n_gpu*update_freq))CLIP${clip_norm}"
hyperparams+="/dp${dropout}-attn_dp${attn_dropout}-act_dp${act_dropout}-wd${weight_decay}-sandwich${sandwich_ln}"
if [ "$flag" = 'true' ]; then
  hyperparams+="-flag${flag}-flag_m${flag_m}-stepsize${flag_step_size}-mag${flag_mag}"
fi

hyperparams+="-dist_head${dist_head}"
if [ "${dist_head}" = "gbf" ] || [ "${dist_head}" = "gbf3d" ]; then
  hyperparams+="-ndhk${num_dist_head_kernel}-net${num_edge_types}"
fi
hyperparams+="-SEED${seed}"


save_dir=$save_path/$hyperparams
tsb_dir=$save_dir/tsb
WANDB_PROJECT_NAME="MD-bind"
export WANDB_NAME="$hyperparams"
export WANDB_RUN_ID=`echo -n $save_dir | md5sum | awk '{print $1}'`
export WANDB_SAVE_DIR=$save_dir/wandb

mkdir -p "$save_dir"
mkdir -p "$WANDB_SAVE_DIR"

echo -e "\n\n"
echo "=====================================ARGS======================================"
echo "arg0: $0"
echo "seed: ${seed}"
echo "batch_size: $((batch_size*n_gpu*update_freq))"
echo "n_layers: ${layers}"
echo "lr: ${lr}"
echo "warmup_steps: ${warmup_steps}"
echo "total_steps: ${total_steps}"
echo "max_epoch: ${max_epoch}"
echo "clip_norm: ${clip_norm}"
echo "hidden_size: ${hidden_size}"
echo "ffn_size: ${ffn_size}"
echo "num_head: ${num_head}"
echo "update_freq: ${update_freq}"
echo "dropout: ${dropout}"
echo "attn_dropout: ${attn_dropout}"
echo "act_dropout: ${act_dropout}"
echo "weight_decay: ${weight_decay}"
echo "adam_betas: ${adam_betas}"
echo "adam_eps: ${adam_eps}"
echo "flag_m: ${flag_m}"
echo "flag_step_size: ${flag_step_size}"
echo "flag_mag: ${flag_mag}"
echo "save_dir: ${save_dir}"
echo "tsb_dir: ${tsb_dir}"
echo "data_dir: ${data_path}"
echo "dist_head: ${dist_head}"
echo "num_dist_head_kernel: $num_dist_head_kernel"
echo "num_edge_types: $num_edge_types"
echo "==============================================================================="

# ENV
echo -e "\n\n"
echo "======================================ENV======================================"
echo 'Environment'
ulimit -c unlimited;
echo -e "\n\nhostname"
hostname
echo -e "\n\nnvidia-smi"
nvidia-smi
echo "which python"
which python
echo "PATH"
echo $PATH
echo "torch version"
python -c "import torch; print(torch.__version__)"
echo "ddp_options"
echo $ddp_options
echo "==============================================================================="

echo -e "\n\n"
echo "==================================ACTION ARGS==========================================="
action_args=""
if [ "$sandwich_ln" = "true" ]; then
  action_args+="--sandwich-ln "
fi

if [ "$flag" = 'true' ]; then
  action_args+="--flag-m $flag_m --flag-step-size $flag_step_size --flag-mag $flag_mag "
fi

if [ "$fingerprint" = 'true' ]; then
  action_args+="--fingerprint "
fi

echo "action_args: ${action_args}"
echo "========================================================================================"


python -m torch.distributed.run --nproc_per_node=${n_gpu} --master_port 29501 ${ddp_options} \
  $(which fairseq-train) \
  --user-dir "$(realpath ./dynaformer)" \
  --num-workers 4 --ddp-backend=legacy_ddp \
  --dataset-name "$dataset_name" \
  --dataset-source pyg --data-path "$data_path" \
  --batch-size $batch_size --data-buffer-size 20 \
  --task $task --criterion $loss --arch graphormer_base --num-classes 1 \
  --lr $lr --end-learning-rate $end_lr --lr-scheduler polynomial_decay --power 1 \
  --warmup-updates $warmup_steps --total-num-update $total_steps --max-update $total_steps --update-freq $update_freq --patience $patience \
  --encoder-layers $layers --encoder-attention-heads $num_head \
  --encoder-embed-dim $hidden_size --encoder-ffn-embed-dim $ffn_size \
  --attention-dropout $attn_dropout --act-dropout $act_dropout --dropout $dropout --weight-decay $weight_decay \
  --optimizer adam --adam-betas $adam_betas --adam-eps $adam_eps $action_args --clip-norm $clip_norm \
  --fp16 --save-dir "$save_dir" --tensorboard-logdir $tsb_dir --seed $seed \
  --max-nodes 600 --dist-head $dist_head \
  --num-dist-head-kernel $num_dist_head_kernel --num-edge-types $num_edge_types 2>&1 | tee "$save_dir/train_log.txt"
