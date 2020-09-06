layers=$1
epoch=$2
batch_size=$3
steps=$4
multiplier=$5
mode1=$6
mode2=$7
remark=$8
#for seed in {6..12..6}
for seed in 2
do
    echo $seed
    # update alpha 
    #srun --mpi=pmi2 --partition=AD --gres=gpu:1 -n1 --ntasks-per-node=1 --kill-on-bad-exit=1 --job-name=29 \
    python train_search_cost.py --epochs ${epoch} --batch_size ${batch_size} --layers $layers --init_channels 16 --seed $seed --drop_path_prob 0 --${mode1} --${mode2} --cal_stat \
    --steps ${steps} --multiplier $multiplier --remark ${remark}

done
