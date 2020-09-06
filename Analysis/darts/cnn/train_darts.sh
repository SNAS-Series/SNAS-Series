layers=$1
batch_size=$2
steps=$3
multiplier=$4
remark=$5

#for seed in {6..12..6}
for seed in 2
do
    echo $seed
    # update alpha 
    #srun --mpi=pmi2 --partition=AD --gres=gpu:1 -n1 --ntasks-per-node=1 --kill-on-bad-exit=1 --job-name=29 \
    python train_search.py --epochs 50 --batch_size ${batch_size} --layers $layers --init_channels 16 --seed $seed --drop_path_prob 0 \
    --steps ${steps} --multiplier $multiplier --remark ${remark}

done
