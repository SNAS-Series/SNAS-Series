path=$1
network=$2
layers=$3
batch_size=$4
steps=$5
multiplier=$6
remark=$7
#for seed in {6..12..6}
for seed in 6
do
    echo $seed
    # calculate cost at initialization
    #srun --mpi=pmi2 --partition=AD --gres=gpu:1 -n1 --ntasks-per-node=1 --kill-on-bad-exit=1 --job-name=29 \
    python train_search.py --${network} --epochs 50 --batch_size ${batch_size} --layers $layers --init_channels 16 --seed $seed --nsample 1 --order --drop_path_prob 0 \
    --gen_max_child --edge_reward --steps ${steps} --multiplier $multiplier --child_reward_stat --remark_fur ${remark} 
   # random sample
    echo $batch_size
    #srun --mpi=pmi2 --partition=AD --gres=gpu:1 -n1 --ntasks-per-node=1 --kill-on-bad-exit=1 --job-name=29 \
    python train_search.py --${network} --epochs 150 --batch_size ${batch_size} --layers $layers --init_channels 16 --seed $seed --nsample 1 --order --drop_path_prob 0 \
    --gen_max_child --edge_reward --steps ${steps} --multiplier $multiplier --random_sample --remark_fur ${remark}
    for name in $path/*
    do
        #if [[ $name == *_layer_${layers}_*_seed_${seed}_*_steps_${steps}_${remark} ]]
        if [[ $name == *_layer_${layers}_*seed_${seed}_*random_sample_${remark} ]]
        then
            echo $name
            #srun --mpi=pmi2 --partition=AD --gres=gpu:1 -n1 --ntasks-per-node=1 --kill-on-bad-exit=1 --job-name=25 \
            python train_search.py --${network} --epochs 50 --batch_size ${batch_size} --layers $layers --seed $seed --child_reward_stat  --nsample 1 --order --drop_path_prob 0 \
            --gen_max_child --edge_reward --steps $steps --multiplier $multiplier --init_channels 16 --resume --resume_epoch 0 \
            --resume_path $name --remark_fur random_sample_150epoch_steps_cal_child_stat_${remark}
        fi
    done
done
