network=$1
layers=$2
epoch=$3
batch_size=$4
order=$5
steps=$6
multiplier=$7
mode1=$8
mode2=$9
remark=${10}
# for seed in {6..12..6}
for seed in 6
	do
	echo $seed
	#srun --mpi=pmi2 --partition=AD --gres=gpu:1 -n1 --ntasks-per-node=1 --kill-on-bad-exit=1 --job-name=29 \
	python train_search_cost_entropy_loss.py --${network} --epochs ${epoch} --batch_size ${batch_size} --layers $layers --init_channels 16 --seed $seed --nsample 1 --${order} --drop_path_prob 0 --${mode1} --${mode2} \
	--gen_max_child --edge_reward --steps ${steps} --multiplier $multiplier --remark_fur ${remark}
done
