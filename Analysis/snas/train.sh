network=$1
layers=$2
epoch=$3
batch_size=$4
order=$5
steps=$6
multiplier=$7
mode1=$8
mode2=$9
mode3=${10}
mode4=${11}
mode5=${12}
remark=${13}
# for seed in {6..12..6}
for seed in 6
	do
	echo $seed
	python train_search.py --${network} --epochs ${epoch} --batch_size ${batch_size} --layers $layers --init_channels 16 --seed $seed --nsample 1 --${order} --drop_path_prob 0 --${mode1} --${mode2} \
	--gen_max_child --edge_reward --steps ${steps} --multiplier $multiplier --${mode3} --${mode4} --remark_fur ${remark}
done
