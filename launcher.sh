for t in 1 4
do
	for lr in 0.0001 0.005 0.001
	do
		for run in 1 2 3 4 5
		do
			echo $run $t $lr
			python async_dqn.py --experiment breakout --game "MountainCar-v0" --num_concurrent ${t} --learning_rate ${lr} > dqn_${t}_${lr//.}_${run}.txt
			#python async_dqn.py --experiment breakout --game "MountainCar-v0" --num_concurrent ${t} --learning_rate ${lr}
		done
	done
done
