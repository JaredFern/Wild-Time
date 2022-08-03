#CORAL
python main.py --dataset=precipitation --method=coral --eval_next_timesteps=4 --coral_lambda=0.9 --momentum=0.99 --num_groups=3 --group_size=4 --mini_batch_size=128 --train_update_iter=500 --lr=0.001 --split_time=7 --random_seed=1 --log_dir=./checkpoints

#GroupDRO
python main.py --dataset=precipitation --method=groupdro --eval_next_timesteps=4 --num_groups=3 --group_size=4 --mini_batch_size=128 --train_update_iter=500 --lr=0.001 --split_time=7 --random_seed=1 --log_dir=./checkpoints

#IRM
python main.py --dataset=precipitation --method=irm --eval_next_timesteps=4 --irm_lambda=1.0 --irm_penalty_anneal_iters=0 --num_groups=3 --group_size=4 --mini_batch_size=128 --train_update_iter=500 --lr=0.001 --split_time=7 --random_seed=1 --log_dir=./checkpoints

#ERM
python main.py --dataset=precipitation --method=erm --eval_next_timesteps=4 --mini_batch_size=128 --train_update_iter=500 --lr=0.001 --split_time=7 --random_seed=1 --log_dir=./checkpoints

#LISA
python main.py --dataset=precipitation --method=erm --lisa --eval_next_timesteps=4 --mix_alpha=2.0 --mini_batch_size=128 --train_update_iter=500 --lr=0.001 --split_time=7 --random_seed=1 --log_dir=./checkpoints

#A-GEM
python main.py --dataset=precipitation --method=agem --buffer_size=1000 --eval_next_timesteps=4 --mini_batch_size=128 --train_update_iter=500 --lr=0.001 --split_time=7 --random_seed=1 --log_dir=./checkpoints

#EWC
python main.py --dataset=precipitation --method=ewc --ewc_lambda=0.1 --online --eval_next_timesteps=4 --mini_batch_size=128 --train_update_iter=500 --lr=0.001 --split_time=7 --random_seed=1 --log_dir=./checkpoints

#Fine-tuning
python main.py --dataset=precipitation --method=ft --eval_next_timesteps=4 --mini_batch_size=128 --train_update_iter=500 --lr=0.001 --split_time=7 --random_seed=1 --log_dir=./checkpoints

#SI
python main.py --dataset=precipitation --method=si --si_c=0.1 --epsilon=0.001 --eval_next_timesteps=4 --mini_batch_size=128 --train_update_iter=500 --lr=0.001 --split_time=7 --random_seed=1 --log_dir=./checkpoints