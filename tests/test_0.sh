#!/bin/bash 

python ../rl-toolkit/sb3/tune_sb3.py -e conservation-v2 --study-name trash_test -a sac --n-timesteps 10 --n-trials=5
python ../rl-toolkit/sb3/train_sb3.py -e conservation-v2 --study-name trash_test -a sac --n-timesteps 10

rm -r studies/

python ../rl-toolkit/sb3/tune_sb3.py -e conservation-v2 --study-name trash_test -a ppo --n-timesteps 10 --n-trials=5
python ../rl-toolkit/sb3/train_sb3.py -e conservation-v2 --study-name trash_test -a ppo --n-timesteps 10

rm -r studies/

python ../rl-toolkit/sb3/tune_sb3.py -e conservation-v2 --study-name trash_test -a ddpg --n-timesteps 10 --n-trials=5
python ../rl-toolkit/sb3/train_sb3.py -e conservation-v2 --study-name trash_test -a ddpg --n-timesteps 10

rm -r studies/

python ../rl-toolkit/sb3/tune_sb3.py -e conservation-v2 --study-name trash_test -a a2c --n-timesteps 10 --n-trials=5
python ../rl-toolkit/sb3/train_sb3.py -e conservation-v2 --study-name trash_test -a a2c --n-timesteps 10

rm -r studies/

python ../rl-toolkit/sb3/tune_sb3.py -e conservation-v2 --study-name trash_test -a td3 --n-timesteps 10 --n-trials=5
python ../rl-toolkit/sb3/train_sb3.py -e conservation-v2 --study-name trash_test -a td3 --n-timesteps 10

rm -r studies