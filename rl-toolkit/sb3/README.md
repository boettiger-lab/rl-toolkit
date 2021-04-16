The scripts in here have not been rigorously stress-tested so please send a message on the rl slack channel if an issue comes up. Run `python tune_sb3.py/train_sb3.py -h` to see what the requisite command line flags are. 

General Usage:`tune_sb3.py` is the tuning script that uses SB3. At the moment, it only supports SAC, TD3, DDPG, A2C and PPO. It will work with any environment, but make sure that you import the corresponding gym in the import block. If you want to change the hyperparameters, you will need to change the trial suggest blocks in `hyperparams_utils.py`. After running this tuning script, a .db table will be saved in `studies/` that will record all the trials' performance.  

`train_sb3.py` currently has the ability to take manually inputted hyperparameters or the hyperparameters from the best tuning trial of the specified study. To use manually inputted hyperparameters, create a yaml file with the environment's name as the title. In the yaml file, specify the algorithm and hyperparameters as follows:
```
algorithm:
  hyperparameter: value
```