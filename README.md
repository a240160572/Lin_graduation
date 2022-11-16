# Lin_graduation

## Content
The repository structure is adapted from [
gym-pybullet-drones ](https://github.com/utiasDSL/gym-pybullet-drones). The main training, evaluation and visualization code files are under folder `/experiments/learning`.

- `1000_data` contains the 1000 success/failure scenario data for visualization.
- `csv` contains the tensorboard log.
- `model_logs` contains the final trained policies.
> Folder name `08.30_18.08_t_ppo_opt90_25_1_adam_cpu___FIN` is constructed with `date(08.30_18.08)_environment(t)_algorithm(ppo)_note(opt90_25_1_adam_cpu)`.

> The addition note `opt90_25_1_adam_cpu` stands for 90% optimization rate; 25 for PPO batch size; 1 for PPO epoch number; adam optimizer and trained on CPU.


## Example
The training, evaluation and visualization code files are under folder `/experiments/learning`.

### Training
The libraries used in this project are: [stable-baselines3](https://github.com/DLR-RM/stable-baselines3) and [
stable-baselines3-contrib](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib).

- `a0_training_w_dataset.py` is for policy training in the default two-obstacle-environment.
```
$ python3 a0_training_w_dataset.py --it 500 --note trail_training
```
> with arguments
> - `--exp`: define the direction of a policy for continue training.
> - `--it`: define the training iteration (1 iteration = `n_steps`).
> - `--note`: add addition note to the policy folder.
> - `--vel`: set the observation for history obstacle position to obstacle velocity.

```
After calling the training file, the algorithm selection is presented:
Please select the algorithm
 0) PPO; 1) A2C; 2) TRPO; 3) RPPO (seperate indexes with comma)
```
Typing in `0,0,0,1,1,1` represents 3 trainings for PPO and 3 training for A2C. 

### Evaluation
- `a0_model_eva.py` is for the success rate evaluation and trajectory visualization.
```
$ python3 a0_model_eva.py --exp ./simple_model_logs/08.30_18.08_t_ppo_opt90_25_1_adam_cpu___FIN --model 13200 --subs
```
> with arguments
> - `--exp`: define the direction of the policy.
> - `--model`: define which policy to evaluate. Default setting is `success`; `best` or iteration number (like `13200`) can also be used.
> - `--env`: define which environment to load. Default environment is two-obstacle-environment; `orbit` or `act` is also available. 
> - `--subs`: define whether to plot the subplot figure for multiple scenarios.
> - `--it`: define the amount of test scenario for success rate computation.

Choose the policy to evaluate via:
```
Please type the algorithm index to continue training
(0, '0_PPO')
(1, '0_PPO1')
(2, '0_PPO2')
(3, '1_A2C')
(4, '1_A2C1')
(5, '1_A2C2')
(seperate indexes with comma or type 'all')
```
### Animation
The animation code is adapted from [
gym-pybullet-drones ](https://github.com/utiasDSL/gym-pybullet-drones).
The urdf models are under folder `/gym_pybullet_drones/assets`

- `a0_animation.py` is for generating trajectory animation.
```
$ python3 a0_animation.py --exp ./simple_model_logs/08.30_18.08_t_ppo_opt90_25_1_adam_cpu___FIN/0_PPO1/subs_log
```
> `a0_animation.py` reads the `eva_result_{}.npy` file under argument `--exp`
