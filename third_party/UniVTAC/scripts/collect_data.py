import os
import sys
import time
import yaml
import json
import torch
import argparse
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Literal

sys.path.append('.')

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Collect data"
)
parser.add_argument(
    "task",
    type=str,
    help="Task file name",
)
parser.add_argument(
    "config",
    type=str,
    help="Config file name",
    default="demo.yml"
)
parser.add_argument(
    "--episode_num",
    type=int,
    default=-1,
)
parser.add_argument(
    "--start_seed",
    type=int,
    default=-1,
)
parser.add_argument(
    "--max_seed",
    type=int,
    default=-1,
)
parser.add_argument(
    "--gpu",
    type=str,
    default=None,
)

args_cli = parser.parse_args()
if args_cli.gpu is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = args_cli.gpu

# 启动issac lab
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)

# parse the arguments
args_cli.enable_cameras = True
args_cli.num_envs = 1

def get_config(file, default_root:Path, type:Literal['yaml', 'json']):
    if type == 'yaml':
        if file.endswith('.yml') or file.endswith('.yaml'):
            file = Path(file)
        else:
            file = default_root / f'{file}.yml'
        with open(file, 'r') as f:
            config = yaml.load(f.read(), Loader=yaml.FullLoader)
        return config, file
    else:
        if file.endswith('.json'):
            file = Path(file)
        else:
            file = default_root / f'{file}.json'
        with open(file, 'r') as f:
            config = json.load(f)
        return config, file

task_config, task_config_file = get_config(
    args_cli.config, 
    default_root=Path(__file__).parent.parent / 'task_config', 
    type='yaml'
)

if task_config.get('render_frequency', 1) == 0:
    args_cli.livestream = 2

# launch omniverse app, must done before importing anything from omni.isaac
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib
if TYPE_CHECKING:
    from envs._base_task import BaseTask, BaseTaskCfg

log_path = Path('./log')
def log(msg):
    global log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)

    msg = f"[{time.strftime(r'%Y-%m-%d %H:%M:%S')}] {msg}"
    with open(log_path, 'a') as f:
        f.write(msg + '\n')
    print(msg)

def run(task: 'BaseTask', episode_num, use_seed, start_seed, max_seed, save_all=False):
    suc_num, seed = 0, 0
    suc_map = []
    # When save_all is set, every non-erroringattempt is kept regardless of check_success, 
    # and the episode budget counts *saved*  episodes (mostly failures) 
    # rather than successes. suc_map still records the true success/fail label per seed; 
    # the budget is driven by saved hdf5 files.
    saved_num = 0

    if start_seed != -1:
        seed = start_seed
        log(f"Starting from seed {seed}.")
    elif use_seed:
        suc_map_path = task.save_root / 'suc_map.txt'
        if suc_map_path.exists():
            with open(suc_map_path, 'r') as f:
                suc_map = f.read().strip().split(' ')
            suc_num = sum([1 for s in suc_map if s == '1'])
            seed = len(suc_map)
            log(f"Use seed with {suc_num} successful episodes. Starting from seed {seed}.")
        if save_all:
            hdf5_dir = task.save_root / 'hdf5'
            saved_num = len(list(hdf5_dir.glob('*.hdf5'))) if hdf5_dir.exists() else 0
            log(f"save_all: resuming with {saved_num} saved episodes.")

    def budget():
        return saved_num if save_all else suc_num

    mean_steps = 0.0
    while budget() < episode_num and (max_seed == -1 or seed <= max_seed):
        try:
            start_t = time.perf_counter()
            # 初始化环境，
            task.reset(seed=seed)
            # 各个任务的play once
            task.play_once()
            cost_t = time.perf_counter() - start_t
        except Exception as e:
            log(f"[{budget():<3d}] Seed {seed} failed with error: {traceback.format_exc()}")
            suc_map.append('0')
            task.clean_cache(mean_steps=mean_steps, result='error')
        else:
            is_success = task.plan_success and task.check_success() and not task.check_early_stop()
            if is_success or save_all:
                task.save_to_hdf5()
                label = 'success' if is_success else 'fail'
                suc_map.append('1' if is_success else '0')
                if is_success:
                    suc_num += 1
                    if mean_steps > 0:
                        mean_steps = ((suc_num - 1) * mean_steps + task.step_count) / suc_num
                    else:
                        mean_steps = task.step_count
                saved_num += 1
                log(f"[{budget():<3d}] Seed {seed} saved ({label}) in {cost_t:.2f} s.\n"
                    f"steps: {task.step_count:<5d}, save frames: {task.save_count:<5d}.\n")
                task.clean_cache(mean_steps=mean_steps, result=label)
            else:
                log(f"[{budget():<3d}] Seed {seed} failed in {cost_t:.2f} s.\n"
                    f"Plan {task.plan_success}, Check {task.check_success()}")
                suc_map.append('0')
                task.clean_cache(mean_steps=mean_steps, result='fail')

        with open(task.save_root / 'suc_map.txt', 'w') as f:
            f.write(' '.join([s for s in suc_map]))

        seed += 1

    if save_all:
        log(f'Complete collection, saved {saved_num} episodes over {seed} seeds '
            f'({suc_num} of them successful).')
    else:
        log(f'Complete collection, success rate: {suc_num}/{seed} ({(suc_num / seed) * 100:.2f}%)')

    task.close()
    simulation_app.close()

def main():
    global args_cli, task_config, task_config_file, log_path
    task_file_name = args_cli.task

    episode_num = task_config.get("episode_num", -1)
    if args_cli.episode_num != -1:
        episode_num = args_cli.episode_num
    start_seed = task_config.get("start_seed", -1)
    if args_cli.start_seed != -1:
        start_seed = args_cli.start_seed
    max_seed = task_config.get("max_seed", -1)
    if args_cli.max_seed != -1:
        max_seed = args_cli.max_seed
    
    task_config.update({
        "episode_num": episode_num,
        "start_seed": start_seed,
        "max_seed": max_seed,
    })
    # 加载任务模块和配置
    task_module = importlib.import_module(f"envs.{task_file_name}")
    env_cfg:'BaseTaskCfg' = task_module.TaskCfg()
    env_cfg.tactile_sensor_type = task_config.get('sensor_type', 'gsmini')
    env_cfg.save_dir = Path(task_config.get("save_dir", "./data")) / task_file_name / task_config_file.stem
    env_cfg.decimation = task_config.get("decimation", env_cfg.decimation)
    env_cfg.save_frequency = task_config.get("save_frequency", env_cfg.save_frequency)
    env_cfg.video_frequency = task_config.get("video_frequency", env_cfg.video_frequency)
    env_cfg.render_frequency = task_config.get("render_frequency", env_cfg.render_frequency)
    env_cfg.obs_data_type = task_config.get("observations", {})
    env_cfg.random_texture = task_config.get("random_texture", False)

    # Optional negative-collection knobs: only set those the task actually declares.
    for k in ("neg_noise_scale", "neg_drop_prob"):
        if k in task_config and hasattr(env_cfg, k):
            setattr(env_cfg, k, task_config[k])

    env_cfg.scene.num_envs = 1
    
    init_start = time.perf_counter()
    # 创建任务，包括场景，机器人，物体，相机，传感器
    task:'BaseTask' = task_module.Task(env_cfg, mode='collect')
    init_cost = time.perf_counter() - init_start
    
    log_path = task.save_root / f"{time.strftime(r'%Y-%m-%d_%H:%M:%S')}.log"
    log(f"Task Name: {task_file_name}")
    log(f"Config Name: {task_config_file.stem}")
    log(f"Task Config: \n{json.dumps(task_config, ensure_ascii=False, indent=4)}\n{'-' * 20}\n")
    log(f"Env Config: \n{env_cfg}\n{'-' * 20}\n")
    log(f"Init cost {init_cost:.2f} seconds, devices: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    run(
        task,
        episode_num=episode_num,
        use_seed=task_config.get("use_seed", True),
        start_seed=start_seed,
        max_seed=max_seed,
        save_all=task_config.get("save_all", False),
    )

if __name__ == "__main__":
    main()