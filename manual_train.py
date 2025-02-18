from __future__ import print_function

import gzip
import os
import pickle
import gymnasium as gym
import numpy as np
from play_joystick import play
from gymnasium.core import ActType, ObsType

DATA_DIR = 'data'
DATA_FILE = 'data.gzip'
N_EPISODES = 20

env = gym.make('CarRacing-v3', 
               render_mode='rgb_array',
               lap_complete_percent=0.95)
env.reset()
total_reward = 0
episode = 1

# if the file exists, append
if os.path.exists(os.path.join(DATA_DIR, DATA_FILE)):
    with gzip.open(os.path.join(DATA_DIR, DATA_FILE), 'rb') as f:
        observations = pickle.load(f)
else:
    observations = list()

def env_callback(
        obs_t: ObsType,
        obs_tp1: ObsType,
        action: ActType,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ):
    global observations, total_reward, episode

    # Happens after reset
    if type(obs_t) is tuple:
        obs_t = obs_t[0]

    observations.append((obs_t, action, obs_tp1, reward, (terminated or truncated)))
    total_reward += reward

    if terminated or truncated:
        if episode == N_EPISODES:
            # store generated data
            data_file_path = os.path.join(DATA_DIR, DATA_FILE)
            print("Saving observations to " + data_file_path)

            if not os.path.exists(DATA_DIR):
                os.mkdir(DATA_DIR)

            with gzip.open(data_file_path, 'wb') as f:
                pickle.dump(observations, f)

            env.close()
            return

        print("Episodes %i reward %0.2f" % (episode, total_reward))

        episode += 1
        env.reset()


if __name__ == '__main__':

    # Controller mapping is as following:
    # (axis_id, scale_factor, invert_axis) -> (min_action_value, max_action_value)
    #
    # The values below are for Thrustmaster T150 on my system
    play(env,
            controller_to_action={
                (0, 2.0, False): (-1.0, 1.0),
                (2, 1.0, True): (0.0, 1.0),
                (1, 1.0, True): (0.0, 1.0)
            },
            noop=np.array([0, 0, 0], dtype=np.float32),
            callback=env_callback
    )
