from __future__ import print_function

import os
import gymnasium as gym
import numpy as np
import torch

from train import data_transform, actions_set, Net, DATA_DIR, MODEL_FILE
from pyglet.window import key
import pyglet

# Press gas for first N iterations to get the car rolling
MAX_THROTTLE_ITER = 10

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def play(model):
    """
    Let the agent play
    :param model: the network
    """
    env = gym.make('CarRacing-v3', render_mode='human')

    # initialize environment
    state, _ = env.reset()
    throttle_iter = 0

    while 1:
        env.render()

        state = np.moveaxis(state, 2, 0)  # change shape from (96, 96, 3) to (3, 96, 96)

        # numpy to tensor
        state = torch.from_numpy(np.flip(state, axis=0).copy())
        state = data_transform(state).to(device)   # apply transformations
        state = state.unsqueeze(0)  # add additional dimension

        # forward
        with torch.set_grad_enabled(False):
            outputs = model(state)

        normalized = torch.nn.functional.softmax(outputs, dim=1)
        # translate from net output to env action
        max_action = np.argmax(normalized.cpu().numpy()[0])

        if throttle_iter <= MAX_THROTTLE_ITER:
            # Send it
            action = actions_set[3]
            throttle_iter += 1
        else:
            action = actions_set[max_action]

        action = np.array(action, dtype=np.float32)

        # one step
        state, _, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            env.close()
            return


if __name__ == '__main__':
    m = Net()
    m.load_state_dict(torch.load(os.path.join(DATA_DIR, MODEL_FILE)))
    m.eval()
    play(m)
