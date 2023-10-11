#!/usr/bin/env python3

import time
import argparse
import numpy as np
import gym
from escape_room import EscapeRoom
from window import Window
import torch

def redraw():
    img = env.render_agent_view("RGB")
    window.show_img(img)

def reset():
    if args.seed != -1:
        env.seed(args.seed)
    obs = env.reset()
    redraw()

def step(action):
    obs, reward, done, info = env.step(action)
    state = env.get_state()

    print('step=%s, reward=%.2f' % (env.step_count, reward))

    if done:
        print('done!')
        reset()
    else:
        redraw()

def key_handler(event):
    print('pressed', event.key)

    if event.key == 'escape':
        window.close()
        return

    if event.key == 'enter':
        reset()
        return

    if event.key == 'left':
        step(np.array([0., -1.]))
        return
    if event.key == 'right':
        step(np.array([0., 1.]))
        return
    if event.key == 'up':
        step(np.array([1., 0.]))
        return
    if event.key == 'down':
        step(np.array([-1., 0.]))
        return

parser = argparse.ArgumentParser()

parser.add_argument(
    "--seed",
    type=int,
    help="random seed to generate the environment with",
    default=-1
)

parser.add_argument(
    '--agent_view',
    default=False,
    help="draw the agent sees (partially observable view)",
    action='store_true'
)


args = parser.parse_args()

env = EscapeRoom()

if args.seed != -1:
    env.seed(args.seed)

window = Window('EscapeRoom')
window.reg_key_handler(key_handler)

reset()

# Blocking event loop
window.show(block=True)
