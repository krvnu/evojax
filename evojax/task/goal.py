# Copyright 2022 The EvoJAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Tuple
from PIL import Image
from PIL import ImageDraw
import numpy as np

import jax
import jax.numpy as jnp
from jax import random, lax
from flax.struct import dataclass

from evojax.task.base import TaskState
from evojax.task.base import VectorizedTask

FORCE_SCALING = 1.0

SCREEN_W = 600
SCREEN_H = 600
CART_W = 40
CART_H = 20
VIZ_SCALE = 100
WHEEL_RAD = 5

START_X = 300.0
START_Y = 300.0
START_X_DOT = 0.0
START_Y_DOT = 0.0


@dataclass
class State(TaskState):
    obs: jnp.ndarray
    state: jnp.ndarray
    steps: jnp.int32
    key: jnp.ndarray


def get_new_goal(key): 
    ang = random.truncated_normal(key, 0.0, 1.0, [1])[0] * 2 * jnp.pi
    g_x = (jnp.cos(ang) * 250) + 300
    g_y = (jnp.sin(ang) * 250) + 300

    return g_x, g_y

def in_goal(x, y, g_x, g_y):
    x_dist = jnp.absolute(g_x-x)
    y_dist = jnp.absolute(g_y-y)
    return jnp.bitwise_and(jnp.where(x_dist < 5.0, 1, 0), jnp.where(y_dist < 5.0, 1, 0))

def get_init_state_easy(key: jnp.ndarray) -> jnp.ndarray:
    goal_x, goal_y = get_new_goal(key)

    return jnp.array([START_X, 
                        START_Y, 
                        START_X_DOT, 
                        START_Y_DOT, 
                        goal_x, 
                        goal_y, 
                        0, 
                        0], dtype='float32')


def get_init_state_hard(key: jnp.ndarray) -> jnp.ndarray:
    return get_init_state_easy(key)

def get_obs(state: jnp.ndarray) -> jnp.ndarray:
    x, y, x_dot, y_dot, g_x, g_y, t, c = state
    return jnp.array([x, y, x_dot, y_dot, g_x, g_y, t, c])


def get_reward(state: jnp.ndarray) -> jnp.float32:
    x, y, _, _, g_x, g_y, t, _ = state
    dist_x = jnp.absolute(g_x - x)
    dist_y = jnp.absolute(g_y - y)

    is_goal = in_goal(x, y, g_x, g_y)
    hmmm = jnp.array([0])
    reward = lax.cond(is_goal > 0, lambda x: jnp.array([1000]), lambda x: x, hmmm)

    return (1 / ((dist_x + dist_y) * ((t+0.01)/20) + 0.0001)) + reward[0]


def update_goal(key, x):
    c, g_x, g_y = x
    c = c + 1
    g_x, g_y = get_new_goal(key)
    return jnp.array([c, g_x, g_y])


def update_state(action: jnp.ndarray, state: jnp.ndarray, key) -> jnp.ndarray:
    x_action = jnp.clip(action, -1.0, 1.0)[0] * FORCE_SCALING
    y_action = jnp.clip(action, -1.0, 1.0)[1] * FORCE_SCALING
    x, y, x_dot, y_dot, g_x, g_y, t, c = state

    x_dot = jnp.clip(jnp.array([x_dot + x_action]), -3.0, 3.0)[0]
    y_dot = jnp.clip(jnp.array([y_dot + y_action]), -3.0, 3.0)[0]

    x = jnp.clip(jnp.array([x + x_dot]), 0, 600)[0]
    y = jnp.clip(jnp.array([y + y_dot]), 0, 600)[0]

    t = t + 1

    is_goal = in_goal(x, y, g_x, g_y)
    operands = jnp.array([c, g_x, g_y])
    out = lax.cond(is_goal > 0, lambda x: update_goal(key, x), lambda x: x, operands)
    c, g_x, g_y = out

    return jnp.array([x, y, x_dot, y_dot, g_x, g_y, t, c])

   
def finished_state(state: jnp.ndarray) -> jnp.float32:
    x, y, _, _, g_x, g_y, _, _ = state
    # x_dist = jnp.absolute(g_x-x)
    # y_dist = jnp.absolute(g_y-y)
    # return jnp.bitwise_and(jnp.where(x_dist < 5.0, 1, 0), jnp.where(y_dist < 5.0, 1, 0))
    return 0

class Goal(VectorizedTask):
    """Goal task."""

    def __init__(self,
                 max_steps: int = 1000,
                 harder: bool = False,
                 test: bool = False):

        self.max_steps = max_steps
        self.obs_shape = tuple([8, ])
        self.act_shape = tuple([2, ])
        self.test = test
        if harder:
            get_init_state_fn = get_init_state_hard
        else:
            get_init_state_fn = get_init_state_easy

        def reset_fn(key):
            next_key, key = random.split(key)
            state = get_init_state_fn(key)
            return State(state=state, obs=get_obs(state),
                         steps=jnp.zeros((), dtype=int), key=next_key)
        self._reset_fn = jax.jit(jax.vmap(reset_fn))

        def step_fn(state, action):
            cur_state = update_state(action=action, state=state.state, key=state.key)
            reward = get_reward(state=cur_state)
            steps = state.steps + 1
            done = jnp.bitwise_or(finished_state(cur_state), steps >= max_steps)
            steps = jnp.where(done, jnp.zeros((), jnp.int32), steps)
            next_key, key = random.split(state.key)
            cur_state = jax.lax.cond(
                done, lambda x: get_init_state_fn(key), lambda x: x, cur_state)
            return State(state=cur_state, obs=get_obs(state=cur_state),
                         steps=steps, key=next_key), reward, done
        self._step_fn = jax.jit(jax.vmap(step_fn))

    def reset(self, key: jnp.ndarray) -> State:
        return self._reset_fn(key)

    def step(self,
             state: State,
             action: jnp.ndarray) -> Tuple[State, jnp.ndarray, jnp.ndarray]:
        return self._step_fn(state, action)

    @staticmethod
    def render(state: State, task_id: int) -> Image:
        """Render a specified task."""

        # Blank screen
        img = Image.new('RGB', (SCREEN_W, SCREEN_H), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        x, y, _, _, g_x, g_y, _, _ = np.array(state.state[task_id])

        # Draw the goal
        draw.rectangle(
            (g_x - 10, g_y - 10,
             g_x + 10, g_y + 10),
            fill=(0, 0, 255), width=6)

        draw.rectangle(
            (x - 10, y - 10,
             x + 10, y + 10),
            fill=(255, 0, 0), width=6)

        return img
