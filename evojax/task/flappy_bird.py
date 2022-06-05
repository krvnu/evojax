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

"""Implementation of a Flappy Bird task.
"""

from typing import Tuple
from PIL import Image
from PIL import ImageDraw

import jax
import jax.numpy as jnp
from jax import random
from jax import tree_util
from flax.struct import dataclass

from evojax.task.base import TaskState
from evojax.task.base import VectorizedTask

from jax.config import config

config.update("jax_debug_nans", True)

BUFFER = 50 # buffer for rendering
PIXELS_IN_ONE = 4
SCREEN_H = 500

NUM_OBS = 4
NUM_ACT = 1

AGENT_RADIUS = 5
MAX_HEIGHT = 100.0

@dataclass
class AgentState(object):
    pos_y: jnp.float32

@dataclass
class OpeningState(object):
    top: jnp.float32
    bottom: jnp.float32
    dist: jnp.float32

@dataclass
class State(TaskState):
    agent_state: AgentState
    opening_state: OpeningState
    obs: jnp.ndarray
    steps: jnp.int32
    key: jnp.ndarray


def get_random_agent_state(key):
    y = random.uniform(key, shape=(), minval = 0.0, maxval=10.0)
    return AgentState(pos_y=y)


def get_random_opening_state(key):
    k_top, k_bottom = random.split(key, 2)
    top = random.uniform(k_top, shape=(), minval=15.0, maxval=100.0)
    bottom = top - 10.0
    return OpeningState(top=top, bottom=bottom, dist=MAX_HEIGHT)


def get_random_opening_arr(key):
    k_top, k_bottom = random.split(key, 2)
    top = random.uniform(k_top, shape=(), minval=15.0, maxval=MAX_HEIGHT)
    bottom = top - 10.0
    return jnp.array([top, bottom, 100.0])


def get_init_game_state_fn(key: jnp.ndarray):
    agent_state = get_random_agent_state(key)
    opening_state = get_random_opening_state(key)
    return agent_state, opening_state


def update_state(action, state: State, key):
    # Handle action
    jump = jnp.clip(action, 0.0, 1.0)[0] * 3.0
    new_pos_y = state.agent_state.pos_y + jump - 1.0

    pos_y = jnp.where(new_pos_y > MAX_HEIGHT, MAX_HEIGHT, new_pos_y)

    agent_state = AgentState(pos_y=pos_y)

    # Handle opening and rewards
    opening_dist = state.opening_state.dist - 1.0

    reward = 0.0

    # If bird falls below screen
    reward += jnp.where(agent_state.pos_y < 0.0, -10.0, 0.0)

    # If bird is inbetween opening top/bottom while being away from wall
    reward += jnp.where(
        jnp.bitwise_and(
            agent_state.pos_y < state.opening_state.top,
            agent_state.pos_y > state.opening_state.bottom
        ), 0.5, 0.0)

    # If dist 0 and between top and bottom, reward
    reward += jnp.where(jnp.bitwise_and(
        opening_dist == 0.0, 
        jnp.bitwise_and(
            agent_state.pos_y < state.opening_state.top,
            agent_state.pos_y > state.opening_state.bottom
        )
    ), 1.0, 0.0)

    # If dist 0 and between top and bottom, bad
    reward += jnp.where(jnp.bitwise_and(
        opening_dist == 0.0, 
        jnp.bitwise_or(
            agent_state.pos_y > state.opening_state.top,
            agent_state.pos_y < state.opening_state.bottom
        )
    ), -10.0, 0.0)

    # Create a new wall if needed
    opening_arr = jnp.where(opening_dist < 0.0, 
        get_random_opening_arr(key),
        jnp.array([state.opening_state.top, state.opening_state.bottom, opening_dist]))
    opening_state = OpeningState(top=opening_arr[0], bottom=opening_arr[1], dist=opening_arr[2])

    # Create obs to return
    obs = jnp.array([
        agent_state.pos_y,
        opening_state.top,
        opening_state.bottom,
        opening_state.dist
    ])

    return agent_state, opening_state, reward, obs

def get_obs(state: State):
    obs = jnp.array([
        state.agent_state.pos_y,
        state.opening_state.top,
        state.opening_state.bottom,
        state.opening_state.dist
    ])

    return obs

class FlappyBird(VectorizedTask):
    """Flappy bird training version."""

    def __init__(self,
                 max_steps: int = 1000,
                 test: bool = False):

        self.multi_agent_training = False
        self.max_steps = max_steps
        self.test = test
        self.obs_shape = tuple([NUM_OBS, ])
        self.act_shape = tuple([NUM_ACT, ])

        def reset_fn(key):
            next_key, key = random.split(key)
            agent_state, opening_state = get_init_game_state_fn(key)
            state = State(agent_state=agent_state, opening_state=opening_state, obs=jnp.array([]),
                         steps=jnp.zeros((), dtype=int), key=next_key)
            return State(agent_state=agent_state, opening_state=opening_state, obs=get_obs(state),
                         steps=jnp.zeros((), dtype=int), key=next_key)
        self._reset_fn = jax.jit(jax.vmap(reset_fn))

        def step_fn(state, action):
            next_key, key = random.split(state.key)
            agent_state, opening_state, reward, obs = update_state(
                action=action, state=state, key=key)

            steps = state.steps + 1
            done = jnp.where(steps >= max_steps, 1, 0)
            return State(agent_state=agent_state, opening_state=opening_state, obs=obs,
                         steps=steps, key=next_key), reward, done
        self._step_fn = jax.jit(jax.vmap(step_fn))

    def reset(self, key: jnp.array) -> State:
        return self._reset_fn(key)

    def step(self,
             state: State,
             action: jnp.ndarray) -> Tuple[State, jnp.ndarray, jnp.ndarray]:
        return self._step_fn(state, action)

    @staticmethod
    def render(state: State, task_id: int = 0) -> Image:
        img = Image.new('RGB', (500, 500), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        state = tree_util.tree_map(lambda s: s[task_id], state)

        # Draw circle that shows whether current height is good or not
        bad_height = state.agent_state.pos_y < state.opening_state.bottom or state.agent_state.pos_y > state.opening_state.top
        if state.opening_state.dist == 0.0 and bad_height:
            draw.ellipse(
                (450, 20,
                 480, 50),
                fill=(255, 0, 0), outline=(0, 0, 0))
        elif bad_height:
            draw.ellipse(
                (450, 20,
                 480, 50),
                fill=(255, 255, 0), outline=(0, 0, 0))
        elif not bad_height:
            draw.ellipse(
                (450, 20,
                 480, 50),
                fill=(0, 255, 0), outline=(0, 0, 0))

        # Draw bird
        draw.ellipse(
                (BUFFER - AGENT_RADIUS, SCREEN_H - BUFFER - state.agent_state.pos_y * PIXELS_IN_ONE - AGENT_RADIUS,
                 BUFFER + AGENT_RADIUS, SCREEN_H - BUFFER - state.agent_state.pos_y * PIXELS_IN_ONE + AGENT_RADIUS),
                fill=(0, 255, 0), outline=(0, 0, 0))

        # Draw top line
        draw.line((BUFFER + PIXELS_IN_ONE * state.opening_state.dist, 
            0, 
            BUFFER + PIXELS_IN_ONE * state.opening_state.dist,
            BUFFER + PIXELS_IN_ONE * (MAX_HEIGHT - state.opening_state.top)),
            fill=(0,0,255),
            width=1,
        )

        # Draw bottom line
        draw.line((BUFFER + PIXELS_IN_ONE * state.opening_state.dist, 
            BUFFER + PIXELS_IN_ONE * (MAX_HEIGHT - state.opening_state.bottom), 
            BUFFER + PIXELS_IN_ONE * state.opening_state.dist,
            SCREEN_H),
            fill=(255,0,0),
            width=1,
        )

        return img
