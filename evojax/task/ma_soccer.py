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

"""Implementation of a multi-agents Soccer task.

Ref: https://mobile.aau.at/~welmenre/papers/fehervari-2010-Evolving_Neural_Network_Controllers_for_a_Team_of_Self-organizing_Robots.pdf
"""

from typing import Tuple
from functools import partial
from PIL import Image
from PIL import ImageDraw
import numpy as np

import jax
import jax.numpy as jnp
from jax import random
from jax import tree_util
from flax.struct import dataclass

from evojax.task.base import TaskState
from evojax.task.base import VectorizedTask

from jax.config import config
config.update("jax_debug_nans", True)

PIXELS_IN_METER = 10
FIELD_W = 110
FIELD_H = 75
BUFFER = 50 # buffer for rendering
SCREEN_W = FIELD_W * PIXELS_IN_METER + BUFFER * 2
SCREEN_H = FIELD_H * PIXELS_IN_METER + BUFFER * 2
MAX_BALL_SPEED = 5.0 # TODO

NUM_TEAMS = 2
TEAM_1 = 0
TEAM_2 = 1

NUM_OBS = 16
NUM_ACT = 4

DIST_TO_BALL_N = 0
DIST_TO_BALL_S = 1
DIST_TO_BALL_W = 2
DIST_TO_BALL_E = 3
DIST_TO_OPP_N = 4
DIST_TO_OPP_S = 5
DIST_TO_OPP_W = 6
DIST_TO_OPP_E = 7
DIST_TO_TM_N = 8
DIST_TO_TM_S = 9
DIST_TO_TM_W = 10
DIST_TO_TM_E = 11
DIST_TO_FB_N = 12
DIST_TO_FB_S = 13
DIST_TO_FB_W = 14
DIST_TO_FB_E = 15

ACT_TURN = 0
ACT_DASH = 1
ACT_KICK_POWER = 2
ACT_KICK_DIRECTION = 3

KICKING_RANGE = 1
TEAM_1_GOAL = 0
TEAM_2_GOAL = 1

AGENT_RADIUS = 10

# BUBBLE_RADIUS = 5
# MIN_DIST = 2 * BUBBLE_RADIUS
# MAX_RANGE = 60
# NUM_RANGE_SENSORS = 30
# DELTA_ANG = 2 * 3.14 / NUM_RANGE_SENSORS

# TYPE_VOID = 0
# TYPE_WALL = 1
# TYPE_FOOD = 2
# TYPE_POISON = 3
# TYPE_AGENT = 4
# SENSOR_DATA_DIM = 6

# ACT_UP = 0
# ACT_DOWN = 1
# ACT_LEFT = 2
# ACT_RIGHT = 3

@dataclass
class AgentStatus(object):
    pos_x: jnp.float32
    pos_y: jnp.float32
    theta: jnp.float32
    vel: jnp.float32
    kick: jnp.int32
    team: jnp.int32
    id: jnp.int32

@dataclass
class BallStatus(object):
    pos_x: jnp.float32
    pos_y: jnp.float32
    theta: jnp.float32
    vel: jnp.float32

@dataclass
class State(TaskState):
    agent_state: AgentStatus
    ball_state: BallStatus
    obs: jnp.ndarray
    steps: jnp.int32
    key: jnp.ndarray


@partial(jax.vmap, in_axes=(0, None))
def create_agents(key: jnp.ndarray, team: jnp.int32) -> AgentStatus:
    k_pos_x, k_pos_y, k_id = random.split(key, 3)
    id = random.uniform(k_id, (), dtype=jnp.float32, minval=0.0, maxval=99999999.9)
    vel = 1.0
    theta = 0.0
    pos_x = jnp.where(team == 1, FIELD_W / 4,  (FIELD_W / 4) * 3)
    pos_y = FIELD_H / 2
    pos_x = random.uniform(k_pos_x, (), dtype=jnp.float32) * 20 + pos_x
    pos_y = random.uniform(k_pos_y, (), dtype=jnp.float32) * 20 + pos_y
    return AgentStatus(pos_x=pos_x, pos_y=pos_y, theta=theta, vel=vel, team=team, id=id, kick=0)

def create_ball() -> BallStatus:
    return BallStatus(pos_x=FIELD_W/2, pos_y=FIELD_H/2, theta=0.0, vel=0.0)

def get_reward(agent: AgentStatus,
               ball: BallStatus) -> Tuple[AgentStatus, jnp.float32]:
    
    # dist = jnp.sqrt(jnp.square(agent.pos_x - items.pos_x) +
    #                 jnp.square(agent.pos_y - items.pos_y))
    # rewards = (jnp.where(items.bubble_type == TYPE_FOOD, 1., -1.) *
    #            items.valid * jnp.where(dist < MIN_DIST, 1, 0))
    # poison_cnt = jnp.sum(jnp.where(rewards == -1., 1, 0)) + agent.poison_cnt
    # reward = jnp.sum(rewards)
    # items_valid = (dist >= MIN_DIST) * items.valid
    # agent_state = BubbleStatus(
    #     pos_x=agent.pos_x, pos_y=agent.pos_y,
    #     vel_x=agent.vel_x, vel_y=agent.vel_y,
    #     bubble_type=agent.bubble_type,
    #     valid=agent.valid, poison_cnt=poison_cnt)
    # items_state = BubbleStatus(
    #     pos_x=items.pos_x, pos_y=items.pos_y,
    #     vel_x=items.vel_x, vel_y=items.vel_y,
    #     bubble_type=items.bubble_type,
    #     valid=items_valid, poison_cnt=items.poison_cnt)

    

    dist = jnp.sqrt(jnp.square(agent.pos_x - ball.pos_x) +
                    jnp.square(agent.pos_y - ball.pos_y))

    return agent, (1 / dist)


@partial(jax.vmap, in_axes=(0, None, None))
def get_rewards(agent: AgentStatus,
                agents: AgentStatus,
                ball: BallStatus) -> Tuple[AgentStatus, jnp.ndarray]:
    
    b_dist = jnp.sqrt(jnp.square(agent.pos_x - ball.pos_x) +
                    jnp.square(agent.pos_y - ball.pos_y))

    
    reward = ball.vel * 100.0

    return agent, reward


# @jax.vmap
# def update_item_state(item: BubbleStatus) -> BubbleStatus:
#     vel_x = item.vel_x
#     vel_y = item.vel_y
#     pos_x = item.pos_x + vel_x
#     pos_y = item.pos_y + vel_y
#     # Collide with the west wall.
#     vel_x = jnp.where(pos_x < 1, -vel_x, vel_x)
#     pos_x = jnp.where(pos_x < 1, 1, pos_x)
#     # Collide with the east wall.
#     vel_x = jnp.where(pos_x > SCREEN_W - 1, -vel_x, vel_x)
#     pos_x = jnp.where(pos_x > SCREEN_W - 1, SCREEN_W - 1, pos_x)
#     # Collide with the north wall.
#     vel_y = jnp.where(pos_y < 1, -vel_y, vel_y)
#     pos_y = jnp.where(pos_y < 1, 1, pos_y)
#     # Collide with the south wall.
#     vel_y = jnp.where(pos_y > SCREEN_H - 1, -vel_y, vel_y)
#     pos_y = jnp.where(pos_y > SCREEN_H - 1, SCREEN_H - 1, pos_y)
#     return BubbleStatus(
#         pos_x=pos_x, pos_y=pos_y, vel_x=vel_x, vel_y=vel_y,
#         bubble_type=item.bubble_type, valid=item.valid,
#         poison_cnt=item.poison_cnt)

# TODO: Fix this vmap, combine clips
@jax.vmap
def update_agent_state(agent: AgentStatus,
                    action: jnp.ndarray) -> AgentStatus:
    
    # Need to flip actions for this player if TEAM 2
    # action = jnp.where(agent.id == TEAM_1, 
    #     raw_action,
    #     jnp.array([
    #         raw_action[0] * -1.0, # Turn flipped
    #         raw_action[1],
    #         raw_action[2],
    #         raw_action[3],
    #     ])
    # )

    # Handle turning
    turn = jnp.clip(action, -1.0, 1.0)[0] * (jnp.pi/16)
    theta = agent.theta + turn

    # Handle dash
    dash = jnp.clip(action, -1.0, 1.0)[1]
    vel = agent.vel + dash
    vel = jnp.where(vel > 2, 2, vel)
    vel = jnp.where(vel < 0, 0, vel)
    vel = vel * 0.95

    pos_x = agent.pos_x + jnp.cos(theta) * vel
    pos_y = agent.pos_y + jnp.sin(theta) * vel
    # Collide with the west wall.
    vel = jnp.where(pos_x < 1, 0, vel)
    pos_x = jnp.where(pos_x < 1, 1, pos_x)
    # Collide with the east wall.
    vel = jnp.where(pos_x > FIELD_W - 1, 0, vel)
    pos_x = jnp.where(pos_x > FIELD_W - 1, FIELD_W - 1, pos_x)
    # Collide with the north wall.
    vel = jnp.where(pos_y > FIELD_H - 1, 0, vel)
    pos_y = jnp.where(pos_y > FIELD_H - 1, FIELD_H - 1, pos_y)
    # Collide with the south wall.
    vel = jnp.where(pos_y < 1, 0, vel)
    pos_y = jnp.where(pos_y < 1, 1, pos_y)

    return AgentStatus(
                pos_x=pos_x, pos_y=pos_y, theta=theta, vel=vel, team=agent.team, id=agent.id, kick=agent.kick
            )


def update_ball_state(agent: AgentStatus,
                    action: jnp.ndarray,
                    ball: BallStatus) -> BallStatus:

    # TODO: Is ball in goal?

    # Calc ball states
    ball_states = calc_ball_state(agent, action, ball)

    # Average to get overall change
    pos_x = jnp.mean(ball_states.pos_x)
    pos_y = jnp.mean(ball_states.pos_y)
    vel = jnp.mean(ball_states.vel)
    theta = jnp.mean(ball_states.theta)

    return BallStatus(pos_x=pos_x, pos_y=pos_y, theta=theta, vel=vel)

# TODO: Fix vmap
@partial(jax.vmap, in_axes=(0, 0, None))
def calc_ball_state(agent: AgentStatus,
                    action: jnp.ndarray,
                    ball: BallStatus) -> BallStatus:

    # Might need to do this afterwards?
    # Need to flip actions for this player if TEAM 2
    # action = jnp.where(agent.id == TEAM_1, 
    #     raw_action,
    #     jnp.array([
    #         raw_action[0],
    #         raw_action[1],
    #         raw_action[2],
    #         raw_action[3] * -1.0, # Need to flip kick direction
    #     ])
    # )

    # Handle kick
    kick_power = jnp.clip(action, 0.0, 1.0)[2] * 3
    kick_direction = jnp.clip(action, -1.0, 1.0)[3] * (jnp.pi)

    x_diff, y_diff, dist_to_ball = distance_between(
        jnp.array([agent.pos_x]), 
        jnp.array([ball.pos_x]), 
        jnp.array([agent.pos_y]), 
        jnp.array([ball.pos_y])
    )
    dist_to_ball = dist_to_ball[0]

    theta = jnp.where(
        dist_to_ball < 1.0,
        kick_direction,
        ball.theta
    )

    vel = jnp.where(
        dist_to_ball < 1.0,
        kick_power,
        ball.vel * 0.95
    )

    pos_x = ball.pos_x + jnp.cos(theta) * vel
    pos_y = ball.pos_y + jnp.sin(theta) * vel
    # Collide with the west wall.
    vel = jnp.where(pos_x < 1, 0, vel)
    pos_x = jnp.where(pos_x < 1, 1, pos_x)
    # Collide with the east wall.
    vel = jnp.where(pos_x > FIELD_W - 1, 0, vel)
    pos_x = jnp.where(pos_x > FIELD_W - 1, FIELD_W - 1, pos_x)
    # Collide with the north wall.
    vel = jnp.where(pos_y > FIELD_H - 1, 0, vel)
    pos_y = jnp.where(pos_y > FIELD_H - 1, FIELD_H - 1, pos_y)
    # Collide with the south wall.
    vel = jnp.where(pos_y < 1, 0, vel)
    pos_y = jnp.where(pos_y < 1, 1, pos_y)

    return BallStatus(pos_x=pos_x, pos_y=pos_y, theta=theta, vel=vel)


# @jax.vmap
# def get_line_seg_intersection(x1: jnp.float32,
#                               y1: jnp.float32,
#                               x2: jnp.float32,
#                               y2: jnp.float32,
#                               x3: jnp.float32,
#                               y3: jnp.float32,
#                               x4: jnp.float32,
#                               y4: jnp.float32) -> Tuple[np.bool, jnp.ndarray]:
#     """Determine if line segment (x1, y1, x2, y2) intersects with line
#     segment (x3, y3, x4, y4), and return the intersection coordinate.
#     """
#     denominator = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
#     ua = jnp.where(
#         jnp.isclose(denominator, 0.0), 0,
#         ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denominator)
#     mask1 = jnp.bitwise_and(ua > 0., ua < 1.)
#     ub = jnp.where(
#         jnp.isclose(denominator, 0.0), 0,
#         ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denominator)
#     mask2 = jnp.bitwise_and(ub > 0., ub < 1.)
#     intersected = jnp.bitwise_and(mask1, mask2)
#     x_intersection = x1 + ua * (x2 - x1)
#     y_intersection = y1 + ua * (y2 - y1)
#     up = jnp.where(intersected,
#                    jnp.array([x_intersection, y_intersection]),
#                    jnp.array([SCREEN_W, SCREEN_W]))
#     return intersected, up


# @jax.vmap
# def get_line_dot_intersection(x1: jnp.float32,
#                               y1: jnp.float32,
#                               x2: jnp.float32,
#                               y2: jnp.float32,
#                               x3: jnp.float32,
#                               y3: jnp.float32) -> Tuple[np.bool, jnp.ndarray]:
#     """Determine if a line segment (x1, y1, x2, y2) intersects with a dot at
#     (x3, y3) with radius BUBBLE_RADIUS, if so return the point of intersection.
#     """
#     point_xy = jnp.array([x3, y3])
#     v = jnp.array([y2 - y1, x1 - x2])
#     v_len = jnp.linalg.norm(v)
#     d = jnp.abs((x2 - x1) * (y1 - y3) - (x1 - x3) * (y2 - y1)) / v_len
#     up = point_xy + v / v_len * d
#     ua = jnp.where(jnp.abs(x2 - x1) > jnp.abs(y2 - y1),
#                    (up[0] - x1) / (x2 - x1),
#                    (up[1] - y1) / (y2 - y1))
#     ua = jnp.where(d > BUBBLE_RADIUS, 0, ua)
#     intersected = jnp.bitwise_and(ua > 0., ua < 1.)
#     return intersected, up

@jax.vmap
def distance_between(x1, x2, y1, y2) -> Tuple[jnp.float32, jnp.float32, jnp.float32]:
    x_diff = x1 - x2
    y_diff = y1 - y2
    dist = jnp.sqrt(jnp.square(x_diff) + jnp.square(y_diff))
    return x_diff, y_diff, dist

@jax.vmap
def distance_between_exluding_self_opponents(x1, x2, y1, y2, team1, team2, id1, id2) -> Tuple[jnp.float32, jnp.float32, jnp.float32]:
    x_diff = x1 - x2
    y_diff = y1 - y2
    dist = jnp.sqrt(jnp.square(x_diff) + jnp.square(y_diff))

    # Remove self and opponents
    dist = jnp.where(jnp.bitwise_or(id1 == id2, team1 != team2), 9999, dist)

    return x_diff, y_diff, dist

@jax.vmap
def distance_between_exluding_team(x1, x2, y1, y2, team1, team2) -> Tuple[jnp.float32, jnp.float32, jnp.float32]:
    x_diff = x1 - x2
    y_diff = y1 - y2
    dist = jnp.sqrt(jnp.square(x_diff) + jnp.square(y_diff))

    # Remove self
    dist = jnp.where(team1 == team2, 9999, dist)

    return x_diff, y_diff, dist

# Main thing here is that all players obs should be going to the right
# Team 1 -> doesnt need to be switch
# Team 2 -> does need to be switched
@partial(jax.vmap, in_axes=(0, None, None))
def get_obs(agent: AgentStatus,
            agents: AgentStatus,
            ball: BallStatus) -> jnp.ndarray:

    agent_xy = jnp.array([agent.pos_x, agent.pos_y]).ravel()
    ball_xy = jnp.array([ball.pos_x, ball.pos_y]).ravel()

    # Dist to ball
    x_diff, y_diff, dist_to_ball = distance_between(
        jnp.array([agent_xy[0]]), 
        jnp.array([ball_xy[0]]), 
        jnp.array([agent_xy[1]]), 
        jnp.array([ball_xy[1]])
    )
    x_diff = x_diff[0]
    y_diff = y_diff[0]
    dist_to_ball = dist_to_ball[0]
    dist_to_ball_n = jnp.where(agent_xy[1] > ball_xy[1], 0, jnp.abs(y_diff)/jnp.square(dist_to_ball))
    dist_to_ball_s = jnp.where(agent_xy[1] < ball_xy[1], 0, jnp.abs(y_diff)/jnp.square(dist_to_ball))
    dist_to_ball_w = jnp.where(agent_xy[0] < ball_xy[0], 0, jnp.abs(x_diff)/jnp.square(dist_to_ball))
    dist_to_ball_e = jnp.where(agent_xy[0] < ball_xy[0], 0, jnp.abs(x_diff)/jnp.square(dist_to_ball))

    # Dist to nearest opp
    n_agents = len(agents.team)
    x_diff, y_diff, dists_to_agents = distance_between_exluding_team(
        jnp.ones(n_agents) * agent_xy[0], 
        jnp.ones(n_agents) * agent_xy[1],
        agents.pos_x, 
        agents.pos_y,
        jnp.ones(n_agents) * agent.team,
        agents.team)

    # Remove friendlies and self
    # dist_to_opps = jnp.where(
    #     jnp.bitwise_and(
    #         jnp.not_equal(agent.team * jnp.ones(n_agents), agents.team),
    #         jnp.not_equal(dists_to_agents, 0.0 * jnp.ones(n_agents)),
    #     ),
    #     dists_to_agents,
    #     1000, 
    # )

    i_opp = jnp.argmin(dists_to_agents)
    opp = AgentStatus(
        agents.pos_x[i_opp], 
        agents.pos_y[i_opp], 
        agents.theta[i_opp], 
        agents.vel[i_opp], 
        agents.kick[i_opp],
        agents.team[i_opp],
        agents.id[i_opp]
    )
    opp_xy = jnp.array([opp.pos_x, opp.pos_y]).ravel()
    x_diff, y_diff, dist_to_opp = distance_between(
        jnp.array([agent_xy[0]]), 
        jnp.array([opp_xy[0]]), 
        jnp.array([agent_xy[1]]), 
        jnp.array([opp_xy[1]])
    )
    x_diff = x_diff[0]
    y_diff = y_diff[0]
    dist_to_opp = dist_to_opp[0]
    dist_to_opp_n = jnp.where(agent_xy[1] > opp_xy[1], 0, jnp.abs(y_diff)/jnp.square(dist_to_opp))
    dist_to_opp_s = jnp.where(agent_xy[1] < opp_xy[1], 0, jnp.abs(y_diff)/jnp.square(dist_to_opp))
    dist_to_opp_w = jnp.where(agent_xy[0] < opp_xy[0], 0, jnp.abs(x_diff)/jnp.square(dist_to_opp))
    dist_to_opp_e = jnp.where(agent_xy[0] < opp_xy[0], 0, jnp.abs(x_diff)/jnp.square(dist_to_opp))

    # Remove opponents and self
    x_diff, y_diff, dist_to_tm = distance_between_exluding_self_opponents(
        jnp.ones(n_agents) * agent_xy[0], 
        jnp.ones(n_agents) * agent_xy[1],
        agents.pos_x, 
        agents.pos_y,
        jnp.ones(n_agents) * agent.team,
        agents.team,
        jnp.ones(n_agents) * agent.id,
        agents.id)

    # dist_to_tm = jnp.where(
    #     jnp.bitwise_and(
    #         jnp.not_equal(agent.team * jnp.ones(n_agents), agents.team),
    #         jnp.not_equal(dists_to_agents, 0.0 * jnp.ones(n_agents)),
    #     ),
    #     dists_to_agents,
    #     1000, 
    # )

    i_tm = jnp.argmin(dist_to_tm)
    tm = AgentStatus(
        agents.pos_x[i_tm], 
        agents.pos_y[i_tm], 
        agents.theta[i_tm], 
        agents.vel[i_tm], 
        agents.kick[i_opp],
        agents.team[i_tm],
        agents.id[i_tm]
    )
    tm_xy = jnp.array([tm.pos_x, tm.pos_y]).ravel()
    x_diff, y_diff, dist_to_tm = distance_between(
        jnp.array([agent_xy[0]]), 
        jnp.array([tm_xy[0]]), 
        jnp.array([agent_xy[1]]), 
        jnp.array([tm_xy[1]])
    )
    x_diff = x_diff[0]
    y_diff = y_diff[0]
    dist_to_tm = dist_to_tm[0]
    dist_to_tm_n = jnp.where(agent_xy[1] > tm_xy[1], 0, jnp.abs(y_diff)/jnp.square(dist_to_tm))
    dist_to_tm_s = jnp.where(agent_xy[1] < tm_xy[1], 0, jnp.abs(y_diff)/jnp.square(dist_to_tm))
    dist_to_tm_w = jnp.where(agent_xy[0] < tm_xy[0], 0, jnp.abs(x_diff)/jnp.square(dist_to_tm))
    dist_to_tm_e = jnp.where(agent_xy[0] < tm_xy[0], 0, jnp.abs(x_diff)/jnp.square(dist_to_tm))
    
    # Dist to field borders
    dist_to_fb_n = FIELD_H - agent_xy[0]
    dist_to_fb_s = agent_xy[0]
    dist_to_fb_w = agent_xy[0]
    dist_to_fb_e = FIELD_W - agent_xy[0]

    # TODO: Normalize distances

    # TODO: Need to know own speed / angle probably
    # Flip observations if on team 2
    obs = jnp.where(agent.id == TEAM_1, 
        jnp.array([
            dist_to_ball_n,
            dist_to_ball_s,
            dist_to_ball_w,
            dist_to_ball_e,
            dist_to_opp_n,
            dist_to_opp_s,
            dist_to_opp_w,
            dist_to_opp_e,
            dist_to_tm_n,
            dist_to_tm_s,
            dist_to_tm_w,
            dist_to_tm_e,
            dist_to_fb_n,
            dist_to_fb_s,
            dist_to_fb_w,
            dist_to_fb_e,
        ]),
        jnp.array([
            dist_to_ball_n,
            dist_to_ball_s,
            dist_to_ball_e,
            dist_to_ball_w,
            dist_to_opp_n,
            dist_to_opp_s,
            dist_to_opp_e,
            dist_to_opp_w,
            dist_to_tm_n,
            dist_to_tm_s,
            dist_to_tm_e,
            dist_to_tm_w,
            dist_to_fb_n,
            dist_to_fb_s,
            dist_to_fb_e,
            dist_to_fb_w,
        ])
    )

    return obs


class MultiAgentSoccer(VectorizedTask):
    """Soccer, multi-agents training version."""

    def __init__(self,
                 num_agents: int = 12,
                 max_steps: int = 1000,
                 test: bool = False):

        self.multi_agent_training = True
        self.max_steps = max_steps
        self.test = test
        self.obs_shape = tuple([num_agents, NUM_OBS, ])
        self.act_shape = tuple([num_agents, NUM_ACT])

        def reset_fn(key):
            next_key, key = random.split(key)
            ks = random.split(key, num_agents)
            num_half = num_agents//2
            team_1 = create_agents(ks[:num_half], TEAM_1)
            team_2 = create_agents(ks[num_half:], TEAM_2)
            agents = tree_util.tree_multimap(
                lambda x, y: jnp.concatenate([x, y], axis=0),
                team_1, team_2)
            ball = create_ball()
            obs = get_obs(agents, agents, ball)
            return State(agent_state=agents, ball_state=ball, obs=obs,
                         steps=jnp.zeros((), dtype=jnp.int32), key=next_key)
        self._reset_fn = jax.jit(jax.vmap(reset_fn))

        def step_fn(state, actions):
            next_key, key = random.split(state.key)
            # action_keys = random.split(key, num_agents)

            agents = update_agent_state(state.agent_state, actions)
            ball = update_ball_state(state.agent_state, actions, state.ball_state)
            _, rewards = get_rewards(agents, agents, ball)

            steps = state.steps + 1
            done = jnp.where(steps >= max_steps, 1, 0)
            obs = get_obs(agents, agents, ball)
            return State(agent_state=agents, ball_state=ball, obs=obs,
                         steps=steps, key=next_key), rewards, done
        self._step_fn = jax.jit(jax.vmap(step_fn))

    def reset(self, key: jnp.array) -> State:
        return self._reset_fn(key)

    def step(self,
             state: State,
             action: jnp.ndarray) -> Tuple[State, jnp.ndarray, jnp.ndarray]:
        return self._step_fn(state, action)

    @staticmethod
    def render(state: State, task_id: int = 0) -> Image:
        img = Image.new('RGB', (SCREEN_W, SCREEN_H), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        state = tree_util.tree_map(lambda s: s[task_id], state)

        # Draw field border
        draw.line((BUFFER, 
            BUFFER, 
            BUFFER + (FIELD_W * PIXELS_IN_METER),
            BUFFER),
            fill=(0,0,0),
            width=1,
        )

        draw.line((BUFFER, 
            BUFFER, 
            BUFFER,
            BUFFER+ (FIELD_H * PIXELS_IN_METER)),
            fill=(0,0,0),
            width=1,
        )

        draw.line((BUFFER + (FIELD_W * PIXELS_IN_METER), 
            BUFFER, 
            BUFFER + (FIELD_W * PIXELS_IN_METER),
            BUFFER + (FIELD_H * PIXELS_IN_METER)),
            fill=(0,0,0),
            width=1,
        )

        draw.line((BUFFER, 
            BUFFER + (FIELD_H * PIXELS_IN_METER), 
            BUFFER + (FIELD_W * PIXELS_IN_METER),
            BUFFER + (FIELD_H * PIXELS_IN_METER)),
            fill=(0,0,0),
            width=1,
        )


        # Draw the agents
        agents = state.agent_state
        for i, (x, y, team) in enumerate(zip(agents.pos_x, agents.pos_y, agents.team)):
            # print(x)
            # print(y)
            color = (255, 255, 0) if team == TEAM_1 else (0, 0, 255)

            draw.ellipse(
                (BUFFER + (x * PIXELS_IN_METER) - AGENT_RADIUS, BUFFER + (y * PIXELS_IN_METER) - AGENT_RADIUS,
                 BUFFER + (x * PIXELS_IN_METER) + AGENT_RADIUS, BUFFER + (y * PIXELS_IN_METER) + AGENT_RADIUS),
                fill=color, outline=(0, 0, 0))
        
        # Draw the ball
        ball = state.ball_state
        draw.ellipse(
                (BUFFER + (ball.pos_x * PIXELS_IN_METER) - AGENT_RADIUS, BUFFER + (ball.pos_y * PIXELS_IN_METER) - AGENT_RADIUS,
                 BUFFER + (ball.pos_x * PIXELS_IN_METER) + AGENT_RADIUS, BUFFER + (ball.pos_y * PIXELS_IN_METER) + AGENT_RADIUS,),
                fill=(255, 0, 0), outline=(0, 0, 0))

        return img
