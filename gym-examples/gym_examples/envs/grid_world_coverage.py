import gym
from gym import spaces
import pygame
import numpy as np
import random

from gym.spaces import MultiBinary, Box


class GridWorldCoverageEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=10):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window

        self.time_steps = 0

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict({"map": Box(low=0, high=1, shape=[self.size,self.size], dtype=np.int),#MultiBinary([self.size, self.size]),
                                              "agent": Box(low=0, high=1, shape=[self.size,self.size], dtype=np.int),
                                              "remaining": Box(low=0, high=self.size * self.size, shape=[(1)], dtype=np.int)
                                              })  # Box(low=-1, high=2, shape=(size, size), dtype=np.int)
        # We have 4 actions, corresponding to "right", "up", "left", "down", "right"
        self.action_space = spaces.Discrete(4)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        #print(self.agent_map)
        return {
            "map": self.map ,
            "agent" :self.agent_map ,
            "remaining": np.array([self.remaining])
        }

    def _get_info(self):
        return {
            "Covered": self.size * self.size - np.count_nonzero(self.map)
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        # super().reset(seed=seed)
        self.map = np.zeros((self.size, self.size), dtype=int)
        self.agent_map = np.zeros((self.size, self.size), dtype=int)
        self.time_steps = 0
        self.remaining=self.size*self.size
        self.last_direction = np.array([0, 0])
        # Choose the agent's location uniformly at randomself._agent_location = np.random.randint(1, self.size, size=2)
        self._agent_location = np.random.randint(1, self.size, size=2)
        self.agent_map[self._agent_location[1], self._agent_location[0]] = 1
        self.map[self._agent_location[1], self._agent_location[0]] = 1

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        self.time_steps = self.time_steps + 1;
        reward = 0
        direction = self._action_to_direction[action]
        if (direction == (-1 * self.last_direction)).all==True:
            print(direction,-1*self.last_direction)
            reward = reward - 1 / self.size
        self.last_direction = direction
        self.agent_map[self._agent_location[1], self._agent_location[0]] = 0
        if ((self._agent_location + direction) < 0).any() or ((self._agent_location + direction) >= self.size).any():
            reward = reward - 1 / self.size

        # We use `np.clip` to make sure we don't leave the gridCheck
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )
        if self.map[self._agent_location[1], self._agent_location[0]] == 0:
            reward = reward + 1 / self.size
            #print(reward)
        else :
          reward = reward - 1 / self.size
        self.agent_map[self._agent_location[1], self._agent_location[0]] = 1
        self.map[self._agent_location[1], self._agent_location[0]] = 1
        self.remaining=self.size*self.size-np.count_nonzero(self.map)
        terminated = self.remaining==0

        if terminated:
            reward = reward + 1 * self.size

        observation = self._get_obs()
        info = self._get_info()
        # dist = info.get('distance')

        if self.render_mode == "human":
            self._render_frame()
            print(reward)

        return observation, reward, terminated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            #pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
                self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        for i in range(self.size):
            for j in range(self.size):
                if self.map[i, j] == 1:
                    pygame.draw.rect(
                        canvas,
                        (255, 0, 0),
                        pygame.Rect(
                            pix_square_size * np.array([j, i]),
                            (pix_square_size, pix_square_size),
                        ),
                    )

        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            #print(self.map)


            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()