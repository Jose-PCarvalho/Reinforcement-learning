from gym.spaces import Discrete, MultiDiscrete, Dict, MultiBinary, Box
from ipywidgets import Output
from IPython import display
import numpy as np
import pygame

# Ray imports.
import ray
from ray.rllib.env.multi_agent_env import MultiAgentEnv


class MultiAgentArena(MultiAgentEnv):
    def __init__(self, config=None):
        config = config or {}
        # Dimensions of the grid.
        self.width = config.get("width", 6)
        self.height = config.get("height", 6)
        self.size = self.width  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window
        self.window = None
        self.clock = None

        self.observation_space = Dict({"agents": MultiDiscrete([self.width * self.height, self.width * self.height]),
                                       "coverage": MultiBinary(self.height*self.height)})
        self.action_space = Discrete(5)

        # Reset env.
        self.reset()

        # For rendering.
        if config.get("render"):
            self.render_mode = "human"

    def reset(self):
        """Returns initial observation of next(!) episode."""
        self.metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
        # Row-major coords.
        self.agent1_pos = [0, 0]  # upper left corner
        self.agent2_pos = [self.height - 1, self.width - 1]  # lower bottom corner

        # Accumulated rewards in this episode.
        self.agent1_R = 0.0
        self.agent2_R = 0.0

        # Reset agent1's visited fields.
        self.agent1_visited_fields = set([tuple(self.agent1_pos)])
        self.agent1_visited_fields.add(tuple(self.agent2_pos))
        self.map=np.zeros((self.height, self.width), dtype=int)
        self.coverage_map = np.zeros((self.height* self.width), dtype=int)
        self.coverage_map[self._discrete_pos(self.agent1_pos)] = 1
        self.coverage_map[self._discrete_pos(self.agent2_pos)] = 1
        self.map[self.agent1_pos[0],self.agent1_pos[1]] = 1
        self.map[self.agent2_pos[0], self.agent2_pos[1]] = 1
        self.remaining = self.height * self.width - 2;
        # How many timesteps have we done in this episode.
        self.timesteps = 0
        # Did we have a collision in recent step?
        self.collision = False
        # How many collisions in total have we had in this episode?
        self.num_collisions = 0
        # Return the initial observation in the new episode.
        return self._get_obs()

    def step(self, action: dict):
        """
        Returns (next observation, rewards, dones, infos) after having taken the given actions.

        e.g.
        `action={"agent1": action_for_agent1, "agent2": action_for_agent2}`
        """

        # increase our time steps counter by 1.
        self.timesteps += 1
        # An episode is "done" when we reach the time step limit.
        # Agent2 always moves first.
        # events = [collision|agent1_new_field]
        events = self._move(self.agent2_pos, action["agent2"], is_agent1=False)
        events |= self._move(self.agent1_pos, action["agent1"], is_agent1=True)
        r1 = 0
        r2 = 0

        # Useful for rendering.
        self.collision = "collision" in events
        if self.collision is True:
            self.num_collisions += 1

        # Get observations (based on new agent positions).
        obs = self._get_obs()

        # Determine rewards based on the collected events:
        if "Blocked1" in events:
            r1 -= 0.5
        if "Blocked2" in events:
            r2 -= 0.5
        if "collision" in events:
            r1 -= 0.5
            r2 -= 0.5
        if "agent1_new_field" in events:
            r1 += 1
        if "agent2_new_field" in events:
            r2 += 1
        if action["agent1"] == 4:
            r1 -= 0.1
        else:
            r1 -= 0.05
        if action["agent2"] == 4:
            r2 -= 0.1
        else:
            r2 -= 0.05

        is_done = self.remaining == 0
        if is_done:
            r1 = 100
            r2 = 100

        self.agent1_R += r1
        self.agent2_R += r2

        rewards = {
            "agent1": r1,
            "agent2": r2,
        }

        # Generate a `done` dict (per-agent and total).
        dones = {
            "agent1": is_done,
            "agent2": is_done,
            # special `__all__` key indicates that the episode is done for all agents.
            "__all__": is_done,
        }

        return obs, rewards, dones, {}  # <- info dict (not needed here).

    def _get_obs(self):
        """
        Returns obs dict (agent name to discrete-pos tuple) using each
        agent's current x/y-positions.
        """
        ag1_discrete_pos = self._discrete_pos(self.agent1_pos)
        ag2_discrete_pos = self._discrete_pos(self.agent2_pos)
        return {
            "agent1": {"agents": np.array([ag1_discrete_pos, ag2_discrete_pos]), "coverage": self.coverage_map},
            "agent2": {"agents": np.array([ag2_discrete_pos, ag1_discrete_pos]), "coverage": self.coverage_map}
        }

    def _discrete_pos(self, coords):

        return coords[0] * self.width + \
               (coords[1] % self.width)

    def _move(self, coords, action, is_agent1):
        """
        Moves an agent (agent1 iff is_agent1=True, else agent2) from `coords` (x/y) using the
        given action (0=up, 1=right, etc..) and returns a resulting events dict:
        Agent1: "new" when entering a new field. "bumped" when having been bumped into by agent2.
        Agent2: "bumped" when bumping into agent1 (agent1 then gets -1.0).
        """
        if action == 4:
            return {"no_move"}
        orig_coords = coords[:]
        # Change the row: 0=up (-1), 2=down (+1)
        coords[0] += -1 if action == 0 else 1 if action == 2 else 0
        # Change the column: 1=right (+1), 3=left (-1)
        coords[1] += 1 if action == 1 else -1 if action == 3 else 0


        # Solve collisions.
        # Make sure, we don't end up on the other agent's position.
        # If yes, don't move (we are blocked).
        if (is_agent1 and coords == self.agent2_pos) or (not is_agent1 and coords == self.agent1_pos):
            coords[0], coords[1] = orig_coords
            # Agent2 blocked agent1 (agent1 tried to run into agent2)
            # OR Agent2 bumped into agent1 (agent2 tried to run into agent1)
            return {"collision"}

        # No agent blocking -> check walls.
        blocked = False
        if coords[0] < 0:
            coords[0] = 0
            blocked = True
        elif coords[0] >= self.height:
            coords[0] = self.height - 1
            blocked = True
        if coords[1] < 0:
            coords[1] = 0
            blocked = True
        elif coords[1] >= self.width:
            coords[1] = self.width - 1
            blocked = True
        if blocked and is_agent1:
            return {"Blocked1"}
        if blocked and not is_agent1:
            return {"Blocked2"}

        # If agent1 -> "new" if new tile covered.
        if is_agent1 and not tuple(coords) in self.agent1_visited_fields:
            self.agent1_visited_fields.add(tuple(coords))
            self.coverage_map[self._discrete_pos(self.agent1_pos)] = 1
            self.map[self.agent1_pos[0], self.agent1_pos[1]] = 1
            self.remaining -= 1;
            return {"agent1_new_field"}
        if not is_agent1 and not tuple(coords) in self.agent1_visited_fields:
            self.agent1_visited_fields.add(tuple(coords))
            self.coverage_map[self._discrete_pos(self.agent2_pos)] = 1
            self.map[self.agent2_pos[0], self.agent2_pos[1]] = 1
            self.remaining -= 1;
            return {"agent2_new_field"}
        # No new tile for agent1.
        return set()

    def render(self, mode=None):

        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
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
            (np.flip(np.array(self.agent1_pos)) + 0.5) * pix_square_size,
            pix_square_size / 3,
        )
        pygame.draw.circle(
            canvas,
            (0, 255, 0),
            (np.flip(np.array(self.agent2_pos)) + 0.5) * pix_square_size,
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

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
