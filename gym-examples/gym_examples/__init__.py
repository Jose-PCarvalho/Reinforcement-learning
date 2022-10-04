from gym.envs.registration import register

register(
    id="gym_examples/GridWorldTarget-v0",
    entry_point="gym_examples.envs:GridWorldTargetEnv",
)

register(
    id="gym_examples/GridWorldCoverage-v0",
    entry_point="gym_examples.envs:GridWorldCoverageEnv",
)
