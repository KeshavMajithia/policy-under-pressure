from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import os

class RLAgentFactory:
    @staticmethod
    def create(env, verbose=1, tensorboard_log=None):
        """
        Creates a PPO agent for the given environment.
        """
        # PPO Hyperparameters (tuned for continuous control simple tasks)
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
            verbose=verbose,
            tensorboard_log=tensorboard_log
        )
        return model

    @staticmethod
    def load(path, env):
        return PPO.load(path, env=env)
