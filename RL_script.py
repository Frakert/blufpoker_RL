from stable_baselines3 import PPO
from stable_baselines3.common.env_util import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

from blufpoker_env import DiceBluffEnv

if __name__ == "__main__":
    # Wrap in DummyVecEnv
    env = DummyVecEnv([lambda: DiceBluffEnv(num_players=4)])

    # Optional: Save checkpoints every 10000 steps
    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./models/',
                                             name_prefix='ppo_dicebluff')

    # Instantiate PPO agent
    model = PPO(
        "MultiInputPolicy", env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        tensorboard_log="./tb_dicebluff/"
    )

    # Train
    model.learn(total_timesteps=500_000, callback=checkpoint_callback)

    # Save final model
    model.save("ppo_dicebluff_final")