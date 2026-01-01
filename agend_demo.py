"""
run_trained_agent.py

Demo script to run a trained PPO agent in DiceBluffEnv.
"""
import time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from blufpoker_env import DiceBluffEnv, Phase, ActionType, POKER_VALUE, PENALTY_LOSE_ROUND

def decode_throw_action(action):
    if not (ActionType.THROW_START <= action < ActionType.DECLARE_START):
        return "Invalid Throw Action"
    rel_action = action - ActionType.THROW_START
    die_actions = []
    temp_action = rel_action
    action_map = {0: "Keep,Hide", 1: "Keep,Show", 2: "Roll,Hide", 3: "Roll,Show"}
    for _ in range(3):
        die_actions.append(action_map[temp_action % 4])
        temp_action //= 4
    die_actions.reverse()
    using_cup = any("Roll,Hide" in a for a in die_actions)
    desc = f"THROW/REROLL: Die 1: {die_actions[0]}, Die 2: {die_actions[1]}, Die 3: {die_actions[2]}"
    if using_cup:
        desc += " (Cup Used!)"
    return desc

def run_agent_demo(model_path, rounds=3, render_delay=0.5):
    # Wrap environment in DummyVecEnv
    env = DummyVecEnv([lambda: DiceBluffEnv(num_players=4)])
    # Load trained PPO model
    model = PPO.load(model_path, env=env)

    for r in range(rounds):
        obs = env.reset()
        terminated = False
        step_count = 0
        print(f"\n========== ROUND {r+1} ==========")

        while not terminated:
            # Predict action using the trained agent
            action, _states = model.predict(obs, deterministic=True)
            action_val = int(action[0])

            # Extract observation for player 0
            phase = Phase(obs['phase'][0])
            player_index = obs['player_index'][0]
            prev_declared_value = obs['prev_declared_value'][0]
            poker_attempt = obs['poker_attempt'][0]

            # Describe action
            if phase == Phase.BELIEVE:
                action_desc = "BELIEVE" if action_val == ActionType.BELIEVE else "DOUBT"
            elif phase == Phase.THROW:
                action_desc = decode_throw_action(action_val)
            elif phase == Phase.DECLARE:
                action_desc = f"DECLARE {action_val - ActionType.DECLARE_START}"
            elif phase == Phase.POKER:
                action_desc = f"POKER THROW (Attempt {poker_attempt + 1})"
            else:
                action_desc = str(action_val)

            print(f"\nStep {step_count} >> Player {player_index} ({phase.name}) decides: {action_desc}")

            # Step environment
            obs, rewards, dones, infos = env.step([action_val])

            # Render environment
            env.envs[0].render()

            # Print reward and info if any
            if rewards[0] != 0:
                print(f"Reward: {rewards[0]}")
            if infos and infos[0]:
                print(f"Info: {infos[0]}")

            step_count += 1
            terminated = dones[0]

            time.sleep(render_delay)

        print(f"\nROUND {r+1} ENDED! LOSER: Player {env.envs[0].loser} (Penalty: {PENALTY_LOSE_ROUND})")
        print("===================================")

if __name__ == "__main__":
    run_agent_demo("ppo_dicebluff_final.zip", rounds=3, render_delay=0.2)
