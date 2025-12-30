"""
Demo script to run a trained PPO agent in DiceBluffEnv.
"""

import time

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import DummyVecEnv
from blufpoker_env import DiceBluffEnv, Phase, ActionType, PENALTY_LOSE_ROUND

# --- Decode throw action for readability ---
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

# --- Main demo ---
def run_agent_demo(model_path, rounds=3, render_delay=0.5):
    env = DummyVecEnv([lambda: DiceBluffEnv(num_players=4)])
    model = PPO.load(model_path, env=env)

    for r in range(rounds):
        obs, _ = env.reset()
        terminated, truncated = False, False
        step_count = 0
        print(f"\n========== ROUND {r+1} ==========")
        while not terminated and not truncated:
            action, _states = model.predict(obs, deterministic=True)
            # VecEnv returns array of shape (1, ), take first element
            action_val = int(action[0])
            phase = Phase(obs[0]['phase'])
            action_desc = ""

            if phase == Phase.BELIEVE:
                action_desc = "BELIEVE" if action_val == ActionType.BELIEVE else "DOUBT"
            elif phase == Phase.THROW:
                action_desc = decode_throw_action(action_val)
            elif phase == Phase.DECLARE:
                action_desc = f"DECLARE {action_val - ActionType.DECLARE_START}"
            elif phase == Phase.POKER:
                attempt_num = obs[0]['poker_attempt'] + 1
                action_desc = f"POKER THROW (Attempt {attempt_num})"

            print(f"\nStep {step_count} >> Player {obs[0]['player_index']} ({phase.name}) decides: {action_desc}")
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            if reward[0] != 0:
                print(f"Reward: {reward[0]}")
            if info and info[0]:
                print(f"Info: {info[0]}")

            step_count += 1
            time.sleep(render_delay)

        print(f"\nROUND {r+1} ENDED! LOSER: Player {env.envs[0].loser} (Penalty: {PENALTY_LOSE_ROUND})")
        print("===================================")

if __name__ == "__main__":
    # Replace with your trained model path
    run_agent_demo("ppo_dicebluff_final.zip", rounds=3, render_delay=0.2)
