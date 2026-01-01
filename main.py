import numpy as np
from blufpoker_env import DiceBluffEnv, Phase, ActionType, POKER_VALUE, MAX_NORMAL_DECLARE, NUM_DICE, MAX_DIE_VALUE

class RandomBluffPokerAgent:
    def __init__(self, rng=None, doubt_prob=0.25, poker_prob=0.05):
        self.rng = rng or np.random.default_rng()
        self.doubt_prob = doubt_prob
        self.poker_prob = poker_prob

    def act(self, obs):
        phase = Phase(obs["phase"])

        if phase == Phase.BELIEVE:
            return self._believe()
        if phase == Phase.THROW:
            return self._throw()
        if phase == Phase.DECLARE:
            return self._declare(obs)
        if phase == Phase.POKER:
            return self._throw()

        raise RuntimeError("Unknown phase")

    def _believe(self):
        return (
            ActionType.DOUBT
            if self.rng.random() < self.doubt_prob
            else ActionType.BELIEVE
        )

    def _throw(self):
        return self.rng.integers(
            ActionType.THROW_START,
            ActionType.DECLARE_START
        )

    def _declare(self, obs):
        prev = obs["prev_declared_value"]

        # Occasionally attempt poker
        if prev < POKER_VALUE and self.rng.random() < self.poker_prob:
            return ActionType.DECLARE_START + POKER_VALUE

        min_val = max(prev + 1, 111)
        if min_val > MAX_NORMAL_DECLARE:
            return ActionType.DECLARE_START + POKER_VALUE

        val = self.rng.integers(min_val, MAX_NORMAL_DECLARE + 1)
        return ActionType.DECLARE_START + val

def describe_action(action):
    if action == ActionType.BELIEVE:
        return "BELIEVE"
    if action == ActionType.DOUBT:
        return "DOUBT"

    if ActionType.THROW_START <= action < ActionType.DECLARE_START:
        rel = action - ActionType.THROW_START
        acts = []
        for i in range(3):
            a = rel % 4
            rel //= 4
            acts.append(
                ["Keep+Hide", "Keep+Show", "Roll+Hide", "Roll+Show"][a]
            )
        return f"THROW [{', '.join(acts)}]"

    if action >= ActionType.DECLARE_START:
        val = action - ActionType.DECLARE_START
        return f"DECLARE {val}"

    return f"UNKNOWN ({action})"


if __name__ == "__main__":
    env = DiceBluffEnv(num_players=5)
    agent = RandomBluffPokerAgent()

    obs, _ = env.reset()
    round_idx = 1
    step_idx = 0

    while round_idx <= 100:
        step_idx += 1

        print(f"\nRound {round_idx} | Step {step_idx}")
        env.render()

        action = agent.act(obs)
        print(f"Action chosen: {describe_action(action)}, raw: {action}")

        obs, reward, terminated, truncated, info = env.step(action)

        print(f"Reward: {reward: .2f}")

        if info:
            print(f"Info: {info}")

        if terminated:
            print("\n*** ROUND ENDED ***")
            print(f"Loser index: {env.loser}")
            print("-" * 40)

            obs, _ = env.reset()
            round_idx += 1
            step_idx = 0