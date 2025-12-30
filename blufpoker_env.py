import gymnasium as gym
from gymnasium import spaces
import numpy as np
from enum import IntEnum

# --- Game Constants ---
NUM_DICE = 3
MAX_DIE_VALUE = 6
MAX_NORMAL_DECLARE = 665
POKER_VALUE = 1000
MAX_DECLARE_VAL = 1000

# --- Rewards (Survival Focused) ---
# The goal is to NOT lose. 
# There is no real "winning", only making someone else lose.
REWARD_SURVIVE = 0.05       # Small reward for successfully passing the turn
REWARD_CATCH_BLUFF = 0.05   # Equivalent to surviving (round ends, you didn't lose)
PENALTY_LOSE_ROUND = -1.0   # Big penalty: You buy the drinks
PENALTY_INVALID = -0.5

class Phase(IntEnum):
    DECLARE = 0
    BELIEVE = 1
    THROW = 2
    POKER = 3

class ActionType(IntEnum):
    BELIEVE = 0
    DOUBT = 1
    # Throw actions cover indices 2 to 65 (64 combinations)
    THROW_START = 2
    # Declare actions cover indices 66 to 1066
    DECLARE_START = 66

class DiceBluffEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, num_players=3):
        super().__init__()
        assert 3 <= num_players <= 6, "Player count must be between 3 and 6."
        self.num_players = num_players

        # --- Action Space ---
        # 0: Believe
        # 1: Doubt
        # 2..65: Throw Actions (64 combos of Reroll/Visibility)
        # 66..1066: Declare Value (0 to 1000)
        self.action_space = spaces.Discrete(
            ActionType.DECLARE_START + MAX_DECLARE_VAL + 1
        )

        # --- Observation Space ---
        self.observation_space = spaces.Dict({
            "dice": spaces.Box(0, 6, shape=(3,), dtype=np.int8),
            "cup_mask": spaces.Box(0, 1, shape=(3,), dtype=np.int8), # 1=Hidden, 0=Visible
            "declared_value": spaces.Discrete(MAX_DECLARE_VAL + 1),
            "prev_declared_value": spaces.Discrete(MAX_DECLARE_VAL + 1),
            "phase": spaces.Discrete(len(Phase)),
            "player_index": spaces.Discrete(num_players),
            "poker_attempt": spaces.Discrete(4) 
        })

        self.np_random = np.random.default_rng()
        self.loser = 0 
        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_player = self.loser
        
        # Initial State
        self.dice = self._roll_dice([True, True, True]) 
        self.cup_mask = np.ones(3, dtype=np.int8) # All hidden under cup
        
        self.declared_value = 0
        self.prev_declared_value = 0
        self.phase = Phase.DECLARE
        self.poker_attempts = 0
        
        return self._get_obs(), {}

    def step(self, action):
        reward = 0.0
        terminated = False
        truncated = False
        info = {}

        if not self._is_action_valid_for_phase(action):
            return self._get_obs(), PENALTY_INVALID, False, False, {"error": "Invalid action for phase"}

        if self.phase == Phase.BELIEVE:
            reward, terminated = self._handle_believe(action)
        
        elif self.phase == Phase.THROW:
            self._handle_throw(action)
            
        elif self.phase == Phase.DECLARE:
            valid_declare = self._handle_declare(action)
            if not valid_declare:
                return self._get_obs(), PENALTY_INVALID, False, False, {"error": "Declaration must be higher"}
            
        elif self.phase == Phase.POKER:
            reward, terminated = self._handle_poker(action)

        return self._get_obs(), reward, terminated, truncated, info

    # --- Phase Handlers ---

    def _handle_believe(self, action):
        if action == ActionType.DOUBT:
            # Reveal all
            self.cup_mask = np.zeros(3, dtype=np.int8) 
            actual_val = self._calculate_value(self.dice)
            
            is_bluff = actual_val < self.declared_value
            
            if is_bluff:
                # Declarer lied -> Declarer loses
                self.loser = (self.current_player - 1) % self.num_players
                # Current player survived/caught them
                return REWARD_CATCH_BLUFF, True
            else:
                # Truth -> Doubter (current) loses
                self.loser = self.current_player
                return PENALTY_LOSE_ROUND, True

        elif action == ActionType.BELIEVE:
            if self.declared_value == POKER_VALUE:
                self.phase = Phase.POKER
                self.poker_attempts = 0
                return 0.0, False
            else:
                self.phase = Phase.THROW
                return REWARD_SURVIVE, False
        
        return 0.0, False

    def _handle_throw(self, action):
        """
        Action: 2..65 (Base-4 encoded for 3 dice)
        Per Die Action: 0=(Keep,Hide), 1=(Keep,Show), 2=(Roll,Hide), 3=(Roll,Show)
        
        PHYSICAL CONSTRAINT: 
        If ANY die is being Rolled+Hidden (Action 2), it means the cup is being used.
        Therefore, ANY die that is Kept (Action 0 or 1) MUST be outside the cup.
        We force Action 0 (Keep+Hide) -> Action 1 (Keep+Show) in this scenario.
        """
        rel_action = action - ActionType.THROW_START
        
        # 1. Decode Actions
        die_actions = []
        temp_action = rel_action
        for _ in range(3):
            die_actions.append(temp_action % 4)
            temp_action //= 4
            
        # 2. Check Physical Constraint
        # Is the player trying to roll anything under the cup?
        using_cup_for_roll = any(a == 2 for a in die_actions)
        
        if using_cup_for_roll:
            for i in range(3):
                # If we are trying to Keep+Hide (0) while using the cup for others,
                # gravity/physics forces us to reveal the kept die.
                if die_actions[i] == 0:
                    die_actions[i] = 1 # Force Keep+Show

        # 3. Execute
        reroll_mask = []
        for i in range(3):
            da = die_actions[i]
            # 2 or 3 means Roll
            should_roll = (da >= 2)
            # 0 or 2 means Hide
            should_hide = (da % 2 == 0)
            
            reroll_mask.append(should_roll)
            self.cup_mask[i] = 1 if should_hide else 0
            
        new_rolls = self._roll_dice(reroll_mask)
        for i in range(3):
            if reroll_mask[i]:
                self.dice[i] = new_rolls[i]
                
        self.phase = Phase.DECLARE

    def _handle_declare(self, action):
        val = action - ActionType.DECLARE_START
        if val <= self.prev_declared_value:
            return False
        self.declared_value = val
        self.prev_declared_value = val
        self.current_player = (self.current_player + 1) % self.num_players
        self.phase = Phase.BELIEVE
        return True

    def _handle_poker(self, action):
        self.poker_attempts += 1
        reroll_mask = [False, False, False]
        
        if self.poker_attempts == 1:
            reroll_mask = [True, True, True]
        else:
            if not (ActionType.THROW_START <= action < ActionType.DECLARE_START):
                return PENALTY_INVALID, False
            rel_action = action - ActionType.THROW_START
            temp_action = rel_action
            for i in range(3):
                if (temp_action % 4) >= 2: # Roll
                    reroll_mask[i] = True
                temp_action //= 4
        
        new_rolls = self._roll_dice(reroll_mask)
        for i in range(3):
            if reroll_mask[i]:
                self.dice[i] = new_rolls[i]
                # In poker phase, usually all are hidden or revealed at end.
                # We'll keep them hidden (1) during attempts.
                self.cup_mask[i] = 1 
                
        if self._calculate_value(self.dice) == POKER_VALUE:
            # Poker achieved! The person who doubted/sent me here loses.
            self.loser = (self.current_player - 1) % self.num_players
            return REWARD_CATCH_BLUFF, True # I survived
            
        if self.poker_attempts >= 3:
            # Failed
            self.loser = self.current_player
            return PENALTY_LOSE_ROUND, True
            
        return 0.0, False

    # --- Helpers ---

    def _is_action_valid_for_phase(self, action):
        if self.phase == Phase.BELIEVE:
            return action in [ActionType.BELIEVE, ActionType.DOUBT]
        if self.phase == Phase.THROW:
            return ActionType.THROW_START <= action < ActionType.DECLARE_START
        if self.phase == Phase.DECLARE:
            return action >= ActionType.DECLARE_START
        if self.phase == Phase.POKER:
            return ActionType.THROW_START <= action < ActionType.DECLARE_START
        return False

    def _get_obs(self):
        obs_dice = self.dice.copy()
        if self.phase == Phase.BELIEVE:
            for i in range(3):
                if self.cup_mask[i] == 1:
                    obs_dice[i] = 0     
        return {
            "dice": obs_dice,
            "cup_mask": self.cup_mask.copy(),
            "declared_value": self.declared_value,
            "prev_declared_value": self.prev_declared_value,
            "phase": int(self.phase),
            "player_index": self.current_player,
            "poker_attempt": self.poker_attempts
        }

    def _roll_dice(self, mask):
        new_vals = [0, 0, 0]
        rolls = self.np_random.integers(1, 7, size=sum(mask))
        roll_idx = 0
        for i in range(3):
            if mask[i]:
                new_vals[i] = rolls[roll_idx]
                roll_idx += 1
            else:
                new_vals[i] = self.dice[i]
        return np.array(new_vals, dtype=np.int8)

    def _calculate_value(self, dice):
        if dice[0] == dice[1] == dice[2]:
            return POKER_VALUE
        s = sorted(dice, reverse=True)
        return int(s[0]) * 100 + int(s[1]) * 10 + int(s[2])


    def render(self):
        phases = ["DECLARE", "BELIEVE", "THROW", "POKER"]
        print(f"--- Player {self.current_player} ({phases[self.phase]}) ---")
        dice_str = []
        for i in range(3):
            val = str(self.dice[i])
            if self.cup_mask[i] == 1:
                val = f"[{val}]"
            dice_str.append(val)
        print(f"Dice: {' '.join(dice_str)}")
        print(f"Declared: {self.declared_value} (Prev: {self.prev_declared_value})")