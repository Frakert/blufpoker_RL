import time
import numpy as np
# Assuming the updated environment is saved as dice_bluff_env.py
from blufpoker_env import DiceBluffEnv, Phase, ActionType, POKER_VALUE, PENALTY_LOSE_ROUND

def print_separator(char='-', length=60):
    print(char * length)

def decode_throw_action(action):
    """Decodes the throw action for logging purposes."""
    if not (ActionType.THROW_START <= action < ActionType.DECLARE_START):
        return "Invalid Throw Action"
        
    rel_action = action - ActionType.THROW_START
    die_actions = []
    temp_action = rel_action
    
    action_map = {
        0: "Keep,Hide", 1: "Keep,Show", 
        2: "Roll,Hide", 3: "Roll,Show"
    }
    
    for _ in range(3):
        die_actions.append(action_map[temp_action % 4])
        temp_action //= 4
    
    die_actions.reverse()
    
    # Check if this action uses the cup for rolling (Action 2 = Roll,Hide)
    using_cup_for_roll = any("Roll,Hide" in act for act in die_actions)
    
    if rel_action == 0:
        return "BLIND PASS (Keep All, Hide All)"
    
    desc = f"THROW/REROLL: Die 1: {die_actions[0]}, Die 2: {die_actions[1]}, Die 3: {die_actions[2]}"
    if using_cup_for_roll:
        desc += " (Cup Used!)"
    return desc

def run_test_game(render_delay=0.5):
    """
    Runs a simulation of the Blufpoker environment with automatic valid actions.
    """
    # Increased to 4 players for more interaction complexity
    env = DiceBluffEnv(num_players=4) 
    obs, info = env.reset()
    
    terminated = False
    truncated = False
    round_count = 1

    print_separator('=')
    print(f"STARTING NEW GAME (4 Players)")
    print_separator('=')
    
    env.render()

    for i in range(10): # Run until interrupted or max rounds reached
        current_phase = obs['phase']
        prev_val = obs['prev_declared_value']
        
        action = 0
        action_desc = ""

        # --- Smart Random Agent Logic ---
        if current_phase == Phase.BELIEVE:
            # 70% chance to Believe, 30% chance to Doubt
            action = np.random.choice([ActionType.BELIEVE, ActionType.DOUBT], p=[0.7, 0.3])
            action_desc = "BELIEVE" if action == ActionType.BELIEVE else "DOUBT"
            
            if prev_val == POKER_VALUE:
                 action_desc += " (Poker Declared!)"

        elif current_phase == Phase.THROW:
            # Pick a random valid Throw action (2 to 65)
            # The environment will auto-correct the mask if the physical constraint is violated.
            action = np.random.randint(ActionType.THROW_START, ActionType.DECLARE_START)
            action_desc = decode_throw_action(action)

        elif current_phase == Phase.DECLARE:
            # Generate a strictly higher declaration
            
            # Start slightly higher
            declare_val = prev_val + np.random.randint(1, 4) 
            
            # 5% chance to declare Poker, unless we are already very high
            if np.random.random() < 0.05 and declare_val < POKER_VALUE: 
                declare_val = POKER_VALUE
            
            # If we cross the 665 boundary, must jump to 1000 or declare a normal high number
            if 665 < declare_val < POKER_VALUE:
                # Randomly either cap at 665 (invalid, but let env punish) or jump to 1000
                if np.random.random() < 0.5:
                    declare_val = 666 # Will be rejected unless env allows 'nonsense' above 665.
                else:
                    declare_val = POKER_VALUE

            # Ensure declaration is at least 1, never 0
            declare_val = max(declare_val, 1)

            action = ActionType.DECLARE_START + declare_val
            action_desc = f"DECLARE {declare_val}"

        elif current_phase == Phase.POKER:
            # Random throw action for poker attempts (3-throw rule)
            action = np.random.randint(ActionType.THROW_START, ActionType.DECLARE_START)
            attempt_num = obs['poker_attempt'] + 1
            action_desc = f"POKER THROW (Attempt {attempt_num})"

        # --- Execute Step ---
        print(f"\n>> P{obs['player_index']} ({Phase(current_phase).name}) decides: {action_desc}")
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Visualization ---
        env.render()
        
        if info:
            print(f"Info: {info}")
        if reward != 0:
            print(f"Reward: {reward}")

        if terminated:
            loser = env.loser
            print_separator('*')
            print(f"ROUND {round_count} ENDED!")
            print(f"LOSER: Player {loser} (Reward: {PENALTY_LOSE_ROUND})")
            
            if reward > 0:
                print("Result: BLUFF CAUGHT / POKER SUCCESSFULLY MADE")
            else:
                print("Result: FALSE DOUBT / POKER FAILED")
                
            print_separator('*')
            
            # Start New Round
            round_count += 1
            obs, info = env.reset()
            
            # Render new start state
            print_separator('=')
            print(f"STARTING ROUND {round_count}. Player {env.loser} starts.")
            print_separator('=')
            env.render()


        time.sleep(render_delay)

if __name__ == "__main__":
    # Run the game continuously until manually stopped
    try:
        run_test_game(render_delay=0.2)
    except KeyboardInterrupt:
        print("\n\nTest run stopped by user.")