import gymnasium as gym
from gymnasium import spaces

def get_propositions_doorkey(env, _state, _action, _new_state):
    """
    Looks at the internal MiniGrid state and returns a list of events.
    k: Agent is carrying the key
    o: The door is unlocked and open
    g: Agent reached the goal
    """
    props = []
    unwrapped = env.unwrapped

    # 1. Check if holding the key
    # carrying is an object, we check its type
    if unwrapped.carrying is not None and unwrapped.carrying.type == 'key':
        props.append("k")

    # 2. Check if the door is open
    # Iterate the grid to find the door object and check its state
    for obj in unwrapped.grid.grid:
        if obj is not None and obj.type == 'door':
            if obj.is_open:
                props.append("o")
            break # There is only one door

    # 3. Check if at goal
    # Compare agent position to the grid contents
    current_cell = unwrapped.grid.get(*unwrapped.agent_pos)
    if current_cell is not None and current_cell.type == 'goal':
        props.append("g")

    return props
class MiniGridDiscreteWrapper(gym.ObservationWrapper):
    """
    Wraps MiniGrid to output a single integer state dynamically 
    based on the actual grid size.
    """
    def __init__(self, env):
        super().__init__(env)
        
        # Dynamically get the true grid size
        self.w = self.unwrapped.width
        self.h = self.unwrapped.height
        
        # State space: w * h * 4 (dir) * 2 (key) * 2 (door)
        self.observation_space = gym.spaces.Discrete(self.w * self.h * 4 * 2 * 2)

    def observation(self, _):
        unwrapped = self.unwrapped
        
        agent_x, agent_y = unwrapped.agent_pos
        agent_dir = unwrapped.agent_dir
        
        # 1 if carrying key, else 0
        has_key = 1 if (unwrapped.carrying and unwrapped.carrying.type == 'key') else 0
        
        # 1 if door is open, else 0
        door_open = 0
        for obj in unwrapped.grid.grid:
            if obj is not None and obj.type == 'door':
                if obj.is_open:
                    door_open = 1
                break
                
        # Encode dynamically based on actual height
        state = ((((agent_x * self.h) + agent_y) * 4 + agent_dir) * 2 + has_key) * 2 + door_open
        return state
