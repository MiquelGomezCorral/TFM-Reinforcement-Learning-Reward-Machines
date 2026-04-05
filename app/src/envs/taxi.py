

def get_propositions(env, state):
    """
    Looks at the state transition and returns a string of events that occurred.
    Passenger locations: 0=R, 1=G, 2=Y, 3=B, 4=in taxi
    """
    # decode returns: (taxi_row, taxi_col, p_location, destination)
    _, _, p_loc, destination = env.unwrapped.decode(state)
    
    props = []

    # 'p' is true as long as the passenger is in the taxi (index 4)
    if p_loc == 4:
        props.append("p")
    
    if p_loc == destination:
        props.append("d")
    
    return props
