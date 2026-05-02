

def get_propositions_taxi(env, state):
    """
    Looks at the state transition and returns a string of events that occurred.
    Passenger locations: 0=R, 1=G, 2=Y, 3=B, 4=in taxi
    """
    # decode returns: (taxi_row, taxi_col, p_location, destination)
    _, _, p_loc, destination = env.unwrapped.decode(state)
    
    props = []

    if p_loc == 0:
        props.append("r")
    if p_loc == 1:
        props.append("g")
    if p_loc == 2:
        props.append("y")
    if p_loc == 3:
        props.append("b")

    if destination == 0:
        props.append("dr")
    if destination == 1:
        props.append("dg")
    if destination == 2:
        props.append("dy")
    if destination == 3:
        props.append("db")

    # 'p' is true as long as the passenger is in the taxi (index 4)
    if p_loc == 4:
        props.append("p")
        # if p_loc == 0:
        #     props.append("pr")
        # if p_loc == 1:
        #     props.append("pg")
        # if p_loc == 2:
        #     props.append("py")
        # if p_loc == 3:
        #     props.append("pb")
    
    # get_propositions_taxi
    if p_loc == 0 and destination == 0:  # delivered to R
        props.append("del")
    if p_loc == 1 and destination == 1:  # delivered to G
        props.append("del")
    if p_loc == 2 and destination == 2:  # delivered to Y
        props.append("del")
    if p_loc == 3 and destination == 3:  # delivered to B
        props.append("del")
        
    return props
