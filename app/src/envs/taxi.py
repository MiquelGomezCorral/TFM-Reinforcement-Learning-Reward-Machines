

# def get_propositions_taxi(env, state):
#     """
#     Looks at the state transition and returns a string of events that occurred.
#     Passenger locations: 0=R, 1=G, 2=Y, 3=B, 4=in taxi
#     """
#     # decode returns: (taxi_row, taxi_col, p_location, destination)
#     _, _, p_loc, destination = env.unwrapped.decode(state)
    
#     props = []

#     if p_loc == 0:
#         props.append("r")
#     if p_loc == 1:
#         props.append("g")
#     if p_loc == 2:
#         props.append("y")
#     if p_loc == 3:
#         props.append("b")

#     if destination == 0:
#         props.append("dr")
#     if destination == 1:
#         props.append("dg")
#     if destination == 2:
#         props.append("dy")
#     if destination == 3:
#         props.append("db")

#     # 'p' is true as long as the passenger is in the taxi (index 4)
#     if p_loc == 4:
#         props.append("p")
#         # if p_loc == 0:
#         #     props.append("pr")
#         # if p_loc == 1:
#         #     props.append("pg")
#         # if p_loc == 2:
#         #     props.append("py")
#         # if p_loc == 3:
#         #     props.append("pb")
    
#     # get_propositions_taxi
#     if p_loc == 0 and destination == 0:  # delivered to R
#         props.append("del")
#     if p_loc == 1 and destination == 1:  # delivered to G
#         props.append("del")
#     if p_loc == 2 and destination == 2:  # delivered to Y
#         props.append("del")
#     if p_loc == 3 and destination == 3:  # delivered to B
#         props.append("del")
        
#     return props


def get_propositions_taxi(env, state):
    """
    Looks at the state transition and returns a string of events that occurred.
    Passenger locations: 0=R, 1=G, 2=Y, 3=B, 4=in taxi
    """
    # decode returns: (taxi_row, taxi_col, p_location, destination)
    _, _, p_loc, destination = env.unwrapped.decode(state)
    
    props = []

    if p_loc == 4:
        props.append("p")
    
    if p_loc == destination:
        props.append("d")
        
    return props

def get_propositions_multi_taxi(env, state):
    """
    Evaluates propositions for 2 passengers.
    p1, p2 = picked up (in taxi)
    d1, d2 = dropped correctly (at destination)
    """
    _, _, p1_loc, p1_dest, p2_loc, p2_dest = env.unwrapped.decode(state)
    
    props = []

    # Passenger 1
    if p1_loc == 4:
        props.append("p1")
    elif p1_loc == p1_dest:
        props.append("d1")
        
    # Passenger 2
    if p2_loc == 4:
        props.append("p2")
    elif p2_loc == p2_dest:
        props.append("d2")
        
    return props