def get_propositions_taxi(env, state, _action, new_state):
    """
    Looks at the state transition and returns a string of events that occurred.
    Passenger locations: 0=R, 1=G, 2=Y, 3=B, 4=in taxi
    """
    _, _, previous_location, _ = env.unwrapped.decode(state)
    _, _, passenger_location, destination = env.unwrapped.decode(new_state)
    props = []

    location = previous_location if previous_location < 4 else passenger_location
    if location < 4:
        props.append(("r", "g", "y", "b")[location])

    props.append(("dr", "dg", "dy", "db")[destination])

    if passenger_location == 4:
        props.append("p")

    if passenger_location == destination:
        props.append("d")
        props.append("del")

    return props


def get_propositions_multi_taxi(env, _state, _action, new_state):
    """
    Evaluates propositions for 2 passengers.
    p1, p2 = picked up (in taxi)
    d1, d2 = dropped correctly (at destination)
    """
    _, _, p1_loc, p1_dest, p2_loc, p2_dest = env.unwrapped.decode(new_state)
    props = []

    if p1_loc == 4:
        props.append("p1")
    elif p1_loc == p1_dest:
        props.append("d1")

    if p2_loc == 4:
        props.append("p2")
    elif p2_loc == p2_dest:
        props.append("d2")

    return props
