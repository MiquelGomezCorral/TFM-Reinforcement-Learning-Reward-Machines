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


def get_propositions_multi_taxi(env, state, _action, new_state):
    """
    Return pickup and delivery events for passenger state transitions.

    p1, p2 are emitted only when a passenger enters the taxi. d1, d2 are
    emitted only when a passenger is delivered at its destination.
    """
    previous = env.unwrapped.decode(state)
    decoded = env.unwrapped.decode(new_state)
    props = []

    for i in range(env.unwrapped.num_passengers):
        previous_location = previous[2 + i * 2]
        passenger_location, destination = decoded[2 + i * 2:4 + i * 2]
        if previous_location != 4 and passenger_location == 4:
            props.append(f"p{i + 1}")
        elif previous_location == 4 and passenger_location == destination:
            props.append(f"d{i + 1}")

    return props
