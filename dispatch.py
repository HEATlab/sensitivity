
from stn import STN, loadSTNfromJSONfile
from util import STNtoDCSTN, PriorityQueue
from dc_stn import DC_STN
from relax import relaxSearch
import empirical
import random
import json
import string

# For faster checking in safely_scheduled
import simulation as sim

##
# \file dispatch.py
# \brief Hosts method to dispatch STNUs that were modified by the
#        old dynamic checking algorithm
# \note More detailed explanation about the dispatch algorithm can be found in:
#       https://pdfs.semanticscholar.org/0313/af826f45d090a63fd5d787c92321666115c8.pd

ZERO_ID = 0


##
# \fn simulate_and_save(file_names, size, out_name)
# \brief Keep track of dispatch results on networks
def simulate_and_save(file_names: list, size: int, out_name: str):
    rates = {}
    # Loop through files and record the dispatch success rates and
    # approximated probabilities
    for name in file_names:
        success_rate = simulate_file(name, size)
        rates[name] = success_rate

    # Save the results
    with open(out_name, 'w') as out_json:
        out_json.dump(rates)
    print("Results saved to", out_name)


##
# \fn simulate_file(file_name, size)
# \brief Record dispatch result for single file
def simulate_file(file_name, size, verbose=False, gauss=False, relaxed=False) -> float:
    network = loadSTNfromJSONfile(file_name)
    result = simulation(network, size, verbose, gauss, relaxed)
    if verbose:
        print(f"{file_name} worked {100*result}% of the time.")
    return result


##
# \fn simulation(network, size)
# @param size               The number of time the simulation runs?
def simulation(network: STN, size: int, verbose=False, gauss=False, relaxed=False) -> float:
    # Collect useful data from the original network
    contingent_pairs = network.contingentEdges.keys()
    contingents = {src: sink for (src, sink) in contingent_pairs}
    # this prints a set of uncontrollable events 
    uncontrollables = set(contingents.values())
    # print("uncontrollables: ", uncontrollables)
    uncontrolled_size = len(uncontrollables)
    
    # creating a dictionary to store all datapoints for each contingent data point - relative to last contingent timepoint
    dict_of_list = {}

    # creating a dictionary to store all datapoints for each contingent data point - relative to the zero timepoint
    dict_of_list_zero = {}


    # create uncontrolled_size amount of new lists in the dict
    for events in uncontrollables:
        dict_of_list[events] = []
        dict_of_list_zero[events] = []

    if relaxed:
        dispatching_network, count, cycles, weights = relaxSearch(getMinLossBounds(network.copy(), 2))
        if dispatching_network == None:
            dispatching_network = network
    else:
        dispatching_network = network

    total_victories = 0
    dc_network = STNtoDCSTN(dispatching_network)
    dc_network.addVertex(ZERO_ID)

    controllability = dc_network.is_DC()
    if verbose:
        print("Finished checking DC...")

    # Detect if the network has an inconsistency in a fixed edge
    verts = dc_network.verts.keys()
    for vert in verts:
        if (vert, vert) in dc_network.edges:
            if verbose:
                print("Checking", vert)
            edge = dc_network.edges[vert, vert][0]
            if edge.weight < 0:
                dc_network.edges[(vert, vert)].remove(edge)
                dc_network.verts[vert].outgoing_normal.remove(edge)
                dc_network.verts[vert].incoming_normal.remove(edge)
                del dc_network.normal_edges[(vert, vert)]

    # Run the simulation
    for j in range(size):
        realization = generate_realization(network, gauss)
        copy = dc_network.copy()
        result, final_schedule = dispatch(dispatching_network, copy, realization, contingents,
                          uncontrollables, verbose = False)

        # make a list of controllable events in the ordering of the final_schudule
        event_order = []
        for events in list(final_schedule.keys()):
            if events in uncontrollables:
                event_order.append(events)

        # intializing this as 0 for the first time point to compare to 
        last_event_time = 0 
        for events in event_order:
            dict_of_list_zero[events].append(round(final_schedule[events]/1000,4))
            dict_of_list[events].append(round((final_schedule[events]-last_event_time)/1000,4)) 
            last_event_time = final_schedule[events]
        if verbose:
            print("Completed a simulation.")
        if result:
            total_victories += 1

    goodie = float(total_victories / size)
    if verbose:
        print(f"Worked {100*goodie}% of the time.")

    return goodie, dict_of_list, dict_of_list_zero, event_order

##
# \fn getMinLossBounds(network, numSig)
# \brief Create copy of network with bounds related to spread
def getMinLossBounds(network: STN, numSig):
    for nodes, edge in network.edges.items():
        if edge.type == 'pstc' and edge.dtype == 'gaussian':
            sigma = edge.sigma
            mu = edge.mu
            edge.Cij = mu + numSig * sigma
            edge.Cji = -(mu - numSig * sigma)
    return network


##
# \fn dispatch(network, dc_network, realization, contingent_map,
#           uncontrollable_events, verbose)
# \brief Run an early-first scheduling algorithm on a network
#
# @param network                The original STNU we are scheduling on
# @param dc_network             The modified STNU with inferred constraints
# @param realization            An assignment of values for contingent edges
# @param contingent_map         A dictionary for contingent edges
# @param uncontrollable_events  A collection of uncontrollables
# @param verbose                Prints extra statements when set to True
#
# @post A flag which is True precisely when dispatch is succeeds
def dispatch(network: STN,
             dc_network: DC_STN,
             realization: dict,
             contingent_map: dict,
             uncontrollable_events,
             verbose=False) -> bool:

    # Dispatch the modified network and assume we have a zero reference point
    enabled = {ZERO_ID}
    not_executed = set(dc_network.verts.keys())
    executed = set()
    current_time = 0.0

    schedule = {}

    time_windows = {event: [0, float('inf')] for event in not_executed}
    current_event = ZERO_ID
    if verbose:
        print("Beginning dispatch...")
    while len(not_executed) > 0:
        # Find next event to execute
        min_time = float('inf')

        if verbose:
            print("\n\nNetwork looks like: ")
            print(dc_network)

            # print("Realization (random pick values for contingent edges): ", realization)
            print("Current time windows: ", time_windows)
            print("Currently enabled: ", enabled)
            print("Already executed: ", executed)
            print("Still needs to be executed: ", not_executed)

        # Pick an event to schedule
        for event in enabled:
            if verbose:
                print("checking enabled event", event)
            lower_bound = time_windows[event][0]
            if event in uncontrollable_events:
                if lower_bound < min_time:
                    min_time = lower_bound
                    current_event = event
            else:
                # Check that the wait constraints on the event are satisfied
                waits = dc_network.verts[event].outgoing_upper
                lower_bound = time_windows[event][0]

                for edge in waits:
                    if edge.parent != event:
                        if (edge.parent not in executed):
                            if edge.j not in executed:
                                continue
                            lower_bound = max(lower_bound,
                                              schedule[edge.j] - edge.weight)

                if lower_bound < min_time:
                    min_time = lower_bound
                    current_event = event

        is_uncontrollable = current_event in uncontrollable_events

        if verbose:
            if is_uncontrollable:
                print("This event is uncontrollable!")
        current_time = min_time
        schedule[current_event] = current_time
        if verbose:
            print('event', current_event,'is scheduled at', current_time)


        # If the executed event was a contingent source
        if current_event in contingent_map:
            uncontrollable = contingent_map[current_event]
            delay = realization[uncontrollable]
            set_time = current_time + delay
            enabled.add(uncontrollable)
            time_windows[uncontrollable] = [set_time, set_time]

        if is_uncontrollable:
            # Remove waits
            original_edges = list(dc_network.upper_case_edges.items())
            for nodes, edge in original_edges:
                if edge.parent == current_event:
                    if (current_event != edge.i) and (current_event != edge.j):
                        # Modifying the network
                        dc_network.remove_upper_edge(edge.i, edge.j)
        if current_event in not_executed:
            not_executed.remove(current_event)
        else:
            return False
        if current_event in enabled:
            enabled.remove(current_event)
        executed.add(current_event)

        # Propagate the constraints
        for nodes, edge in dc_network.normal_edges.items():
            if edge.i == current_event:
                new_upper_bound = edge.weight + current_time
                if new_upper_bound < time_windows[edge.j][1]:
                    time_windows[edge.j][1] = new_upper_bound
            if edge.j == current_event:
                new_lower_bound = current_time - edge.weight
                if new_lower_bound > time_windows[edge.i][0]:
                    time_windows[edge.i][0] = new_lower_bound

        # Add newly enabled events
        for event in not_executed:
            if verbose:
                print("***")
                print("Checking event", event)
            if (event not in enabled) and (event not in uncontrollable_events):

                ready = True
                outgoing_reqs = dc_network.verts[event].outgoing_normal
                # Check required constraints
                for edge in outgoing_reqs:
                    # For required
                    if edge.weight < 0:
                        if edge.j not in executed:
                            if verbose:
                                print(event, "was not enabled because of",
                                      edge)
                            ready = False
                            break
                    elif edge.weight == 0:
                        if (edge.j, edge.i) in dc_network.edges:
                            if dc_network.edges[(edge.j, edge.i)][0].weight != 0:
                                if edge.j not in executed:
                                    ready = False
                                    break
                        else:
                            if edge.j not in executed:
                                ready = False
                                break

                # Check wait constraints
                outgoing_upper = dc_network.verts[event].outgoing_upper
                for edge in outgoing_upper: 
                    if edge.weight < 0:
                        label_wait = (edge.parent not in executed)
                        main_wait = (edge.j not in executed)
                        if label_wait and main_wait:
                            ready = False
                            if verbose:
                                print(event, "was not enabled because of",
                                      edge)
                            break

                if ready:
                    if verbose:
                        print("Looks like we enabled", event)
                    enabled.add(event)

    # The realization should be preserved for src, sink in contingent_map
    if verbose:
        print("\n\nFinal schedule is: ")
        print(schedule)
        # print("Network is: ")
        # print(network)
        # print("uncontrollable is: ")
        # print(uncontrollable_events)

    good = empirical.scheduleIsValid(network, schedule)
    if verbose:
        msg = "We're safe!" if good else "We failed!"
        print(msg)
        # print(good) -> this prints T/False
    return good, schedule


##
# \fn generate_realization(network)
# \brief Uniformly at random pick values for contingent edges in STNU
def generate_realization(network: STN, gauss=True) -> dict:
    realization = {}
    for nodes, edge in network.contingentEdges.items():
        assert edge.dtype != None

        if edge.dtype() == "gaussian":
            generated = random.gauss(edge.mu, edge.sigma)
            while generated < min(-edge.Cji, edge.Cij) or generated > max(-edge.Cji, edge.Cij):
                generated = random.gauss(edge.mu, edge.sigma)
            realization[nodes[1]] = generated
        elif edge.dtype() == "uniform":
            generated = random.uniform(edge.dist_lb, edge.dist_ub)
            counter = 0
            while generated < min(-edge.Cji, edge.Cij) or generated > max(-edge.Cji, edge.Cij):
                generated = random.uniform(edge.dist_lb, edge.dist_ub)
                counter += 1
                if counter > 1000:
                    print('over 1000 regenerations')

            realization[nodes[1]] = generated
    return realization