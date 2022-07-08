from typing import final
from stn import STN, gammaempirical, loadSTNfromJSONfile
from util import STNtoDCSTN, PriorityQueue
from dc_stn import DC_STN
from relax import relaxSearch
import empirical
import random
import json
import glob
import os
import re
import numpy as np
from os import truncate
from scipy.stats import norm
from scipy.stats import gamma


# For faster checking in safely_scheduled
import simulation as sim

##
# \file dispatch.py
# \brief Hosts method to dispatch STNUs that were modified by the
#        old dynamic checking algorithm
# \note More detailed explanation about the dispatch algorithm can be found in:
#       https://pdfs.semanticscholar.org/0313/af826f45d090a63fd5d787c92321666115c8.pd

ZERO_ID = 0


def simulate_and_save_files(file_path, size:int, out_name:str, compare_path="", relaxed=False, dist=False, allow=False):
    file_names = glob.glob(os.path.join(file_path, '*.json'))
    file_names = sorted(file_names, key=lambda s: int(re.search(r'\d+', s).group()))
    if compare_path != "":
        compare_files = glob.glob(os.path.join(compare_path, '*.json'))
        compare_files = sorted(compare_files, key=lambda s: int(re.search(r'\d+', s).group()))
    else:
        compare_files = []
    results = simulate_and_save(file_names, size, out_name, compare_files, relaxed=relaxed, dist=dist)
    return results
## \class Pair_info
#  \brief Represents the relationship
class Pair_info(object):

    ## \brief Pair_info Constructor
    #  \param count              The number of times the row index has happened immediately after the col index
    #  \param frequency          count/total number of simulation
    #  \param total_run          total number of times the simulation run
    #  \param time_differences   a list of time differences between the two events
    #  \param distribution       guessing a type of distribution to be fitted; U for unknown, N for normal, G for gemma
    def __init__(self, count, frequency, total_run, time_differences, distribution):

        self.count = count
        self.frequency = frequency
        self.total_run = total_run
        self.time_differences = time_differences
        self.distribution = distribution

    ## \brief update the pair_info 
    #  \param time_difference    the time difference between the two vertices for a particular simulation
    def update(self, time_difference):
        self.count += 1
        self.frequency = self.count/self.total_run
        self.time_differences.append(time_difference)
        # add: distribution fitter
        # self.distribution = fit()

    def __repr__(self):
        return f"count: {self.count}, frequency: {self.frequency}, time difference: {self.time_differences}, distribution: {self.distribution}"

##
# \fn simulate_and_save(file_names, size, out_name)
# \brief Keep track of dispatch results on networks
def simulate_and_save(file_names: list, size: int, out_name: str, compare_files=[], relaxed=False, dist = True, allow=False):
    rates = {}
    # Loop through files and record the dispatch success rates and
    # approximated probabilities
    for i in range(len(file_names)):
        if len(compare_files) != 0:
            compare = compare_files[i]
        else: 
            compare = False
        success_rate = simulate_file(file_names[i], size, compare, relaxed=relaxed, gauss=dist, allow=allow)
        rates[file_names[i]] = success_rate
        print(file_names[i], success_rate)

    # Save the results
    with open(out_name, 'w') as out_json:
        json.dump(rates, out_json)
    print("Results saved to", out_name) 
    return rates

##
# \fn simulate_file(file_name, size)
# \brief Record dispatch result for single file
def simulate_file(file_name, size, compare=False, verbose=False, gauss=True, relaxed=False, risk=0.05, allow=False) -> float:
    network = loadSTNfromJSONfile(file_name)
    compareNetwork = loadSTNfromJSONfile(compare) if compare else None
    goodie, dict_of_list, dict_of_list_zero, event_order = simulation(network, size, compareNetwork, verbose, gauss, relaxed, risk, allow)
    if verbose:
        print(f"{file_name} worked {100*goodie}% of the time.")
    return goodie

##
# \fn simulation(network, size)
def simulation(simulationNetwork: STN, size: int, strategyNetwork=None, verbose=False, dist=True, relaxed=False, risk=0.05, allow=False) -> float:
    # Collect useful data from the original network
    contingent_pairs = simulationNetwork.contingentEdges.keys()
    contingents = {src: sink for (src, sink) in contingent_pairs}
    # this prints a set of uncontrollable events 
    uncontrollables = set(contingents.values())

    # print("uncontrollables: ", uncontrollables)
    uncontrolled_size = len(uncontrollables)
    
    # creating a dictionary to store all datapoints for each contingent data point
    # relative to last time point, whether contingent or not
    dict_of_list = {}

    # creating a dictionary to store all datapoints for each contingent data point - relative to the zero timepoint
    dict_of_list_zero = {}

    # create uncontrolled_size amount of new lists in the dict
    for events in uncontrollables:
        dict_of_list[events] = []
        dict_of_list_zero[events] = []

    if strategyNetwork:
        guessNetwork = strategyNetwork
    else:
        guessNetwork = simulationNetwork.copy()
    
    # initial guess (gaussian distribution based on min and max) of the distribution between contingent pairs
    for nodes, edge in guessNetwork.edges.items():
        if edge.type == 'Empirical':
            mu = (edge.Cij + edge.Cji)/2 - edge.Cji
            sigma = (edge.Cij + edge.Cji)/10
            setattr(edge, 'distribution', 'N_'+str(mu/1000)+'_'+str(sigma/1000))

    # initial dispatching_network is determined based on using cut methods or not
    if relaxed:
        # dispatching_network, count, cycles, weights = relaxSearch(getMinLossBounds(network.copy(), risk))
        dispatching_network = relaxSearch(getMinLossBounds(guessNetwork, risk))[0]

        if dispatching_network == None:
            dispatching_network = guessNetwork
    else:
        dispatching_network = guessNetwork
        
    total_victories = 0

    # turn the dispatch network into a dc_network, check for its dc_controllability, and find inconsistency
    dc_network = dcInconsistency(dispatching_network)

    # a list of all final schedules keys 
    final_schedule_combo = []
    final_cont_schedule = []
    contingent_time_difference = {}


    # create the matrix that store the pairwise relationship between vertices
    num = len(simulationNetwork.verts) if 0 in simulationNetwork.verts else len(simulationNetwork.verts) + 1
    matrix = {}

    for i in range(num):
        row = {}
        for j in range(num):
            block = Pair_info(0,0,size,[],"U")
            row['vertex_'+str(j)] = block
        matrix['vertex_'+str(i)] = row
    
    # Run the simulation, each j is one simulation
    for j in range(size):
        realization = generate_realization(simulationNetwork, dist, allow)
        dc_network = dcInconsistency(dispatching_network)
        copy = dc_network.copy()

        print(realization)
        x = dispatch(simulationNetwork, copy, realization, contingents,
                          uncontrollables, False)
        if x != False:
            result, final_schedule = x
        else:
            return 0.0, [], [], []

        final_keys = list(final_schedule.keys())
        final_schedule_combo.append(final_keys)

        # get the time difference between each pair of events happened adjacently
        for i in range(len(final_keys)-1):
            time_difference = final_schedule[final_keys[i+1]]-final_schedule[final_keys[i]]
            matrix["vertex_"+str(final_keys[i])]["vertex_"+str(final_keys[i+1])].update(time_difference)


        # make a list of uncontrollable events in the ordering of the final_schudule
        event_order = []
        for events in final_keys:
            if events in uncontrollables:
                event_order.append(events)

        final_cont_schedule.append(event_order)

        # record time differences between a contingent event and the previous event for all contingent vertices
        for i in range(len(final_keys)):
            if final_keys[i] in event_order:
                dict_of_list_zero[final_keys[i]].append(round(final_schedule[final_keys[i]]/1000,4))
                if i == 0:
                    dict_of_list[final_keys[i]].append(round(final_schedule[final_keys[i]]/1000,4))
                else:
                    dict_of_list[final_keys[i]].append(round(final_schedule[final_keys[i]]-final_schedule[final_keys[i-1]]/1000,4))

        for src in contingents:
            sink = contingents[src]
            if (src, sink) in contingent_time_difference.keys():
                contingent_time_difference[(src, sink)].append(final_schedule[sink] - final_schedule[src])
            else:
                contingent_time_difference[(src, sink)]=[final_schedule[sink] - final_schedule[src]]
        print(contingent_time_difference)

        if verbose:
            print("Completed a simulation.")
        if result:
            total_victories += 1


        orderChanged = False
        if j >= 20 and j%10 == 0:
            dispatching_network, orderChanged = updateDispatch(dispatching_network, contingent_time_difference)
        if orderChanged:
            # update the contingent map and pairs
            contingent_pairs = dispatching_network.contingentEdges.keys()
            contingents = {src: sink for (src, sink) in contingent_pairs}
            uncontrollables = set(contingents.values())
            recalculatePastData(dispatching_network)


    # clean up the matrix
    for i in range(num):
        for j in range(num):
            time = matrix['vertex_'+str(i)]['vertex_'+str(j)].time_differences
            if all([t==0 for t in time]):
                matrix['vertex_'+str(i)]['vertex_'+str(j)].time_differences = ["all zero"]
            if matrix['vertex_'+str(i)]['vertex_'+str(j)].count == 0:
                del matrix['vertex_'+str(i)]['vertex_'+str(j)]

    goodie = float(total_victories / size)
    if verbose:
        print(f"Worked {100*goodie}% of the time.")

    return goodie, dict_of_list, dict_of_list_zero, final_schedule, matrix

def updateDispatch(dispatching_network, time_difference):
    new_dispatch = dispatching_network.copy()
    for nodes, edge in new_dispatch.edges.items():
        src = edge.i
        sink = edge.j
        if edge.type == 'Empirical':
            data = time_difference[(src, sink)]
            fits = gammaempirical.fitdist([data], [len(data)], gammaFlag=False, plot=True, types=['norm', 'gamma'])
            distribution = list(fits.keys())[0]
            name, firstPar, secondPar, size, negative = distribution
            if name =='norm':
                setattr(edge, 'distribution', 'N_'+str(firstPar/1000)+'_'+str(secondPar/1000))
            elif name == 'gamma':
                beta, loc = secondPar
                setattr(edge, 'distribution', 'N_'+str(firstPar/1000)+'_('+str(beta)+', '+str(loc/1000)+')')

    orderChanged = False
    return new_dispatch, orderChanged


def recalculatePastData(dispatching_network):
    return dispatching_network


    return new_dispatch, orderChanged
def dcInconsistency(dispatching_network:STN):
    dc_network = STNtoDCSTN(dispatching_network)
    dc_network.addVertex(ZERO_ID)
    controllability = dc_network.is_DC()
    # if verbose:
    #     print("Finished checking DC...")

    # Detect if the network has an inconsistency in a fixed edge
    verts = dc_network.verts.keys()
    for vert in verts:
        if (vert, vert) in dc_network.edges:
            # if verbose:
            #     print("Checking", vert)
            edge = dc_network.edges[vert, vert][0]
            if edge.weight < 0:
                dc_network.edges[(vert, vert)].remove(edge)
                dc_network.verts[vert].outgoing_normal.remove(edge)
                dc_network.verts[vert].incoming_normal.remove(edge)
                del dc_network.normal_edges[(vert, vert)]
    return dc_network

##
# \fn getMinLossBounds(network, numSig)
# \brief Create copy of network with bounds related to spread
def getMinLossBounds(network: STN, risk):
    numSig = norm.ppf(1-risk/2)
    for nodes, edge in network.edges.items():
        if edge.type == 'Empirical' and edge.dtype() == 'gaussian':
            sigma = edge.sigma
            mu = edge.mu
            edge.Cij = min(mu + numSig * sigma, edge.Cij)
            edge.Cji = min(-(mu - numSig * sigma), edge.Cji)
        elif edge.type == 'Empirical' and edge.dtype() == 'gamma':
            alpha = edge.alpha
            beta = edge.beta
            loc = float(f'{edge.loc:.2f}')
            lower = gamma.ppf(risk/2, a=alpha, scale = 1/beta) +loc
            upper = gamma.ppf(1-risk/2, a=alpha, scale=1/beta) + loc
            # lower, upper = minLossGamma(alpha, beta, loc, risk, res = 0.01)
            edge.Cij = min(1000*upper, edge.Cij)
            edge.Cji = min(-(1000*lower), edge.Cji)
        elif edge.isContingent():
            print("here", edge.type, edge.dtype())
            sigma = (edge.Cij + edge.Cji)/4
            mu = (edge.Cij - edge.Cji)/2
            print(mu, " is mu and ", sigma, " is sigma")
            edge.Cij = min(mu + numSig * sigma, edge.Cij)
            edge.Cji = min(-(mu - numSig * sigma), edge.Cji)
    return network

def recursiveGamma(upper, lower, risk, weights, sum, res = 0.01):
    upper = float(f'{upper:.2f}')
    lower = float(f'{lower:.2f}')

    if sum < 1-risk:
        if lower>0:
            if weights[upper] >= weights[lower]:
                sum += weights[upper]
                upper += res
                return recursiveGamma(upper, lower, risk, weights, sum)
            else:
                sum += weights[lower]
                lower -= res
                return recursiveGamma(upper, lower, risk, weights, sum)
        else:
            sum += weights[upper]
            return recursiveGamma(upper + 0.02, 0, risk, weights, sum)
    else:
        return (lower, upper)

def bestGamma(risk, weights, units, res, mode, loc):
    dp = [0 for i in range(units)]
    key = 0.00
    sumIndex = int(mode/res)
    lower, upper = (sumIndex-1, sumIndex+1) if mode != 0.0 else (0, 1)
    for i in range(units):
        key = float(f'{key:.2f}')
        dp[i] = weights[key]
        key += res
    sum = dp[sumIndex]
    for i in range(units):
        if lower == 0 or lower == loc*1000:
            return 0, 0
        elif dp[upper] >= dp[lower]:
            sum += dp[upper]
            if sum >= 1-risk:
                return 1000*lower*res, 1000*upper*res
            upper += 1
        else:
            sum += dp[lower]
            if sum >= 1-risk:
                return 1000*lower*res, 1000*upper*res
            lower -=1

def minLossGamma(alpha, beta, loc, risk, res=.01):
    mode = float(f'{loc+1/beta*(alpha-1):.2f}') if alpha >1 else 0.00 # zero frame mode
    theta = 1/beta
    ub = gamma.ppf(q=0.99, a=alpha, scale = theta) + loc  # zero frame ub
    units = round(ub/res)+1
    X = []
    lb = 0
    for i in range(units):
        X.append(gamma.pdf(x=lb-loc, a=alpha, scale = theta))  # zero frame 
        lb += res
    # get the weights for all entrances
    X = [ w/sum(X) for w in X ]
    weights = {}
    x = 0
    for i in range(units):
        weights[float(f'{x:.2f}')] = X[i]
        x += res
    # lower, upper = recursiveGamma(mode+res, mode-res, risk, weights, weights[float(f'{mode:.2f}')], res = 0.01)
    lower, upper = bestGamma(risk, weights, units, res, mode, loc)
    if upper == 0:
        return 1000*loc, 1000*(gamma.ppf(q=1-risk, a=alpha, scale = theta) + loc)
    else:
        return lower, upper

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

            print("Realization (random pick values for contingent edges): ", realization)
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
            # print(current_event, "current_event is ")
            uncontrollable = contingent_map[current_event]
            delay = realization[uncontrollable] # the length of the contingent edge
            set_time = current_time + delay
            # print("current, delay", current_time, delay, set_time)
            enabled.add(uncontrollable)
            time_windows[uncontrollable] = [set_time, set_time] #update the time windows for the contingent sink
            # print(time_windows)
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
            print("huh")
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
        print("good, ", good)
        msg = "We're safe!" if good else "We failed!"
        print(msg)
        # print(good) -> this prints T/False
    return good, schedule


##
# \fn generate_realization(network)
# \brief Uniformly at random pick values for contingent edges in STNU
def generate_realization(network: STN, dist=False, allow=False) -> dict:
    realization = {}
    if dist:
        for nodes, edge in network.contingentEdges.items():
            assert edge.dtype != None
            if edge.dtype() == "gaussian":
                generated = random.gauss(edge.mu, edge.sigma)
                if not allow:
                    while generated < min(-edge.Cji, edge.Cij) or generated > max(-edge.Cji, edge.Cij):
                        # print(generated, -edge.Cji, edge.Cij)
                        generated = random.gauss(edge.mu, edge.sigma)
                realization[nodes[1]] = generated
                print("generated", generated)
            elif edge.dtype() == "uniform":
                generated = random.uniform(edge.dist_lb, edge.dist_ub)
                if not allow:
                    while generated < min(-edge.Cji, edge.Cij) or generated > max(-edge.Cji, edge.Cij):
                        print("oop")
                        generated = random.uniform(edge.dist_lb, edge.dist_ub)
                realization[nodes[1]] = generated
            elif edge.dtype() == "gamma":
                rand = np.random.gamma(edge.alpha, 1/edge.beta)
                generated = 1000*(edge.loc+rand)
                if not allow:
                    while generated < min(-edge.Cji, edge.Cij) or generated > max(-edge.Cji, edge.Cij):
                        generated = 1000*(edge.loc+np.random.gamma(edge.alpha, 1/edge.beta))
                realization[nodes[1]] = generated
    else:
        for nodes, edge in network.contingentEdges.items():
            mu = (edge.Cij - edge.Cji)/2
            sigma = (edge.Cij - mu)/2
            generated = random.gauss(mu, sigma)
            realization[nodes[1]] = generated
        
    return realization


if __name__ == '__main__':
    data = 'mr_x.json'
    data2 = 'mr_x2.json'
    stn = loadSTNfromJSONfile(data)
    stn2 = loadSTNfromJSONfile(data2)