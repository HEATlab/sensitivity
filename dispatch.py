from stn import STN, loadSTNfromJSONfile
from stn.stn import Vertex
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
normal169 = glob.glob(os.path.join('dataset/PSTNS_normal','*.json'))
gamma169 = glob.glob(os.path.join('dataset/PSTNS_gamma','*.json'))
normaDream = glob.glob(os.path.join('dataset/dream_normal','*.json'))
gammaDream = glob.glob(os.path.join('dataset/dream_gamma','*.json'))


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


def simulate_and_save_files(file_path, size:int, out_name:str, strategy=None, relaxed=False, allow=False):
    file_names = glob.glob(os.path.join(file_path, '*.json'))
    file_names = sorted(file_names, key=lambda s: int(re.search(r'\d+', s).group()))
    results = simulate_and_save(file_names, size, out_name, strategy, relaxed, allow)
    return results

##
# \fn simulate_and_save(file_names, size, out_name)
# \brief Keep track of dispatch results on networks
def simulate_and_save(file_names: list, size: int, out_name: str, strategy=None, relaxed=True, allow=True):
    rates = {}
    # Loop through files and record the dispatch success rates and
    # approximated probabilities
    for i in range(len(file_names)):
        success_rate = simulate_file(file_names[i], size, strategy=strategy, relaxed=relaxed,  allow=allow)
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
def simulate_file(file_name, size, strategy=None, verbose=False, relaxed=False, risk=0.05, allow=True) -> float:
    network = loadSTNfromJSONfile(file_name)
    goodie = simulation(network, size, strategy, verbose, relaxed, risk, allow)[0]
    if verbose:
        print(f"{file_name} worked {100*goodie}% of the time.")
    return goodie

##
# \fn simulation(network, size)
def simulation(simulationNetwork: STN, size: int, strategy=None, verbose=False, relaxed=False, risk=0.01, allow=True) -> float:
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

    gammaDict = {}

    # create uncontrolled_size amount of new lists in the dict
    for events in uncontrollables:
        dict_of_list[events] = []
        dict_of_list_zero[events] = []

    guessNetwork = simulationNetwork.copy()

    if relaxed:
        # dispatching_network, count, cycles, weights = relaxSearch(getMinLossBounds(network.copy(), risk))
        dispatching_network = relaxSearch(getMinLossBounds(guessNetwork, risk, gammaDict, strategy))[0]

        # if there is no way to relax, we let the original network to be the dispactching network
        if dispatching_network == None:
            dispatching_network = guessNetwork
    else:
        dispatching_network = guessNetwork

    total_victories = 0
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

    # a list of all final schedules keys 
    final_schedule_combo = []
    final_cont_schedule = []

    # create the matrix that store the pairwise relationship between vertices
    num = len(simulationNetwork.verts) if 0 in simulationNetwork.verts else len(simulationNetwork.verts) + 1
    
    ####### start a matrix to keep track of temporal difference btn events
    # matrix = {}
    # for i in range(num):
    #     row = {}
    #     for j in range(num):
    #         block = Pair_info(0,0,size,[],"U")
    #         row['vertex_'+str(j)] = block
    #     matrix['vertex_'+str(i)] = row

    # Run the simulation, each j is one simulation
    for j in range(size):
        # generate realization for each iteration
        realization = generate_realization(simulationNetwork, allow)
        # print("simulated realization is,",realization)
        
        copy = dc_network.copy()

        # we feed the dispatching strategies and realizations into the process 
        x = dispatch(dispatching_network, copy, realization, contingents,
                          uncontrollables, False)
        # if the schedule successes, we record the data 
        if x != False:
            result, final_schedule = x
        else:
            return 0.0, [], [], []
        final_keys = list(final_schedule.keys())
        final_schedule_combo.append(final_keys)

        # for i in range(len(final_keys)-1):
        #     time_difference = final_schedule[final_keys[i+1]]-final_schedule[final_keys[i]]
        #     matrix["vertex_"+str(final_keys[i])]["vertex_"+str(final_keys[i+1])].update(time_difference)

        # make a list of controllable events in the ordering of the final_schudule
        event_order = []
        for events in final_keys:
            if events in uncontrollables:
                event_order.append(events)
        final_cont_schedule.append(event_order)

        # record data point
        for i in range(len(final_keys)):
            if final_keys[i] in event_order:
                dict_of_list_zero[final_keys[i]].append(round(final_schedule[final_keys[i]]/1000,4))
                if i == 0:
                    dict_of_list[final_keys[i]].append(round(final_schedule[final_keys[i]]/1000,4))
                else:
                    dict_of_list[final_keys[i]].append(round((final_schedule[final_keys[i]]-final_schedule[final_keys[i-1]])/1000,4))

        if verbose:
            print("Completed a simulation.")
        if result:
            total_victories += 1

    # clean up the matrix
    # for i in range(num):
    #     for j in range(num):
    #         time = matrix['vertex_'+str(i)]['vertex_'+str(j)].time_differences
    #         if all([t==0 for t in time]):
    #             matrix['vertex_'+str(i)]['vertex_'+str(j)].time_differences = ["all zero"]
    #         if matrix['vertex_'+str(i)]['vertex_'+str(j)].count == 0:
    #             del matrix['vertex_'+str(i)]['vertex_'+str(j)]

    goodie = float(total_victories / size)
    if verbose:
        print(f"Worked {100*goodie}% of the time.")

    return goodie, dict_of_list, dict_of_list_zero, event_order, final_schedule,[]

##
# \fn getMinLossBounds(network, numSig)
# \brief Create copy of network with bounds related to spread
def getMinLossBounds(network: STN, risk, gammaDict={}, strategy=None):
    numSig = norm.ppf(1-risk/2)
    
    uncut = list(network.edges.values())

    while len(uncut) != 0:
        edge = uncut[0]
        # cut for normal distribution
        if edge.type == 'Empirical' and edge.dtype() == 'gaussian':
            sigma = edge.sigma
            mu = edge.mu
            if not strategy:
                edge.Cij = min(mu + numSig * sigma, edge.Cij)
                edge.Cji = min(-(mu - numSig * sigma), edge.Cji)
            elif strategy == 'gamma':
                # cut tail 
                upper = norm.ppf(q = 1-risk, loc=mu, scale=sigma) 
                edge.Cij = min(upper, edge.Cij)
                # edge.Cji = min(-(lower), edge.Cji)
        # cut for gamma distribution
        elif edge.type == 'Empirical' and edge.dtype() == 'gamma':
            alpha = edge.alpha
            beta = edge.beta
            alphaKey = float(f'{alpha:.2f}')
            betaKey = float(f'{beta:.2f}')
            loc = float(f'{edge.loc:.1f}')

            if not strategy:
                # cut equal amount of probability on both ends
                if (alphaKey, betaKey) in gammaDict:
                    lower, upper, locPrev = gammaDict[(alphaKey, betaKey)]
                    lower = lower - locPrev + loc*1000
                    upper = upper - locPrev + loc*1000
                else:
                    lower, upper = minLossGamma(alpha, beta, loc, risk, res = 0.1)
                    gammaDict[(alphaKey, betaKey)] = (lower, upper, loc*1000)
            elif strategy == 'minCut':
                # cut in the middle and add one more edge

                # if (alphaKey, betaKey) in gammaDict:
                #     lower, upper, locPrev = gammaDict[(alphaKey, betaKey)]
                #     lower = lower - locPrev  + loc*1000
                #     upper = upper - locPrev + loc*1000
                # else:
                lp = gamma.cdf(x=-edge.Cji/1000-loc, a= alpha, scale = 1/beta)
                up = 1-gamma.cdf(x=edge.Cij/1000-loc, a= alpha, scale = 1/beta)
                if lp+up <risk:   
                    lower, upper = minCutGamma(alpha, beta, loc, risk-lp-up, res = 0.1)
                    # gammaDict[(alphaKey, betaKey)] = (lower, upper, loc*1000)
                else:
                    uncut.remove(edge)
                    continue
                    
                tempL = len(network.verts)
                tempU = tempL+1
                network.addVertex(tempL)
                network.addVertex(tempU)
                # network.addEdge(tempL, tempU, 0, upper-lower, type='stc')
                network.addEdge(edge.i,tempL, -edge.Cji, lower, type='Empirical')
                network.removeEdge(edge.i, edge.j)
                if (edge.Cij-upper) > 0:
                    network.addEdge(tempU, edge.j, 0, edge.Cij-upper, type='Empirical')
            elif strategy == 'cutHead':
                # cut the head for dist with alpha less than 1, and cut same for dist with alpha greater than 1
                if alpha <= 1:
                    upper = (gamma.ppf(q=0.9999, a= alpha, scale=1/beta) +loc)*1000
                    lower = (gamma.ppf(q=risk, a= alpha, scale=1/beta) +loc)*1000
                else:
                    lower = (gamma.ppf(q=risk/2, a=alpha, scale=1/beta)+loc)*1000
                    upper = (gamma.ppf(q=1-risk/2, a=alpha, scale=1/beta)+loc)*1000
            elif strategy == 'normal':
                # cut the same amount of probabilities
                lower = (gamma.ppf(q=risk*0.5, a=alpha, scale=1/beta)+loc)*1000
                upper = (gamma.ppf(q=1-risk*0.5, a=alpha, scale=1/beta)+loc)*1000
            
            edge.Cij = min(upper, edge.Cij)
            edge.Cji = min(-(lower), edge.Cji)
            
        elif edge.isContingent():
            print("here", edge.type, edge.dtype())
            sigma = (edge.Cij + edge.Cji)/4
            mu = (edge.Cij - edge.Cji)/2
            print(mu, " is mu and ", sigma, " is sigma")
            edge.Cij = min(mu + numSig * sigma, edge.Cij)
            edge.Cji = min(-(mu - numSig * sigma), edge.Cji)
        uncut.remove(edge)
    return network

def recursiveGamma(upper, lower, risk, weights, sum, res = 0.1):
    upper = float(f'{upper:.1f}')
    lower = float(f'{lower:.1f}')
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

# cut the probability with max uncertainties from tails
def bestGamma(risk, weights, units, res, mode, loc):
    dp = [0 for i in range(units)]
    key = 0.0
    sumIndex = int(mode/res)
    lower, upper = (sumIndex-1, sumIndex+1) if mode != 0.0 else (0, 1)
    for i in range(units):
        key = float(f'{key:.1f}')
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

# cut the probability with least uncertainties from the mode
def minGamma(risk, weights, units, res, mode, loc):
    dp = [0 for i in range(units)]
    key = 0.0
    sumIndex = int(mode/res)
    lower, upper = (sumIndex-1, sumIndex+1) if mode != 0.0 else (0, 1)
    for i in range(units):
        key = float(f'{key:.1f}')
        dp[i] = weights[key]
        key += res
    sum = dp[sumIndex]
    for i in range(units):
        if lower == 0 or lower == loc*1000:
            return 0, 0
        elif dp[upper] >= dp[lower]:
            sum += dp[upper]
            if sum >= risk:
                return 1000*lower*res, 1000*upper*res
            upper += 1
        else:
            sum += dp[lower]
            if sum >= risk:
                return 1000*lower*res, 1000*upper*res
            lower -=1

# lose the least amount of uncertainties
def minCutGamma(alpha, beta, loc, risk, res=.1):
    mode = float(f'{loc+1/beta*(alpha-1):.1f}') if alpha >1 else 0.00 # zero frame mode
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
        weights[float(f'{x:.1f}')] = X[i]
        x += res
    lower, upper = minGamma(risk, weights, units, res, mode, loc)
    if lower == upper == 0:
        return 1000*loc, 1000*(gamma.ppf(q=1-risk, a=alpha, scale = theta) + loc)
    return lower, upper

# lose the max amount of uncertainties
def minLossGamma(alpha, beta, loc, risk, res=.1):
    mode = float(f'{loc+1/beta*(alpha-1):.1f}') if alpha >1 else 0.00 # zero frame mode
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
        weights[float(f'{x:.1f}')] = X[i]
        x += res
    lower, upper = bestGamma(risk, weights, units, res, mode, loc)
    if lower == upper == 0:
        return 1000*loc, 1000*(gamma.ppf(q=1-risk, a=alpha, scale = theta) + loc)
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
            uncontrollable = contingent_map[current_event]
            delay = realization[uncontrollable] # the length of the contingent edge
            set_time = current_time + delay
            enabled.add(uncontrollable)
            time_windows[uncontrollable] = [set_time, set_time] #update the time windows for the contingent sink

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

    for event in contingent_map:
        sink = contingent_map[event]
        schedule[sink] = schedule[event]+realization[sink]
    
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
def generate_realization(network: STN, allow=True) -> dict:
    realization = {}

    for nodes, edge in network.contingentEdges.items():
        assert edge.dtype != None
        if edge.dtype() == "gaussian":
            generated = random.gauss(edge.mu, edge.sigma)
            if not allow:
                while generated < min(-edge.Cji, edge.Cij) or generated > max(-edge.Cji, edge.Cij):
                    # print(generated, -edge.Cji, edge.Cij)
                    generated = random.gauss(edge.mu, edge.sigma)
            realization[nodes[1]] = generated
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
        
    return realization


if __name__ == '__main__':
    data = 'mr_x.json'
    data2 = 'mr_x2.json'
    stn = loadSTNfromJSONfile(data)
    stn2 = loadSTNfromJSONfile(data2)
    stn3 = loadSTNfromJSONfile('mr_x3.json')
    test = loadSTNfromJSONfile('dataset/PSTNS_gamma/test1.json')