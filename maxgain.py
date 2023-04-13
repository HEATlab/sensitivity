from yaml import load
from stn import STN, loadSTNfromJSONfile, invcdf_norm, invcdf_uniform
from util import STNtoDCSTN, PriorityQueue
from dc_stn import DC_STN
import empirical
import random
import glob
import json
from algorithm import *
from math import floor, ceil
from dispatch import *
from stn.distempirical import *
# import time
import timeout_decorator
import numpy
import re
import os
from scipy.stats import lognorm

def maxgain(inputstn,
            debug=False,
            returnAlpha=True,
            lb=0.0,
            ub=0.999, betaFlag = False):

    stncopy = inputstn.copy()

    # a,b,c,d = DC_Checker(stncopy)
    # if a:
    #     return stncopy
    # dictionary of alphas for binary search, a dictionary of {n:n/1000}

    alphas = {i: i / 1000.0 for i in range(1001)}

    # bounds for binary search
    lower = ceil(lb * 1000) - 1
    upper = floor(ub * 1000) + 1

    result = None

    # list of untightened edges
    tedges = stncopy.contingentEdges
    result = None

    # rounds equals the number of contingent edges
    rounds = 0

    while len(tedges) > 0:
        # print("tedges before everything starts",tedges, len(tedges))
        rounds += 1

        while upper - lower > 1:
            print("===================================================================")
            
            # line 7 of the algorithm 
            alpha = alphas[(upper + lower) // 2]

            if debug:
                print('trying alpha = {}'.format(alpha))
                
            stncopy = alphaUpdate(stncopy, tedges, alpha, betaFlag = betaFlag)
            #print(stncopy)

            if alpha == .999:
                print("Maxgain finished")
                return stncopy #stncopy    # stncopy maxgain inputstn maxgain+                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
                

            dc, conflicts, bounds, weight = DC_Checker(stncopy)
            # print("difference bigger than 1 bounds")
            # print(bounds)
            # print("conflicts",conflicts,len(conflicts))
            
            # line 9
            if dc:
                upper = (upper + lower) // 2
                print("dc, decrease upper")
                result = (alpha, stncopy)
            # line 11
            else:
                print("not dc, increase lower")
                lower = (upper + lower) // 2
            
            if debug:
                print("lower", lower/1000, "upper", upper/1000)

            # finished our search, load the smallest alpha decoupling
            # similar to line 3 of the algorithm 
            if upper - lower <= 1:
                print("the difference between lower and upper is smaller than 1")
                if result is not None:
                    if debug:
                        print('finished binary search, with lower', lower/1000,
                              'alpha', result[0], 'upper', upper/1000, 'rounds', rounds)
                    print(result[0],"final alpha")
                    stncopy = alphaUpdate(stncopy, tedges, result[0], betaFlag =betaFlag)
                    loweststn = alphaUpdate(stncopy, tedges, result[0]-.001, betaFlag =betaFlag)
                    dc, conflicts, bounds, weight = DC_Checker(loweststn)
                    print('difference smaller than 1 BOUNDS')
                    print(bounds)
                    if dc:
                        # how is it possible that the loweststn could still work??????????
                        return loweststn
                    else:
                        # find the tightest contingent edges
                        # tightest includes all the contingent edges after running dc_checker on the first stn that is not working 
                        # tedges includes all the contingent edges in the original stn 
                        tightest = bounds['contingent']
                        print("FINISHED")
                        #print(conflicts)

                        # shouldn't tightest and tedges always have the same length 
                        #print(len(tightest))
                        #print(len(tedges))
                        #print("tightest")
                        #print(tightest)
                        #print("tedges")
                        #print(tedges)
                        for i, j in list(tightest.keys()):
                            edge, bound = tightest[i, j]
                            #print('edge',edge)
                            #print('bound', bound)
                            if (edge.i, edge.j) in tedges.keys():
                                tedges.pop((edge.i, edge.j))
                            #print('tedges',tedges)

                else:
                    if debug:
                        print('could not produce dynamically controllable STNU.')
                    return None
        lower = ceil(lb * 1000) - 1
        upper = result[0]*1000
    print("RETURNING THE FINAL STN")
    # print(stncopy)
    return stncopy

# WHERE THE HDI WILL WORK

# EXTRACT-STNU-EDGE
def alphaUpdate(inputstn, tedges, alpha, betaFlag = False):
    stncopy = inputstn.copy()
    # update the edges based on the list of target edges
    if alpha <= 0:
        return stncopy
    else:
        for (i, j), edge in list(tedges.items()):
            if edge.type == "stcu":
                p_ij = invcdf_uniform(1.0 - alpha * 0.5, -edge.Cji, edge.Cij)
                p_ji = -invcdf_uniform(alpha * 0.5, -edge.Cji, edge.Cij)
                stncopy.modifyEdge(i, j, p_ij)
                stncopy.modifyEdge(j, i, p_ji)
            else:
                assert edge.dtype != None

                if edge.dtype() == "gaussian":
                    # print("gaussian")
                    p_ij = invcdf_norm(1.0 - alpha * 0.5, edge.mu, edge.sigma)
                    p_ji = -invcdf_norm(alpha * 0.5, edge.mu, edge.sigma)
                    stncopy.modifyEdge(i, j, p_ij)
                    stncopy.modifyEdge(j, i, p_ji)

                elif edge.dtype() == "uniform":
                    # print("uniform")
                    p_ij = invcdf_uniform(
                        1.0 - alpha * 0.5, edge.dist_lb, edge.dist_ub)
                    p_ji = -invcdf_uniform(alpha * 0.5,
                                           edge.dist_lb, edge.dist_ub)
                    stncopy.modifyEdge(i, j, p_ij)
                    stncopy.modifyEdge(j, i, p_ji)
                elif betaFlag == True and edge.dtype() == "gamma":
                    # print('max',edge.Cij)
                    # print('min',edge.Cji)
                    # p_ij = (edge.gammastart + invcdf_gamma(1.0 - alpha * 0.5, edge.alpha, edge.beta))*1000
                    # p_ji = (-edge.gammastart  -invcdf_gamma(alpha * 0.5, edge.alpha, edge.beta))*1000
                    # print('gamma p_ij',p_ij)
                    # print('gamma p_ji',p_ji)
                    #duration = edge.Cij + edge.Cji
                    p_ij = ( edge.Cij/1000- (edge.gammastart + invcdf_gamma(alpha * 0.5, edge.alpha, edge.beta) ) - edge.Cji/1000)*1000
                    p_ji = -(edge.Cij/1000 - (edge.gammastart + invcdf_gamma(1 - alpha * 0.5, edge.alpha, edge.beta)) - edge.Cji/1000)*1000
                    #p_ij = edge.Cji + invcdf_gamma(1.0 - alpha * 0.5, edge.alpha, edge.beta)
                    #p_ji = - (edge.Cji) -invcdf_gamma(alpha * 0.5, edge.alpha, edge.beta)
                    # print('alpha',edge.alpha)
                    # print('beta',edge.beta)
                    # print('beta p_ij',p_ij)
                    # print('beta p_ji',p_ji)
                    stncopy.modifyEdge(i, j, p_ij)
                    stncopy.modifyEdge(j, i, p_ji)
                elif edge.dtype() == "gamma":
                    print('max',edge.Cij)
                    print('min',edge.Cji)
                    #duration = edge.Cij + edge.Cji
                    p_ij = (edge.gammastart + invcdf_gamma(1.0 - alpha * 0.5, edge.alpha, edge.beta))*1000
                    p_ji = (-edge.gammastart  -invcdf_gamma(alpha * 0.5, edge.alpha, edge.beta))*1000
                    #p_ij = edge.Cji + invcdf_gamma(1.0 - alpha * 0.5, edge.alpha, edge.beta)
                    #p_ji = - (edge.Cji) -invcdf_gamma(alpha * 0.5, edge.alpha, edge.beta)
                    # print('alpha',edge.alpha)
                    # print('beta',edge.beta)
                    # print('gamma p_ij',p_ij)
                    # print('gamma p_ji',p_ji)
                    stncopy.modifyEdge(i, j, p_ij)
                    stncopy.modifyEdge(j, i, p_ji)

                elif edge.dtype() == "exponential":
                    p_ij = (np.log(alpha *0.5)/(-edge.expo_lambda) + edge.expo_start) * 1000
                    p_ji = -(np.log(1.0 - alpha * 0.5)/(-edge.expo_lambda) + edge.expo_start) * 1000
                    # print('expo_p_ij',p_ij)
                    # print('expo p_ji',p_ji)
                    stncopy.modifyEdge(i, j, p_ij)
                    stncopy.modifyEdge(j, i, p_ji)
                elif edge.dtype() == "lognormal":
                    # ppf(q, s, loc=0, scale=1)
                    p_ij = lognorm.ppf(1.0 - alpha * 0.5, s = edge.lognormal_sigma, loc = edge.lognormal_start, scale = math.exp(edge.lognormal_mu)) * 1000
                    p_ji = -lognorm.ppf(alpha * 0.5, s = edge.lognormal_sigma, loc = edge.lognormal_start, scale = math.exp(edge.lognormal_mu)) * 1000
                    # print('lognormal_p_ij',p_ij)
                    # print('lognormal p_ji',p_ji)
                    stncopy.modifyEdge(i, j, p_ij)
                    stncopy.modifyEdge(j, i, p_ji)
        return stncopy
                

                
        
"""  
for (i, j), edge in list(tedges.items()):
            if edge.type == "stcu":
                if bounds['contingent'][(i, j)][1] == 'UPPER':
                    p_ij = invcdf_uniform(1.0 - alpha, -edge.Cji, edge.Cij)
                    stncopy.modifyEdge(i, j, p_ij)
                else:
                    p_ji = -invcdf_uniform(alpha, -edge.Cji, edge.Cij)
                    stncopy.modifyEdge(j, i, p_ji)
for (i, j) in list(STN.contingentEdges.keys()):
    if j not in list(epsilons.keys()):
        continue
    edge = STN.contingentEdges[(i, j)]
    if bounds['contingent'][(i, j)][1] == 'UPPER':
        STN.modifyEdge(i, j, edge.Cij - epsilons[j])
    else:
        STN.modifyEdge(j, i, edge.Cji - epsilons[j])
"""


        
def simulate_maxgain(network, shrinked_network, size=200, verbose=False, gauss=True, betaFlag = False):
    # Collect useful data from the original network
    contingent_pairs = network.contingentEdges.keys()
    contingents = {src: sink for (src, sink) in contingent_pairs}
    uncontrollables = set(contingents.values())

    total_victories = 0
    dc_network = STNtoDCSTN(shrinked_network)
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
        realization = generate_realization(network, gauss, betaFlag = betaFlag)
        # print('REALIZATION',realization)
        copy = dc_network.copy()
        # schedule
        result, schedule= dispatch(network, copy, realization, contingents,
                          uncontrollables, verbose)
        if verbose:
            print("Completed a simulation.")

        
        relative_schedule = {}

        num_not_within_bound = 0 
        for i,j in list(shrinked_network.edges):
            relative_schedule[(i,j)] = schedule[j] - schedule[i]
            
            # within_bound = relative_schedule[(i,j)] <= shrinked_network.edges[(i,j)].Cij and relative_schedule[(i,j)] >= -shrinked_network.edges[(i,j)].Cji
            # if not within_bound:
            #     print("schedule not within bound")
            #     num_not_within_bound += 1 
            # if (i, j) in list(shrinked_network.contingentEdges.keys()) and realization[j] != relative_schedule[(i,j)] and realization[j] >= relative_schedule[(i,j)]:
            #     print(realization[j], relative_schedule[(i,j)], "THIS IS REALLY REALLY INTERSTING")
        if result:
            total_victories += 1
            # print("this is a success")
            # for i,j in list(relative_schedule.keys()):
            #     for node in list(realization.keys()):
            #         if j == node:
            #             if (realization[j] != relative_schedule[(i,j)]) and i != 0: 
            #                 print("this is really interesting")
            #                 print("realization",realization)
            #                 print("relative_schedule",relative_schedule)
            #                 print("schedule",schedule)
                            
        #     if num_not_within_bound != 0:
        #         print("THIS IS also  REALLY INTERSTING")
            
        #     print("WHY MAXGAIN succeeded")
        #     print("schedule:",schedule)
        #     print("relative schedule",relative_schedule)
        #     print("shrinked network",shrinked_network)
        #     print("realization",realization)
        
        # else:
        #     if num_not_within_bound == 0:
        #         print("FAILED THIS IS REALLY INTERSTING")
        #     print("WHY MAXGAIN FAILED")
        #     print("schedule:",schedule)
        #     print("relative schedule",relative_schedule)
        #     print("shrinked network",shrinked_network)
        #     print("realization",realization)
        print("===========================================================================================")
    goodie = float(total_victories / size)
    if verbose:
        print(f"Worked {100*goodie}% of the time.")

    return goodie

def get_valid_interval_stn(stn,valid_intervals):
    for (i,j) in (valid_intervals.keys()):
        if (i, j) in list(stn.contingentEdges.keys()):
            stn.modifyEdge(i, j, valid_intervals[(i,j)][1])
            stn.modifyEdge(j, i, -valid_intervals[(i,j)][0])
    return stn


if __name__ == "__main__":
    file_names = glob.glob(os.path.join("dataset/dream_exp", '*.json'))
    #file_names = sorted(file_names, key= lambda s: int(re.search(r'\d+',s).group()))
    print(len(file_names))
    data_list = [] 
    for i in range(1,170):
        #if i == 3 or i==13 or i == 34 or i ==45: 
            #continue
        data = f'dataset/PSTNS_gamma/test{i}.json'
        data_list.append(data)
    data_list_x = ['dataset/mrx_gamma.json']
    data_list_one = ['dataset/PSTNS_gamma_fixed/test1.json']
    data_list_two = ['PSTNS_normal/test51.json','PSTNS_normal/test103.json']
    total_dispatch_rate = 0
    dispatch_rate_list = []
    total_num = 0 


    #stn = loadSTNfromJSONfile('PSTNS_normal/test2.json')
        
        
    # valid_intervals = {(0, 1): (0, 0), (0, 2): (39666.67532046465, 56058.1023726658), (0, 3): (36584.81572968031, 49222.862214705405), (0, 4): (67130.99730406152, 73999.3490496011), (0, 5): (0, 0), (0, 6): (100941.74729258296, 130057.2897317994), (0, 7): (0, 0), (0, 8): (0, 0), (0, 9): (114436.30161146066, 144743.10089840408), (0, 10): (99293.20415285249, 130544.53465789538), (0, 11): (137672.7214053811, 163669.10474982442), (0, 12): (51437.26001510919, 73296.0035970535), (0, 13): (54791.90463184493, 92816.902341733), (0, 14): (24779.610539430003, 35297.76690661208), (0, 15): (46118.55134844475, 70632.89910218997), (0, 16): (87693.59328240211, 109631.19886554297), (0, 17): (0, 0), (0, 18): (38328.515405338854, 54411.3841067036), (0, 19): (43703.70758492393, 62510.38332072177), (0, 20): (54528.631568715144, 85409.39081362991), (0, 21): (96951.35435435304, 121730.3520430151), (0, 22): (45750.246213310595, 71839.89111687812), (1, 2): (39666.67532046465, 56058.1023726658), (3, 4): (24675.103319994872, 35041.40481644177), (5, 6): (100941.74729258296, 130057.2897317994), (7, 4): (67130.99730406152, 73999.3490496011), (8, 9): (114436.30161146066, 144743.10089840408), (10, 11): (29975.220005060415, 47310.06009731359), (12, 13): (2801.701392841642, 21581.51446583397), (8, 12): (51437.26001510919, 73296.0035970535), (14, 3): (9801.855769532129, 19669.9806570593), (15, 16): (35807.25920587056, 54872.87227687606), (17, 11): (137672.7214053811, 163669.10474982442), (2, 15): (3401.414297064606, 17778.648204867328), (17, 18): (38328.515405338854, 54411.3841067036), (19, 20): (9600.749085372117, 25461.235075795914), (21, 10): (900.0376241393678, 15603.003016308168), (13, 9): (47056.06896699304, 73048.81848004718), (18, 22): (6000.455733402036, 19789.74835140165), (5, 19): (43703.70758492393, 62510.38332072177), (22, 21): (42786.92483267526, 64128.43913904626), (1, 16): (87693.59328240211, 109631.19886554297), (7, 14): (24779.610539430003, 35297.76690661208), (20, 6): (39757.026213818666, 62172.74013505723)}
    # newstn = get_valid_interval_stn(stn,valid_intervals)
    # dispat = simulate_maxgain(
    #         stn, newstn, size=200, verbose=True, gauss=True)
    # print("dispatch rate for",dispat)
    

    @timeout_decorator.timeout(5*60) 
    def doStuff(data): 
        print("simulating", data)
        stn = loadSTNfromJSONfile(data)
        # newstn = loadSTNfromJSONfile('test6prime.json')
        newstn = maxgain(stn, debug=False, betaFlag = True)
        print("newstn",newstn)
        dispat = simulate_maxgain(
            stn, newstn, size=200, verbose=False, gauss=True, betaFlag = True)
        # total_dispatch_rate += dispat
        
        print("dispatch rate for",dispat)
        return dispat
    #loop code goes here 

    for data in data_list:   
        try: 
            dispat = doStuff(data) 
            total_dispatch_rate += dispat
            dispatch_rate_list.append((data, dispat))
            total_num += 1
            print(total_dispatch_rate,total_num)
        except: 
            print(data, "has failed")
            print(total_dispatch_rate, len(dispatch_rate_list),dispatch_rate_list)
            pass
    print(total_dispatch_rate/len(data_list),dispatch_rate_list)
'''
if __name__ == "__main__":
    directory = "dataset/dynamically_controllable"
    
    # data_list = glob.glob(os.path.join(directory, '*.json'))
    # data_list = ['dataset/uncontrollable_full/uncontrollable6.json']
    # data_list = ['dataset/dreamdata/STN_a4_i4_s5_t10000/original_0.json']
    #data_list = ['dataset/dreamdata/STN_a2_i4_s1_t4000/original_9.json']
    # data_list = ['dataset/dreamdata/STN_a3_i8_s1_t2000/original_8.json', 'dataset/dreamdata/STN_a3_i8_s1_t2000/original_4.json', 'dataset/dreamdata/STN_a3_i8_s1_t2000/original_5.json', 'dataset/dreamdata/STN_a3_i8_s1_t2000/original_9.json', 'dataset/dreamdata/STN_a3_i8_s1_t2000/original_2.json', 'dataset/dreamdata/STN_a3_i8_s1_t2000/original_0.json', 'dataset/dreamdata/STN_a3_i8_s1_t2000/original_7.json', 'dataset/dreamdata/STN_a3_i8_s3_t6000/original_9.json', 'dataset/dreamdata/STN_a3_i8_s3_t6000/original_1.json', 'dataset/dreamdata/STN_a4_i8_s1_t1000/original_4.json', 'dataset/dreamdata/STN_a4_i8_s1_t1000/original_5.json', 'dataset/dreamdata/STN_a4_i8_s1_t1000/original_9.json', 'dataset/dreamdata/STN_a4_i8_s1_t1000/original_3.json', 'dataset/dreamdata/STN_a4_i8_s1_t1000/original_0.json', 'dataset/dreamdata/STN_a4_i8_s1_t1000/original_1.json', 'dataset/dreamdata/STN_a4_i8_s1_t1000/original_6.json', 'dataset/dreamdata/STN_a4_i8_s1_t1000/original_7.json', 'dataset/dreamdata/STN_a3_i4_s5_t10000/original_3.json', 'dataset/dreamdata/STN_a2_i4_s1_t1000/original_5.json', 'dataset/dreamdata/STN_a2_i4_s1_t1000/original_9.json', 'dataset/dreamdata/STN_a2_i4_s1_t1000/original_2.json', 'dataset/dreamdata/STN_a3_i8_s3_t3000/original_4.json', 'dataset/dreamdata/STN_a3_i8_s3_t3000/original_9.json', 'dataset/dreamdata/STN_a3_i8_s3_t3000/original_3.json', 'dataset/dreamdata/STN_a3_i8_s3_t3000/original_6.json', 'dataset/dreamdata/STN_a3_i8_s3_t3000/original_7.json', 'dataset/dreamdata/STN_a4_i8_s1_t4000/original_8.json', 'dataset/dreamdata/STN_a4_i8_s1_t4000/original_5.json', 'dataset/dreamdata/STN_a4_i8_s1_t4000/original_2.json', 'dataset/dreamdata/STN_a4_i8_s1_t4000/original_0.json', 'dataset/dreamdata/STN_a4_i8_s1_t4000/original_1.json', 'dataset/dreamdata/STN_a4_i8_s1_t4000/original_6.json', 'dataset/dreamdata/STN_a4_i8_s1_t4000/original_7.json', 'dataset/dreamdata/STN_a3_i4_s5_t20000/original_3.json', 'dataset/dreamdata/STN_a3_i4_s5_t20000/original_7.json', 'dataset/dreamdata/STN_a4_i8_s5_t5000/original_1.json', 'dataset/dreamdata/STN_a4_i8_s5_t5000/original_7.json', 'dataset/dreamdata/STN_a4_i4_s5_t5000/original_3.json', 'dataset/dreamdata/STN_a4_i4_s5_t5000/original_0.json', 'dataset/dreamdata/STN_a2_i8_s5_t20000/original_4.json', 'dataset/dreamdata/STN_a2_i8_s5_t20000/original_9.json', 'dataset/dreamdata/STN_a2_i8_s1_t1000/original_8.json', 'dataset/dreamdata/STN_a2_i8_s1_t1000/original_4.json', 'dataset/dreamdata/STN_a2_i8_s1_t1000/original_2.json', 'dataset/dreamdata/STN_a2_i8_s1_t1000/original_3.json', 'dataset/dreamdata/STN_a2_i8_s1_t1000/original_0.json', 'dataset/dreamdata/STN_a2_i8_s1_t1000/original_1.json', 'dataset/dreamdata/STN_a2_i8_s1_t1000/original_6.json', 'dataset/dreamdata/STN_a3_i4_s3_t3000/original_5.json', 'dataset/dreamdata/STN_a3_i4_s3_t3000/original_1.json', 'dataset/dreamdata/STN_a3_i4_s3_t3000/original_6.json', 'dataset/dreamdata/STN_a3_i4_s3_t3000/original_7.json', 'dataset/dreamdata/STN_a2_i8_s5_t10000/original_5.json', 'dataset/dreamdata/STN_a3_i8_s5_t20000/original_6.json', 'dataset/dreamdata/STN_a4_i8_s5_t20000/original_4.json', 'dataset/dreamdata/STN_a4_i4_s1_t1000/original_8.json', 'dataset/dreamdata/STN_a4_i4_s1_t1000/original_0.json', 'dataset/dreamdata/STN_a4_i4_s1_t1000/original_7.json', 'dataset/dreamdata/STN_a2_i8_s5_t5000/original_6.json', 'dataset/dreamdata/STN_a2_i8_s5_t5000/original_7.json', 'dataset/dreamdata/STN_a3_i8_s5_t10000/original_8.json', 'dataset/dreamdata/STN_a3_i8_s5_t10000/original_2.json', 'dataset/dreamdata/STN_a3_i8_s5_t10000/original_1.json', 'dataset/dreamdata/STN_a3_i4_s1_t2000/original_3.json', 'dataset/dreamdata/STN_a3_i4_s1_t2000/original_7.json', 'dataset/dreamdata/STN_a4_i8_s5_t10000/original_5.json', 'dataset/dreamdata/STN_a4_i8_s5_t10000/original_9.json', 'dataset/dreamdata/STN_a4_i8_s5_t10000/original_2.json', 'dataset/dreamdata/STN_a4_i8_s5_t10000/original_1.json', 'dataset/dreamdata/STN_a4_i8_s5_t10000/original_7.json', 'dataset/dreamdata/STN_a3_i4_s3_t6000/original_3.json', 'dataset/dreamdata/STN_a2_i8_s1_t4000/original_8.json', 'dataset/dreamdata/STN_a2_i8_s1_t4000/original_4.json', 'dataset/dreamdata/STN_a2_i8_s1_t4000/original_9.json', 'dataset/dreamdata/STN_a2_i8_s1_t4000/original_2.json', 'dataset/dreamdata/STN_a2_i8_s1_t4000/original_0.json', 'dataset/dreamdata/STN_a2_i8_s1_t4000/original_1.json', 'dataset/dreamdata/STN_a2_i8_s1_t4000/original_7.json', 'dataset/dreamdata/STN_a4_i4_s1_t2000/original_5.json', 'dataset/dreamdata/STN_a4_i4_s1_t2000/original_3.json', 'dataset/dreamdata/STN_a4_i4_s1_t2000/original_7.json', 'dataset/dreamdata/STN_a4_i4_s3_t6000/original_8.json', 'dataset/dreamdata/STN_a2_i8_s3_t3000/original_8.json', 'dataset/dreamdata/STN_a2_i8_s3_t3000/original_9.json', 'dataset/dreamdata/STN_a2_i8_s3_t3000/original_2.json', 'dataset/dreamdata/STN_a2_i8_s3_t3000/original_3.json', 'dataset/dreamdata/STN_a2_i8_s3_t3000/original_1.json', 'dataset/dreamdata/STN_a2_i8_s3_t3000/original_7.json', 'dataset/dreamdata/STN_a3_i4_s1_t1000/original_8.json', 'dataset/dreamdata/STN_a3_i4_s1_t1000/original_4.json', 'dataset/dreamdata/STN_a3_i4_s1_t1000/original_5.json', 'dataset/dreamdata/STN_a3_i4_s1_t1000/original_6.json', 'dataset/dreamdata/STN_a3_i4_s1_t1000/original_7.json', 'dataset/dreamdata/STN_a4_i4_s3_t12000/original_6.json', 'dataset/dreamdata/STN_a3_i4_s3_t12000/original_8.json', 'dataset/dreamdata/STN_a4_i4_s3_t3000/original_5.json', 'dataset/dreamdata/STN_a4_i4_s3_t3000/original_7.json', 'dataset/dreamdata/STN_a3_i4_s1_t4000/original_5.json', 'dataset/dreamdata/STN_a2_i8_s3_t6000/original_8.json', 'dataset/dreamdata/STN_a2_i8_s1_t2000/original_8.json', 'dataset/dreamdata/STN_a2_i8_s1_t2000/original_4.json', 'dataset/dreamdata/STN_a2_i8_s1_t2000/original_5.json', 'dataset/dreamdata/STN_a2_i8_s1_t2000/original_3.json', 'dataset/dreamdata/STN_a2_i8_s1_t2000/original_0.json', 'dataset/dreamdata/STN_a2_i8_s1_t2000/original_6.json', 'dataset/dreamdata/STN_a2_i8_s1_t2000/original_7.json', 'dataset/dreamdata/STN_a3_i8_s1_t4000/original_8.json', 'dataset/dreamdata/STN_a3_i8_s1_t4000/original_4.json', 'dataset/dreamdata/STN_a3_i8_s1_t4000/original_9.json', 'dataset/dreamdata/STN_a3_i8_s1_t4000/original_2.json', 'dataset/dreamdata/STN_a3_i8_s1_t4000/original_3.json', 'dataset/dreamdata/STN_a3_i8_s1_t4000/original_1.json', 'dataset/dreamdata/STN_a3_i8_s1_t4000/original_6.json', 'dataset/dreamdata/STN_a3_i8_s1_t4000/original_7.json', 'dataset/dreamdata/STN_a3_i8_s5_t5000/original_8.json', 'dataset/dreamdata/STN_a3_i8_s5_t5000/original_4.json', 'dataset/dreamdata/STN_a3_i8_s5_t5000/original_9.json', 'dataset/dreamdata/STN_a3_i8_s5_t5000/original_3.json', 'dataset/dreamdata/STN_a3_i8_s5_t5000/original_6.json', 'dataset/dreamdata/STN_a2_i4_s1_t2000/original_4.json', 'dataset/dreamdata/STN_a2_i4_s1_t2000/original_0.json', 'dataset/dreamdata/STN_a2_i8_s3_t12000/original_5.json', 'dataset/dreamdata/STN_a4_i8_s3_t3000/original_9.json', 'dataset/dreamdata/STN_a4_i8_s3_t3000/original_3.json', 'dataset/dreamdata/STN_a4_i8_s3_t3000/original_0.json', 'dataset/dreamdata/STN_a4_i8_s3_t3000/original_1.json', 'dataset/dreamdata/STN_a4_i8_s3_t3000/original_6.json', 'dataset/dreamdata/STN_a4_i8_s3_t3000/original_7.json', 'dataset/dreamdata/STN_a3_i8_s1_t1000/original_8.json', 'dataset/dreamdata/STN_a3_i8_s1_t1000/original_4.json', 'dataset/dreamdata/STN_a3_i8_s1_t1000/original_5.json', 'dataset/dreamdata/STN_a3_i8_s1_t1000/original_9.json', 'dataset/dreamdata/STN_a3_i8_s1_t1000/original_3.json', 'dataset/dreamdata/STN_a3_i8_s1_t1000/original_0.json', 'dataset/dreamdata/STN_a3_i8_s1_t1000/original_1.json', 'dataset/dreamdata/STN_a3_i8_s1_t1000/original_6.json', 'dataset/dreamdata/STN_a4_i8_s1_t2000/original_8.json', 'dataset/dreamdata/STN_a4_i8_s1_t2000/original_4.json', 'dataset/dreamdata/STN_a4_i8_s1_t2000/original_5.json', 'dataset/dreamdata/STN_a4_i8_s1_t2000/original_9.json', 'dataset/dreamdata/STN_a4_i8_s1_t2000/original_3.json', 'dataset/dreamdata/STN_a4_i8_s1_t2000/original_0.json', 'dataset/dreamdata/STN_a4_i8_s1_t2000/original_6.json', 'dataset/dreamdata/STN_a4_i8_s3_t6000/original_4.json', 'dataset/dreamdata/STN_a4_i8_s3_t6000/original_5.json', 'dataset/dreamdata/STN_a4_i8_s3_t6000/original_2.json', 'dataset/dreamdata/STN_a4_i8_s3_t6000/original_3.json', 'dataset/dreamdata/STN_a4_i8_s3_t12000/original_5.json', 'dataset/dreamdata/STN_a4_i8_s3_t12000/original_9.json', 'dataset/dreamdata/STN_a4_i8_s3_t12000/original_2.json', 'dataset/dreamdata/STN_a3_i8_s3_t12000/original_8.json', 'dataset/dreamdata/STN_a3_i8_s3_t12000/original_5.json', 'dataset/dreamdata/STN_a3_i8_s3_t12000/original_9.json', 'dataset/dreamdata/STN_a3_i8_s3_t12000/original_7.json']
    # data_list = ['small_examples/dynamic1.json']
    # data_list = ['paperexample.json']
    # data_list = ['dataset/mrx.json']
    # data_list = ['dataset/mrx_gamma.json']
    # data_list = ['untounchedmrx.json']
    # data_list = ['smallresult/test2.json']
    data_list = ['dataset/PSTNS_exp/test1.json']
    ##testing dream data ##
    # directory = 'dataset/dreamdata/'
    # folders = os.listdir(directory)
    # data_list = []
    # for folder in folders:
    #     data = glob.glob(os.path.join(directory, folder, '*.json'))
    #     data_list += data
    # data_list = ['dataset/dreamdata/STN_a2_i4_s1_t4000/original_9.json']
    comparison = []
    improvement = 0
    tied = 0
    failed = []
    count = 0
    bad_data = []
    for data in data_list:
        print("simulating", data)
        print("load")
        stn = loadSTNfromJSONfile(data)
        print("maxgain")
        print(stn)
        newstn = maxgain(stn, debug=False)
        dispat = simulate_maxgain(
            stn, newstn, size=20, verbose=False, gauss=True)
        print("dispact", dispat)
        print('hotham')
        # if a:
        #     result = simulate_maxgain(stn, 100, verbose = False)
        #     print(result)
        #     break
        # print(stn)
        print("new stn")
        print(newstn)
        # newresult = simulate_maxgain(stn, newstn,50)
        # print('c')
        # oldresult = simulation(stn,50, verbose = False)
        # if a and oldresult < .9:
        #     bad_data += [(data, oldresult)]
        # comparison += [(newresult, oldresult, data)]
        # count += 1
        # if newresult > oldresult:
        #     improvement += 1
        # elif newresult == oldresult:
        #     tied += 1
        #     if newresult == 0.0:
        #         failed += [data]
        # comparison += [(newresult, oldresult)]
        print("comparison")
        print(comparison)
    # text_file = open("weird.txt", "w")
    # text_file.write(str(bad_data))
    # text_file.close()
    # text_file = open("failed.txt", "w")
    # text_file.write(str(failed))
    # text_file.close()
# 51 103
'''
 
  
  
