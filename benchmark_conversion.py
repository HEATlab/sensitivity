import random
from stn import STN, loadSTNfromJSONfile



def convert_one_event(min_duration, max_duration,step):
    time_prob = {}
    # range: (min_duration, max_duration)
    for i in range(min_duration, max_duration+1, step):
        # generate a random number between 0 to 1 for this integer in between
        time_prob[i] =  random.random()
        sum += time_prob[i] 

    # normalization so the sum equal to 1
    for i in range(min_duration, max_duration+1):
        time_prob[i] = time_prob[i]/sum
    
    return time_prob


def convert_network(file_path,step):
    network = loadSTNfromJSONfile(file_path)
    # nested dictionary network{event_num:{time_prob{first time: ... , second time: ...}}}
    network_prob = {}
    for event in network.verts:
        min_duration = -event.Cji
        max_duration = event.Cij
        time_prob = {}
        # range: (min_duration, max_duration)
        for i in range(min_duration, max_duration+1, step):
            # generate a random number between 0 to 1 for this integer in between
            time_prob[i] =  random.random()
            sum += time_prob[i] 

        # normalization so the sum equal to 1
        for i in range(min_duration, max_duration+1):
            time_prob[i] = time_prob[i]/sum
        network_prob[event] = time_prob
    
    return network_prob
