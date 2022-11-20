from algorithm import *
from pulp import *
from scipy.stats import gamma
from ast import literal_eval as make_tuple
from scipy.optimize import minimize
import numpy as np

## \file relax.py
#  \brief relaxation algorithm for dynamic controllability


##
#  \fn addConstraint(constraint,problem)
#  \brief Adds an LP constraint to the given LP
#
#  @param constraint A constraint that need to be added to the LP problem
#  @param problem    An input LP problem
#
#  @post LP problem with new constraint added
def addConstraint(constraint, problem):
    problem += constraint


##
# \fn getShrinked(STN, bounds, epsilons)
# \brief compute the shrinked contingent intervals and find contingent intervals
#        that are actually shrinked
#
# @param STN            An input STN
# @param bounds         A dictionary of bounds we can relax to resolve conflict
# @param epsilons       A dictionary of variables returned by LP to resolve
#                       the conflict
#
# @return dictionaries of original and shrinked contingent intervals and a list
#         of changed contingent constarints
def getShrinked(STN, bounds, epsilons):
    contingent = bounds['contingent']
    original = {}
    shrinked = {}
    changed = []

    for (i, j) in list(STN.contingentEdges.keys()):
        edge = STN.contingentEdges[(i, j)]
        orig = (-edge.Cji, edge.Cij)
        original[(i, j)] = orig

        if (i, j) not in contingent or epsilons[j].varValue == 0:
            shrinked[(i, j)] = orig
        else:
            eps = epsilons[j].varValue
            _, bound = contingent[(i, j)]
            low, high = orig

            if bound == 'UPPER':
                high -= eps
            else:
                low += eps

            shrinked[(i, j)] = (low, high)
            changed.append((i, j))

    return original, shrinked, changed


##
# \fn relaxNLP(bounds, weight, debug=False)
# \brief run LP to compute amount of uncertainty need to be removed to
#        resolve the negative cycle
#
# @param bounds       A dictionary of bounds we can relax to resolve conflict
# @param weight       Weight of the semi-reducible negative cycle
# @param debug        Flag indicating wehther we want to report information
#
# @return A dictionary Lp variable epsilons (amount of uncertain need to be
#         removed from each contingent interval)
def relaxNLP(bounds, weight, debug=False):
    contingent = bounds['contingent']
    epsilons = {}

    prob = LpProblem('Relaxation LP', LpMinimize)

    eps = []
    for i, j in list(contingent.keys()):
        edge, bound = contingent[(i, j)]
        length = edge.Cij + edge.Cji

        epsilons[j] = LpVariable('eps_%i' % j, lowBound=0, upBound=length)
        eps.append(1.0 / length * epsilons[j])

    s = sum([epsilons[j] for j in epsilons])
    addConstraint(s >= -weight, prob)

    Obj = sum(eps)
    prob += Obj, "Minimize the Uncertainty Removed"

    # write LP into file for debugging (optional)
    if debug:
        prob.writeLP('relax.lp')
        LpSolverDefault.msg = 10

    try:
        prob.solve()
    except Exception:
        print("The model is invalid.")
        return 'Invalid', None

    # Report status message
    status = LpStatus[prob.status]
    if debug:
        print("Status: ", status)

        for v in prob.variables():
            print(v.name, '=', v.varValue)

    if status != 'Optimal':
        print('hi', status)
        print("The solution for LP is not optimal")
        return status, None

    return status, epsilons, eps


##
# \fn relaxDeltaLP(bounds, weight, debug=False)
# \brief run delta LP to compute amount of uncertainty need to be removed to
#        resolve the negative cycle
#
# @param bounds       A dictionary of bounds we can relax to resolve conflict
# @param weight       Weight of the semi-reducible negative cycle
# @param debug        Flag indicating wehther we want to report information
#
# @return A dictionary Lp variable epsilons (amount of uncertain need to be
#         removed from each contingent interval)
def relaxDeltaLP(bounds, weight, debug=False):
    contingent = bounds['contingent']
    epsilons = {}

    prob = LpProblem('Relaxation Delta LP', LpMinimize)
    delta = LpVariable('delta', lowBound=0, upBound=1)

    for i, j in list(contingent.keys()):
        edge, bound = contingent[(i, j)]
        length = edge.Cij + edge.Cji

        epsilons[j] = LpVariable('eps_%i' % j, lowBound=0, upBound=length)
        addConstraint(epsilons[j] == delta * length, prob)

    s = sum([epsilons[j] for j in epsilons])
    addConstraint(s >= -weight, prob)

    Obj = delta
    prob += Obj, "Minimize the Proportion of Uncertainty Removed"

    # write LP into file for debugging (optional)
    if debug:
        prob.writeLP('relax_delta.lp')
        LpSolverDefault.msg = 10

    try:
        prob.solve()
    except Exception:
        print("The model is invalid.")
        return 'Invalid', None

    # Report status message
    status = LpStatus[prob.status]
    if debug:
        print("Status: ", status)

        for v in prob.variables():
            print(v.name, '=', v.varValue)

    if status != 'Optimal':
        print("The solution for LP is not optimal")
        return status, None

    return status, epsilons


def relaxMaxLP(bounds, weight, debug=False):
    contingent = bounds['contingent']
    epsilons = {}

    prob = LpProblem('Relaxation Delta LP', LpMaximize)
    # delta = LpVariable('delta', lowBound=0, upBound=1)
    cutLength = {}
    for i, j in list(contingent.keys()):
        edge, bound = contingent[(i, j)]
        length = edge.Cij + edge.Cji

        epsilons[j] = LpVariable('eps_%i' % j, lowBound=0, upBound=length)
        # addConstraint(epsilons[j] == delta * length, prob)
        cutLength[j] = length - epsilons[j]
        print(length)

    print(cutLength, weight)
    s = sum([cutLength[j] for j in cutLength])
    addConstraint(s >= -weight, prob)
    Obj = 1
    for j in epsilons:
        Obj += epsilons[j]
    prob += Obj, "Maximize the Proportion of Uncertainty"

    # write LP into file for debugging (optional)
    if debug:
        prob.writeLP('relax_vol.lp')
        LpSolverDefault.msg = 10

    try:
        prob.solve()
    except Exception:
        print("The model is invalid.")
        return 'Invalid', None

    # Report status message
    status = LpStatus[prob.status]
    if debug:
        print("Status: ", status)

        for v in prob.variables():
            print(v.name, '=', v.varValue)

    if status != 'Optimal':
        print("The solution for LP is not optimal")
        return status, None

    return status, epsilons



##
# \fn optimalRelax(bounds, weight)
# \brief optimal solution for compute relax strategy
#
# @param bounds       A dictionary of bounds we can relax to resolve conflict
# @param weight       Weight of the semi-reducible negative cycle
#
# @return A dictionary of epsilons (amount of uncertain need to be
#         removed from each contingent interval)
def optimalRelax(bounds, weight):
    contingent = [bounds['contingent'][x][0] for x in \
                                list(bounds['contingent'].keys())]
    contingent.sort(key=lambda x: x.Cij + x.Cji, reverse=False)

    length = [e.Cij + e.Cji for e in contingent]
    S = sum(length) + weight
    n = len(contingent)
    if S < 0:
        return None

    m = None
    for i in range(n):
        previous = length[:i]
        test_sum = sum(previous) + (n - i) * length[i]
        if test_sum >= S:
            m = i
            break

    A = (S - sum(length[:m])) / (n - m)
    epsilons = {}
    for e in contingent[m:]:
        epsilons[e.j] = e.Cij + e.Cji - A
    return epsilons


def relaxTotalLength(bounds, weight):
    return None

##
# \fn relaxSearch(STN)
# \brief run relaxation algorithm on an STNU so that it becomes dynamically
#        controllable
#
# @param STN       An STNU we want to relax/process
#
# @return The dynamically controllable relaxed STNU and the number of conflict
#         need to be resolved
def relaxSearch(STN, origin):
    result, conflicts, bounds, weight = DC_Checker(STN.copy(), report=False)
    print(result, conflicts, bounds, weight)
    count = 0
    cycles = []
    weights = [] 
    print(result)  
    while not result:
        edges = [x[0] for x in list(bounds['contingent'].values())]
        cycles.append(edges)
        weights.append(weight)
        print(weights)
        epsilons = optimalRelax(bounds, weight)

        if not epsilons:
            print("The STNU cannot resolve the conflict...")
            return None, 0, None
        print("epsilons:",epsilons)
        epsilonsN = relaxGammaNLP(bounds, weight)
        print('epsilonsN', epsilonsN)
        

        for (i, j) in list(STN.contingentEdges.keys()):
            if j not in list(epsilonsN.keys()):
                continue
            edge = STN.contingentEdges[(i, j)]

            dist, mu, param = (bounds['contingent'][(i, j)][2]).split("_")
           
            if dist == 'U' or origin==True:
                if j not in list(epsilons.keys()):
                    continue
                if bounds['contingent'][(i, j)][1] == 'UPPER':
                    STN.modifyEdge(i, j, edge.Cij - epsilons[j])
                else:
                    STN.modifyEdge(j, i, edge.Cji - epsilons[j])
            elif dist == 'G':                
                if bounds['contingent'][(i, j)][1] == 'UPPER':
                    print("before the edge is", (i,j), STN.edges[(i,j)])
                    STN.modifyEdge(i, j, edge.Cij - 1000*epsilonsN[j])
                else:
                    STN.modifyEdge(j, i, edge.Cji - 1000*epsilonsN[j])
                print("after the edge is", (i,j), STN.edges[(i,j)])

            elif dist == 'N':
                STN.modifyEdge(i, j, edge.Cij - epsilons[j]/2)
                STN.modifyEdge(j, i, edge.Cji - epsilons[j]/2)
            # elif dist == 'G':
            #     if j not in list(epsilons.keys()):
            #         continue
            #     goal = epsilons[j]
            #     # relaxGammaNLP(bounds, weight)
            #     lowerE, upperE = relaxGamma(goal, mu, param, -edge.Cji, edge.Cij)
            #     print(lowerE, upperE, edge.Cij, edge.Cji)
            #     STN.modifyEdge(i, j, edge.Cij - upperE)
            #     STN.modifyEdge(j, i, edge.Cji - lowerE)
            #     print("new",lowerE, upperE, edge.Cij, edge.Cji)
        count += 1
        result, conflicts, bounds, weight = DC_Checker(
            STN.copy(), report=False)
        print(result, bounds, weight)
    return STN, count, cycles, weights

def relaxGamma(goal, alpha, param, lb, ub):
    scale, loc = make_tuple(param)
    alpha = float(alpha)
    remaining = (ub-lb - goal)/1000
    mode = alpha - 1
    initialL, initialU = mode - remaining/2, mode + remaining/2
    if initialL < 0:
        # the window shifts to the right together
        initialU = initialU - initialL + loc
        initialL = loc
        lowerE = initialL - lb/1000
        upperE = - initialU + ub/1000
        return 1000*lowerE, 1000*upperE
    elif initialU > ub/1000:
        print("wrong")
    else:
        pdfL = gamma.pdf(x = initialL, a = alpha, scale = 1)
        pdfU = gamma.pdf(x = initialU, a = alpha, scale = 1)
        distL = gamma.ppf(q = 0.001, a = alpha, scale = 1)
        distU = gamma.ppf(q = 0.999, a = alpha, scale = 1)
        unit = (distU - distL)/100
        while pdfU > pdfL:
            initialL += unit
            initialU += unit 
            pdfL = gamma.pdf(x = initialL, a = alpha, scale = 1)
            pdfU = gamma.pdf(x = initialU, a = alpha, scale = 1)

        initialU += loc
        initialL += loc
        lowerE = initialL - lb/1000
        upperE = - initialU + ub/1000
        return 1000*lowerE, 1000*upperE
  
def relaxGammaNLP(bounds, weight, debug=False):
    contingent = [bounds['contingent'][x][0] for x in \
                                list(bounds['contingent'].keys())]
    dirs = [bounds['contingent'][(i,j)][1] for (i, j) in list(bounds['contingent'])]
    dists= [(bounds['contingent'][(i, j)][2]).split("_") for (i, j) in list(bounds['contingent'])]
    alphas = [float(mu) for (dist, mu, param) in dists]
    length = [e.Cij + e.Cji for e in contingent]
    posWeight = -weight
    epsilons = optimalRelax(bounds, weight)

    bs = []
    vars = []

    def objective(x):
        expectation = []
        for j in range(len(x)):
            vars.append(x[j])
            if dirs[j] == 'UPPER':
                expectation.append(x[j]*(1-gamma.cdf((length[j]/1000 - x[j]), a=alphas[j], scale=1)))
            else:
                expectation.append(x[j]*(gamma.cdf(x[j], a=alphas[j], scale=1)))
        return sum(expectation)

    def constraint1(x):
        return np.sum(x)-posWeight/1000

    for j in range(len(contingent)):
        bs.append((0,length[j]/1000))
    bnds = tuple(bs)
    con1 = {'type': 'ineq', 'fun': constraint1}
    cons = ([con1])
    x0=[]
    for (i,j) in bounds['contingent']:
        # initial guess using average
        x0.append(weight/1000/len(bounds['contingent']))

        # initial guess using optimal relax
        # if j in epsilons:
        #     x0.append(epsilons[j]/1000)
        # else:
        #     x0.append(0)

    # x0 = [epsilons[j]/1000 for (i,j) in bounds['contingent']]
    # x0 = [0 for i in range(len(bounds['contingent']))]
    solution = minimize(objective,x0,method='SLSQP',\
                        bounds=bnds,constraints=cons)
    result = {}
    index = 0
    for (i, j) in list(bounds['contingent']):
        result[j] = solution.x[index]
        index += 1
    print("relax result is :::::::",result)
    return result


