#PRUNE and DATABASE available

import numpy as np
import math
import sys
import json
import pandas as pd
import time
import time
import random
import matplotlib.pyplot as plt
import itertools
from deap import base
from deap import creator
from deap import tools

import simulation_codes  #Andre's package that used in simulation opt.

# reproducability
random.seed(60)
np.random.seed(60)


###Prune function has to defined earlier 
def optimal_server_number(priority, FailureRates, ServiceRates, holding_costs, penalty_cost, skill_cost, machineCost):
    
    '''
    returns number of servers and total cost for allocating
    that number of servers 
    
    '''

    min_nserver=int(sum(np.array(FailureRates)/np.array(ServiceRates)))+1    #min required servers
    assignment=np.ones((min_nserver, len(FailureRates)))
    
    #########RISK AVERSE #################
    holding_backorder_CostList =  simulation_codes.SimulationInterface.simulation_optimization_bathrun_priority(FailureRates, ServiceRates,holding_costs, penalty_cost, assignment,priority,
                                  numberWarmingUpRequests = 5000,
                                  stopNrRequests = 100000,
                                  replications =40)
    
    Server_Cost=min_nserver*machineCost
    
    #############################BELOW code calculates risk averse expected backorder and holding cost###########
    holding_backorder_CostList=np.array(holding_backorder_CostList)
    var = np.percentile(holding_backorder_CostList, 100 - var_level) #95%
    cvar_plus = holding_backorder_CostList[holding_backorder_CostList > var].mean() 
    var_level_p = var_level / 100.0
    cdf_var = holding_backorder_CostList[holding_backorder_CostList <= var].size / (1.0*holding_backorder_CostList.size)
    lamda = (cdf_var-var_level_p)/(1-var_level_p)
    risk_averse_cost = lamda*var+ (1-lamda)*cvar_plus
    
    TotalCost = risk_averse_cost+Server_Cost  #this is now risk averse code 
    
    ######################################
    while True:
        min_nserver += 1 
        assignment=np.ones((min_nserver, len(FailureRates)))
        temp_holding_backorder_CostList =  simulation_codes.SimulationInterface.simulation_optimization_bathrun_priority(FailureRates, ServiceRates,holding_costs, penalty_cost, assignment,priority,
                                  numberWarmingUpRequests = 5000,
                                  stopNrRequests = 100000,
                                  replications =40)
        temp_Server_Cost=min_nserver*machineCost

        #########RISK AVERSE #################
        temp_holding_backorder_CostList=np.array(temp_holding_backorder_CostList)
        temp_var = np.percentile(temp_holding_backorder_CostList, 100 - var_level) #95%
        temp_cvar_plus = temp_holding_backorder_CostList[temp_holding_backorder_CostList > temp_var].mean() 
        temp_cdf_var = temp_holding_backorder_CostList[temp_holding_backorder_CostList <= temp_var].size / (1.0*temp_holding_backorder_CostList.size)
        lamda = (temp_cdf_var-var_level_p)/(1-var_level_p)
        temp_risk_averse_cost = lamda*temp_var + (1-lamda)*temp_cvar_plus 
    
        temp_TotalCost=temp_risk_averse_cost+temp_Server_Cost
    
        if temp_TotalCost > TotalCost:
            min_nserver-=1
            break
        else:
            TotalCost=temp_TotalCost
            Server_Cost= temp_Server_Cost
    
    return min_nserver, TotalCost, Server_Cost, TotalCost-Server_Cost


####PRUNE and DATABASE ###################
def set_multipliers(len_sku):
    #global multipliers
    multipliers = 2**np.arange(len_sku-1,-1,-1)
    return multipliers


def encode(x):
    """ 
    Encode x as a sorted list of base10 representation to detect symmetries.
        
    First, for each cluster determine which SKU types are available.
        x = [1,1,2] means in cluster 1 there are SKU1 and SKU2 in cluster 2 there is only SKU3
        So the binary representation(skill-server assignment) is below. 
        Each inner list represents a server and binary values denote if that server has the skill or not. 
        
        Examples of solutions that are symmetric to x, their binary and base 10 represetations:
               x     binary representation   base10 representation
            -------  ---------------------    ------------------
            [1,1,2]  [[1,1,0], [0,0,1]]             [6,1]
            [1,1,3]  [[1,1,0], [0,0,1]]             [6,1]
            [2,2,1]  [[0,0,1], [1,1,0]]             [1,6]
            [2,2,3]  [[1,1,0], [0,0,1]]             [6,1]
            [3,3,1]  [[0,0,1], [1,1,0]]             [1,6]
            [3,3,2]  [[0,0,1], [1,1,0]]             [1,6]
    :param x, sample solution
    """
    multipliers=set_multipliers(len(x))
    return tuple([sum([multipliers[j] for j,c2 in enumerate(x) if c2 == c]) for c in set(x)]) 


def prune_and_evaluate(priority, cache, failure_rates, service_rates, holding_costs, penalty_cost, skill_cost, machine_cost):
    priority_rep = encode(priority)
    if priority_rep in cache.keys():
        return cache[priority_rep]
    #may need to check cache length
    _, TotalCost, _, _= optimal_server_number(priority, failure_rates, service_rates, holding_costs, penalty_cost, skill_cost, machine_cost)
    
    if len(cache)<10000: #len of cache
        cache[priority_rep]=TotalCost
    return TotalCost


###Fitness Evaulation function
#### CaLls simulation 
###cache is the database 
def Fitness(cache, FailureRates, ServiceRates, holding_costs, penalty_cost, skillCost, machineCost,  priority):
    '''
    input: -Individual representing clustering scheme
           -Failure rates and corresponding service rates of each SKU
           -Related cost terms holding costs for SKUs(array), backorder, skill and server (per server and per skill)
           -Simulation Module with Priorites called --> to find Queue length dist. of failed SKUs
                                                
           
           -OptimizeStockLevels calculates EBO and S for giving clustering (Queue length dist.)
           
     output: Returns best total cost and other cost terms, Expected backorder (EBO) and stocks (S) for each SKU, # of 
             servers at each cluster
    
     evalOneMax function evaluates the fitness of individual chromosome by:
           (1) chromosome converted a clustering scheme
           (2) for each SKU in each cluster at the clustering scheme queue length dist. evaluated by calling MMC solver
           (3) OptimzeStockLevels function is called by given queue length dist. and initial costs are calculated???
           (4) Local search is performed by increasing server numbers in each cluster by one and step (2) and (3) repeted
           (5) Step (4) is repated if there is a decrease in total cost
    
    Warning !! type matching array vs list might be problem (be careful about type matching)
    
    '''
    TotalCost = prune_and_evaluate(priority, cache, FailureRates, ServiceRates, holding_costs, penalty_cost, skillCost,  machineCost)
    return TotalCost,  
        

def generatePriority(numSKUs, numPriorityClass=1):
    '''
    -assing priority classes to each SKU randomly for a given number of priority class
    -returns a np array 
    '''
    
    return np.random.choice(np.arange(1, numPriorityClass+1), numSKUs)


# VNS Neighborhood Structures
def ns_throas_mutation(x, *args):
    """ Randomly select three element positions(i,j,k) of x. 
         value at i becomes value at j 
         value at j becomes value at k
         value at k becomes value at i """
    x_new = x[:]
    n_skus = len(x_new)
    idx1, idx2, idx3 = random.sample(range(0, n_skus), 3)
    x_new[idx2] = x[idx1]
    x_new[idx3] = x[idx2]
    x_new[idx1] = x[idx3]   
    return x_new  


def ns_center_inverse_mutation(x, *args):
    """ Randomly select a position i, mirror genes referencing i 
    Example: [1,2,3,4,5,6] if i is 3 result is [3,2,1,6,5,4]"""
    idx = random.randint(0, len(x)-1)
    return  x[idx::-1] + x[:idx:-1]   


def ns_two_way_swap(x, *args):
    """ Randomly swaps two elements of x. """
    x_new = x[:]
    n_skus = len(x_new)
    idx1, idx2 = random.sample(range(0, n_skus), 2)
    x_new[idx1] = x[idx2]
    x_new[idx2] = x[idx1]
    return x_new


def ns_shuffle(x, *args):
    """ Returns a permutation of elements of x. """
    x_new = x[:]
    random.shuffle(x_new)
    return x_new


def ns_mutate_random(x, min_cluster, max_cluster):
    """ Changes value of a random element of x. 
        The new values are between min_cluster and max_cluster inclusive. """
    x_new = x[:]
    n_skus = len(x_new)
    idx = random.randint(0, n_skus-1)
    ex_cluster_number = x[idx]
    numbers = list(range(min_cluster, ex_cluster_number)) + list(range(ex_cluster_number + 1, max_cluster))
    x_new[idx] = random.choice(numbers)
    return x_new #if filter_out_symmetric_solutions([x_new]) else ns_mutate_random(x, min_cluster, max_cluster)


def ns_mutate_random2(x, min_cluster, max_cluster):
    """ Changes values of two random elements of x.
        The new values are between min_cluster and max_cluster inclusive. """
    x_new = x[:]
    n_skus = len(x_new)
    idx1, idx2 = random.sample(range(0, n_skus), 2)
    ex_cluster_number = x[idx1]
    numbers = list(range(min_cluster, ex_cluster_number)) + list(range(ex_cluster_number + 1, max_cluster))
    x_new[idx1] = random.choice(numbers)
    ex_cluster_number = x[idx2]
    numbers = list(range(min_cluster, ex_cluster_number)) + list(range(ex_cluster_number + 1, max_cluster))
    x_new[idx2] = random.choice(numbers)
    
    return x_new 


def solve_rvns( cache, initial_priority, ngf, min_cluster, max_cluster, failure_rates, service_rates, holding_costs, penalty_cost, skill_cost, machine_cost, max_iters=1000):
    """Finds best solution x given an initial solution x,
       list shaking functions ngf, and
    """
    x = initial_priority 
    tcost_x = prune_and_evaluate(x, cache, failure_rates, service_rates, holding_costs, penalty_cost, skill_cost, machine_cost)
    iter_since_last_best = 0
    same_consecutive_count = 0
    prev_best = 0
    while(iter_since_last_best < 100 and same_consecutive_count < 10 ):
        k = 0
        better_found = False
        while k < len(nsf):
            # create neighborhood solution using kth ngf
            x1 = ngf[k](x, min_cluster, max_cluster)
            tcost_x1 = prune_and_evaluate(x1, cache, failure_rates, service_rates, holding_costs, penalty_cost, skill_cost, machine_cost)
            if tcost_x1 <= tcost_x:
                print("=== NEW lower total cost: {:.4f}, iter_slb:{}".format(tcost_x1, iter_since_last_best))
                x = x1
                tcost_x = tcost_x1
                k = 0
                better_found = True
                if prev_best == tcost_x1 :
                    same_consecutive_count += 1
                else:
                    same_consecutive_count = 0
                    prev_best = tcost_x1
            else:
                k += 1                
        
        # check for improvement
        if not better_found:
            iter_since_last_best += 1
        else:
            iter_since_last_best = 0
    return tcost_x, x, cache


def VNS_Priority(cache, failure_rates, service_rates, holding_costs, penalty_cost, skill_cost, machine_cost, numPriorityClass=1): 
   
    # 1 is for maximization -1 for minimization
    # Minimize total cost just EBO cost and holding cost at the moment 
    creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    

    def generatePriority(numSKUs, numPriorityClass=1):
        '''
        -assing priority classes to each SKU randomly for a given number of priority class
        -returns a np array 
        '''
    
        return creator.Individual(np.random.choice(np.arange(1, numPriorityClass+1), numSKUs))

    
    toolbox = base.Toolbox()
    toolbox.register("individual", generatePriority, len(failure_rates), numPriorityClass)
    # define the population to be a list of individuals
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    #----------
    # Operator registration
    #----------
    # register the goal / fitness function
    toolbox.register("evaluate", Fitness, cache, failure_rates, service_rates, holding_costs, penalty_cost, skill_cost, machine_cost)
    
    # register the crossover operators

    # operator for selecting individuals for breeding the next
    # generation: each individual of the current generation
    # is replaced by the 'fittest' (best) of three individuals
    # drawn randomly from the current generation.
    toolbox.register("select", tools.selTournament, tournsize=10)

    #----------

    # random.seed(64)

    # create an initial population of 50 individuals (where
    # each individual is a list of integers)
    start_time = time.time() #start time
    pop = toolbox.population(n=1)
    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    
    # Extracting all the fitnesses of 
    fits = [ind.fitness.values[0] for ind in pop]
    best_cost=min(fits)
    best_priority=tools.selBest(pop, 1)[0]
    # Variable keeping track of the number of generations
    best_cost, best_priority, cache = solve_rvns(cache, best_priority, nsf, 2, len(failure_rates), failure_rates, service_rates, holding_costs, penalty_cost, skill_cost, machine_cost)
    stop_time = time.time() - start_time
    
    return best_cost, best_priority, stop_time, cache


json_case=[]
with open("GAPoolingAll_4a.json", "r") as json_file:
    #json_file.readline()
    for line in json_file:
        json_case.append(json.loads(line))
        
# shaking functions
indices = [0,1]
fname = "".join(map(str,indices))
nsf = [ns_mutate_random, ns_mutate_random2, ns_shuffle, ns_two_way_swap, ns_throas_mutation, ns_center_inverse_mutation]
nsf = [nsf[i] for i in indices]

# risk averse parameters
var_level = 5

#################INTEGRATING pruning and cache TO GA#####################################

Results=[]
tot_cases = 0
for case in json_case[0]:
    if case['simulationGAresults']['holding_costs_variant']==2: #HPB cost variant
        if case['simulationGAresults']['skill_cost_factor']==0.1:
            if case['simulationGAresults']['utilization_rate'] ==0.8:
                tot_cases += 1 
                if tot_cases == 4:
                    break         
                FailureRates=np.array(case['simulationGAresults']["failure_rates"])
                ServiceRates=np.array(case['simulationGAresults']["service_rates"])
                holding_costs=np.array(case['simulationGAresults']["holding_costs"])
                penalty_cost=case['simulationGAresults']["penalty_cost"]
                skillCost=100 #NOT USING THIS ATM
                machineCost=case['simulationGAresults']['machine_cost']
                        
                VNS_SimOpt={}
                numPriorityClass=len(FailureRates) #making equal to number of SKUs
                        
                ##############VNS RUN############
                print "running VNS for ", case["caseID"]
                cache={}
                best_cost, best_priority, run_time, cache  =VNS_Priority(cache, FailureRates, ServiceRates,holding_costs, penalty_cost,skillCost, machineCost, numPriorityClass) 
                        
                print "Best Cost", best_cost
                print "Best Priority", best_priority
                        
                        
                ####################RECORDING RESULTS##################
                ###VNS Outputs
                VNS_SimOpt["VNS_best_cost"]=best_cost
                VNS_SimOpt["VNS_best_priority"]=best_priority
                VNS_SimOpt["VNS_run_time"]=run_time
                        
                        
                ###inputs
                VNS_SimOpt["CaseID"]=case["caseID"]
                VNS_SimOpt["FailureRates"]=FailureRates.tolist()
                VNS_SimOpt["ServiceRates"]=ServiceRates.tolist()
                VNS_SimOpt["holding_costs"]=holding_costs.tolist()
                VNS_SimOpt["penalty_cost"]=penalty_cost  
                VNS_SimOpt["skillCost"]=skillCost
                VNS_SimOpt["machineCost"]=machineCost
                VNS_SimOpt["num_SKU"]=len(FailureRates)
                VNS_SimOpt["utilization_rate"]= case['simulationGAresults']['utilization_rate']
                VNS_SimOpt["holding_cost_min"]= case['simulationGAresults']['holding_cost_min']
                #####

                ### risk averse parameter
                VNS_SimOpt["var_level"]= var_level*100
            
                Results.append(VNS_SimOpt)        
         
                

with open("db_"+str(var_level*100)+"_"+fname+'_RiskAverse_VNS_Priority_'+str(tot_cases)+'_instance.json', 'w') as outfile:
    json.dump(Results, outfile)
