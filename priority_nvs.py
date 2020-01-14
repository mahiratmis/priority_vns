# To add a new markdown cell, type '# %% [markdown]'

#Capacity Optimizer ADDED
#LOCAL SEARCH ADDED
#INITAL POPULATION GENERATOR ADDED
#PRUNE AND DATABASE added to Local search
#PRUNE and DATABASE added to GA
#Database size set to 10000 over this limit not recorded to cache


####Running condations########
#100 pop size 50 gen size
#100000 simulation length 5000 warmup period
#10 replications
#0.8 crossover prob -0.4 mutation prob. 
#equal probability to choose any crossover and mutation type
#tournoment size 10


import numpy as np
import math
import sys
import json
import pandas as pd
import time
#from bokeh.charts import BoxPlot, output_notebook, show
#import seaborn as sns
#import pulp
#from pulp import *
import time
#from gurobipy import *
import random
import matplotlib.pyplot as plt


from deap import base
from deap import creator
from deap import tools

import itertools

import simulation_codes  #Andre's package that used in simulation opt.
#import scipy.stats as st


###Prune function has to defined earlier 
def optimal_server_number(priority, FailureRates, ServiceRates, holding_costs, penalty_cost, skill_cost, machineCost):
    
    '''
    returns number of servers and total cost for allocating
    that number of servers 
    
    '''

    min_nserver=int(sum(np.array(FailureRates)/np.array(ServiceRates)))+1    #min required servers
    
    assignment=np.ones((min_nserver, len(FailureRates)))
    
    
    holding_backorder_CostList =                     simulation_codes.SimulationInterface.simulation_optimization_bathrun_priority(FailureRates, ServiceRates,                                            holding_costs, penalty_cost, assignment,                                             priority,
                                  numberWarmingUpRequests = 5000,
                                  stopNrRequests = 100000,
                                  replications =10)
                    
    Server_Cost=min_nserver*machineCost
    
    TotalCost=np.mean(holding_backorder_CostList)+Server_Cost
    
        
    while True:
        #print min_nserver
        min_nserver+=1 
        #print min_nserver
        assignment=np.ones((min_nserver, len(FailureRates)))
        
        temp_holding_backorder_CostList =                     simulation_codes.SimulationInterface.simulation_optimization_bathrun_priority(FailureRates, ServiceRates,                                            holding_costs, penalty_cost, assignment,                                             priority,
                                  numberWarmingUpRequests = 5000,
                                  stopNrRequests = 100000,
                                  replications =10)
        
        temp_Server_Cost=min_nserver*machineCost
        temp_TotalCost=np.mean(temp_holding_backorder_CostList)+temp_Server_Cost
    
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
    
    priority_rep=encode(priority)
    
    #print priority_rep
    
    if priority_rep in cache.keys():
        
        #print "in cache"
        
        return cache[priority_rep]
    
    else:
        
        #may need to check cache length
        #print "update cache"
        
        _, TotalCost, _, _=optimal_server_number(priority, failure_rates, service_rates, holding_costs, penalty_cost, skill_cost, machine_cost)
        
        
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
    
    #take cache as input and use 
    
    TotalCost=prune_and_evaluate(priority, cache, FailureRates, ServiceRates, holding_costs, penalty_cost, skillCost,  machineCost)
    
    print "After Prune Final number: ", TotalCost
    return TotalCost,  
#we may add invenorty and other cost at the end  of optimization with one more run for optimized schedules 
#priority, totalCostList, 
#priority, totalCostList, 
#DONT FORGET COME AT THE END!!!
        


def generatePriority(numSKUs, numPriorityClass=1):
    '''
    -assing priority classes to each SKU randomly for a given number of priority class
    -returns a np array 
    '''
    
    return np.random.choice(np.arange(1, numPriorityClass+1), numSKUs)


def swicthGen(numPriorityClass,  n, priority):
    
    #to keep orginal probabilty of switching to other cluster during iteration
    
    #when n=1 only one gen is switched 

    priority_new = priority[:]

    numSKUs = len(priority_new)

    idx1, idx2 = random.sample(range(0, numSKUs), 2)

    ex_priority_number = priority[idx1]

    #excluding current priority class
    
    numbers = [x for x in range(1,numPriorityClass+1) if x != ex_priority_number]
    
    priority_new[idx1] = random.choice(numbers)

    if n == 2:

        ex_priority_number = priority[idx2]

        numbers = [x for x in range(1,numPriorityClass+1) if x != ex_priority_number]

        priority_new[idx2] = random.choice(numbers)

    return creator.Individual(priority_new)
    


def shuffleGen(priority):
    priority_new = priority[:]
    
    random.shuffle(priority_new )
    
    return creator.Individual(priority_new) 

def swapGen(priority):
    
    priority_new = priority[:]
    numSKUs = len(priority_new)
    idx1, idx2 = random.sample(range(0, numSKUs), 2)
    priority_new[idx1] = priority[idx2]
    priority_new[idx2] = priority[idx1]
    return creator.Individual(priority_new)

def chooseMutation(numPriorityClass, priority): 
    r = random.uniform(0, 1)
    if r <= 0.25: 
        return shuffleGen(priority)
    
    if r <= 0.5:
        return swapGen(priority)
    
    if r <= 0.75:
        return swicthGen(numPriorityClass,  1, priority)

    return swicthGen(numPriorityClass,  2, priority)


#########Generating different populations###########
#1-FCFS->
#2-Shortest and longest proccessing->
#3-min and max holding cost->
#4-hxmu long lowest and highest->
#5-2 priority class versions of all above 



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
    
    return x_new #if filter_out_symmetric_solutions([x_new]) else ns_mutate_random2(x, min_cluster, max_cluster)


def population_generator(failure_rates, service_rates, holding_costs, penalty_cost, skill_cost, machine_cost):
    '''
    
    '''
    population=[]
    
    fcfs_rule=np.ones(len(failure_rates))
    
    population.append(fcfs_rule)
    
    ####faliure rate#########################
    
    priority=np.ones(len(failure_rates))
    
    k=1
    for i in np.argsort(failure_rates):
        priority[i]=k
        k+=1
    
    population.append(priority)
    population.append([2 if x >= len(priority)/2.0 else 1 for x in priority]) #two class based on failure rates
   
    
    priority=np.ones(len(failure_rates))
    
    k=1
    for i in np.flip(np.argsort(failure_rates),0):
        priority[i]=k
        k+=1
    
    population.append(priority)
    population.append([2 if x >= len(priority)/2.0 else 1 for x in priority])
    #####service rate#######################
    
    priority=np.ones(len(service_rates))
    
    k=1
    for i in np.argsort(service_rates):
        priority[i]=k
        k+=1
    
    population.append(priority)
    population.append([2 if x >= len(priority)/2.0 else 1 for x in priority])
    
    priority=np.ones(len(service_rates))
    
    k=1
    for i in np.flip(np.argsort(service_rates),0):
        priority[i]=k
        k+=1
    
    population.append(priority)
    population.append([2 if x >= len(priority)/2.0 else 1 for x in priority])
    ###holding cost#######################
    
    priority=np.ones(len(holding_costs))
    
    k=1
    for i in np.argsort(holding_costs):
        priority[i]=k
        k+=1
    
    population.append(priority)
    population.append([2 if x >= len(priority)/2.0 else 1 for x in priority])
    
    
    
    priority=np.ones(len(holding_costs))
    
    k=1
    for i in np.flip(np.argsort(holding_costs),0):
        priority[i]=k
        k+=1
    
    population.append(priority)
    population.append([2 if x >= len(priority)/2.0 else 1 for x in priority])
    
    ###holding *service rate##############
    
    priority=np.ones(len(holding_costs))
    
    k=1
    for i in np.argsort(np.array(holding_costs)*np.array(service_rates)):
        priority[i]=k
        k+=1
    
    population.append(priority)
    population.append([2 if x >= len(priority)/2.0 else 1 for x in priority])
    
    priority=np.ones(len(holding_costs))
    
    k=1
    for i in np.flip(np.argsort(np.array(holding_costs)*np.array(service_rates)),0):
        priority[i]=k
        k+=1
    
    population.append(priority)
    population.append([2 if x >= len(priority)/2.0 else 1 for x in priority])
    
    return [list(x) for x in population]


def solve_rvns( cache, initial_priority, nsf, min_cluster, max_cluster, failure_rates, service_rates, holding_costs, penalty_cost, skill_cost, machine_cost, max_iters=1000):
    """Finds best solution x given an initial solution x,
       list shaking functions nsf, and
       list of neighbor list generation functions  nlgf. """
    #logging.basicConfig(filename=f'app_rvns{fname_possix}.log', level=logging.DEBUG, format=f'%(levelname)s - %(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
    x = initial_priority 
    tcost_x = prune_and_evaluate(x, cache, failure_rates, service_rates, holding_costs, penalty_cost, skill_cost, machine_cost)
    #total_search_time=0.
    #pbar = tqdm.tqdm(total = 200, desc=f"{case_id}")
    iter_since_last_best = 0
    epoch=0
    same_consecutive_count = 0
    prev_best = 0
    while(iter_since_last_best < 100 and same_consecutive_count < 5 ):
        k = 0
        epoch += 1
        better_found = False
        #start_time = time.time()
        while k < len(nsf):
            # create neighborhood solution using kth ngf
            x1 = nsf[k](x, min_cluster, max_cluster)
            tcost_x1 = prune_and_evaluate(x1, cache, failure_rates, service_rates, holding_costs, penalty_cost, skill_cost, machine_cost)
            if tcost_x1 <= tcost_x:
                #logging.debug(f"{case_id} === NEW lower total cost: {tcost_x1:.4f} epoch{epoch} ===")
                print("=== NEW lower total cost: {:.4f}".format(tcost_x1))
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
            #pbar.update(1)
        else:
            iter_since_last_best = 0
            #pbar.close()
            #pbar = tqdm.tqdm(total = 250, desc=f"{case_id}")

        #stop_time = time.time() - start_time
        #total_search_time += stop_time
        #logging.debug(f"{case_id} epoch{epoch} time:{stop_time:.4f} total search time:{total_search_time:.4f} min_cost:{tcost_x}")  
    #pbar.close()
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

    # the goal ('fitness') function to be maximized
    #for objective function call pooling optimizer !!!
    # what values need for optimizer !!!
    
    
    #def evalOneMax(individual):
    #    return sum(individual),

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

    random.seed(64)

    # create an initial population of 50 individuals (where
    # each individual is a list of integers)
    
    start_time = time.time() #start time
    
    # pop = [creator.Individual(x) for x in population_generator(failure_rates, service_rates, holding_costs, penalty_cost, skill_cost, machine_cost)]
             
    pop = toolbox.population(n=1)
   
    print "Population", len(pop), pop
    
    #print("Start of evolution")
    
    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    
    print "Fitness", fitnesses
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    
    print("  Evaluated %i individuals" % len(pop))

    # Extracting all the fitnesses of 
    fits = [ind.fitness.values[0] for ind in pop]
    best_cost=min(fits)
    best_priority=tools.selBest(pop, 1)[0]
    
    print best_priority 
    # Variable keeping track of the number of generations

    best_cost, best_priority, cache = solve_rvns(cache, best_priority, nsf, 2, len(failure_rates), failure_rates, service_rates, holding_costs, penalty_cost, skill_cost, machine_cost)

    stop_time = time.time() - start_time
    
    best_ind = tools.selBest(pop, 1)[0]
    
    return best_cost,best_priority, stop_time, cache

#returns cache now

#if __name__ == "__main__":
#    main()







def local_search(list_priority, cache, global_best_cost, failure_rates, service_rates, holding_costs, penalty_cost, skill_cost, machine_cost):
    
    '''
    list_priority: list of priority assigments that want to be included in the local search
    
    returns: 
    
    '''
    
    best_priority_list=[]
    
    #global_best_cost=float("inf")
    
    global_best_priority=[]
    
    global_best_priority[:]=list_priority[0]
    
    
    #print cache 
    
    for priority in list_priority:
        
        
        TotalCost=prune_and_evaluate(priority, cache, failure_rates, service_rates, holding_costs, penalty_cost, skill_cost, machine_cost)
         
       
        best_priority=[]
        
        temp_best_priority=[]
        
        best_priority[:]=priority
        
        temp_best_priority[:]=priority
        
        best_TotalCost=TotalCost
        
        gene_set=set(priority)
        
        #print gene_set
        
        #print  best_TotalCost, global_best_cost
        
        for i in range(len(priority)):
            
            for gene in gene_set:
                
                temp_best_priority[:]=best_priority
                
                if best_priority[i]!=gene:
                    
                    temp_best_priority[i]=gene
                    
                    TotalCost=prune_and_evaluate(temp_best_priority, cache, failure_rates, service_rates, holding_costs, penalty_cost, skill_cost, machine_cost)
        
                  
                    if TotalCost < best_TotalCost:
                        
                        best_TotalCost=TotalCost
                        
                        best_priority[:]=temp_best_priority
                        
                        if best_TotalCost< global_best_cost:
                            
                            global_best_cost= best_TotalCost
                            
                            global_best_priority[:] =best_priority
                
                #print best_TotalCost, global_best_cost      
                    
            if len(gene_set) < len(failure_rates): # to assign an independed priority class!
                
                diff_priority=set(range(1,len(failure_rates)+1)).difference(gene_set)
                
                for diff in [max(diff_priority), min(diff_priority)]:
                    
                    temp_best_priority[:]=best_priority
                
                    temp_best_priority[i]=diff
                
                    TotalCost=prune_and_evaluate(temp_best_priority, cache, failure_rates, service_rates, holding_costs, penalty_cost, skill_cost, machine_cost)
        
                
                    if TotalCost < best_TotalCost:
                
                        best_TotalCost=TotalCost
                        
                        best_priority[:]=temp_best_priority
                        
                        if best_TotalCost< global_best_cost:
                            
                            global_best_cost= best_TotalCost
                            
                            global_best_priority[:] =best_priority
                            
  
        best_priority_list.append([best_priority, best_TotalCost])       
                
    return global_best_cost, global_best_priority, best_priority_list
    


json_case=[]
with open("GAPoolingAll_4a.json", "r") as json_file:
    #json_file.readline()
    for line in json_file:
        json_case.append(json.loads(line))
        
# shaking functions
nsf = [ns_mutate_random, ns_mutate_random2, ns_shuffle, ns_two_way_swap, ns_throas_mutation, ns_center_inverse_mutation]

#################INTEGRATING pruning and cache TO GA#####################################

Results=[]
for case in json_case[0]:
    if case['simulationGAresults']['holding_costs_variant']==2: #HPB cost variant
        if case['simulationGAresults']['skill_cost_factor']==0.1:
                    
            FailureRates=np.array(case['simulationGAresults']["failure_rates"])
            ServiceRates=np.array(case['simulationGAresults']["service_rates"])
            holding_costs=np.array(case['simulationGAresults']["holding_costs"])
            penalty_cost=case['simulationGAresults']["penalty_cost"]
            skillCost=100 #NOT USING THIS ATM
            machineCost=case['simulationGAresults']['machine_cost']
                    
            VNS_SimOpt={}
            numPriorityClass=len(FailureRates) #making equal to number of SKUs
                    
            ##############GA RUN############
            print "running VNS"
            print case["caseID"]
            cache={}
            best_cost, best_priority, run_time, cache  =VNS_Priority(cache, FailureRates, ServiceRates,holding_costs, penalty_cost,skillCost, machineCost, numPriorityClass) 
                    
            print best_cost, best_priority
                    
            print "VNS is over"
                    
            print "=============="
                    
            start_time = time.time() #start time
    
            list_priority=[best_priority] #selection procedure is needed here !!!
                    
            global_best_cost, global_best_priority, best_priority_list=local_search(list_priority,  cache, best_cost, FailureRates, ServiceRates, holding_costs, penalty_cost, skillCost, machineCost)
        
            end_time=time.time()-start_time
            
            print global_best_cost, global_best_priority, best_priority_list

            ####################RECORDING RESULTS##################
            ###GA Outputs
            VNS_SimOpt["VNS_best_cost"]=best_cost
            VNS_SimOpt["VNS_best_priority"]=best_priority
            VNS_SimOpt["VNS_run_time"]=run_time
                    
            ###Local Search Outputs
            VNS_SimOpt["global_best_cost"]=global_best_cost
            VNS_SimOpt["global_best_priority"]=global_best_priority
            VNS_SimOpt["best_priority_list"]=best_priority_list
            VNS_SimOpt["global_run_time"]=run_time+end_time
                    
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
        
            Results.append(VNS_SimOpt)
                   
            print "run completed"
                
                
#analysis number of priority classes
#assigment per priority class
#min required sever vs optimized server number
#cost distribution machine vs holding vs backorder
#comparision with benchmark
#comparison with regular GA

with open('Improved_VNS_Priority_32_instance.json', 'w') as outfile:
    json.dump(Results, outfile)


json_case2=[]
with open("Improved_GA_Priority_32_instance.json", "r") as json_file2:
    #json_file.readline()
    for line in json_file2:
        json_case2.append(json.loads(line))
        


df=pd.DataFrame.from_dict(json_case2[0])


df.keys()


df['global_best_cost']=df['global_best_cost'].map(lambda x:  x[0] if type(x)==list else x)


df[["CaseID","GA_best_cost", "global_best_cost"]]

