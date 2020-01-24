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
        
        min_nserver, TotalCost, Server_Cost, _=optimal_server_number(priority, failure_rates, service_rates, holding_costs, penalty_cost, skill_cost, machine_cost)
        
        
        if len(cache)<10000: #len of cache
        
            cache[priority_rep]=TotalCost
        
            #print "update cache"
        #else:
            
            #print "cache is full"
        
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


def GA_Priority(cache, failure_rates, service_rates, holding_costs, penalty_cost, skill_cost, machine_cost, numPriorityClass=1): 
   
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
    
    
    toolbox.register("mate", tools.cxTwoPoint)

    
    
    toolbox.register("mate2", tools.cxUniform)
    
    
    # register a mutation operator with a probability to
    toolbox.register("mutate", chooseMutation, numPriorityClass)  #1=num gen mutate
    
    #swicthGen(priority, numPriorityClass=1, n=1)
    #toolbox.register("mutate", swicthtoOtherMutation)

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
    
    pop = [creator.Individual(x) for x in population_generator(failure_rates, service_rates, holding_costs, penalty_cost, skill_cost, machine_cost)]
             
    pop=pop+toolbox.population(n=100-len(pop))
   
    print "Population", pop
    # CXPB  is the probability with which two individuals
    #       are crossed
    #
    # MUTPB is the probability for mutating an individual
    CXPB, MUTPB = 0.8, 0.4
    CXPB_type=0.5 #crossover type selection probability 
    
    indp=0.5 #probability of swap genes for uniform crossover
    
    #ADD Hall of Fame for the next version !!!
    #hof = tools.HallOfFame(1)
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
    
    gen_max = 50
    g=0
    
    # Begin the evolution
    #while max(fits) < 100 and g < 100:

   
    record_of_min=[]
    best_cost_record=[best_cost]
    
    while g < gen_max: #max generation here
        # A new generation
        g = g + 1
        
        print "generation: ", g
        #print("-- Generation %i --" % g)
        
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))
    
        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # cross two individuals with probability CXPB
            if random.random() < CXPB:
                if random.random() <CXPB_type:
                    toolbox.mate(child1, child2)
                else:
                    
                    toolbox.mate2(child1, child2, indp)
                    
                    
                # fitness values of the children
                # must be recalculated later
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:

            # mutate an individual with probability MUTPB
            if random.random() < MUTPB:
                
                mutant=toolbox.mutate(mutant)
                
                del mutant.fitness.values
    
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        #print("  Evaluated %i individuals" % len(invalid_ind))
        
        # The population is entirely replaced by the offspring
        pop[:] = offspring
        
        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]
        
        record_of_min.append(min(fits)) #record min found in each iteration
        
        if min(fits)< best_cost:
            best_cost=min(fits)
            best_priority=tools.selBest(pop, 1)[0]
        
        best_cost_record.append(best_cost) #record of the best found so far
        
        
        #length = len(pop)
        #mean = sum(fits) / length
        #sum2 = sum(x*x for x in fits)
        #std = abs(sum2 / length - mean**2)**0.5
        
        #print("  Min %s" % min(fits))
        
    #    print("  Max %s" % max(fits))
    #    print("  Avg %s" % mean)
    #    print("  Std %s" % std)
    
    #print("-- End of (successful) evolution --")
    
    stop_time = time.time() - start_time
    
    best_ind = tools.selBest(pop, 1)[0]
    
    return best_ind.fitness.values, best_ind, stop_time, best_cost_record, record_of_min, best_cost,best_priority, cache

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
                    
            GA_SimOpt={}
            numPriorityClass=len(FailureRates) #making equal to number of SKUs
                    
            ##############GA RUN############
            print "running GA for", case["caseID"]
            cache={}
            best_cost, best_priority, run_time,  best_cost_record, record_of_min, _,_, cache            =GA_Priority(cache, FailureRates, ServiceRates,                                                       holding_costs, penalty_cost,                                                       skillCost, machineCost, numPriorityClass) 
            print best_cost, best_priority
            print "GA is over"
            print "=============="
                    
            # start_time = time.time() #start time
            # list_priority=[best_priority] #selection procedure is needed here !!!
            # global_best_cost, global_best_priority, best_priority_list=local_search(list_priority,  cache, best_cost, FailureRates, ServiceRates, holding_costs, penalty_cost, skillCost, machineCost)
            # end_time=time.time()-start_time
            # print global_best_cost, global_best_priority, best_priority_list

            ####################RECORDING RESULTS##################
            ###GA Outputs
            GA_SimOpt["GA_best_cost"]=best_cost[0]
            GA_SimOpt["GA_best_priority"]=best_priority
            GA_SimOpt["GA_run_time"]=run_time
            GA_SimOpt["best_cost_record"]=best_cost_record
            GA_SimOpt["record_of_min"]=record_of_min
                    
            ###Local Search Outputs
            # GA_SimOpt["global_best_cost"]=global_best_cost
            # GA_SimOpt["global_best_priority"]=global_best_priority
            # GA_SimOpt["best_priority_list"]=best_priority_list
            # GA_SimOpt["global_run_time"]=run_time+end_time
                    
            ###inputs
            GA_SimOpt["CaseID"]=case["caseID"]
            GA_SimOpt["FailureRates"]=FailureRates.tolist()
            GA_SimOpt["ServiceRates"]=ServiceRates.tolist()
            GA_SimOpt["holding_costs"]=holding_costs.tolist()
            GA_SimOpt["penalty_cost"]=penalty_cost  
            GA_SimOpt["skillCost"]=skillCost
            GA_SimOpt["machineCost"]=machineCost
            GA_SimOpt["num_SKU"]=len(FailureRates)
            GA_SimOpt["utilization_rate"]= case['simulationGAresults']['utilization_rate']
            GA_SimOpt["holding_cost_min"]= case['simulationGAresults']['holding_cost_min']
            #####
        
            Results.append(GA_SimOpt)
            print "statistics cached for run"
                
                
#analysis number of priority classes
#assigment per priority class
#min required sever vs optimized server number
#cost distribution machine vs holding vs backorder
#comparision with benchmark
#comparison with regular GA

with open('Improved_GA_Priority_32_instance.json', 'w') as outfile:
    json.dump(Results, outfile)
