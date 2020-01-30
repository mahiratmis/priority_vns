import numpy as np
import math
import sys
import json
import pandas as pd
import time
import random
import matplotlib.pyplot as plt
from deap import base
from deap import creator
from deap import tools
import itertools

import simulation_codes  #Andre's package that used in simulation opt.

# reproducability
random.seed(60)
np.random.seed(60)

lamda=0.5
var_level=0.5
prefix = "simple_ga_v{}_l{}_".format(int(var_level*100), int(lamda*100))  


def optimal_server_number(priority, FailureRates, ServiceRates, holding_costs, penalty_cost, skill_cost, machineCost):
    
    '''
    returns number of servers and total cost for allocating
    that number of servers 
    
    '''

    
    min_nserver=int(sum(np.array(FailureRates)/np.array(ServiceRates)))+1    #min required servers
    assignment=np.ones((min_nserver, len(FailureRates)))
    
    holding_backorder_CostList =simulation_codes.SimulationInterface.simulation_optimization_bathrun_priority_riskaverse(lamda, var_level, FailureRates, ServiceRates,holding_costs, penalty_cost, assignment,priority,
                                  numberWarmingUpRequests = 5000,
                                  stopNrRequests = 100000,
                                  replications =40)
                    
    Server_Cost=min_nserver*machineCost
    TotalCost=np.mean(holding_backorder_CostList)+Server_Cost
        
    while True:
        #print min_nserver
        min_nserver+=1 
        #print min_nserver
        assignment=np.ones((min_nserver, len(FailureRates)))
        
        temp_holding_backorder_CostList =simulation_codes.SimulationInterface.simulation_optimization_bathrun_priority_riskaverse(lamda, var_level,FailureRates, ServiceRates,holding_costs, penalty_cost, assignment,priority,
                                  numberWarmingUpRequests = 5000,
                                  stopNrRequests = 100000,
                                  replications =40)
        
        temp_Server_Cost=min_nserver*machineCost
        temp_TotalCost=np.mean(temp_holding_backorder_CostList)+temp_Server_Cost
    
        if temp_TotalCost > TotalCost:
            min_nserver-=1
            break
        
        else:
            TotalCost=temp_TotalCost
            Server_Cost= temp_Server_Cost
        
    
    return min_nserver, TotalCost, Server_Cost, TotalCost-Server_Cost


def Fitness(FailureRates, ServiceRates, holding_costs, penalty_cost, skillCost, machineCost,  priority):
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
    _, TotalCost, _, _=optimal_server_number(priority, FailureRates, ServiceRates, holding_costs, penalty_cost, skillCost, machineCost)
    return TotalCost,  
        

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


def GA_Priority(failure_rates, service_rates, holding_costs, penalty_cost, skill_cost, machine_cost, numPriorityClass=1): 
   
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
    # register the goal / fitness function
    toolbox.register("evaluate", Fitness, failure_rates, service_rates, holding_costs, penalty_cost, skill_cost, machine_cost)

    toolbox.register("mate", tools.cxOnePoint)
    # register a mutation operator with a probability to
    toolbox.register("mutate", swicthGen, numPriorityClass, 1)  #1=num gen mutate

    # operator for selecting individuals for breeding the next
    # generation: each individual of the current generation
    # is replaced by the 'fittest' (best) of three individuals
    # drawn randomly from the current generation.
    toolbox.register("select", tools.selTournament, tournsize=10)
    random.seed(64)
    
    start_time = time.time() #start time
    pop = toolbox.population(n=100)
    # CXPB  is the probability with which two individuals
    #       are crossed
    #
    # MUTPB is the probability for mutating an individual
    CXPB, MUTPB = 0.8, 0.4
    
   
    #ADD Hall of Fame for the next version !!!
    #hof = tools.HallOfFame(1)
    #print("Start of evolution")
    
    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    
    #print fitnesses
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    # Extracting all the fitnesses of 
    fits = [ind.fitness.values[0] for ind in pop]
    best_cost=min(fits)
    best_priority=tools.selBest(pop, 1)[0]
    
    # Variable keeping track of the number of generations
    gen_max = 50
    g=0
    
    # Begin the evolution
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
                toolbox.mate(child1, child2)
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
        
        # The population is entirely replaced by the offspring
        pop[:] = offspring
        
        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]
        record_of_min.append(min(fits)) #record min found in each iteration
        
        if min(fits)< best_cost:
            best_cost=min(fits)
            best_priority=tools.selBest(pop, 1)[0]
        
        best_cost_record.append(best_cost) #record of the best found so far
    
    stop_time = time.time() - start_time
    best_ind = tools.selBest(pop, 1)[0]
    return best_ind.fitness.values, best_ind, stop_time, best_cost_record, record_of_min, best_cost,best_priority


json_case=[]
with open("GAPoolingAll_4a.json", "r") as json_file:
    #json_file.readline()
    for line in json_file:
        json_case.append(json.loads(line))
        

#################INTEGRATING pruning and cache TO GA#####################################
tot_cases = 0
Results=[]
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
                        
                GA_SimOpt={}
                numPriorityClass=len(FailureRates) #making equal to number of SKUs
                        
                ##############GA RUN############
                print "running GA for", case["caseID"]
                cache={}
                best_cost, best_priority, run_time,  best_cost_record, record_of_min, _,_            =GA_Priority(FailureRates, ServiceRates,                                                       holding_costs, penalty_cost,                                                       skillCost, machineCost, numPriorityClass) 
                print best_cost, best_priority
                        

                ####################RECORDING RESULTS##################
                ###GA Outputs
                GA_SimOpt["GA_best_cost"]=best_cost[0]
                GA_SimOpt["GA_best_priority"]=best_priority
                GA_SimOpt["GA_run_time"]=run_time
                GA_SimOpt["best_cost_record"]=best_cost_record
                GA_SimOpt["record_of_min"]=record_of_min
                        
                        
                ###inputs
                GA_SimOpt["penalty_cost"]=penalty_cost  
                GA_SimOpt["skillCost"]=skillCost
                GA_SimOpt["machineCost"]=machineCost
                GA_SimOpt["num_SKU"]=len(FailureRates)
                GA_SimOpt["utilization_rate"]= case['simulationGAresults']['utilization_rate']
                GA_SimOpt["holding_cost_min"]= case['simulationGAresults']['holding_cost_min']
                #####
            
                ### risk averse parameter
                GA_SimOpt["var_level"]= var_level
                GA_SimOpt["lamda"]= lamda

                Results.append(GA_SimOpt)
                print case["caseID"], "run completed"
                
                
#analysis number of priority classes
#assigment per priority class
#min required sever vs optimized server number
#cost distribution machine vs holding vs backorder
#comparision with benchmark
#comparison with regular GA
with open(prefix+'Risk_Averse_Priority_16_instance.json', 'w') as outfile:
    json.dump(Results, outfile)