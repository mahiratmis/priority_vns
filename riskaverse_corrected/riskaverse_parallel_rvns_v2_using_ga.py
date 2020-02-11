# PRUNE and DATABASE available

import math
import json
import time
import random
from deap import base
from deap import creator
from deap import tools
import multiprocessing
import numpy as np
# import psutil 
from simulation_codes.SimulationInterface import simulation_optimization_bathrun_priority_riskaverse as sobpr

# reproducability
random.seed(60)
np.random.seed(60)

lamda = 0.5  # test 0, 0.5, 1
var_level = 0.05
case_start, case_end = 1, 4
prefix = "hybrid_parallel_vns_ga_db_v{}_l{}_p{}_{}_".format(int(var_level*100), 
                                                          int(lamda*100), 
                                                          case_start, 
                                                          case_end)

# num_cpus = psutil.cpu_count(logical=True)
# print(num_cpus)
n_cores = 2

MAX_ISLB = 1 # maximum iter_since_last_best default 100
MAX_SCC = 1 # maximum same_consecutive_count default 10

# 1 is for maximization -1 for minimization
# Minimize total cost just EBO cost and holding cost at the moment 
creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)


def optimal_server_number(priority, FailureRates, ServiceRates, holding_costs, penalty_cost, skill_cost, machineCost):
    '''
    returns number of servers and total cost for allocating
    that number of servers
    '''
    # min required servers
    min_nserver = int(sum(np.array(FailureRates)/np.array(ServiceRates)))+1
    assignment = np.ones((min_nserver, len(FailureRates)))
    holding_backorder_CostList = sobpr( lamda,
                                        var_level,
                                        FailureRates,
                                        ServiceRates,
                                        holding_costs,
                                        penalty_cost,
                                        assignment,
                                        priority,
                                        numberWarmingUpRequests=5000,
                                        stopNrRequests=100000,
                                        replications=5)

    Server_Cost = min_nserver*machineCost
    TotalCost = np.mean(holding_backorder_CostList)+Server_Cost

    while True:
        # print min_nserver
        min_nserver += 1
        # print min_nserver
        assignment = np.ones((min_nserver, len(FailureRates)))
        temp_holding_backorder_CostList = sobpr(lamda,
                                                var_level,
                                                FailureRates,
                                                ServiceRates,
                                                holding_costs,
                                                penalty_cost,
                                                assignment,
                                                priority,
                                                numberWarmingUpRequests=5000,
                                                stopNrRequests=100000,
                                                replications=5)

        temp_Server_Cost = min_nserver*machineCost
        temp_TotalCost = np.mean(temp_holding_backorder_CostList)+temp_Server_Cost

        if temp_TotalCost > TotalCost:
            min_nserver -= 1
            break
        else:
            TotalCost = temp_TotalCost
            Server_Cost = temp_Server_Cost
    return min_nserver, TotalCost, Server_Cost, TotalCost-Server_Cost


####PRUNE and DATABASE ###################
def set_multipliers(len_sku):
    # global multipliers
    multipliers = 2**np.arange(len_sku-1, -1, -1)
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
    multipliers = set_multipliers(len(x))
    return tuple([sum([multipliers[j] for j, c2 in enumerate(x) if c2 == c]) for c in set(x)])


def prune_and_evaluate(cache, priority, failure_rates, service_rates, holding_costs, penalty_cost, skill_cost, machine_cost):
    priority_rep = encode(priority)
    if priority_rep in cache.keys():
        return cache[priority_rep]
    # may need to check cache length
    _, TotalCost, _, _= optimal_server_number(priority, failure_rates, service_rates, holding_costs, penalty_cost, skill_cost, machine_cost)
    
    if len(cache)<10000: #len of cache
        cache[priority_rep]=TotalCost
    return TotalCost


def prune_and_evaluate_parallel(q, l, cache, priority, failure_rates, service_rates, holding_costs, penalty_cost, skill_cost, machine_cost):
    
    priority_rep=encode(priority)
    l.acquire()
    if priority_rep in cache.keys():
        q.put((priority, cache[priority_rep]))
        l.release()
    else:
        l.release()
        _, TotalCost, _, _= optimal_server_number(priority, failure_rates, service_rates, holding_costs, penalty_cost, skill_cost, machine_cost)
        l.acquire()
        if len(cache)<10000: #len of cache
            cache[priority_rep]=TotalCost
            q.put((priority, TotalCost))
        l.release()


def Fitness(cache, FailureRates, ServiceRates, holding_costs, penalty_cost, skillCost, machineCost, priority ):
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
    
    TotalCost = prune_and_evaluate(cache, priority, FailureRates, ServiceRates, holding_costs, penalty_cost, skillCost,  machineCost)
    return TotalCost,  


def generatePriority(numSKUs, numPriorityClass=1):
    '''
    -assing priority classes to each SKU randomly for a given number of priority class
    -returns a np array 
    '''
    
    return np.random.choice(np.arange(1, numPriorityClass+1), numSKUs)


# VNS Neighborhood Structures Start
def ns_throas_mutation(q, x, *args):
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
    q.put(x_new) 


def ns_center_inverse_mutation(q, x, *args):
    """ Randomly select a position i, mirror genes referencing i 
    Example: [1,2,3,4,5,6] if i is 3 result is [3,2,1,6,5,4]"""
    idx = random.randint(0, len(x)-1)
    q.put(x[idx::-1] + x[:idx:-1])   


def ns_two_way_swap(q, x, *args):
    """ Randomly swaps two elements of x. """
    x_new = x[:]
    n_skus = len(x_new)
    idx1, idx2 = random.sample(range(0, n_skus), 2)
    x_new[idx1] = x[idx2]
    x_new[idx2] = x[idx1]
    q.put(x_new)


def ns_shuffle(q, x, *args):
    """ Returns a permutation of elements of x. """
    x_new = x[:]
    random.shuffle(x_new)
    q.put(x_new)


def ns_mutate_random(q, x, min_cluster, max_cluster):
    """ Changes value of a random element of x. 
        The new values are between min_cluster and max_cluster inclusive. """
    x_new = x[:]
    n_skus = len(x_new)
    idx = random.randint(0, n_skus-1)
    ex_cluster_number = x[idx]
    numbers = list(range(min_cluster, ex_cluster_number)) + list(range(ex_cluster_number + 1, max_cluster))
    x_new[idx] = random.choice(numbers)
    q.put(x_new) #if filter_out_symmetric_solutions([x_new]) else ns_mutate_random(x, min_cluster, max_cluster)


# VNS Neighborhood Structures End
def ns_mutate_random2(q, x, min_cluster, max_cluster):
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
    
    q.put(x_new) #if filter_out_symmetric_solutions([x_new]) else ns_mutate_random2(x, min_cluster, max_cluster)


def solve_parallel_rvns_v2( cache, min_cluster, max_cluster, failure_rates, service_rates, holding_costs, penalty_cost, skill_cost, machine_cost, ngf, initial_priority=None ):
    """Finds best solution x given an initial solution x,
       list shaking functions ngf, and
    """
    if initial_priority is None:
        numSKUs = len(failure_rates)
        x = list(np.random.choice(np.arange(1, numSKUs+1), numSKUs))
    else:
        x = initial_priority
    tcost_x = prune_and_evaluate(cache, x, failure_rates, service_rates, holding_costs, penalty_cost, skill_cost, machine_cost)
    iter_since_last_best = 0
    same_consecutive_count = 0
    prev_best = 0
    q = multiprocessing.Queue()
    q2 = multiprocessing.Queue()
    lck = multiprocessing.Lock()
    # print("x", x, "ngf", ngf)
    k = len(ngf)
    while(iter_since_last_best < MAX_ISLB and same_consecutive_count < MAX_SCC):
        process_ngf_id = 0
        better_found = False
        local_better_found = True
        while local_better_found or process_ngf_id < k:
            local_better_found = False
            processes = []
            # half from first neighborhood
            for _ in range(n_cores//2):
                p = multiprocessing.Process(target=ngf[process_ngf_id],
                                                       args=(q, x, min_cluster, max_cluster))
                processes.append(p)
                p.start()
            # half from other neighborhood
            if process_ngf_id < k-1:
                process_ngf_id += 1
            for _ in range(n_cores//2):
                p = multiprocessing.Process(target=ngf[process_ngf_id],
                                                       args=(q, x, min_cluster, max_cluster))
                processes.append(p)
                p.start()
            
            # wait for processes to finish their tasks 
            for prcs in processes:
                prcs.join()

            # get results (generated priorities)
            x_list = []
            while not q.empty():
                x_list.append(q.get())
            
            # print("x_list", x_list)
            # calculate total costs
            processes2 = []
            for x_prime in x_list:
                p2 = multiprocessing.Process(target=prune_and_evaluate_parallel,
                                             args=(q2, lck, cache, x_prime, failure_rates, service_rates, holding_costs, penalty_cost, skill_cost, machine_cost))
                processes2.append(p2)
                p2.start()
            
            # wait for processes to finish their tasks 
            for prcs2 in processes2:
                prcs2.join()

            # get results (x_primes and their corresponding total costs)
            x_tc_list = []
            while not q2.empty():
                x_tc_list.append(q2.get())
            # print("x_tc_list", x_tc_list)
            # find min cost and priority
            min_x_prime, min_tcost = min(x_tc_list, key = lambda t : t[1])


            process_ngf_id += 1
            if min_tcost <= tcost_x:
                # print("=== NEW lower total cost: {:.4f}, iter_slb:{}".format(min_tcost, iter_since_last_best))
                x = min_x_prime
                tcost_x = min_tcost
                better_found = True
                local_better_found = True
                process_ngf_id = 0  # start from the first neighborhood
                if prev_best == min_tcost :
                    same_consecutive_count += 1
                else:
                    same_consecutive_count = 0
                    prev_best = min_tcost              
        
        # check for improvement
        if not better_found:
            iter_since_last_best += 1
        else:
            iter_since_last_best = 0
    return tcost_x , x, cache


def NGF_Fitness(cache, min_cluster, max_cluster, failure_rates, service_rates, holding_costs, penalty_cost, skill_cost, machine_cost, ngf, initial_priority=None ):
    t_cost, _, _ = solve_parallel_rvns_v2(cache, min_cluster, max_cluster, failure_rates, service_rates, holding_costs, penalty_cost, skill_cost, machine_cost, ngf, initial_priority)
    return (t_cost,)


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


"""
pop_max_fitness_vals={}
def population_fitness(cache, min_cluster, max_cluster, failure_rates, service_rates, holding_costs, penalty_cost, skill_cost, machine_cost, f, g, population, ch_i):
    best_tc = float('-inf') 
    numSKUs = len(failure_rates)
    print(g)
    # if max val of this generation is computed dont recalculate
    if g in pop_max_fitness_vals:
        best_tc = pop_max_fitness_vals[g]
    else:
        for j, ch_j in enumerate(population):
            ngf = [ngfs[i] for i in set(ch_j)]
            print(j, ngf)
            initial_priority = list(np.random.choice(np.arange(1, numSKUs+1), numSKUs))
            tc_j, _, _ = solve_parallel_rvns_v2(cache, initial_priority, ngf, min_cluster, max_cluster, failure_rates, service_rates, holding_costs, penalty_cost, skill_cost, machine_cost) 
            print(tc_j)
            if tc_j > best_tc:
                best_tc = tc_j
        pop_max_fitness_vals[g] = best_tc
        
    # calculate total cst for ch_i
    ngf = [ngfs[i] for i in set(ch_i)]
    initial_priority = list(np.random.choice(np.arange(1, numSKUs+1), numSKUs))
    tc_i, _, _ = solve_parallel_rvns_v2(cache, initial_priority, ngf, min_cluster, max_cluster, failure_rates, service_rates, holding_costs, penalty_cost, skill_cost, machine_cost)
    print("Tci",tc_i, "Best TC", best_tc, "Res", max(0, f * best_tc - tc_i))
    return max(0, f * best_tc - tc_i)

def parent_selection_prob(cache, min_cluster, max_cluster, failure_rates, service_rates, holding_costs, penalty_cost, skill_cost, machine_cost, f, population, ch_i):
    sum_pop = sum([population_fitness(cache, min_cluster, max_cluster, failure_rates, service_rates, holding_costs, penalty_cost, skill_cost, machine_cost, f, population, ch_j) for ch_j in population])
    f_ci = population_fitness(cache, min_cluster, max_cluster, failure_rates, service_rates, holding_costs, penalty_cost, skill_cost, machine_cost, f, population, ch_i)
    return f_ci / sum_pop
"""


def VNS_Priority(cache, failure_rates, service_rates, holding_costs, penalty_cost, skill_cost, machine_cost, numPriorityClass=1): 

    def generatePriority(numSKUs, numPriorityClass=1):
        '''
        -assing priority classes to each SKU randomly for a given number of priority class
        -returns a np array 
        '''
        return creator.Individual(np.random.choice(np.arange(1, numPriorityClass+1), numSKUs))

    def generateNGFS(numNGFs):
        '''
        -assing priority classes to each SKU randomly for a given number of priority class
        -returns a np array 
        '''
        return creator.Individual(np.random.choice(np.arange(numNGFs), numNGFs))

    toolbox = base.Toolbox()
    
    toolbox.register("individual", generatePriority, len(failure_rates), numPriorityClass)
    # define the population to be a list of individuals
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    # register the goal / fitness function
    toolbox.register("evaluate", Fitness, cache, failure_rates, service_rates, holding_costs, penalty_cost, skill_cost, machine_cost)
    # operator for selecting individuals for breeding the next
    # generation: each individual of the current generation
    # is replaced by the 'fittest' (best) of three individuals
    # drawn randomly from the current generation.
    toolbox.register("select", tools.selTournament, tournsize=10)

    # hybrid vns ga setup     
    toolbox2 = base.Toolbox()
    #function to generate individuals and individual length
    toolbox2.register("individual", generateNGFS, len(ngfs))
    toolbox2.register("population", tools.initRepeat, list, toolbox2.individual)
    toolbox2.register("evaluate", NGF_Fitness, cache, 2, len(failure_rates), failure_rates, service_rates, holding_costs, penalty_cost, skill_cost, machine_cost)
    toolbox2.register("mate", tools.cxOnePoint)
    toolbox2.register("mutate", swicthGen, len(ngfs), 1)  #1=num gen mutate
    toolbox2.register("select", tools.selTournament, tournsize=10)
    start_time = time.time()   
    # initial population of n individuals
    Xg = toolbox2.population(n=3)
    # MUTPB is the probability for mutating an individual
    CXPB, MUTPB = 0.8, 0.4
    # Evaluate entire population fitness
    fitnesses = [toolbox2.evaluate([ngfs[i] for i in set(ngf)]) for ngf in Xg]
    for ind, fit in zip(Xg, fitnesses):
        ind.fitness.values = fit

    # # Extract fitnesses values 
    fits = [ind.fitness.values[0] for ind in Xg]
    # Variable keeping track of the number of generations
    gen_max = 2
    # Begin the evolution
    record_of_min=[]
    best_cost_record=[]
    best_cost=min(fits)
    best_ngf=list(set(tools.selBest(Xg, 1)[0]))
    print("Initial Best cost", best_cost, "Best ngf", best_ngf)
    for g in range(1, gen_max+1):
        print ("generation: ", g)
        
        # Select the next generation individuals
        offspring = toolbox2.select(Xg, len(Xg))
        # Clone the selected individuals
        offspring = list(map(toolbox2.clone, offspring))
    
        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            # cross two individuals with probability CXPB
            if random.random() < CXPB:
                toolbox2.mate(child1, child2)
                # fitness values of the children
                # must be recalculated later
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            # mutate an individual with probability MUTPB
            if random.random() < MUTPB:
                mutant=toolbox2.mutate(mutant)
                del mutant.fitness.values
    
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = [toolbox2.evaluate([ngfs[i] for i in set(ngf)]) for ngf in Xg]
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        # The population is entirely replaced by the offspring
        Xg[:] = offspring
        
        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in Xg]
        record_of_min.append(min(fits)) #record min found in each iteration
        
        if min(fits)< best_cost:
            best_cost=min(fits)
        
        best_cost_record.append(best_cost) #record of the best found so far
    
    best_ngf = tools.selBest(Xg, 1)[0]
    best_ngf = [ngfs[i] for i in set(ngf)]
    print("Best cost of GA best neighbor ", best_cost, best_ngf)
    best_briority = []
    best_cost, best_priority,cache = solve_parallel_rvns_v2(cache, 2, len(failure_rates), failure_rates, service_rates, holding_costs, penalty_cost, skill_cost, machine_cost, best_ngf)
    stop_time = time.time() - start_time
    print(best_cost, best_priority, "time", stop_time)
    
    return best_cost, best_priority, stop_time, cache


json_case=[]
with open("GAPoolingAll_4a.json", "r") as json_file:
    #json_file.readline()
    for line in json_file:
        json_case.append(json.loads(line))
        
# shaking functions
indices = [0,1]
fname = "".join(map(str,indices))
fname += '_RiskAverse_Priority_16_instance.json'
ngfs = [ns_mutate_random, ns_mutate_random2, ns_shuffle, ns_two_way_swap, ns_throas_mutation, ns_center_inverse_mutation]
# ngfs = [ngfs[i] for i in indices]



#################INTEGRATING pruning and cache TO GA#####################################
Results=[]
tot_cases = 0
for case in json_case[0]:
    if case['simulationGAresults']['holding_costs_variant']==2: #HPB cost variant
        if case['simulationGAresults']['skill_cost_factor']==0.1:
            if case['simulationGAresults']['utilization_rate'] ==0.8:
                tot_cases += 1 
                if tot_cases<case_start or tot_cases>case_end:
                    continue         
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

                ### risk averse parameters
                VNS_SimOpt["var_level"]= var_level
                VNS_SimOpt["lamda"]= lamda

                Results.append(VNS_SimOpt)        
                print case["caseID"], "run completed"
  
with open(prefix+fname, 'w') as outfile:
    json.dump(Results, outfile)
