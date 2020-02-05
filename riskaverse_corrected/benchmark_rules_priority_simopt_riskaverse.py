#FCFS, SPT, H rule and HXmu rules analyzed under priority

#Risk aversion, var_level=5 
import numpy as np
import math
import sys
import json
import pandas as pd
import time
import time
import random
import matplotlib.pyplot as plt
from deap import base
from deap import creator
from deap import tools
import itertools

import simulation_codes  #Andre's package that used in simulation opt.


###Prune function has to defined earlier 
def optimal_server_number3(priority, FailureRates, ServiceRates, holding_costs, penalty_cost, skill_cost, machineCost):
    
    '''
    returns number of servers and total cost for allocating
    that number of servers 
    
    '''

    min_nserver=int(sum(np.array(FailureRates)/np.array(ServiceRates)))+1    #min required servers
    Min_server=min_nserver
    assignment=np.ones((min_nserver, len(FailureRates)))
    
    
    holding_backorder_CostList =simulation_codes.SimulationInterface.simulation_optimization_bathrun_priority_riskaverse(lamda, var_level, FailureRates, ServiceRates,holding_costs, penalty_cost, assignment,priority,
                                  numberWarmingUpRequests = 5000,
                                  stopNrRequests = 100000,
                                  replications =40)
                    
    Server_Cost=min_nserver*machineCost
    TotalCost=np.mean(holding_backorder_CostList)+Server_Cost
  
    return Min_server, min_nserver, TotalCost, Server_Cost, TotalCost-Server_Cost

#########Generating different populations###########
#1-FCFS->
#2-Shortest and longest proccessing->
#3-min and max holding cost->
#4-hxmu long lowest and highest->
def population_generator2(failure_rates, service_rates, holding_costs, penalty_cost, skill_cost, machine_cost):
    '''
    
    '''
    population=[]
    fcfs_rule=np.ones(len(failure_rates))
    population.append(fcfs_rule)
    #####service rate#######################SPT
    ###Fast repairing items have higher priority######
    priority=np.ones(len(service_rates))
    
    k=1
    for i in np.flip(np.argsort(service_rates),0):
        priority[i]=k
        k+=1
    
    population.append(priority)
    
    ###holding cost#######################
    ###Expensive items have higher priortiy##############
    priority=np.ones(len(holding_costs))
    
    k=1
    for i in np.argsort(holding_costs):
        priority[i]=k
        k+=1
    
    population.append(priority)
     
    ###holding *service rate##############
    priority=np.ones(len(holding_costs))
    
    k=1
    for i in np.argsort(np.array(holding_costs)*np.array(service_rates)):
        priority[i]=k
        k+=1
    
    population.append(priority)
    
    return [list(x) for x in population]


lamda=0.5 #test 0, 0.5, 1
var_level=0.05


json_case2=[]
with open("./riskaverse/results/riskaverse.json", "r") as json_file2:
    #json_file.readline()
    for line in json_file2:
        json_case2.append(json.loads(line))

df=pd.DataFrame.from_dict(json_case2[0])
#df.keys()
# df['global_best_cost']=df['global_best_cost'].map(lambda x:  x[0] if type(x)==list else x)
#df[["CaseID","GA_best_cost", "global_best_cost"]]


SPT=[]
H_rule=[]
HMU_rule=[]
FCFS=[]

for i in range(len(df)):
    if df['utilization_rate'][i]==0.80:
        FailureRates=np.array(df["FailureRates"][i])
        ServiceRates=np.array(df["ServiceRates"][i])
        holding_costs=np.array(df["holding_costs"][i])
        penalty_cost=df["penalty_cost"][i]
        skill_cost=df["skillCost"][i] 
        machineCost=df["machineCost"][i]
    
        pop=population_generator2(FailureRates, ServiceRates, holding_costs, penalty_cost, skill_cost, machineCost)
        _, _, TotalCost, _, _=optimal_server_number3(pop[0], FailureRates, ServiceRates, holding_costs, penalty_cost, skill_cost, machineCost)
        FCFS.append(TotalCost)
    
        _, _, TotalCost, _, _=optimal_server_number3(pop[1], FailureRates, ServiceRates, holding_costs, penalty_cost, skill_cost, machineCost)
        SPT.append(TotalCost)
    
        _, _, TotalCost, _, _=optimal_server_number3(pop[2], FailureRates, ServiceRates, holding_costs, penalty_cost, skill_cost, machineCost)
        H_rule.append(TotalCost)
    
        _, _, TotalCost, _, _=optimal_server_number3(pop[3], FailureRates, ServiceRates, holding_costs, penalty_cost, skill_cost, machineCost)
        HMU_rule.append(TotalCost)
        
    else:
        FCFS.append("NaN")
        SPT.append("NaN")
        H_rule.append("NaN")
        HMU_rule.append("NaN")

df["FCFS_cost"]=pd.Series(FCFS)
df["SPT_cost"]=pd.Series(SPT)
df["H_rule_cost"]=pd.Series(H_rule)
df["HMU_rule_cost"]=pd.Series(HMU_rule)

### risk averse parameter
df["var_level"] = var_level

df.to_json(str(var_level)+"_bechmark_rules_priority_simopt_riskaverse.json")
