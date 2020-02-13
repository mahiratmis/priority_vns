import numpy as np
import json
import pandas as pd

import simulation_codes  # Andre's package that used in simulation opt.


lamda = 0.5  # test 0, 0.5, 1
var_level = 0.05
out_fname = "metadata_benchmark_rules_vns_db_v{}_l{}_priority_simopt_riskaverse.json".format(int(var_level*100), int(lamda*100))
# input_fname = "results/combined_vns_db_v{}_l{}_priority_simopt_riskaverse.json".format(int(var_level*100), int(lamda*100))

db_nodb = "db"
pth = "results/" + db_nodb + "/combined/json/"
dynamic_part = "_{}_v{}_l{}_".format(db_nodb,
                                     int(var_level*100),
                                     int(lamda*100))
sub_pth = "/combined_vns" + dynamic_part + "priority_simopt_riskaverse.json"
input_fname = pth + sub_pth


def optimal_server_number(priority, FailureRates, ServiceRates, holding_costs, penalty_cost, skill_cost, machineCost):
    '''
    returns number of servers and total cost for allocating
    that number of servers
    '''
    min_nserver = int(sum(np.array(FailureRates)/np.array(ServiceRates)))+1    # min required servers
    Min_server = min_nserver
    assignment = np.ones((min_nserver, len(FailureRates)))
    holding_backorder_CostList = simulation_codes.SimulationInterface.simulation_optimization_bathrun_priority_riskaverse(lamda,
                                                                                                                          var_level,
                                                                                                                          FailureRates,
                                                                                                                          ServiceRates,
                                                                                                                          holding_costs,
                                                                                                                          penalty_cost,
                                                                                                                          assignment,
                                                                                                                          priority,
                                                                                                                          numberWarmingUpRequests=5000,
                                                                                                                          stopNrRequests=100000,
                                                                                                                          replications=40)
    Server_Cost = min_nserver*machineCost
    TotalCost = np.mean(holding_backorder_CostList)+Server_Cost
    while True:
        # print min_nserver
        min_nserver += 1
        # print min_nserver
        assignment = np.ones((min_nserver, len(FailureRates)))
        temp_holding_backorder_CostList = simulation_codes.SimulationInterface.simulation_optimization_bathrun_priority_riskaverse(lamda,
                                                                                                                                   var_level,
                                                                                                                                   FailureRates,
                                                                                                                                   ServiceRates,
                                                                                                                                   holding_costs,
                                                                                                                                   penalty_cost,
                                                                                                                                   assignment,
                                                                                                                                   priority,
                                                                                                                                   numberWarmingUpRequests=5000,
                                                                                                                                   stopNrRequests=100000,
                                                                                                                                   replications=40)
        temp_Server_Cost = min_nserver*machineCost
        temp_TotalCost = np.mean(temp_holding_backorder_CostList)+temp_Server_Cost
        if temp_TotalCost > TotalCost:
            min_nserver -= 1
            break
        else:
            TotalCost = temp_TotalCost
            Server_Cost = temp_Server_Cost
    return Min_server, min_nserver, TotalCost, Server_Cost, TotalCost-Server_Cost


def optimal_server_number2(priority, FailureRates, ServiceRates, holding_costs, penalty_cost, skill_cost, machineCost):
    '''
    returns number of servers and total cost for allocating
    that number of servers
    '''
    min_nserver = int(sum(np.array(FailureRates)/np.array(ServiceRates)))+1    # min required servers
    Min_server = min_nserver
    assignment = np.ones((min_nserver, len(FailureRates)))
    holding_backorder_CostList = simulation_codes.SimulationInterface.simulation_optimization_bathrun_priority_riskaverse(lamda,
                                                                                                                          var_level,
                                                                                                                          FailureRates,
                                                                                                                          ServiceRates,
                                                                                                                          holding_costs,
                                                                                                                          penalty_cost,
                                                                                                                          assignment,
                                                                                                                          priority,
                                                                                                                          numberWarmingUpRequests=5000,
                                                                                                                          stopNrRequests=100000,
                                                                                                                          replications=40)
    Server_Cost = min_nserver*machineCost
    TotalCost = np.mean(holding_backorder_CostList)+Server_Cost
    return Min_server, min_nserver, TotalCost, Server_Cost, TotalCost-Server_Cost


json_case2 = []
with open(input_fname, "r") as json_file2:
    for line in json_file2:
        json_case2.append(json.loads(line))


json_case2 = [item for sublist in json_case2[0] for item in sublist]
df = pd.DataFrame.from_dict(json_case2)

df['global_best_cost'] = df['VNS_best_cost']
df['global_best_priority'] = df['VNS_best_priority']
df['global_best_cost'] = df['global_best_cost'].map(lambda x:  x[0] if type(x) == list else x)
# analysis number of priority classes
# Add new column to the dataframe
# it is the unique number in the
df["num_priority_class"] = df["global_best_priority"].map(lambda x:  len(set(x)))
# average assigment per priority class
df["num_assignment_class"] = df["global_best_priority"].map(lambda x:  len(x)/float(len(set(x))))
# min required sever vs optimized server number
# need to call function !!
used_server = []
min_server = []
Server_cost = []
EBH_cost = []
for i in range(len(df)):
    priority = np.array(df["global_best_priority"][i])
    FailureRates = np.array(df["FailureRates"][i])
    ServiceRates = np.array(df["ServiceRates"][i])
    holding_costs = np.array(df["holding_costs"][i])
    penalty_cost = df["penalty_cost"][i]
    skill_cost = df["skillCost"][i]
    machineCost = df["machineCost"][i]
    min_ser, used_ser, _, ser_cost, ebh_cost = optimal_server_number(priority, FailureRates, ServiceRates, holding_costs, penalty_cost, skill_cost, machineCost)
    used_server.append(used_ser)
    min_server.append(min_ser)
    Server_cost.append(ser_cost)
    EBH_cost.append(ebh_cost)


df["used_server"] = pd.Series(used_server)
df["min_server"] = pd.Series(min_server)
df["Server_cost"] = pd.Series(Server_cost)
df["EBH_cost"] = pd.Series(EBH_cost)


# comparision with benchmark
# ########Generating different populations###########
# 1-FCFS->
# 2-Shortest and longest proccessing->
# 3-min and max holding cost->
# 4-hxmu long lowest and highest->
def population_generator2(failure_rates, service_rates, holding_costs, penalty_cost, skill_cost, machine_cost):
    '''

    '''
    population = []
    fcfs_rule = np.ones(len(failure_rates))
    population.append(fcfs_rule)
    # ####service rate#######################SPT
    # ##Fast repairing items have higher priority######
    priority = np.ones(len(service_rates))

    k = 1
    for i in np.flip(np.argsort(service_rates), 0):
        priority[i] = k
        k += 1

    population.append(priority)

    # ##holding cost#######################
    # ##Expensive items have higher priortiy##############
    priority = np.ones(len(holding_costs))

    k = 1
    for i in np.argsort(holding_costs):
        priority[i] = k
        k += 1

    population.append(priority)

    # ##holding *service rate##############
    priority = np.ones(len(holding_costs))

    k = 1
    for i in np.argsort(np.array(holding_costs)*np.array(service_rates)):
        priority[i] = k
        k += 1

    population.append(priority)

    return [list(x) for x in population]


SPT = []
H_rule = []
HMU_rule = []
FCFS = []

for i in range(len(df)):
    if df['utilization_rate'][i] == 0.80:
        FailureRates = np.array(df["FailureRates"][i])
        ServiceRates = np.array(df["ServiceRates"][i])
        holding_costs = np.array(df["holding_costs"][i])
        penalty_cost = df["penalty_cost"][i]
        skill_cost = df["skillCost"][i]
        machineCost = df["machineCost"][i]

        pop = population_generator2(FailureRates, ServiceRates, holding_costs, penalty_cost, skill_cost, machineCost)
        _, _, TotalCost, _, _ = optimal_server_number2(pop[0], FailureRates, ServiceRates, holding_costs, penalty_cost, skill_cost, machineCost)
        FCFS.append(TotalCost)

        _, _, TotalCost, _, _ = optimal_server_number2(pop[1], FailureRates, ServiceRates, holding_costs, penalty_cost, skill_cost, machineCost)
        SPT.append(TotalCost)

        _, _, TotalCost, _, _ = optimal_server_number2(pop[2], FailureRates, ServiceRates, holding_costs, penalty_cost, skill_cost, machineCost)
        H_rule.append(TotalCost)

        _, _, TotalCost, _, _ = optimal_server_number2(pop[3], FailureRates, ServiceRates, holding_costs, penalty_cost, skill_cost, machineCost)
        HMU_rule.append(TotalCost)

    else:
        FCFS.append("NaN")
        SPT.append("NaN")
        H_rule.append("NaN")
        HMU_rule.append("NaN")

df["FCFS_cost"] = pd.Series(FCFS)
df["SPT_cost"] = pd.Series(SPT)
df["H_rule_cost"] = pd.Series(H_rule)
df["HMU_rule_cost"] = pd.Series(HMU_rule)

# ## risk averse parameters
df["var_level"] = var_level
df["lamda"] = lamda

# save dataframe to file
df.to_json(out_fname, orient='records')
