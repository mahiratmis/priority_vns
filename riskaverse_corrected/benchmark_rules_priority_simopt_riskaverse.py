# FCFS, SPT, H rule and HXmu rules analyzed under priority
# Risk aversion, var_level=5
import json
import pandas as pd
import numpy as np


import simulation_codes  # Andre's package that used in simulation opt.


lamda = 0.5  # test 0, 0.5, 1
var_level = 0.05
out_fname = "benchmark_rules_vns_db_v{}_l{}_priority_simopt_riskaverse.json".format(int(var_level*100), int(lamda*100))
input_fname = "results/combined_vns_db_v{}_l{}_priority_simopt_riskaverse.json".format(int(var_level*100), int(lamda*100))


def optimal_server_number3(priority, FailureRates, ServiceRates, holding_costs, penalty_cost, skill_cost, machineCost):

    '''
    returns number of servers and total cost for allocating
    that number of servers

    '''
    min_nserver = int(sum(np.array(FailureRates)/np.array(ServiceRates))) + 1  # min required servers
    assignment = np.ones((min_nserver, len(FailureRates)))

    holding_backorder_costList = simulation_codes.SimulationInterface.simulation_optimization_bathrun_priority_riskaverse(lamda,
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

    server_cost = min_nserver*machineCost
    total_cost = np.mean(holding_backorder_costList)+server_cost

    return min_nserver, min_nserver, total_cost, server_cost, total_cost-server_cost


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


json_case2 = []
with open(input_fname, "r") as json_file2:
    # json_file.readline()
    for line in json_file2:
        json_case2.append(json.loads(line))

json_case2 = [item for sublist in json_case2[0] for item in sublist]

df = pd.DataFrame.from_dict(json_case2)

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
        _, _, TotalCost, _, _ = optimal_server_number3(pop[0], FailureRates, ServiceRates, holding_costs, penalty_cost, skill_cost, machineCost)
        FCFS.append(TotalCost)

        _, _, TotalCost, _, _ = optimal_server_number3(pop[1], FailureRates, ServiceRates, holding_costs, penalty_cost, skill_cost, machineCost)
        SPT.append(TotalCost)

        _, _, TotalCost, _, _ = optimal_server_number3(pop[2], FailureRates, ServiceRates, holding_costs, penalty_cost, skill_cost, machineCost)
        H_rule.append(TotalCost)

        _, _, TotalCost, _, _ = optimal_server_number3(pop[3], FailureRates, ServiceRates, holding_costs, penalty_cost, skill_cost, machineCost)
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

df.to_json(out_fname, orient='records')
