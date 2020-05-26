import numpy as np
import pandas as pd
import ast

import simulation_codes

var_level = 0.05
Server_cost = []
H_cost = []
BO_cost = []
T_cost = []  
Used_server = []
Utilization = []
Num_back_orders = []
Parameter_Value = []
Case_id = []
Var_level = []
Lamda = []
Parameter_Name = []  
param_name = "repair_rate"
for lamda in [0, 0.5, 1]:
    fname = 'results/metadata/combined/csv/combined_vns_db_v5_l{}_priority_simopt_riskaverse.csv'.format(int(lamda*100))
    print(fname, "is being processed")
    df = pd.read_csv(fname)  
    for i in range(len(df)):
        print("{}. case".format(i+1))
        for rr_multiplier in [1.1, 1.2, 1.4, 1.6, 1.8]:  # repair(service) rate sensivity
            assignment = np.ones((df["used_server"][i], len(np.array(ast.literal_eval(df["FailureRates"][i])))))
            priority = np.array(ast.literal_eval(df["VNS_best_priority"][i]))  # is it this or 'global_best_priority'
            holding_costs = np.array(ast.literal_eval(df['holding_costs'][i]))
            penalty_cost = df['penalty_cost'][i] 
            FailureRates = np.array(ast.literal_eval(df["FailureRates"][i]))
            ServiceRates = rr_multiplier * np.array(ast.literal_eval(df["ServiceRates"][i]))
            used_server = df["used_server"][i]
            holding_backorder_CostList, holdList, backorderList = \
                        simulation_codes.SimulationInterface.simulation_optimization_bathrun_priority_riskaverse(lamda, var_level, FailureRates, ServiceRates,\
                                                holding_costs, penalty_cost, assignment,\
                                                priority,
                                    numberWarmingUpRequests = 5000,
                                    stopNrRequests = 100000,
                                    replications =40)
                        
            machine_cost = df["machineCost"][i]
            ser_cost = df["used_server"][i] * machine_cost
            t_cost = np.mean(holding_backorder_CostList) + ser_cost
            h_cost = np.mean(holdList)
            bo_cost = np.mean(backorderList)
            utilization = np.sum(FailureRates/ServiceRates) / used_server
            num_back_orders = bo_cost / float(penalty_cost)

            Server_cost.append(ser_cost)
            H_cost.append(h_cost)
            BO_cost.append(bo_cost)
            T_cost.append(t_cost)
            Used_server.append(used_server)
            Utilization.append(utilization)
            Num_back_orders.append(num_back_orders)
            Parameter_Value.append(rr_multiplier)
            Parameter_Name.append(param_name)
            Case_id.append(df["CaseID"][i])
            Var_level.append(var_level)
            Lamda.append(lamda)

dicts = {'Used_server': Used_server, 
        'Server_cost': Server_cost, 
        'H_cost': H_cost,
        'BO_cost': BO_cost,
        'T_cost': T_cost,
        'Utilization': Utilization,
        'Num_back_orders': Num_back_orders,
        'Parameter_Name': Parameter_Name,
        'Parameter_Value': Parameter_Value,
        'Case_id': Case_id,
        'Var_level':Var_level,
        'Lamda':Lamda
        } 
new_df = pd.DataFrame(dicts)
df.to_json('{}.json'.format(param_name), orient='records')
print("Done!")