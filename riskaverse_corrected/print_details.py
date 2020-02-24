import json
import numpy as np
import pandas as pd

lamda = 0  # test 0, 0.5, 1
var_level = 0.05
alg = "ga"  # ga or vns
db_nodb = "nodb"
out_fname = "metadata_benchmark_rules_{}_{}_v{}_l{}_priority_simopt_riskaverse.json".format(alg, db_nodb, int(var_level*100), int(lamda*100))
# input_fname = "results/combined_vns_db_v{}_l{}_priority_simopt_riskaverse.json".format(int(var_level*100), int(lamda*100))

pth = "results/" + alg + "/combined/json"
# pth = "results/" + db_nodb + "/combined/json"
dynamic_part = "_{}_v{}_l{}_".format(db_nodb,
                                     int(var_level*100),
                                     int(lamda*100))
sub_pth = "/combined_" + alg + dynamic_part + "priority_simopt_riskaverse.json"
input_fname = pth + sub_pth

json_case2 = []
with open(input_fname, "r") as json_file2:
    for line in json_file2:
        json_case2.append(json.loads(line))


json_case2 = [item for sublist in json_case2[0] for item in sublist]
df = pd.DataFrame.from_dict(json_case2)

json_case = []
with open("GAPoolingAll_4a.json", "r") as json_file:
    # json_file.readline()
    for line in json_file:
        json_case.append(json.loads(line))

for case in json_case[0]:
    if case['simulationGAresults']['holding_costs_variant'] == 2:  # HPB cost variant
        if case['simulationGAresults']['skill_cost_factor'] == 0.1:
            if case['simulationGAresults']['utilization_rate'] == 0.8:
                FailureRates = np.array(case['simulationGAresults']["failure_rates"])
                ServiceRates = np.array(case['simulationGAresults']["service_rates"])
                holding_costs = np.array(case['simulationGAresults']["holding_costs"])
                GA_SimOpt = {}
                GA_SimOpt["CaseID"] = case["caseID"]
                GA_SimOpt["FailureRates"] = FailureRates.tolist()
                GA_SimOpt["ServiceRates"] = ServiceRates.tolist()
                GA_SimOpt["holding_costs"] = holding_costs.tolist()
                print ',"CaseID" : "{}", "FailureRates" : {}, "ServiceRates" : {}, "holding_costs" : {}'.format(case["caseID"],
                                                                                                     FailureRates.tolist(),
                                                                                                     ServiceRates.tolist(),
                                                                                                     holding_costs.tolist())

