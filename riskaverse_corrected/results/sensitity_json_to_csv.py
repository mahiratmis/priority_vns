from json_utility import json_to_csv

param_types= ["failure_rate", "holding_cost","penalty_cost","repair_rate","var_level","work_force_cost"]
for p_type in param_types:
    json_to_csv(path = "sensitivity/json", out_fname=f"{p_type}.csv", pattern=f"{p_type}.json")