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


#from deap import base
#from deap import creator
#from deap import tools

import itertools

json_case=[]
with open("GAPoolingAll_4a.json", "r") as json_file:
    #json_file.readline()
    for line in json_file:
        json_case.append(json.loads(line))

tot = 0
for case in json_case[0]:
    for size_multiplier in  [3]:
        if len(case['simulationGAresults']["failure_rates"]) == 10:
            if case['simulationGAresults']['holding_costs_variant']==2: #HPB cost variant
                if case['simulationGAresults']['skill_cost_factor']==0.1:
                    tot += 1
                    
                    FailureRates=np.array(size_multiplier*case['simulationGAresults']["failure_rates"])
                    ServiceRates=np.array(size_multiplier*case['simulationGAresults']["service_rates"])
                    holding_costs=np.array(size_multiplier*case['simulationGAresults']["holding_costs"])
                        
                    #print (len(holding_costs))
                    #print (holding_costs)
                    #print ("==========")
                    penalty_cost=case['simulationGAresults']["penalty_cost"]
                        
                    #print (penalty_cost)
                        
                    skillCost=100 #NOT USING THIS ATM
                    machineCost=case['simulationGAresults']['machine_cost']
                        
                    #print (machineCost)
                        
                    print ("Optimization code goes HERE!!")
                    print(case["caseID"],len(FailureRates))

print(tot)