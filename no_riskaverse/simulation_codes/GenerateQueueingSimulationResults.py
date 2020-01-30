import numpy as np
import numpy.random as rnd
import json
from skill_set_generator import generateSkillsMultiServer, checkSkillAssignment, compute_number_skill_combinations
from simulation import simulation_batch_run
import itertools

np.set_printoptions(linewidth=50000)

rnd.seed(1234) #set fixed seed
replications = 35
# stopTimeInput = 15000
stopNrRequests = 100000
maxQueue = 500
cores = None
maxAssignments = 500
maxWaitingTime = 20
waitingTimeSteps = 0.5

waitingTimeBuckets = waitingTimeSteps*np.array(range(int(maxWaitingTime/waitingTimeSteps) + 1))


SKU_numbers=range(2,6)   #outer loop is needed for changing number of SKUs
n_servers=range(2,6)   #outer loop is needed for changing number of SKUs
utilization_rate=[0.6, 0.7, 0.8, 0.9]  #not used this parameter
lambda_variants=range(5)
max_mu_deviations=[0, 1, 2]



file_path = '/Users/andrei/simulated_sets.json'
#file_path = 'simulated_sets_version2.json'

# with open('simulated_sets.json', "a") as json_file:
#     json_file.write("{}\n".format(json.dumps(new_data)))
# output_dict = dict()cs

with open(file_path, "w") as json_file:
    json_file.write("replications: {}, maxArrivals: {} \n".format(replications, stopNrRequests))


totalCounter = 0
case_nr = 0
for (n, sku_num, rho, max_mu_deviation, lmd_variant) \
        in itertools.product(n_servers, SKU_numbers, utilization_rate, max_mu_deviations, lambda_variants):

    alphas = np.random.dirichlet(np.ones(sku_num),size=1)[0]
    serviceRates = np.exp(np.random.uniform(-max_mu_deviation,max_mu_deviation,sku_num))
    failureRates = n*rho*alphas*serviceRates

    if compute_number_skill_combinations(n,sku_num) < maxAssignments:
        case = dict()
        case_nr += 1
        #output_dict["Case: "+ str(case_nr).zfill(4)] = case

        case["caseID"] = "Case: "+ str(case_nr).zfill(4)
        case["n_servers"]   = n
        case["SKU_number"]   = sku_num
        case["failure_rates"] = failureRates.tolist()
        case["service_rates"] = serviceRates.tolist()
        case["assignments"] = dict()

        print n, sku_num, failureRates, serviceRates,


        assignmentID = 0
        for skillServerAssignment in generateSkillsMultiServer(n, sku_num):
            if checkSkillAssignment(skillServerAssignment, failureRates, serviceRates):
                assignmentID +=1
                case["assignments"][assignmentID] = dict()
                case["assignments"][assignmentID]["skill_assignment"] = np.transpose(skillServerAssignment).tolist()

                numberInSystemDistributionAverage, waitingTimeHistogramAverage, utilizationRatesAverage, skillServerAssignmentStatisitcsAverage  \
                    = simulation_batch_run(failureRates, serviceRates, skillServerAssignment,
                                              histogramBucketsInput = waitingTimeBuckets,
                                              # stopTimeInput = stopTimeInput*np.sum(failureRates),
                                              expectedTotalNumberRequests = stopNrRequests,
                                              numberWarmingUpRequests = 1000,
                                              replications=replications,
                                              maxQueue=maxQueue,
                                              nCores=cores)

                case["assignments"][assignmentID]["queue_length_distributions"] = dict()
                case["assignments"][assignmentID]["waiting_time_distributions"] = dict()
                for sk in range(sku_num):
                    case["assignments"][assignmentID]["queue_length_distributions"][sk] \
                        = {k: v for k, v in zip(range(maxQueue), numberInSystemDistributionAverage[sk]) if v != 0} #remove zero values
                    case["assignments"][assignmentID]["waiting_time_distributions"][sk] \
                        = {k: v for k, v in zip(waitingTimeBuckets, waitingTimeHistogramAverage[sk]) if v != 0} #remove zero values
                case["assignments"][assignmentID]["utilization_rates"] = utilizationRatesAverage.tolist()
                case["assignments"][assignmentID]["skill_server_distribution"] = skillServerAssignmentStatisitcsAverage.tolist()

        print assignmentID

        totalCounter += assignmentID
        with open(file_path, "a") as json_file:
            json_file.write("{}\n".format(json.dumps(case)))

#print "totalCounter:", totalCounter, "; approximate execution time:", totalCounter*15/3600/24, "days"


# with open('simulated_sets.json', 'w') as fp:
#     json.dump(output_dict, fp, indent=2, separators=(',', ':')) # compress default separators
    # json.dump(output_dict, fp, separators=(',', ':')) # compress default separators, no indent, makes the JSON file unreadable but compact



# with open('simulated_sets.json', "r") as json_file:
#     for line in json_file:
#         case = json.loads(line)
#         print type(data)

pass