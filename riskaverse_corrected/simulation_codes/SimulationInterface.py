from __future__ import division
import json
# import shelve
import time
import numpy as np
import multiprocessing as mp

# from QueueingModels import OptimizeStockLevelsAndCosts
from simulation import simulation_batch_run, simulation
from skill_set_generator import checkSkillAssignment
import filedict
import os


database = None

stopNrRequestsDefault = 1000000
replicationsDefault = 35
maxQueue = 3000 #increased from 1500
rounding_decimals = 5
nCores = None  # maximum number of cores will be used

def initDatabase(databaseFile = "db_simulations", createNewDatabase = False):
    filename = databaseFile + ".dict"
    if createNewDatabase and os.path.isfile(filename):
        # try to remove old database
        os.remove(filename)
        pass

    global database
    # possible flags are:
    #         'r'	 - Open existing database for reading only (default)
    #         'w'	 - Open existing database for reading and writing
    #         'c'	 - Open database for reading and writing, creating it if it doesn't exist (default)
    #         'n'	 - Always create a new, empty database, open for reading and writing

    # database = shelve.open(databaseFile, protocol=2, flag=flag, writeback=True)
    database = filedict.FileDict(filename = filename)
    pass


# DONE: TODO create one function writing to DB
# DONE: TODO create key forming function
# TODO normalization, currently not essential as all experiments assume normalized lambdas
# DONE: TODO round failure and service rates to 5 digits after comma
# DONE: TODO create sorting and unsorting functionality by the failure rates


def makeCaseKey(n_servers, m_skills, failureRates, serviceRates):
    # str1 = "N:"+str(n_servers)+"|M:" + str(m_skills) + '|FR:[ '+str('  '.join(map(str, failureRates)))+']|SR:[ '+str('  '.join(map(str, serviceRates)))+']'
    # str2 = "N:"+str(n_servers)+"|M:" + str(m_skills) + "|FR:"+str(failureRates) + "|SR:" + str(serviceRates)
    #
    # print str1
    # print str2


    return "N:"+str(n_servers)+"|M:" + str(m_skills) + '|FR:['+str(','.join(map(str, failureRates)))+']|SR:['+str(','.join(map(str, serviceRates)))+']'
    # return "N:"+str(n_servers)+"|M:" + str(m_skills) + "|FR:"+str(failureRates) + "|SR:" + str(serviceRates)

def makeCaseKey2(n_servers, m_skills, failureRates, serviceRates, assignment):
    # str1 = "N:"+str(n_servers)+"|M:" + str(m_skills) + '|FR:[ '+str('  '.join(map(str, failureRates)))+']|SR:[ '+str('  '.join(map(str, serviceRates)))+']'
    # str2 = "N:"+str(n_servers)+"|M:" + str(m_skills) + "|FR:"+str(failureRates) + "|SR:" + str(serviceRates)
    #
    # print str1
    # print str2


    return "N:"+str(n_servers)+"|M:" + str(m_skills) + '|FR:['+str(','.join(map(str, failureRates)))+']|SR:['+str(','.join(map(str, serviceRates)))+']'\
           +'|Ass:['+str(','.join(map(str, ['['+str(','.join(map(str, x)))+']' for x in assignment])))+']'


def writeToDB1(n_servers, m_skills, failureRates, serviceRates, skillServerAssignment, numberInSystemDistributionAverage, utilizationRatesAverage, skillServerAssignmentStatisitcsAverage):
    sorting_order = failureRates.argsort()
    sorting_dict = dict(zip(sorting_order.tolist(), range(m_skills)))
    reverse_order = [sorting_dict[i] for i in sorted(sorting_dict.keys())]  # map(sorting_dict.get, sorting_dict.keys().sort())

    sorted_failureRates = np.round(failureRates[sorting_order], rounding_decimals)
    sorted_serviceRates = np.round(serviceRates[sorting_order], rounding_decimals)
    sorted_assignment = skillServerAssignment[:,sorting_order] # sorted on skills based on increasing failure rates

    queue_characherisitcs_key = makeCaseKey(n_servers, m_skills, sorted_failureRates, sorted_serviceRates )

    case = None
    databaseUpdated = False

    if queue_characherisitcs_key in database:
        case = database[queue_characherisitcs_key]
    else:
        case = {}
        case["m_skills"] = m_skills
        case["n_servers"] = n_servers
        case["failureRates"] = sorted_failureRates
        case["serviceRates"] = sorted_serviceRates
        case["skills_assignments"] = {}
        database[queue_characherisitcs_key] = case

    skillServerAssignment_key = str(sorted(sorted_assignment.tolist()))  # the assignment is sorted on server

    if not skillServerAssignment_key in case["skills_assignments"]:

        case["skills_assignments"][skillServerAssignment_key] = {}
        maxQueue = sum(np.sum(numberInSystemDistributionAverage,axis=0)>0)
        case["skills_assignments"][skillServerAssignment_key]["maxQueue"] = maxQueue
        case["skills_assignments"][skillServerAssignment_key]["skill_assignment"] = np.array(sorted(sorted_assignment.tolist()))
        case["skills_assignments"][skillServerAssignment_key]["numberInSystemDistributionAverage"] = numberInSystemDistributionAverage[sorting_order,:maxQueue]
        # case[skillServerAssignment_key]["utilizationRatesAverage"] = utilizationRatesAverage
        # case[skillServerAssignment_key]["skillServerAssignmentStatisitcsAverage"] = skillServerAssignmentStatisitcsAverage

        database[queue_characherisitcs_key] = case
        # database.sync()    # database updated
        databaseUpdated = True
        # print "databaseUpdated: ", skillServerAssignment.tolist()


    return databaseUpdated

def writeToDB2(n_servers, m_skills, failureRates, serviceRates, skillServerAssignment, numberInSystemDistributionAverage, utilizationRatesAverage, skillServerAssignmentStatisitcsAverage):
    '''
    In this function for each skill assignment a separate record in the database will be created
    '''

    sorting_order = failureRates.argsort()

    sorted_failureRates = np.round(failureRates[sorting_order], rounding_decimals)
    sorted_serviceRates = np.round(serviceRates[sorting_order], rounding_decimals)
    sorted_assignment = skillServerAssignment[:, sorting_order] # sorted on skills based on increasing failure rates

    skillServerAssignment_key = sorted(sorted_assignment.tolist())  # the assignment is sorted on server
    queue_assignment_characherisitcs_key = makeCaseKey2(n_servers, m_skills, sorted_failureRates, sorted_serviceRates, skillServerAssignment_key)

    databaseUpdated = False

    if not queue_assignment_characherisitcs_key in database:
        case = {}
        case["m_skills"] = m_skills
        case["n_servers"] = n_servers
        case["failureRates"] = sorted_failureRates
        case["serviceRates"] = sorted_serviceRates
        case["skills_assignments"] = np.array(sorted(sorted_assignment.tolist()))

        maxQueue = sum(np.sum(numberInSystemDistributionAverage,axis=0)>0)
        case["maxQueue"] = maxQueue
        case["numberInSystemDistributionAverage"] = numberInSystemDistributionAverage[sorting_order,:maxQueue]
        case["utilizationRatesAverage"] = utilizationRatesAverage
        case["skillServerAssignmentStatisitcsAverage"] = skillServerAssignmentStatisitcsAverage

        database[queue_assignment_characherisitcs_key] = case
        databaseUpdated = True

    return databaseUpdated

def getRecordFromDB1(n_servers, m_skills, failureRates, serviceRates, skillServerAssignment):
    sorting_order = failureRates.argsort()
    sorting_dict = dict(zip(sorting_order.tolist(), range(m_skills)))
    reverse_order = [sorting_dict[i] for i in sorted(sorting_dict.keys())]  # map(sorting_dict.get, sorting_dict.keys().sort())

    sorted_failureRates = np.round(failureRates[sorting_order], rounding_decimals)
    sorted_serviceRates = np.round(serviceRates[sorting_order], rounding_decimals)
    sorted_assignment = skillServerAssignment[:,sorting_order] # sorted on skills based on increasing failure rates

    queue_characherisitcs_key = makeCaseKey(n_servers, m_skills, sorted_failureRates, sorted_serviceRates )

    skillServerAssignment_key = str(sorted(sorted_assignment.tolist()))  # the assignment is sorted on server

    case = None
    if queue_characherisitcs_key in database:
        case = database[queue_characherisitcs_key]["skills_assignments"]

    if case: # case found
        if skillServerAssignment_key in case:
            serverAssignment = case[skillServerAssignment_key]
            return serverAssignment["numberInSystemDistributionAverage"][reverse_order]
        else:
            return None


def getRecordFromDB2(n_servers, m_skills, failureRates, serviceRates, skillServerAssignment):
    sorting_order = failureRates.argsort()
    sorting_dict = dict(zip(sorting_order.tolist(), range(m_skills)))
    reverse_order = [sorting_dict[i] for i in sorted(sorting_dict.keys())]  # map(sorting_dict.get, sorting_dict.keys().sort())

    sorted_failureRates = np.round(failureRates[sorting_order], rounding_decimals)
    sorted_serviceRates = np.round(serviceRates[sorting_order], rounding_decimals)
    sorted_assignment = skillServerAssignment[:,sorting_order] # sorted on skills based on increasing failure rates

    skillServerAssignment_key = sorted(sorted_assignment.tolist())  # the assignment is sorted on server
    queue_characherisitcs_key = makeCaseKey2(n_servers, m_skills, sorted_failureRates, sorted_serviceRates, skillServerAssignment_key )

    if queue_characherisitcs_key in database:
        return database[queue_characherisitcs_key]["numberInSystemDistributionAverage"][reverse_order]
    else:
        return None

def OptimizeStockLevelsAndCostsSimBased(holdingCosts, penalty, marginalDistribution):

    if not isinstance(holdingCosts, np.ndarray):
        holdingCosts = np.array(holdingCosts)

    if len(marginalDistribution.shape) == 1:
        marginalDistribution = marginalDistribution.reshape(1,len(marginalDistribution))

    nSKUs=len(holdingCosts)
    maxQueue = marginalDistribution.shape[1]
    n_array = np.array(range(maxQueue))
    S = np.zeros(nSKUs, dtype=int)
    PBO = np.sum(marginalDistribution[:,1:], axis=1)
    EBO = np.sum(marginalDistribution*np.array(range(marginalDistribution.shape[1])), axis=1)

    hb_ratio = holdingCosts/penalty
    for sk in xrange(nSKUs):
        while S[sk]<maxQueue and np.sum(marginalDistribution[sk, S[sk]+1:]) > hb_ratio[sk]:
            S[sk] += 1
            PBO[sk] = np.sum(marginalDistribution[sk,S[sk]+1:]) # -= marginalDistribution[sk, S[sk]]
            EBO[sk] = np.sum(marginalDistribution[sk,S[sk]:]*n_array[:-S[sk]]) #-= PBO[sk]

    totalCost = np.sum(S*holdingCosts) + np.sum(penalty*EBO)

    return totalCost, S, EBO


def OptimizeStockLevelsAndCostsSimBased_RiskAverse(holdingCosts, penalty, marginalDistribution, lamda, var_level):

    if not isinstance(holdingCosts, np.ndarray):
        holdingCosts = np.array(holdingCosts)

    if len(marginalDistribution.shape) == 1:
        marginalDistribution = marginalDistribution.reshape(1,len(marginalDistribution))

    nSKUs=len(holdingCosts)
    maxQueue = marginalDistribution.shape[1]
    n_array = np.array(range(maxQueue))
    S = np.zeros(nSKUs, dtype=int)
    PBO = np.sum(marginalDistribution[:,1:], axis=1)
    EBO = np.sum(marginalDistribution*np.array(range(marginalDistribution.shape[1])), axis=1)
    
    #Prob_BO=np.zeros(nSKUs*maxQueue, dtype=float)
    
    #hb_ratio = holdingCosts/penalty

    #calculate new critical fractions
    print("a")
    rho=(penalty-holdingCosts)/penalty

    #print min(rho), max(rho)

    #update rho based on lamda and var_level

    print("b")
    critical_ratio= np.zeros(nSKUs, dtype=float)
    
    for sk in xrange(nSKUs):
        
        #try to normalize
    
        marginalDistribution[sk]=marginalDistribution[sk]/np.sum(marginalDistribution[sk])

    for sk in xrange(nSKUs):

        if (lamda >=0) and (lamda <=min(1, (1-rho[sk])/ var_level)):

                critical_ratio[sk]= rho[sk]*(1-var_level)/(1-lamda*var_level)                          

        if (lamda >(1-rho[sk])/ var_level) and (lamda <=1):

                critical_ratio[sk]= 1-(1-rho[sk])/lamda


    
    print("c")
    for sk in xrange(nSKUs):
        while S[sk]<maxQueue-1 and np.sum(marginalDistribution[sk, :S[sk]+1]) <= critical_ratio[sk]:
            S[sk] += 1
            PBO[sk] = np.sum(marginalDistribution[sk,S[sk]+1:]) # -= marginalDistribution[sk, S[sk]]
            EBO[sk] = np.sum(marginalDistribution[sk,S[sk]:]*n_array[:-S[sk]]) #-= PBO[sk]

            
            ###probability dist. of P(BO=i) i=0,1,....qmax
            #Prob_BO[sk]=np.append(Pb0, marginalDistribution[sk,S[sk]+1:])

            #print len(marginalDistribution[sk,S[sk]+1:])
            ###find i that achieves cumulative P(BO=i)>1-var_level

    #print np.sum(Prob_BO[0])

    #print S

    S_riskaverse=[]

    print("d")
    for sk in xrange(nSKUs):
        print("Error Here")
        print sk, len(marginalDistribution[sk,S[sk]:])
        Prob_BO=np.zeros(len(marginalDistribution[sk,S[sk]:]), dtype=float) #back order dist. for SKU sk

        Prob_BO[0]=1-PBO[sk] #0 backorder prob

        Prob_BO[1:]=marginalDistribution[sk,S[sk]+1: ] #rest of the backorder probs

        #Find the minimum index over 1-var_level this var backorder level

        #print np.cumsum(Prob_BO)[>=1-var_level]

        a=np.cumsum(Prob_BO)

        idx=np.where(a == a[a>=1-var_level].min())[0][0]

        #print idx #this is var value 

        x=np.arange(idx, len(Prob_BO))

        #normalize conditional probabilities

        #print np.sum(Prob_BO[idx:]/np.sum(Prob_BO[idx:]))

        #print sum(np.multiply(x, Prob_BO[idx:]/np.sum(Prob_BO[idx:]))) #Risk averse BO number

        S_riskaverse.append(sum(np.multiply(x, Prob_BO[idx:]/np.sum(Prob_BO[idx:]))))

      
    #print EBO

    #print S_riskaverse

    print("e")
    totalCost = np.sum(S*holdingCosts) + lamda*np.sum(penalty*EBO)+(1-lamda)*np.sum(penalty*np.array(S_riskaverse))

    #print S

    #print Prob_BO
    print "heeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
    return totalCost, S, EBO

# writeToDB = writeToDB1
# getRecordFromDB = getRecordFromDB1

writeToDB = writeToDB2
getRecordFromDB = getRecordFromDB2

# @profile
def simulation_optimization_run(failureRates, serviceRates, holdingCosts, penalty, skillServerAssignment,
                                stopNrRequests = stopNrRequestsDefault,
                                replications = replicationsDefault,
                                numberWarmingUpRequests = 0,
                                runTheSimulation=True, useDatabase=True, maxQ = maxQueue):

    numberInSystemDistributionAverage = None
    utilizationRatesAverage = None
    skillServerAssignmentStatisitcsAverage = None

    m_skills = skillServerAssignment.shape[1]
    n_servers = skillServerAssignment.shape[0]

    databaseUpdated = False
    if useDatabase:
        numberInSystemDistributionAverage = getRecordFromDB(n_servers, m_skills, failureRates, serviceRates, skillServerAssignment)

    if numberInSystemDistributionAverage is None and runTheSimulation:

        # run the simulaiton
        # print "run simulation:", skillServerAssignment, failureRates, serviceRates
        if checkSkillAssignment(skillServerAssignment, failureRates, serviceRates):
            # print skillServerAssignment.tolist(), failureRates, serviceRates,
            numberInSystemDistributionAverage, _, utilizationRatesAverage, skillServerAssignmentStatisitcsAverage\
                    = simulation_batch_run( failureRates = failureRates,
                                            serviceRates = serviceRates,
                                            skillServerAssignment = skillServerAssignment,
                                            histogramBucketsInput = [],
                                            # stopTimeInput=stopTime,
                                            expectedTotalNumberRequests = stopNrRequests,
                                            numberWarmingUpRequests = numberWarmingUpRequests,
                                            replications=replications,
                                            maxQueue=maxQ,
                                            nCores=nCores)

            if useDatabase:
                databaseUpdated = writeToDB(n_servers, m_skills, failureRates, serviceRates, skillServerAssignment,
                                            numberInSystemDistributionAverage, utilizationRatesAverage, skillServerAssignmentStatisitcsAverage)


    totalCost = float('inf'); S = np.zeros(m_skills, dtype=int); EBO = np.zeros(m_skills)
    if numberInSystemDistributionAverage is not None:
        # print [np.nanmean([numberInSystemDistributionAverage[j][i-1]/numberInSystemDistributionAverage[j][i]*failureRates[j] for i in range(3,15)]) for j in range(m_skills)]
        totalCost, S, EBO = OptimizeStockLevelsAndCostsSimBased(holdingCosts, penalty, numberInSystemDistributionAverage)

    return totalCost, S, EBO, databaseUpdated, utilizationRatesAverage, skillServerAssignmentStatisitcsAverage


def simulation_optimization_runV2(failureRates, serviceRates, holdingCosts, penalty, skillServerAssignment,
                                replications = replicationsDefault,
                                stopNrRequests = stopNrRequestsDefault,
                                numberWarmingUpRequests = 0, maxQ = maxQueue):

    totalCostList = []
    for i in xrange(replications):
        numberInSystemDistributionAverage, _, _, _ = \
            simulation_batch_run(failureRates = failureRates,
                                serviceRates = serviceRates,
                                skillServerAssignment = skillServerAssignment,
                                histogramBucketsInput = [],
                                expectedTotalNumberRequests = stopNrRequests,
                                numberWarmingUpRequests = numberWarmingUpRequests,
                                replications=1,
                                maxQueue=maxQ,
                                startSeed=100+i*30)


        if numberInSystemDistributionAverage is not None:
            totalCost, _, _ = OptimizeStockLevelsAndCostsSimBased(holdingCosts, penalty, numberInSystemDistributionAverage)
            totalCostList.append(totalCost)

    return totalCostList


def simulation_optimization_bathrun(failureRates, serviceRates, holdingCosts, penalty, skillServerAssignment,
                                  replications=replicationsDefault,
                                  stopNrRequests = stopNrRequestsDefault,
                                  numberWarmingUpRequests=0, maxQ=maxQueue):
    totalCostList = []

    pool = mp.Pool(processes=nCores)

    requestArrivalRateInput = np.sum(failureRates)
    skillDistributionInput = np.cumsum(failureRates/requestArrivalRateInput)
    # skillServerCosts = np.ones(skillServerAssignment.shape)

    skillServerRates = np.transpose(np.zeros(skillServerAssignment.shape))
    for i in xrange(serviceRates.shape[0]): skillServerRates[i,:] = skillServerAssignment[:,i] * serviceRates[i]
    skillServerRates = np.transpose(skillServerRates)
    skillServerCosts = skillServerRates

    startSeed = 50
    try:
        results = [pool.apply_async(simulation,
                     args=(requestArrivalRateInput,
                        skillDistributionInput,  #The distribution should be in cummulative form
                        skillServerRates, ##replenishment times can be different for different priority classes
                        skillServerCosts,
                        [],
                        stopNrRequests,
                        numberWarmingUpRequests,
                        startSeed+20*i,
                        ) )
            for i in xrange(replications)]

        for p in results:
            numberInSystemDistribution = p.get()[0]

            numberInSystemDistributionArray = np.zeros((len(failureRates), maxQueue))
            for sk in numberInSystemDistribution:
                for n in numberInSystemDistribution[sk]:
                    if(n<maxQ):
                        numberInSystemDistributionArray[sk,n] += numberInSystemDistribution[sk][n]

            totalCost, _, _ = OptimizeStockLevelsAndCostsSimBased(holdingCosts, penalty, numberInSystemDistributionArray)
            totalCostList.append(totalCost)

        #print "Runs are executed:", len(results)
        pool.close()
    except KeyboardInterrupt:
        print 'got ^C while pool mapping, terminating the pool'
        pool.terminate()
        print 'pool is terminated'
    except Exception, e:
        print 'got exception: %r, terminating the pool' % (e,)
        pool.terminate()
        print 'pool is terminated'
    finally:
        pool.join()

    return totalCostList

def simulation_optimization_bathrun_priority(failureRates, serviceRates, holdingCosts, penalty, skillServerAssignment,priority,
                                  replications=replicationsDefault,
                                  stopNrRequests = stopNrRequestsDefault,
                                  numberWarmingUpRequests=0, maxQ=maxQueue):
    totalCostList = []

    pool = mp.Pool(processes=nCores)

    requestArrivalRateInput = np.sum(failureRates)
    skillDistributionInput = np.cumsum(failureRates/requestArrivalRateInput)
    # skillServerCosts = np.ones(skillServerAssignment.shape)

    skillServerRates = np.transpose(np.zeros(skillServerAssignment.shape))
    for i in xrange(serviceRates.shape[0]): skillServerRates[i,:] = skillServerAssignment[:,i] * serviceRates[i]
    skillServerRates = np.transpose(skillServerRates)
    skillServerCosts = np.array(skillServerRates)*np.array(priority) #

    startSeed = 50
    try:
        results = [pool.apply_async(simulation,
                     args=(requestArrivalRateInput,
                        skillDistributionInput,  #The distribution should be in cummulative form
                        skillServerRates, ##replenishment times can be different for different priority classes
                        skillServerCosts,
                        [],
                        stopNrRequests,
                        numberWarmingUpRequests,
                        startSeed+20*i,
                        ) )
            for i in xrange(replications)]

        for p in results:
            numberInSystemDistribution = p.get()[0]

            numberInSystemDistributionArray = np.zeros((len(failureRates), maxQueue))
            for sk in numberInSystemDistribution:
                for n in numberInSystemDistribution[sk]:
                    if(n<maxQ):
                        numberInSystemDistributionArray[sk,n] += numberInSystemDistribution[sk][n]

            totalCost, _, _ = OptimizeStockLevelsAndCostsSimBased(holdingCosts, penalty, numberInSystemDistributionArray)
            totalCostList.append(totalCost)

        #print "Runs are executed:", len(results)
        pool.close()
    except KeyboardInterrupt:
        print 'got ^C while pool mapping, terminating the pool'
        pool.terminate()
        print 'pool is terminated'
    except Exception, e:
        print 'got exception: %r, terminating the pool' % (e,)
        pool.terminate()
        print 'pool is terminated'
    finally:
        pool.join()

    return totalCostList


def simulation_optimization_bathrun_priority_riskaverse(lamda, var_level, failureRates, serviceRates, holdingCosts, penalty, skillServerAssignment,priority,
                                  replications=replicationsDefault,
                                  stopNrRequests = stopNrRequestsDefault,
                                  numberWarmingUpRequests=0, maxQ=maxQueue):
    totalCostList = []

    pool = mp.Pool(processes=nCores)

    requestArrivalRateInput = np.sum(failureRates)
    skillDistributionInput = np.cumsum(failureRates/requestArrivalRateInput)
    # skillServerCosts = np.ones(skillServerAssignment.shape)

    skillServerRates = np.transpose(np.zeros(skillServerAssignment.shape))
    for i in xrange(serviceRates.shape[0]): skillServerRates[i,:] = skillServerAssignment[:,i] * serviceRates[i]
    skillServerRates = np.transpose(skillServerRates)
    skillServerCosts = np.array(skillServerRates)*np.array(priority) #

    startSeed = 50
    try:
        results = [pool.apply_async(simulation,
                     args=(requestArrivalRateInput,
                        skillDistributionInput,  #The distribution should be in cummulative form
                        skillServerRates, ##replenishment times can be different for different priority classes
                        skillServerCosts,
                        [],
                        stopNrRequests,
                        numberWarmingUpRequests,
                        startSeed+20*i,
                        ) )
            for i in xrange(replications)]

        for p in results:
            numberInSystemDistribution = p.get()[0]

            numberInSystemDistributionArray = np.zeros((len(failureRates), maxQueue))
            for sk in numberInSystemDistribution:
                for n in numberInSystemDistribution[sk]:
                    if(n<maxQ):
                        numberInSystemDistributionArray[sk,n] += numberInSystemDistribution[sk][n]

            totalCost, _, _ = OptimizeStockLevelsAndCostsSimBased_RiskAverse(holdingCosts, penalty, numberInSystemDistributionArray, lamda, var_level)
            totalCostList.append(totalCost)

        #print "Runs are executed:", len(results)
        pool.close()
    except KeyboardInterrupt:
        print 'got ^C while pool mapping, terminating the pool'
        pool.terminate()
        print 'pool is terminated'
    except Exception, e:
        print 'got exception: %r, terminating the pool' % (e,)
        pool.terminate()
        print 'pool is terminated'
    finally:
        pool.join()

    return totalCostList

def createDatabaseFromExistingJSON():
    file_path = "..//simulated_sets_version3.json"

    initDatabase(databaseFile = "..//db_simulations", createNewDatabase = True)

    with open(file_path, "r") as json_file:
        json_file.readline()
        for line in json_file:
            json_case = json.loads(line)

            if True: #json_case["caseID"] == "Case: 0061": #debuging line
                m_skills = json_case["SKU_number"]
                n_servers = json_case["n_servers"]
                failureRates = np.array(json_case["failure_rates"])
                serviceRates = np.array(json_case["service_rates"])

                for assignment in json_case["assignments"].values():
                    skillServerAssignment = np.array(assignment["skill_assignment"]) #map(list, zip(*assignment["skill_assignment"]))) # transpose the assignment

                    #maxQueue = max(len(val) for val in assignment["queue_length_distributions"].values())
                    maxQueue = max([max([int(val) for val in queue_req.keys()]) for queue_req in assignment["queue_length_distributions"].values()]) + 1
                    marginalDistribution = np.zeros((len(assignment["queue_length_distributions"]), maxQueue))
                    for sk in assignment["queue_length_distributions"]:
                        for n in assignment["queue_length_distributions"][sk]:
                            marginalDistribution[int(sk), int(n)] = assignment["queue_length_distributions"][sk][n]

                    upt = writeToDB(n_servers, m_skills, failureRates, serviceRates, skillServerAssignment, marginalDistribution, assignment["utilization_rates"], assignment["skill_server_distribution"])
                    pass

            print json_case["caseID"]

            # if json_case["caseID"] == "Case: 0010":
            #     break

    # database.close()
    pass



if __name__ == '__main__':
    np.set_printoptions(linewidth=50000)

    startTime = time.time()
    #createDatabaseFromExistingJSON()

    # testDatabaseInterface()

    failure_rates = np.array([  2.23738889e-01,   4.94087095e-02,   9.83533638e-03,   2.32382124e-01,   3.31158691e-05,   7.81429552e-02,   2.04179403e-01,   1.43352018e-01,   4.02415285e-02,   1.86859200e-02])
    service_rates = np.array([  1.76753604e+00,   4.69646045e-02,   2.32412359e+00,   3.29198749e-01,   1.99622149e-05,   5.00131046e-01,   4.21953234e-01,   7.21777996e-02,   5.78270470e-02,   1.65334582e-02])
    fullflex_assignment = np.ones((10,10), dtype=int)

    holding_costs = np.array([ 543.85807009,  577.29253873,  817.43195388,  688.21756066,  114.2328496 ,  166.58437791,  910.89245915,  598.73244586,  638.60568814,  901.15711987])
    penalty_cost = 32277.41185467891

    failure_rates = np.array([0.2541783360098542, 0.31538730923198566, 0.9915311240730542, 0.13692735636243197, 0.10197587432267417])
    service_rates = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    holding_cost = np.array([386.4200669605343, 264.8119394757236, 53.76775526966686, 775.1555433515449, 1000.0])

    penalty_cost_factor = 50
    penalty = penalty_cost_factor*sum(failure_rates*holding_cost)/sum(failure_rates)  #24801.5530506

    assignment1    = np.array([[0, 1, 1, 0, 1], [1, 0, 1, 1, 0]])
    assignment2    = np.array([[0, 0, 1, 0, 0], [1, 1, 1, 1, 1]])


    # totalCost, S, EBO, _, utilizationRatesAverage, skillServerAssignmentStatisitcsAverage = \
    #     simulation_optimization_run(failure_rates, service_rates, holding_cost, penalty, assignment2,
    #                                   numberWarmingUpRequests = 5000,
    #                                   stopNrRequests = 1000000,
    #                                   replications = 50,
    #                                   runTheSimulation = True, useDatabase = False)
    #     # simulation_optimization_run(failure_rates, service_rates, holding_costs, penalty_cost, fullflex_assignment, runTheSimulation=True, useDatabase=False)


    totalCostList = \
        simulation_optimization_runV2(failure_rates, service_rates, holding_cost, penalty, assignment2,
                                      numberWarmingUpRequests = 5000,
                                      stopNrRequests = 1000000,
                                      replications = 10)


    print totalCostList
    # print S*holding_cost + penalty*EBO
    # print utilizationRatesAverage
    # print skillServerAssignmentStatisitcsAverage

    print time.time() - startTime
    pass
