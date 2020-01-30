import sys, os
import numpy as np
import time

parentPath = os.path.abspath("..")
if parentPath not in sys.path:
    sys.path.insert(0, parentPath)

import simulation_codes as sc

def run_time_test():
    np.set_printoptions(linewidth=50000)
    startTime = time.time()

    expectedTotalNumberRequests = 50000
    replications = 70
    maxQueue = 500
    nCores = None  # maximum number of cores will be used

    n_servers = 2
    m_skills = 4
    failureRates = np.array([0.21515405370607182, 0.9848459462939283, 0.21515405370607182, 0.9848459462939283])
    serviceRates = np.array([1.0, 1.0, 1.0, 1.0])
    skillServerAssignment = np.array([[0,1,0,1],[1,0,1,0]])


    numberInSystemDistributionAverage, waitingTimeHistogramAverage, utilizationRatesAverage, skillServerAssignmentStatisitcsAverage \
       = sc.simulation_batch_run(failureRates, serviceRates, skillServerAssignment,
                                 histogramBucketsInput = np.array(range(10)),
                                 expectedTotalNumberRequests = expectedTotalNumberRequests,
                                 numberWarmingUpRequests = 1000,
                                 replications=replications,
                                 maxQueue=maxQueue,
                                 nCores=nCores)
    print "n_servers", n_servers
    print "m_skills", m_skills
    print "stopTime", expectedTotalNumberRequests
    print "replications", replications

    print "Runtime:", time.time() - startTime
    pass

if __name__ == '__main__':
    # run_time_test()
    # run_time_test()
    # run_time_test()
    # run_time_test()
    # run_time_test_single_item()
    run_time_test()
    pass