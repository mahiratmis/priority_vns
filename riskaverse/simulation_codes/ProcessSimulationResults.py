import json
import numpy as np
from skill_set_generator import generateSkillsMultiServer, checkSkillAssignment, compute_number_skill_combinations


# json_in_location  = "../simulated_sets_version3.json"
# json_out_location = "simulated_sets_version4.json"
#
# with open(json_out_location, "w") as json_out:
#
#     with open(json_in_location, "r") as json_in:
#         json_out.write(json_in.readline()) # copy the first line, it containes general characteristics of the simulations
#         for line in json_in:
#             json_out_case = dict()
#             json_in_case = json.loads(line) # read each line as separate json object (dictionary)
#
#             json_out_case["caseID"]        = json_in_case["caseID"]
#             json_out_case["m_servers"]     = json_in_case["n_servers"]
#             json_out_case["n_classes"]     = json_in_case["SKU_number"]
#             json_out_case["arrival_rates"] = json_in_case["failure_rates"]
#             json_out_case["service_rates"] = json_in_case["service_rates"]
#             json_out_case["assignments"]   = dict() #json_in_case["assignments"]
#             for assignmentID, assignment in json_in_case["assignments"].items():
#                 json_out_case["assignments"][assignmentID] = dict()
#                 json_out_case["assignments"][assignmentID]["class_server_assignment"]    = assignment["skill_assignment"]
#                 json_out_case["assignments"][assignmentID]["queue_length_distributions"] = assignment["queue_length_distributions"]
#                 json_out_case["assignments"][assignmentID]["utilization_rates"]          = assignment["utilization_rates"]
#                 json_out_case["assignments"][assignmentID]["class_server_distribution"]  = assignment["skill_server_distribution"]
#
#
#             json_out.write("{}\n".format(json.dumps(json_out_case)))
#
#

json_location = "simulated_sets_version4.json"

with open(json_location, "r") as json_file:
    # ignore the first line, it containes general characteristics of the simulations
    json_file.readline()
    total = 0
    for line in json_file:
        json_case = json.loads(line) # read each line as separate json object (dictionary)

        n_classes = json_case["n_classes"]
        m_servers = json_case["m_servers"]
        arrivalRates = np.array(json_case["arrival_rates"])
        serviceRates = np.array(json_case["service_rates"])

        total += len(json_case["assignments"])

    #     print        m_skills, n_servers, sum(arrivalRates/serviceRates)/n_servers, len(json_case["assignments"]), compute_number_skill_combinations(n_servers,m_skills)
    #
    # print total

        for assignmentID, assignment in json_case["assignments"].items():
            # read class-server assignments
            # assignment matrix containes class-server assignments as 0-1 values
            # column corresponds to class, row to server
            assignment_matrix = np.array(assignment["class_server_assignment"])

            # read marginal queue length distributions and convert it to np.array
            # the matrix contains probabilities that each class has q items in the system
            # rows (first index) correspond to classes
            # columns (second index) correspond to the number q of items (clients)
            # of each class in the system
            # i.e. marginalDistribution[cl, q] is the probability
            # that class cl has q items in the system
            maxQueue = max(max(int(n) for n in val.keys())
                           for val in assignment["queue_length_distributions"].values())+1
            marginalDistribution = \
                np.zeros((len(assignment["queue_length_distributions"]), maxQueue))
            for cl in assignment["queue_length_distributions"]:
                for q in assignment["queue_length_distributions"][cl]:
                    marginalDistribution[int(cl), int(q)] = \
                        assignment["queue_length_distributions"][cl][q]

            # read class-server utilization rates
            # utilization rates matrix containes class-server utilizations
            # (utilization of server time capacity)
            # column corresponds to class, row to server
            utilization_rates = np.array(assignment["utilization_rates"])

            # read class-server distributions
            # class server distribution matrix containes percentages
            # of class flows assigned to servers
            # column corresponds to class, row to server
            class_server_distribution = np.array(assignment["class_server_distribution"])

            print arrivalRates, serviceRates, assignment_matrix.tolist(), \
                np.sum(marginalDistribution*np.array(range(maxQueue)), axis=1)