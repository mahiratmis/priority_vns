import numpy as np
from pulp import *


get_bin = lambda x, n: x >= 0 and str(bin(x))[2:].zfill(n) or "-" + str(bin(x))[3:].zfill(n)


def compute_number_skill_combinations(n_servers,n_skills):
    prod=1
    sm=1
    N2 = 2**n_skills-1
    for i in range(n_servers):
        # prod *= (N2+i)/(i+1) # i runs from 1 to n_servers-1, and we need to add 1 to it
        prod *= (N2+i)
        prod /= (i+1) # i runs from 1 to n_servers-1, and we need to add 1 to it
        # don't combine in one line, there will be problems with division
        sm   += prod
    return sm


def generateSkillsMultiServerInner(n_servers, m_skills_start, m_skills_end):
    if n_servers==0:
        yield []
    else:
        for i in range(m_skills_start, m_skills_end+1):
            for vect in generateSkillsMultiServerInner(n_servers-1, i, m_skills_end):
                yield [i]+vect


def generateSkillsMultiServer(n_servers, m_skills):
    for vect_n in generateSkillsMultiServerInner(n_servers,0,2**(m_skills)-1):
        all_data = np.zeros((m_skills,0), dtype=int)
        for j in vect_n:
            all_data = np.append(all_data,
                                 [[int(x)] for x in str(bin(j))[2:].zfill(m_skills)], 1)
        yield all_data

def generateSkillsMultiServerV1(n_servers, m_skills):
    for vect_n in generateSkillsMultiServerInner(n_servers,0,2**(m_skills)-1):
        yield np.array([[int(x) for x in str(bin(v))[2:].zfill(m_skills)] for v in vect_n])



def checkSkillAssignment(assignement, lambdas, mus):
    ''' running this function too frequest might raise an error with too many opened files (nulldev is opened to suppress output)
        possible solutions are
        run:  ulimit -S -n 1024  (currrent value is 256)
    '''
    try:
        # print "Check Assignment:", assignement, lambdas, mus

        # with LpProblem("AssignmentVerification", LpMinimize) as prob:
        prob = LpProblem("AssignmentVerification", LpMinimize)

        m_skills = assignement.shape[1]
        n_servers = assignement.shape[0]

        assert m_skills == len(lambdas)

        alpha = LpVariable.dicts("alpha", itertools.product(range(m_skills), range(n_servers)), lowBound=0.0, upBound=1.0)
        maxRho   = LpVariable("maxRho", lowBound=0.0, upBound=0.999)
        prob += maxRho

        for i in range(m_skills):
            prob += lpSum([alpha[(i,j)] for j in range(n_servers)]) == 1
        for j in range(n_servers):
            prob += lpSum([alpha[(i,j)]*lambdas[i]/mus[i] for i in range(m_skills)]) <= maxRho
        for i in range(m_skills):
            for j in range(n_servers):
                if assignement[j,i] == 0:
                    prob += alpha[(i,j)] <= 0

        status = prob.solve() == 1 # solved to optimum
        retalphas = None
        if status:
            retalphas = np.array([[alpha[(i,j)].value() for i in range(m_skills)] for j in range(n_servers)])

        return status, retalphas
    except:
        print "Something went wrong with LpProblem"
        raise
    pass


if __name__ == '__main__':

    # checkSkillAssignment(np.array([[1,1,1,1],[1,1,1,1]]), )

    # n_servers=int(sys.argv[1])
    # n_skils = int(sys.argv[2])
    #
    # for vect_n in generateSkillsMultiServer(n_servers,0,2**(n_skils)-1):
    #     for j in vect_n:
    #         print get_bin(j, n_skils),
    #     print
        
    pass