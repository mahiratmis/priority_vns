from __future__ import division
import numpy as np

# from memory_profiler import profile

def MMC(lmbda, mu, c, maxQueueLength = 500, eps = 1e-10):

    if maxQueueLength < c:
        raise ValueError("c(" + str(c) + ") must be smaller than maxQueueLength (" + str(maxQueueLength) + ")")

    rho = lmbda/mu/c

    marginalDistribution = np.zeros(maxQueueLength)

    marginalDistribution[0] = 1.0
    for i in xrange(1, c):
        marginalDistribution[i] = marginalDistribution[i-1]*rho*c/i

    p0 = 1/ (np.sum(marginalDistribution) + marginalDistribution[c-1]*rho/(1-rho))


    marginalDistribution *= p0

    i=c
    tail = 1 - np.sum(marginalDistribution)
    while i < maxQueueLength and tail > eps:
        marginalDistribution[i] = marginalDistribution[i-1]*rho
        tail -= marginalDistribution[i]
        i += 1

    i -= 1

    EN = lmbda/mu + marginalDistribution[c]*rho/(1-rho)/(1-rho)

    return marginalDistribution, EN

def MMC_fractional_capacity(lmbda, mu, capacity):
    rho = lmbda/mu/capacity

    if capacity>1:
        lw_cap = int(np.ceil(capacity)-1)
        mu_lw = lmbda/rho/lw_cap
        marginal_lw = MMC(lmbda, mu_lw, lw_cap)

    up_cap = int(np.ceil(capacity))
    mu_up = lmbda/rho/up_cap
    marginal_up = MMC(lmbda, mu_up, up_cap)

    if capacity > 1:
        marginal = np.trim_zeros(marginal_lw*(up_cap - capacity) + marginal_up*(capacity - lw_cap), "b")
    else:
        marginal = np.trim_zeros(marginal_up, "b")

    return marginal


def ComputeMarginalDistributions(lambdas, mus, nServers=1, maxQueueLength = 500):

    marginalDistribution = np.zeros((len(lambdas), maxQueueLength))

    rho = np.sum(lambdas/mus)/nServers
    # single server queue with same service rates

    alpha = lambdas/np.sum(lambdas)
    for k in xrange(len(lambdas)):
        marginalDistribution[k,0] = (1-rho)/(1-(1-alpha[k])*rho)
        for i in xrange(1, maxQueueLength):
            marginalDistribution[k,i] = marginalDistribution[k,i-1]*(alpha[k]*rho)/(1-(1-alpha[k])*rho)

    return marginalDistribution

# @profile
def OptimizeStockLevelsAndCosts(holdingCosts, penalty, marginalDistribution):

    if not isinstance(holdingCosts, np.ndarray):
        holdingCosts = np.array(holdingCosts)

    if len(marginalDistribution.shape) == 1:
        marginalDistribution = marginalDistribution.reshape(1,len(marginalDistribution))

    nSKUs=len(holdingCosts)
    maxQueue = marginalDistribution.shape[1]
    n_array = np.array(range(maxQueue))
    S = np.zeros(nSKUs, dtype=int)
    PBO = np.ones(nSKUs) -  marginalDistribution[:,0]
    EBO = np.sum(marginalDistribution*np.array(range(marginalDistribution.shape[1])), axis=1)

    hb_ratio = holdingCosts/penalty
    for sk in xrange(nSKUs):
        while S[sk]<maxQueue and np.sum(marginalDistribution[sk, S[sk]+1:]) > hb_ratio[sk]:
            S[sk] += 1
            EBO[sk] -= PBO[sk]
            PBO[sk] -= marginalDistribution[sk, S[sk]]
            # EBO[sk] -= PBO[sk]

    totalCost = np.sum(S*holdingCosts) + np.sum(penalty*EBO)

    return totalCost, S, EBO

def optimize_stock_server(lmbda, mu, serverCost, holdingCost, penalty, capacityStep = 0.01):
    ''' finds the optimal stock and server capacity based on '''
    deltaCost = 1
    optCapacity = lmbda/mu+capacityStep
    marginalDistribution = MMC_fractional_capacity(lmbda, mu, optCapacity)
    _, _, optTotalCost = OptimizeStockLevelsAndCosts([holdingCost], penalty, marginalDistribution)
    optTotalCost += optCapacity*serverCost

    while deltaCost > 0:
        marginalDistribution = MMC_fractional_capacity(lmbda, mu, optCapacity+capacityStep)
        _, _, newTotalCost = OptimizeStockLevelsAndCosts([holdingCost], penalty, marginalDistribution)
        newTotalCost += optCapacity*serverCost

        deltaCost = optTotalCost - newTotalCost
        if deltaCost > 0:
            optCapacity += capacityStep
            optTotalCost = newTotalCost


    marginalDistribution = MMC_fractional_capacity(lmbda, mu, optCapacity)
    optS, optEBO, optTotalCost = OptimizeStockLevelsAndCosts([holdingCost], penalty, marginalDistribution)

    optTotalCost += optCapacity*serverCost

    return optS, optCapacity, optEBO, optTotalCost


def OptimizeStockLevelsAndCostsQueueing(lambdas, mus, holdingCosts, penalty, nServers=1, maxQueueLength = 500):
    # =================== compute stock levels, probabiities and costs =======================
    marginalDistribution = ComputeMarginalDistributions(lambdas, mus, nServers=nServers, maxQueueLength = maxQueueLength)

    return OptimizeStockLevelsAndCosts(holdingCosts, penalty, marginalDistribution), marginalDistribution


if __name__ == '__main__':

    pass

#print optimize_stock_server(12, 6.0, 10, 100, 1000)

    #pass
