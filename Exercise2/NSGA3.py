import array
import configparser
import random
import time

import matplotlib.pyplot as plt
import numpy
from deap import base, algorithms
from deap import creator
from deap import tools
from deap.benchmarks.tools import hypervolume, convergence, diversity
from pymop import factory

config = configparser.ConfigParser()
config.read("configuration.ini")
PROBLEM = str(config['Properties']['PROBLEM'])
BOUND_LOW = float(config['Properties']['BOUND_LOW'])
BOUND_UP = float(config['Properties']['BOUND_UP'])
NDIM = int(config['Properties']['DIMENSIONS'])

ALGORITHM = str(config['Properties']['ALGORITHM'])
NGEN = int(config['Properties']['NGEN'])
MU = int(config['Properties']['POPULATION'])
CXPB = float(config['Properties']['CXPB'])
CXETA = float(config['Properties']['CXETA'])
MUTETA = float(config['Properties']['MUTETA'])
MUTPB = 0.9

problem = factory.get_problem(PROBLEM, n_var=NDIM)

creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)


def uniform(low, up, size=None):
    try:
        return [random.uniform(a, b) for a, b in zip(low, up)]
    except TypeError:
        return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]


toolbox = base.Toolbox()
toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, NDIM)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", problem.evaluate, return_values_of=["F"])
toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=CXETA)
toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=MUTETA, indpb=1.0 / NDIM)
toolbox.register("select", tools.selNSGA3, ref_points=factory.get_problem(PROBLEM).pareto_front(n_pareto_points=MU))


def main(seed=None):
    random.seed(seed)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean, axis=0)
    stats.register("std", numpy.std, axis=0)
    stats.register("min", numpy.min, axis=0)
    stats.register("max", numpy.max, axis=0)

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"

    pop = toolbox.population(n=MU)
    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(invalid_ind), **record)

    # Begin the generational process
    for gen in range(1, NGEN):
        offspring = algorithms.varAnd(pop, toolbox, CXPB, MUTPB)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Select the next generation population from parents and offspring
        pop = toolbox.select(pop + offspring, MU)

        # Compile statistics about the new population
        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)

    print("Final population hypervolume is %f" % hypervolume(pop, [11.0, 11.0]))
    return pop, logbook


if __name__ == "__main__":
    start = time.time()
    pop, stats = main()
    time_elapsed = time.time() - start

    pop_fit = numpy.array([ind.fitness.values for ind in pop])
    pf = factory.get_problem(PROBLEM).pareto_front(n_pareto_points=MU)

    pop.sort(key=lambda x: x.fitness.values)
    print(stats)
    print("Time elapsed: ", time_elapsed)
    print("Convergence: ", convergence(pop, pf))
    print("Diversity: ", diversity(pop, pf[0], pf[-1]))

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)

    ax.scatter(pf[:, 0], pf[:, 1], marker="s", s=20, color='black', label="Ideal Pareto Front")
    ax.scatter(pop_fit[:, 0], pop_fit[:, 1], marker="o", s=20, color='red', label="Final Population")
    plt.grid()
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.show()
