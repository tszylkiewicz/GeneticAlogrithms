import array
import random
import json

import numpy
import configparser
from math import sqrt, sin, cos
import matplotlib.pyplot as plt

from deap import algorithms
from deap import base
from deap import benchmarks
from deap.benchmarks.tools import diversity, convergence, hypervolume
from deap import creator
from deap import tools


def fun1(individual):
    return sum(sin(x) * cos(x1)
               for x, x1 in zip(individual[:-1], individual[1:])),


def fun2(individual):
    return sum(sin(x) * cos(x1) + x + x1
               for x, x1 in zip(individual[:-1], individual[1:])),


def fun3(individual):
    return sum(sin(x) * cos(x1) + (x * x) + (x1 + x1)
               for x, x1 in zip(individual[:-1], individual[1:])),


def uniform(low, up, size=None):
    try:
        return [random.uniform(a, b) for a, b in zip(low, up)]
    except TypeError:
        return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]


def main(seed=None):
    random.seed(seed)
    random.randint
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean, axis=0)
    stats.register("std", numpy.std, axis=0)
    stats.register("min", numpy.min, axis=0)
    stats.register("max", numpy.max, axis=0)
    hof = tools.HallOfFame(1)
    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"

    pop = toolbox.population(n=POPULATION)

    fitnesses = toolbox.map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    hof.update(pop)
    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(pop), **record)
    print(logbook.stream)

    # Begin the generational process
    for gen in range(1, MAX_GENERATIONS):
        offspring = toolbox.select(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]

        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            if random.random() <= CXPB:
                toolbox.mate(ind1, ind2)
                del ind1.fitness.values, ind2.fitness.values

        for mutant in offspring:
            toolbox.mutate(mutant)
            del mutant.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        pop[:] = offspring
        hof.update(pop)
        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        print(logbook.stream)
    
    return pop, logbook


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("configuration.ini")
    BOUND_LOW = float(config['Properties']['BOUND_LOW'])
    BOUND_UP = float(config['Properties']['BOUND_UP'])
    MAX_GENERATIONS = int(config['Properties']['MAX_GENERATIONS'])
    POPULATION = int(config['Properties']['POPULATION'])
    DIMENSIONS = int(config['Properties']['DIMENSIONS'])
    CXPB = float(config['Properties']['CROSSOVER_PROBABILITY'])
    MUTPB = float(config['Properties']['MUTATION_PROBABILITY'])

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, DIMENSIONS)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    # benchmarks.schwefel()
    toolbox.register("evaluate", eval(config['Properties']['EVALUATION']))
    if config['Properties']['CROSSOVER'] == "TwoPoint":
        toolbox.register("mate", tools.cxTwoPoint)
    else:
        toolbox.register("mate", tools.cxUniform, indpb=0.5)
    if config['Properties']['MUTATION'] == "Gaussian":
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=MUTPB)
    else:
        toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=1.0,
                         indpb=MUTPB)
    toolbox.register("select", tools.selTournament, tournsize=2)

    pop, logbook = main()

    gen = logbook.select("gen")
    fit_mins = logbook.select("min")
    size_avgs = logbook.select("avg")

    fig, ax1 = plt.subplots()
    # line1 = ax1.plot(gen[60::], fit_mins[60::], "b-", label="Minimum Fitness")
    line1 = ax1.plot(gen, fit_mins, "b-", label="Minimum Fitness")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness", color="b")
    for tl in ax1.get_yticklabels():
        tl.set_color("b")

    # ax2 = ax1.twinx()
    # line2 = ax2.plot(gen[60::], size_avgs[60::], "r-", label="Average Size")
    # line2 = ax2.plot(gen, size_avgs, "r-", label="Average Size")
    # ax2.set_ylabel("Size", color="r")
    # for tl in ax2.get_yticklabels():
    #     tl.set_color("r")

    lns = line1
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc="center right")

    plt.show()
