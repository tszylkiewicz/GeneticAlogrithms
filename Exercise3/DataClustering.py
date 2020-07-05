import collections
import configparser
import random
import scipy.special
import time

import numpy as np
import matplotlib.pyplot as plt

from deap import creator, base, tools, algorithms
from scipy.spatial import distance

from DatasetManager import DatasetManager


def choose_cluster(current_chromosome):
    condition = True
    cluster_center = []
    iterations = 0
    while condition:
        far_enough = True
        cluster_center = np.asarray(random.choice(manager.data))
        for i in range(len(current_chromosome)):
            if distance.euclidean(current_chromosome[i], cluster_center) < min_distance:
                far_enough = False
        if far_enough or iterations == 100:
            condition = False
        iterations += 1
    return cluster_center


def generate_chromosome(min_size, max_size):
    size = random.randint(min_size, max_size)

    chromosome = np.zeros((size, manager.dimension))
    for i in range(size):
        chromosome[i] = choose_cluster(chromosome[0:i])

    return chromosome.reshape(-1)


def kmeans_assignment(individual):
    centroids = np.reshape(individual,
                           (int(len(individual)/manager.dimension), manager.dimension))
    assignments = np.zeros(len(manager.data), dtype=int)
    for i in range(len(manager.data)):
        for j in range(len(centroids)):
            if distance.euclidean(manager.data[i], centroids[j]) < distance.euclidean(manager.data[i], centroids[assignments[i]]):
                assignments[i] = j
    return assignments


def mutCluster(individual, indpb):
    lower_bounds, upper_bounds = manager.get_data_bounds()
    for ind, val in enumerate(individual):
        if random.random() < indpb:
            t_l = (individual[ind] - lower_bounds[ind %
                                                  manager.dimension]) / 2.
            t_u = (upper_bounds[ind %
                                manager.dimension] - individual[ind]) / 2.
            individual[ind] = random.uniform(
                individual[ind]-t_l, individual[ind]+t_u)


def mutAddCluster(individual):

    if int(len(individual)/manager.dimension) < max_clusters:
        new_cluster = choose_cluster(individual)
        individual.extend(new_cluster)


def mutDelCluster(individual):

    if int(len(individual)/manager.dimension) > min_clusters:
        index = random.randrange(int(len(individual)/manager.dimension))
        del individual[index*manager.dimension:(index+1)*manager.dimension]


def evaluate(ind):
    n_clusters = int(len(ind)/manager.dimension)

    centroids = np.reshape(ind, (n_clusters, manager.dimension))
    assignments = kmeans_assignment(ind)

    grouped_data = manager.group_data(assignments, n_clusters)

    min_avg = np.zeros(len(centroids))
    max_diff = np.zeros(len(centroids))
    for k in range(n_clusters):
        first_part = 0
        n = len(grouped_data[k])
        if n != 0:
            for i in range(n):
                for j in range(i + 1, n):
                    first_part += distance.euclidean(
                        grouped_data[k][i], grouped_data[k][j])

                max_d = 10000
                for l in range(n_clusters):
                    if l != k:
                        for j in range(len(grouped_data[l])):
                            if distance.euclidean(grouped_data[k][i], grouped_data[l][j]) < max_d:
                                max_d = distance.euclidean(
                                    grouped_data[k][i], grouped_data[l][j])
                        max_diff[l] += max_d
                min_avg[k] = first_part / n

    return np.sum(min_avg), np.sum(max_diff)


def main(seed=None):
    random.seed(seed)
    random.randint    

    population = toolbox.population(n=MU)
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"

    fitnesses = map(toolbox.evaluate, population)
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    hof.update(population)
    record = stats.compile(population)
    logbook.record(gen=0, evals=len(population), **record)
    print(logbook.stream)

    for g in range(1, NGEN):
        offspring = [toolbox.clone(ind) for ind in population]

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values, child2.fitness.values

        for ind in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(ind)
                del ind.fitness.values
            if random.random() < ADDPB:
                toolbox.addcluster(ind)
                del ind.fitness.values
            if random.random() < DELPB:
                toolbox.delcluster(ind)
                del ind.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        population = toolbox.select(population + offspring, MU)
        hof.update(population)
        record = stats.compile(population)
        logbook.record(gen=g, evals=len(invalid_ind), **record)
        print(logbook.stream)
    print(hof)
    return population, logbook, hof


def plot_histogram(log):
    gen = log.select("gen")
    fit_mins = np.array(log.select("min"))

    fig = plt.figure()

    plt.plot(gen, fit_mins[:, 0])
    plt.plot(gen, fit_mins[:, 1])
    fig.suptitle('Fitness plot')
    plt.legend(['Min average distance inside cluster', 'Max distance between cluster elements'], loc='center right')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.show()


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("configuration.ini")

    min_clusters = int(config['Properties']['MIN_CLUSTERS'])
    max_clusters = int(config['Properties']['MAX_CLUSTERS'])
    min_distance = float(config['Properties']['MIN_DISTANCE'])

    MU = int(config['Properties']['MU'])
    NGEN = int(config['Properties']['NGEN'])
    CXPB = float(config['Properties']['CXPB'])
    INDPB = float(config['Properties']['INDPB'])
    MUTPB = float(config['Properties']['MUTPB'])    
    ADDPB = float(config['Properties']['ADDPB'])
    DELPB = float(config['Properties']['DELPB'])

    manager = DatasetManager()
    manager.load_dataset()

    creator.create("FitnessMulti", base.Fitness, weights=(-1.0, 1.0))
    creator.create("Individual", list, fitness=creator.FitnessMulti)

    toolbox = base.Toolbox()
    toolbox.register("indices", generate_chromosome,
                     min_size=min_clusters, max_size=max_clusters)
    toolbox.register("individual", tools.initIterate,
                     creator.Individual, toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("mutate", mutCluster, indpb=INDPB)
    toolbox.register("addcluster", mutAddCluster)
    toolbox.register("delcluster", mutDelCluster)
    toolbox.register("select", tools.selNSGA2)
    
    start = time.time()
    pop, logbook, hof = main()
    time_elapsed = time.time() - start

    assign = kmeans_assignment(hof[0])
    manager.plot_data()
    manager.plot_clustered_data(hof[0], assign)
    plot_histogram(logbook)
    # print(assign)
    # print(manager.target)
    print("Time elapsed: ", time_elapsed)
    print(collections.Counter(assign))
    print(collections.Counter(manager.target))
