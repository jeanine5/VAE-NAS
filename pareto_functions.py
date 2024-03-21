"""
This file contains the functions used to calculate the Pareto fronts and the crowding distance. Here, we have the
functions for calculating the Pareto fronts, crowding distance, and the non-dominated rank. These functions are
used in the NSGA-II class in EcoNAS/EA/NSGA.py. Note, we have conflicting objectives.
"""

import numpy as np
from vae import VAE, VAEArchitectures


def get_corr_archs(front, architectures: list[VAEArchitectures]):
    """
    Get the architectures corresponding to the indices in the front
    :param front: list of indices
    :param architectures: list of NeuralArchitecture objects
    :return: list of NeuralArchitecture objects
    """
    corr_archs = []
    for idx in front:
        corr_archs.append(architectures[idx])

    return corr_archs


def crowded_comparison_operator(ind1: VAEArchitectures, ind2: VAEArchitectures):
    """
    Crowded comparison operator defined from Deb et al. (2002). https://ieeexplore.ieee.org/document/996017
    :param ind1: NeuralArchitecture object
    :param ind2: NeuralArchitecture object
    :return: True if ind1 is better than ind2, False otherwise
    """
    if ind1.nondominated_rank < ind2.nondominated_rank:
        return True
    elif ind1.nondominated_rank == ind2.nondominated_rank and ind1.crowding_distance > ind2.crowding_distance:
        return True
    else:
        return False


def set_non_dominated(population: list[VAEArchitectures]):
    """
    Set the non-dominated rank for each NeuralArchitecture object in the population
    :param population: list of NeuralArchitecture objects
    :return: None
    """
    pbo = [[ind.objectives['loss'], ind.objectives['OOD']] for ind in population]

    for i in range(len(population)):
        for j in range(len(population)):
            if i != j:
                dominates = is_pareto_dominant(pbo[i], pbo[j])
                if not dominates:
                    population[i].nondominated_rank += 1


def is_pareto_dominant(p, q):
    """
    Check if p dominates q
    :param p: list of fitness values
    :param q: list of fitness values
    :return: True if p dominates q, False otherwise
    """
    dom = p[0] < q[0]

    return dom


def fast_non_dominating_sort(population):
    """
    Fast non-dominated sort algorithm from Deb et al. (2002). https://ieeexplore.ieee.org/document/996017
    Code from: https://github.com/adam-katona/NSGA_2_tutorial/blob/master/NSGA_2_tutorial.ipynb
    :param population:  list of fitness values
    :return: list of Pareto fronts
    """

    domination_sets = []
    domination_counts = []
    for fitnesses_1 in population:
        current_domination_set = set()
        domination_counts.append(0)
        for i, fitnesses_2 in enumerate(population):
            if is_pareto_dominant(fitnesses_1, fitnesses_2):
                current_domination_set.add(i)
            elif is_pareto_dominant(fitnesses_2, fitnesses_1):
                domination_counts[-1] += 1

        domination_sets.append(current_domination_set)

    domination_counts = np.array(domination_counts)
    fronts = []
    while True:
        current_front = np.where(domination_counts == 0)[0]
        if len(current_front) == 0:
            break
        fronts.append(current_front)

        for individual in current_front:
            domination_counts[
                individual] = -1
            dominated_by_current_set = domination_sets[individual]
            for dominated_by_current in dominated_by_current_set:
                domination_counts[dominated_by_current] -= 1

    return fronts


def crowding_distance_assignment(pop_by_obj, front: list):
    """
    Crowding distance assignment from Deb et al. (2002). https://ieeexplore.ieee.org/document/996017
    Code from: https://github.com/adam-katona/NSGA_2_tutorial/blob/master/NSGA_2_tutorial.ipynb
    :param pop_by_obj:
    :param front:
    :return:
    """
    num_objectives = pop_by_obj.shape[1]
    num_individuals = pop_by_obj.shape[0]

    # Normalise each objectives, so they are in the range [0,1]
    # This is necessary, so each objective's contribution have the same magnitude to the crowding metric.
    normalized_fitnesses = np.zeros_like(pop_by_obj)
    for objective_i in range(num_objectives):
        min_val = np.min(pop_by_obj[:, objective_i])
        max_val = np.max(pop_by_obj[:, objective_i])
        val_range = max_val - min_val
        normalized_fitnesses[:, objective_i] = (pop_by_obj[:, objective_i] - min_val) / val_range

    fitnesses = normalized_fitnesses
    crowding_metrics = np.zeros(num_individuals)

    for objective_i in range(num_objectives):

        sorted_front = sorted(front, key=lambda x: fitnesses[x, objective_i])

        crowding_metrics[sorted_front[0]] = np.inf
        crowding_metrics[sorted_front[-1]] = np.inf
        if len(sorted_front) > 2:
            for i in range(1, len(sorted_front) - 1):
                crowding_metrics[sorted_front[i]] += fitnesses[sorted_front[i + 1], objective_i] - fitnesses[
                    sorted_front[i - 1], objective_i]

    return crowding_metrics


def fronts_to_nondomination_rank(fronts):
    """
    :param fronts:
    :return:
    """
    non_domination_rank_dict = {}
    for i, front in enumerate(fronts):
        for x in front:
            non_domination_rank_dict[x] = i
    return non_domination_rank_dict
