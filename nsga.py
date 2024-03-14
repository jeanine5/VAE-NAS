"""
This file contains the implementation of the NSGA-2 algorithm. It is a multi-objective evolutionary algorithm
that is used for the evolutionary search of neural network architectures. The algorithm is implemented in the
NSGA_II class. The algorithm is used in EcoNAS/Training/CIFAR.py.
"""
from matplotlib import pyplot as plt

from EcoNAS.EA.genetic_functions import *
from EcoNAS.EA.pareto_functions import *
import functools
import numpy as np

from EcoNAS.SearchSpace.regression_predictors.cifar_predictor import CIFARBenchmark
from EcoNAS.SearchSpace.regression_predictors.mnist_predictor import MNISTBenchmark


class NSGA_II:
    def __init__(self, population_size, generations, crossover_factor, mutation_factor):
        self.population_size = population_size
        self.generations = generations
        self.crossover_factor = crossover_factor
        self.mutation_factor = mutation_factor

    def initial_population(self, max_hidden_layers, max_hidden_size, data_name):
        """
        Initialize the population pool with random deep neural architectures
        :param max_hidden_layers: Maximum number of hidden layers
        :param max_hidden_size: Maximum number of hidden units per layer
        :return: Returns a list of NeuralArchitecture objects
        """
        if data_name == 'MNIST':
            input_size = 784
            output_size = 10
        else:
            input_size = 3072
            output_size = 10

        archs = []
        for _ in range(self.population_size):
            num_hidden_layers = random.randint(3, max_hidden_layers)
            hidden_sizes = [random.randint(10, max_hidden_size) for _ in range(num_hidden_layers)]
            arch = NeuralArchitecture(input_size, output_size, hidden_sizes)

            archs.append(arch)

        return archs

    def evolve(self, hidden_layers, hidden_size, data_name):
        """
        The NSGA-2 algorithm. It evolves the population for a given number of generations, however
        there is quite a bit of excessive training going on here.
        :param hidden_layers:
        :param hidden_size:
        :param data_name:
        :return: List of the best performing NeuralArchitecture objects of size of at most population_size
        """
        # step 0: initial search space
        if data_name == 'MNIST':
            regression_trainer = MNISTBenchmark('../SearchSpace/precomputed_datasets/trained_mnist_dataset.csv')
        else:
            regression_trainer = CIFARBenchmark('../SearchSpace/precomputed_datasets/trained_cifar10_dataset.csv')

        regression_trainer.train_models()
        regression_trainer.evaluate_models()

        evolutionary_path = []

        # step 1: generate initial population
        archs = self.initial_population(hidden_layers, hidden_size, data_name)

        # step 2 : evaluate the objective functions for each arch
        for a in archs:
            predicted_performance = regression_trainer.predict_performance(a)
            a.objectives = {
                'accuracy': predicted_performance[0],
                'introspectability': predicted_performance[1],
                'flops': predicted_performance[2]
            }

        # step 3: set the non-dominated ranks for the population and sort the architectures by rank
        set_non_dominated(archs)  # fitness vals
        archs.sort(key=lambda arch: arch.nondominated_rank)

        # step 4: create an offspring population Q0 of size N
        offspring_pop = generate_offspring(archs, self.crossover_factor, self.mutation_factor, regression_trainer)

        # step 5: start algorithm's counter
        for generation in range(self.generations):
            print(f'Generation: {generation}')
            # step 6: combine parent and offspring population
            combined_population = archs + offspring_pop  # of size 2N
            set_non_dominated(combined_population)

            population_by_objectives = np.array([[ind.objectives['accuracy'], ind.objectives['introspectability'],
                                                  ind.objectives['flops']] for ind in combined_population])

            generation_metrics = []
            for ind in archs:
                # Convert objectives to a dictionary for clarity
                objectives_dict = {
                    'accuracy': ind.objectives['accuracy'],
                    'introspectability': ind.objectives['introspectability'],
                    'flops': ind.objectives['flops']
                }
                generation_metrics.append({'objectives': objectives_dict})

            evolutionary_path.append(generation_metrics)

            # step 7:
            non_dom_fronts = fast_non_dominating_sort(population_by_objectives)

            # step 8: initialize new parent list and non-dominated front counter
            archs, i = [], 0

            # step 9: calculate crowding-distance in Fi until the parent population is filled
            while len(archs) + len(non_dom_fronts[i]) <= self.population_size:
                corresponding_archs = get_corr_archs(non_dom_fronts[i], combined_population)
                # calculated crowding-distance
                crowding_metric = crowding_distance_assignment(population_by_objectives, non_dom_fronts[i])
                for j in range(len(corresponding_archs)):
                    predicted_performance = regression_trainer.predict_performance(corresponding_archs[j])
                    corresponding_archs[j].objectives = {
                        'accuracy': predicted_performance[0],
                        'introspectability': predicted_performance[1],
                        'flops': predicted_performance[2]
                    }
                    corresponding_archs[j].crowding_distance = crowding_metric[j]
                archs += corresponding_archs
                i += 1

            # step 8: sort front by crowding comparison operator
            last_front_archs = get_corr_archs(non_dom_fronts[i], combined_population)
            last_front_archs.sort(key=functools.cmp_to_key(crowded_comparison_operator), reverse=True)

            archs = archs + last_front_archs[1: self.population_size - len(archs)]

            offspring_pop = generate_offspring(archs, self.crossover_factor, self.mutation_factor, regression_trainer)

        #plot_evolutionary_path(evolutionary_path, 'accuracy', data_name)
        plot_evolutionary_path(evolutionary_path, 'introspectability', data_name)
        plot_evolutionary_path(evolutionary_path, 'flops', data_name)

        return archs


def plot_evolutionary_path(evolutionary_path, metric_name, data_name):
    plt.figure(figsize=(10, 6))

    for i, generation_metrics in enumerate(evolutionary_path):
        if i % 4 == 0:
            # Extract the metric values for each architecture in the current generation
            metric_values = [arch['objectives'][metric_name] for arch in generation_metrics]

            # Plot the metric values for each architecture in the current generation
            plt.plot(metric_values, label=f'Generation {i + 1}')

    plt.xlabel('Architecture Index')
    plt.ylabel(metric_name)
    plt.title(f'Evolutionary Path of NSGA-2 on {data_name}')
    plt.legend()
    plt.show()
