import random
import argparse
import numpy as np

import multiprocessing as mp

global MUTATE_ROUNDS, DEBUG, MONTE_CARLO_TIMES, POPULATION_SIZE, NUM_GENERATIONS, WELL_MUTATE_TRIES, RUNNERS, time_data, fitness_data, well_mutated_cnt

DEBUG = True
MONTE_CARLO_TIMES = 80
POPULATION_SIZE = 40
NUM_GENERATIONS = 15
WELL_MUTATE_TRIES = 0.4
MUTATE_ROUNDS = 2
RUNNERS = 3

parser = argparse.ArgumentParser(description='')
parser.add_argument('-c', type=str, help='test case number')
parser = parser.parse_args()
casenum=parser.c

NET_WORK='cases/Heuristic/map{}/dataset{}'.format(casenum,casenum)
SEED='cases/Heuristic/map{}/seed'.format(casenum)
SEED_B='cases/Heuristic/map{}/seed_balanced'.format(casenum)
OUTPUT='cases/Heuristic/map{}/score'.format(casenum)


time_data = None
fitness_data = None
well_mutated_cnt = None
if DEBUG:
    time_data = []
    well_mutated_cnt = 0
    fitness_data = []
    import time

def evaluate(map, individual, n) -> int :
    flag = [np.zeros(n, dtype=int), np.zeros(n, dtype=int)]
    flag_exposed = [np.zeros(n, dtype=int), np.zeros(n, dtype=int)]
    for iter in range(2):
        q = []
        for i, status in enumerate(individual[iter], 0):
            if status == 1:
                q.append(i)
                flag[iter][i] = 1
                flag_exposed[iter][i] = 1
        while len(q) != 0:
            node = q.pop(0)
            for edge in map[node]:
                if flag[iter][edge[0]] == 1:
                    continue
                flag_exposed[iter][edge[0]] = 1
                val = np.random.rand()
                if (val < edge[iter + 1]):
                    flag[iter][edge[0]] = 1
                    q.append(edge[0])
    ret = 0
    for i in range(n):
        if flag_exposed[0][i] == flag_exposed[1][i]:
            ret += 1
    return ret

def bitwise_or_bytearray(ba1, ba2):
    return bytearray(b1 | b2 for b1, b2 in zip(ba1, ba2))

def fitness(initial_set, individual, map, n, k, monte_carlo_times):
    ret = 0
    cnt = individual[0].count(1) + individual[1].count(1)
    if cnt > k:
        return -cnt
    fin_individual = [bitwise_or_bytearray(initial_set[i], individual[i]) for i in range(2)]
    for _ in range(monte_carlo_times):
        fitness_value = evaluate(map, fin_individual, n)
        ret += fitness_value
    ret /= monte_carlo_times
    return ret

def mutate(individual, n, k, degree):
    ones = individual[0].count(1) + individual[1].count(1)
    select_group = random.randint(0, 1)
    add = random.randint(0, 1)
    if add and ones < k:
        for i in range(int(WELL_MUTATE_TRIES * n)):
            select_point = random.choices(range(n), weights=degree)[0]
            if individual[select_group][select_point] == 0:

                if DEBUG:
                    global well_mutated_cnt
                    well_mutated_cnt += 1

                individual[select_group][select_point] = 1 - individual[select_group][select_point]
                return individual
            select_group = random.randint(0, 1)
    else:
        select_point = random.randint(0, n - 1)
        while individual[select_group][select_point] == 0 and ones >= k:
            select_group = random.randint(0, 1)
            select_point = random.randint(0, n - 1)
        individual[select_group][select_point] = 1 - individual[select_group][select_point]
        return individual

def crossover(parent1, parent2, n):
    crossover_point1 = random.randint(1, n - 2)
    crossover_point2 = random.randint(crossover_point1, n - 1)
    
    child1 = [bytearray(n) for _ in range(2)]
    child2 = [bytearray(n) for _ in range(2)]
    
    for i in range(2):
        child1[i][:crossover_point1] = parent1[i][:crossover_point1]
        child2[i][:crossover_point1] = parent2[i][:crossover_point1]
        
        child1[i][crossover_point2:] = parent1[i][crossover_point2:]
        child2[i][crossover_point2:] = parent2[i][crossover_point2:]
        
        child1[i][crossover_point1:crossover_point2] = parent2[i][crossover_point1:crossover_point2]
        child2[i][crossover_point1:crossover_point2] = parent1[i][crossover_point1:crossover_point2]
    
    return child1, child2

def select(population, fitness_values, runners_up=RUNNERS):
    selected_individuals = []

    for _ in range(2):
        nodes = random.choices(range(len(population)), weights=fitness_values, k=runners_up)

        winner = nodes[0]
        for i in range(1, runners_up):
            if fitness_values[nodes[i]] > fitness_values[winner]:
                winner = nodes[i]
        selected_individuals.append(population[winner])

    return selected_individuals[0], selected_individuals[1]

def initialize_individual(n, k, degree):
    state = [bytearray(n) for _ in range(2)]
    select_cnt = random.randint(1, k - 1)
    samples = random.choices(range(n), weights=degree, k=select_cnt)
    for i in range(select_cnt):
        state[0][samples[i]] = 1
    samples = random.choices(range(n), weights=degree, k=k - select_cnt)
    for i in range(k - select_cnt):
        state[1][samples[i]] = 1
    return state

def thread_worker(map, start, end, tasks, n):
    len_ = len(tasks)
    results = {}
    for i in range(start, end):
        if (i >= len_):
            break
        task = tasks[i]
        results[task[0]] = evaluate(map, task[1], n)
    return results

def evaluate_population(population, map, initial_set, n, monte_carlo_times):
    raw_tasks = []
    results = []
    fitness_values = np.zeros((len(population)))
    id = 0
    for individual in population:
        cnt = individual[0].count(1) + individual[1].count(1)
        if cnt > k:
            fitness_values[id] = -cnt
            id += 1
            continue
        fin_individual = [bitwise_or_bytearray(initial_set[i], individual[i]) for i in range(2)]
        for _ in range(monte_carlo_times):
            raw_tasks.append(((id, _), fin_individual))
        id += 1
    num_threads = 8
    step = len(raw_tasks) // num_threads
    results = []
    pool = mp.Pool(processes=num_threads)
    for i in range(num_threads):
        results.append(pool.apply_async(thread_worker, (map, i * step, (i + 1) * step, raw_tasks, n)))
    pool.close()
    pool.join()

    real_results = np.zeros((len(population), monte_carlo_times))
    for _result in results:
        result = _result.get()
        for key in result:
            real_results[key[0], key[1]] = result[key]
    real_results = np.mean(real_results, axis=1)
    fitness_values += real_results
    return fitness_values

import time

def evolutionary_algorithm(population_size, num_generations, map, initial_set, n, k, degree, monte_carlo_times):

    if DEBUG:
        global time_data, fitness_data
        total_start_time = time.time() 
    
    population = [initialize_individual(n, k, degree) for i in range(population_size)]

    if DEBUG:
        total_end_time = time.time()
        time_data.append(total_end_time - total_start_time)

    fitness_values = evaluate_population(population, map, initial_set, n, monte_carlo_times)

    if DEBUG:
        fitness_data.append(fitness_values)
        total_end_time = time.time()
        time_data.append(total_end_time - total_start_time)
        generation_time = []

    for generation in range(num_generations):

        if DEBUG:
            start_time = time.time()

        new_population = []

        for i in range(population_size):
            parent1, parent2 = select(population, fitness_values)

            child1, child2 = crossover(parent1, parent2, n)

            for _ in range(MUTATE_ROUNDS):
                child1 = mutate(child1, n, k, degree)
                child2 = mutate(child2, n, k, degree)

            new_population.append(child1)
            new_population.append(child2)
        
        new_fitness_values = evaluate_population(new_population, map, initial_set, n, monte_carlo_times)

        # append original values
        new_population.extend(population)
        new_fitness_values = np.append(new_fitness_values, fitness_values)
        sorted_index = np.argsort(new_fitness_values)[::-1]
        new_fitness_values = [new_fitness_values[i] for i in sorted_index][:population_size]
        new_population = [new_population[i] for i in sorted_index][:population_size]
        population = new_population
        fitness_values = new_fitness_values

        if DEBUG:
            end_time = time.time()
            generation_time.append(end_time - start_time)
            fitness_data.append(fitness_values)
            # print("Generation: ", generation, "Fitness: ", max(fitness_values), "Time: ", end_time - start_time)
            print("Generation: {:<3} Fitness: {:<6} Time: {:<5}".format(generation, max(fitness_values), end_time - start_time))

    if DEBUG:
        total_end_time = time.time()
        time_data.append(generation_time)
        time_data.append(total_end_time - total_start_time)
        print("Total Time: ", total_end_time - total_start_time)

    best_pos = np.argmax(fitness_values)
    best_individual = population[best_pos]
    return best_individual, fitness_values[best_pos]

def read_map(network_file):
    with open(network_file, "r") as f:
        lines = f.readlines()
        numbers = lines[0].strip().split()
        n = int(numbers[0])
        m = int(numbers[1])
        degree = np.zeros(n, dtype=int)
        map = [[] for i in range(n)]
        for line in lines[1:]:
            numbers = line.strip().split()
            map[int(numbers[0])].append((int(numbers[1]), float(numbers[2]), float(numbers[3])))
            degree[int(numbers[0])] += 1
    return n, m, map, degree

def read_initial_set(initial_set_file, n):
    initial_set = [bytearray(n) for _ in range(2)]
    with open(initial_set_file, "r") as f:
        lines = f.readlines()
        numbers = lines[0].strip().split(' ')
        isn, ism = int(numbers[0]), int(numbers[1])
        for pos, line in enumerate(lines[1:], 1):
            node = int(line.strip())
            if pos > isn:
                initial_set[1][node] = 1
            else:
                initial_set[0][node] = 1
    return initial_set

if __name__ == "__main__":
    

    n, m, map, degree = read_map(NET_WORK)
    initial_set = read_initial_set(SEED, n)
    k = 5
    best_individual, fit = evolutionary_algorithm(POPULATION_SIZE, NUM_GENERATIONS, map, initial_set, n, k, degree, MONTE_CARLO_TIMES)
    with open(SEED_B, "w") as f:
        f.write(str(best_individual[0].count(1)) + " " + str(best_individual[1].count(1)) + "\n")
        for i in range(n):
            if best_individual[0][i] == 1:
                f.write(str(i) + "\n")
        for i in range(n):
            if best_individual[1][i] == 1:
                f.write(str(i) + "\n")

    if DEBUG:
        import matplotlib.pyplot as plt
        print("Initialize Individual Time Cost:", time_data[0])
        print("First Evaluation Time Cost:", time_data[1])
        print("Total Time Cost:", time_data[-1])
        print("Well Mutated Times:", well_mutated_cnt)
        # plt.plot(time_data[2], label="generation time")
        # plt.show()
        # plt.plot([max(fitness_value) for fitness_value in fitness_data], label="fitness")
        # plt.show()