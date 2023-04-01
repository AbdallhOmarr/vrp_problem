import pandas as pd
import re
import matplotlib.pyplot as plt
import math
import numpy as np
import random
import pandas as pd
from sklearn.cluster import KMeans
import re
from Route import *
from Solution import * 
from sklearn.cluster import KMeans

def load_data():
    # lets load data
    txt_file = r"E:\repos\app2022\vrp_problem\vrp data.txt"
    data = []
    with open(txt_file, "r") as file:
        lines = file.readlines()  # Read all lines in the file
        column_parsed = False
        column_names = []
        if column_parsed == False:
            # Extract column names from first line
            words = re.findall("\S+", lines[0])
            for word in words:
                column_names.append(word)
            column_parsed = True

        for line in lines[1:]:  # Loop through lines, skipping the first line
            # Extract data points from line
            dataline = re.findall("\S+", line)
            line_dict = {}
            for i, point in enumerate(dataline):
                # Convert data point to float and store in dictionary
                line_dict[column_names[i]] = float(point)
            data.append(line_dict)  # Append dictionary to list of data
       


    #converting list of dict to numpy array
    data_array = np.array([[d[col] for col in column_names] for d in data])

    #extract first row into depot array
    DEPOT = data_array[0, :]

    # extract rest of the rows into customers_array
    customers_array = data_array[1:, :]
   
    return customers_array,DEPOT


#getting distance from depot for all points 
def get_distance_between_two_points(x1,y1,x2,y2):
    x_diff = x2-x1
    y_diff = y2-y1
    dist = math.sqrt(math.pow(x_diff, 2) + math.pow(y_diff, 2))
    return dist

def add_distance_feature(customers_array, depot):
    # extract XCOORD and YCOORD columns from customers array
    XCOORD = customers_array[:, 1]
    YCOORD = customers_array[:, 2]
    
    # calculate distances from depot to customers
    distances = np.sqrt((XCOORD - depot[1])**2 + (YCOORD - depot[2])**2)
    
    # add distances as new column to customers array
    customers_array = np.column_stack((customers_array, distances))
    
    return customers_array


def cluster_array(customers_array,  n_clusters):
    # extract columns to cluster from customers array
    columns_to_cluster = [4, 1, 2, 7]  # indices of DUE_DATE, XCOORD, YCOORD, and distance_FROM_DEPOT columns
    
    # normalize the data
    data_norm = (customers_array[:, columns_to_cluster] - customers_array[:, columns_to_cluster].mean(axis=0)) / customers_array[:, columns_to_cluster].std(axis=0)
    
    # cluster the data using KMeans
    kmeans = KMeans(n_clusters=n_clusters).fit(data_norm)
    
    # add cluster labels to customers array
    customers_array = np.column_stack((customers_array, kmeans.labels_))
    
    # sort points by distance from depot within each cluster
    sorted_indices = np.lexsort((customers_array[:, 7], customers_array[:, 5], customers_array[:, 8]))
    customers_array = customers_array[sorted_indices, :]
    
    return customers_array



class Route:
    def __init__(self,customers_array,depot):
        self.customers_array = customers_array
        self.depot = depot
        self.current_time = 0

    def get_customers_ids(self):
        self.customers_ids = self.customers_array[:,0]
        return self.customers_ids

    def get_load(self):
        self.load = self.customers_array[:,3].sum()
        return self.load

    def check_time_constrain(self):

        #current time is between ready time and due date
        # ready_time <= current time <= due date
        #initial current time = first customer readytime
        #ready time row[4]
        #due date row[5]

        self.time_constrain = True
        self.current_time = self.customers_array[0,4]
        for row in self.customers_array:
            if (not self.current_time <= row[4]) or (not self.current_time<=row[5]):
                self.time_constrain =False

                return self.time_constrain
            else:
                self.current_time+=row[6]
        return self.time_constrain

    def check_capacity_constrain(self):
        self.load = self.get_load()
        if self.load>200:
            self.capacity_constrain = False
        else:
            self.capacity_constrain = True
            
        return self.capacity_constrain



    def get_route_distance(self):
        
        #i will get the distance between each customer and the next customer in the array
        #distance row[7]
        self.total_distance = 0 
        for i in range(self.customers_array.shape[0]):
            if i == self.customers_array.shape[0]-1:
                break

            distance = get_distance_between_two_points(self.customers_array[i,1],self.customers_array[i,2],self.customers_array[i+1,1],self.customers_array[i+1,2])
            self.total_distance+=distance

        #calculate distance betwen first customer and depot and last customer and depot 
        distance_1 = get_distance_between_two_points(self.customers_array[0,1],self.customers_array[0,2],self.depot[1],self.depot[2])
        distance_2 = get_distance_between_two_points(self.customers_array[-1,1],self.customers_array[-1,2],self.depot[1],self.depot[2])

        self.total_distance += distance_1
        self.total_distance += distance_2 
        return self.total_distance

    def update_route_variables(self):
        self.get_load()
        self.get_customers_ids()
        self.get_route_distance()
        self.check_capacity_constrain()
        self.check_time_constrain()

    def mutate(self,mutation_rate):
        #get randomly two indices in the customers array and exchange them 
        if self.customers_array.shape[0]<=1:
            return False
        
        if random.random() > mutation_rate:
            return False

        idx_1 = random.randint(0, self.customers_array.shape[0] - 1)
        idx_2 = random.randint(0, self.customers_array.shape[0] - 1)

        first_row = self.customers_array[idx_1,:]
        second_row = self.customers_array[idx_2,:]

        self.customers_array[idx_1,:] = second_row
        self.customers_array[idx_2,:] = first_row 
        self.update_route_variables()
        return True


class Solution:
    def __init__(self,routes):
        self.routes = routes
        # get total distance 
        # check feasibility 

    def prevent_duplicated_customers(self):
        #this function to remove duplicated customers in other routes 
        #check if customer is another route 
        # if customers in another route remove the customer in the first route
        unique_customers =[] 
        unique_customers_idx = []
        duplicated_customers = []
        for route_idx,route in enumerate(self.routes):
            for customer in route.customers_ids:
                if customer not in unique_customers:
                    unique_customers.append(customer)
                    unique_customers_idx.append(route_idx)
                else:
                    duplicated_customers.append({"customer":customer,"route_idx":route_idx})

       
        for customer,route_idx in enumerate(duplicated_customers):
           route = self.routes[route_idx]
           route_customers_array = route.customers_array
           rows_to_delete = [] 
           for i,row in enumerate(route_customers_array.copy()):
               if customer == row[0]:
                    rows_to_delete.append(i)
            route_customers_array = np.delete(route_customers_array, rows_to_delete, axis=0)

            self.routes[route_idx].customers_array = route_customers_array
            self.routes[route_idx].update_route_variables()

            self.update_variables()
                    

    def get_total_solution_distance(self):
        self.total_distance = 0
        for route in self.routes:
            self.total_distance +=route.get_route_distance()

        return self.total_distance 

    def check_feasibility(self):
        self.time_constrain = True
        self.capacity_constrain = True
        for route in self.routes:
            if not route.check_capacity_constrain():
                self.capacity_constrain = False
            if not route.check_time_constrain():
                self.time_constrain = False

        self.feasibility = self.capacity_constrain * self.time_constrain

        return self.feasibility


    def get_total_customers_served(self):
        self.customers_served = []
        for route in self.routes:
            self.customers_served + = route.get_total_customers_served()
        self.total_customers_served = len(self.customers_served)
        return self.total_customers_served 

    def get_number_of_routes(self):
        self.no_of_routes = len(self.routes)
        return self.no_of_routes

    def fitness_func(self):
        self.update_variables()
        self.fitness_value = self.check_feasibility()*(self.total_customers_served*1000 - self.total_distance)
        return self.fitness_value


    def update_variables(self):
        self.get_total_solution_distance()
        self.check_feasibility()
        self.get_total_customers_served()








        









def get_routes(customers_array,n_clusters):
    routes =[]
    for n in range(n_clusters):
        route_customers = customers_array[customers_array[:, 8] == n]
        route = Route(route_customers)
        routes.append(route)

    return routes

def generate_initial_population(customers_array,POPULATION_SIZE,n_cluster):
    population = []
    for i in range(POPULATION_SIZE):
        clustered_customers_array = cluster_array(customers_array,n_cluster)
        routes = get_routes(clustered_customers_array,n_cluster)
        sol = Solution(routes)
        sol.prevent_duplicated_customers()


def generate_initial_population(customers_df,POPULATION_SIZE,n_clusters,DEPOT):
        # Step 1: Generate initial population
        population = []
        for i in range(POPULATION_SIZE):
            customers_df_generated = cluster_df(customers_df, n_clusters)
            length_c_df = customers_df_generated.shape[0]
            routes = get_routes(customers_df_generated, n_clusters,DEPOT)
            sol = Solution(routes)
            population.append(sol)
        return population


def rank_based_selection(population, k=2):
    fitness_list = [sol.fitness_func() for sol in population]
    ranked_list = sorted(range(len(fitness_list)), key=lambda k: fitness_list[k])
    ranks = [len(fitness_list) - i for i in ranked_list]
    probs = [rank / sum(ranks) for rank in ranks]
    selected_indices = random.choices(range(len(population)), weights=probs, k=k)
    return [population[i] for i in selected_indices]


def create_offspring(parent1, parent2, crossover_rate,DEPOT):
    # check if crossover should occur
    if random.random() > crossover_rate:
        # if not, return a clone of parent1 or parent2 at random
        if random.random() < 0.5:
            return parent1
        else:
            return parent2
    p1_routes = parent1.routes
    p2_routes = parent2.routes

    # Choose a random crossover point
    crossover_point = random.randint(1, len(p1_routes) - 1)

    # Create the offspring by combining the parents' genetic material
    offspring_routes = p1_routes[:crossover_point] + p2_routes[crossover_point:]

    # Remove duplicate customers from the offspring
    unique_customers = []
    for route in offspring_routes:
        unique_route_customers_df = pd.DataFrame()
        for customer_id in route.customers_ids:
            if customer_id not in unique_customers:
                parent1.customers_df = parent1.customers_df.reset_index(drop=True)
                # parent1.customers_df.to_excel("parent_df.xlsx")
                print(f"customer id = {customer_id}")
                c_df = parent1.customers_df[parent1.customers_df["CUST_NO"]==int(customer_id)]
                unique_route_customers_df= pd.concat([unique_route_customers_df,c_df])
                unique_route_customers_df=unique_route_customers_df.append(c_df)
                unique_customers.append(customer_id)

        unique_route_customers_df.to_excel(f"{random.randint(1,10000)}.xlsx")
        offspring_routes[offspring_routes.index(route)] = Route(unique_route_customers_df,DEPOT)
        # unique_route_customers_df.to_excel(f"{random.randint(1,1)}-{route.route_no}.xlsx")

    offspring = Solution(offspring_routes)
    return offspring


def main():
    customers_df,DEPOT = load_data()
    customers_df["distance_FROM_DEPOT"] = customers_df.apply(lambda x: get_distance_between_two_points(x["XCOORD"],x["YCOORD"],DEPOT.loc[0,"XCOORD"], DEPOT.loc[0,"YCOORD"]), axis=1)
    #customers_df = cluster_df(customers_df,20)
    n_generation = 300
    n_population = 100 
    co_rate = 0.8
    m_rate = 0.3
    n_clusters = 20
    pop = generate_initial_population(customers_df, n_population,n_clusters,DEPOT)
    pop_fitness = [sol.fitness_func() for sol in pop]

    best_idx = pop_fitness.index(max(pop_fitness))
    best_sol_ever = pop[best_idx]

    for g in range(n_generation):
        print(f"Genration:{g}")
        pop = generate_initial_population(customers_df, n_population,n_clusters,DEPOT)
        print("Population created")
        parent1,parent2 = rank_based_selection(pop,2)
        print("Two parents created")
        offspring = create_offspring(parent1,parent2,co_rate,DEPOT)
        print("an offspring created")
        # Mutate offspring
        for route in offspring.routes:
            route.mutate(m_rate)
        print(f"Route mutated")

        # offspring.add_customers_to_assigned_customers()

        # Calculate fitness for offspring
        offspring_fitness = offspring.fitness_func()
        print(f"offspring fitness:{offspring_fitness}")
        print("offspring solution:")
        print(len(offspring.routes),offspring.get_total_solution_distance(),offspring.get_total_number_of_served_customers(),offspring.check_feasiblity(),offspring.fitness_func())
        print("-"*50)
        offspring_fitness = offspring.fitness_func()
        print(f"offspring fitness2:{offspring_fitness}")

        if offspring.get_total_number_of_served_customers()>100:
            break
        # Replace worst solution in population with offspring
        pop_fitness = [sol.fitness_func() for sol in pop]

        worst_idx = pop_fitness.index(min(pop_fitness))
        print(f"worest solution fitness:{pop_fitness[worst_idx]}")
    

        if offspring_fitness > pop_fitness[worst_idx]:
            pop[worst_idx] = offspring

    
        best_idx = pop_fitness.index(max(pop_fitness))
        best_sol = pop[best_idx]

        if best_sol.fitness_func() > best_sol_ever.fitness_func():
            best_sol_ever = best_sol
        print("best solution till now:")
        print(len(best_sol_ever.routes),best_sol_ever.get_total_solution_distance(),best_sol_ever.get_total_number_of_served_customers(),best_sol_ever.check_feasiblity(),best_sol_ever.fitness_func())
        print("-"*50)


main()