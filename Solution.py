
import pandas as pd
import re
import matplotlib.pyplot as plt
import math
import numpy as np
import random
import pandas as pd
from sklearn.cluster import KMeans
import re

class Solution:
    def __init__(self,routes_list):
        self.routes = routes_list
        self.get_customers_df()
        self.get_total_solution_distance()
        self.get_total_number_of_served_customers()
        self.get_number_of_routes()
        self.check_feasiblity()
        self.assigned_customers = []

    def get_customers_df(self):
        customers_df = pd.DataFrame()
        for route in self.routes:
            customers_df=pd.concat([customers_df,route.customers_df])
            
        self.customers_df = customers_df

    def add_customers_to_assigned_customers(self):
        new_routes = self.routes
        routes_to_remove = []

        for route in new_routes.copy():
            for i, v in route.customers_df.iterrows():
                if not v["CUST_NO"] in self.assigned_customers:
                    self.assigned_customers.append(v["CUST_NO"])
                else:
                    print("customer is duplicated")
                    print("deleting the current customer row")
                    c_df = route.customers_df
                    c_df = c_df.drop(i)
                    c_df = c_df.reset_index(drop=True)
                    print("reconstructing the route obj again without this customer")
                    routes_to_remove.append(route)
                    if c_df.shape[0]>=2:
                        new_route = Route(c_df)
                    else:
                        print("customers dataframe is less than 2")
                        new_route = Route(route.customers_df)
                    print("removing the old route and adding the new one to the route list")
                    new_routes.append(new_route)

        for route in routes_to_remove:
            if route in new_routes:
                new_routes.remove(route)
        
        self.routes = new_routes

    def get_total_solution_distance(self):
        self.solution_total_distance =0
        for route in self.routes:
            self.solution_total_distance+=route.get_total_distance()
        return self.solution_total_distance
    
    def get_total_number_of_served_customers(self):
        self.total_n_customers = 0
        for route in self.routes.copy():
            if route.get_no_of_customers()<=1:
                # self.routes.remove(route)
                pass
            else:
                self.total_n_customers+=route.get_no_of_customers()
        return self.total_n_customers
    
    def get_number_of_routes(self):
        self.no_of_routes = len(self.routes) #equalivent to no. of vehicles as well
        
    def check_feasiblity(self):
        self.feasiblity = True
        for route in self.routes:
            if route.fitness():
                pass
            else:
                self.feasiblity = False
        return self.feasiblity 

    def merge_routes(self,route1,route2):
        #merge_route? 
        #get customers df from the two routes 
        routes = self.routes.copy()
        route1_customers = route1.customers_df
        route2_customers = route2.customers_df

        #merge the two customers 
        merged_customers = pd.concat([route1_customers,route2_customers],ignore_index=True)

        #rename the cluster to be as the first route cluster
        merged_customers["cluster"] = route1_customers.loc[0,"cluster"]

        #create new route obj 
        new_route = Route(merged_customers)
        new_route.sort_route()
        if route1 in routes:
            routes.remove(route1)
        else:
            print("route1 not in self.routes")

        if route2 in routes:
            routes.remove(route2)
        else:
            print("route2 not in self.routes")
        routes.append(new_route)
        
        self.routes = routes.copy()
        self.get_total_solution_distance()
        self.get_total_number_of_served_customers()
        self.get_number_of_routes()
        self.check_feasiblity()

        return new_route
    
    def fitness_func(self):
        capacity_constrain = 1
        time_constrain = 1

        for route in self.routes:
            if not route.check_capacity():
                capacity_constrain = 0 
            if not route.check_time_constrain():
                time_constrain = 0


        fitness_value = capacity_constrain*time_constrain*(self.total_n_customers*10000-self.solution_total_distance)
        if fitness_value == 0 :
            fitness_value -= 1000000000
        return round(fitness_value,2)
            
    def remove_unfeasible_routes(self):
        routes = self.routes.copy()
        for route in self.routes:
            if route.fitness()==0:
                routes.remove(route)
            
        self.routes = routes.copy()
        self.check_feasiblity()
                            
                
    def merge_two_routes_with_least_capacity(self):
        for route in self.routes:
            route.sort_route()

        routes = self.routes.copy()
        sorted_by_load = sorted(routes, key=lambda x: x.load)
        route1 = sorted_by_load[0]
        route2 = sorted_by_load[1]
        self.merge_routes(route1, route2) 

    def merge_two_routes_with_smallest_distance(self):
        for route in self.routes:
            route.sort_route()

        routes = self.routes.copy()
        sorted_by_load = sorted(routes, key=lambda x: x.distance)
        route1 = sorted_by_load[0]
        route2 = sorted_by_load[1]
        self.merge_routes(route1, route2) 
