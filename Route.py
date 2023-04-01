import math
import pandas as pd
import numpy as np 
def get_distance_between_two_points(x1,y1,x2,y2):
    x_diff = x2-x1
    y_diff = y2-y1
    dist = math.sqrt(math.pow(x_diff, 2) + math.pow(y_diff, 2))
    return dist


class Route:
    def  __init__(self,customers_df,DEPOT):
        self.DEPOT = DEPOT
        self.customers_df = customers_df.reset_index(drop=True)
        self.customers_df = self.customers_df.drop_duplicates(subset=["CUST_NO"],keep="first")
        self.get_customer_ids()
        self.route_no = self.customers_df["cluster"][0]        
        self.n_customers = self.customers_df.shape[0]
        # self.calculate_distance_matrix()

    def get_customer_ids(self):
        self.customers_df.to_excel("df.xlsx")
        self.customers_ids = self.customers_df["CUST_NO"]
        self.customers_ids = self.customers_ids.to_list()
        self.customers_ids = [int(x) for x in self.customers_ids]

    def calculate_distance_matrix(self):
        self.dist_matrix = np.zeros((self.n_customers, self.n_customers))
        # Loop over each pair of customers
        for i in range(self.n_customers):
            for j in range(self.n_customers):
                # Calculate the Euclidean distance between the XCOORD and YCOORD values of the two customers
                x_diff = self.customers_df['XCOORD'][i] - \
                    self.customers_df['XCOORD'][j]
                y_diff = self.customers_df['YCOORD'][i] - \
                    self.customers_df['YCOORD'][j]
                dist = math.sqrt(math.pow(x_diff, 2) + math.pow(y_diff, 2))

                # Store the calculated distance in the corresponding position in the distance matrix
                self.dist_matrix[i][j] = dist

        return self.dist_matrix


    def get_total_distance(self):
              # distance between customers + distance between boundary customers and DEPOT
        distances = []
        if self.n_customers > 0:
            for i, customer in self.customers_df.iterrows():

                if i+1 >= self.n_customers:
                    continue

                next_customer = self.customers_df.iloc[i+1]
                distance_between_cust_next_customer = get_distance_between_two_points(
                    customer["XCOORD"], customer["YCOORD"], next_customer["XCOORD"], next_customer["YCOORD"])
                distances.append(distance_between_cust_next_customer)

            # distance between DEPOT and boundry customers
            b1 = self.customers_df.iloc[0]
            distance_between_b1_DEPOT = get_distance_between_two_points(
                b1["XCOORD"], b1["YCOORD"], self.DEPOT["XCOORD"], self.DEPOT["YCOORD"])
            distances.append(distance_between_b1_DEPOT)

            b2 = self.customers_df.iloc[-1]
            distance_between_b2_DEPOT = get_distance_between_two_points(
                b2["XCOORD"], b2["YCOORD"], self.DEPOT["XCOORD"], self.DEPOT["YCOORD"])
            distances.append(distance_between_b2_DEPOT)

            self.distance = 0
            for distance in distances:
                self.distance += distance

            return self.distance
        else:
            self.distance = 0
            return self.distance


    def check_capacity(self):
        self.load =0
        self.capacity_constrain = True
        for i,v in self.customers_df.iterrows():
            self.load +=v["DEMAND"]
        
        if self.load<=200:
            self.capacity_constrain = True
            return True
        else:
            self.capacity_constrain = False
            return False

    def check_time_constrain(self):
        self.time_constrain = True
        self.current_time = self.customers_df.loc[0,"READY_TIME"]
        print(self.current_time)
        print(type(self.customers_df))
        for i,v in self.customers_df.iterrows():
            if self.current_time>v["DUE_DATE"]:
                #means that current time exceeded the upper constrain for serving the customer
                self.time_constrain = False
            elif self.current_time<=v["DUE_DATE"]:
                pass
            else:
                print("condition fails")
            
        return self.time_constrain

    def get_no_of_customers(self):
        return self.n_customers

    def sort_route(self):
        self.get_total_distance()
        self.check_capacity()
        self.check_time_constrain()
        self.customers_df = self.customers_df.sort_values(["cluster","READY_TIME", "distance_FROM_DEPOT"])
        self.customers_df.reset_index(drop=True)

    def fitness(self):
        # calculate route fitness 
        self.check_capacity()
        self.check_time_constrain()
        #self.get_total_distance()
        
        if self.capacity_constrain:
            f1 = 1000
        else:
            f1 = 0 
        
        if self.time_constrain:
            f2 = 1000
        else:
            f2 = 0

        self.fitness_value = f1*f2
        return self.fitness_value


    def mutate(self, mutation_rate):
        # randomly decide if mutation occurs based on mutation_rate
        if np.random.rand() > mutation_rate:
            return
        
        if self.get_no_of_customers()<=1:
            return 
        # randomly select two distinct customers to swap their positions
        i, j = np.random.choice(self.n_customers, size=2, replace=False)
        
        # swap the positions of the two customers
        self.customers_df.iloc[i], self.customers_df.iloc[j] = self.customers_df.iloc[j], self.customers_df.iloc[i]

        # sort the route after mutation
        # self.sort_route()
        # check if the new solution is feasible, otherwise revert back to the original solution
        if not self.check_capacity() or not self.check_time_constrain():
            self.sort_route()
        
        if not self.check_capacity() or not self.check_time_constrain():
            self.customers_df.iloc[i], self.customers_df.iloc[j] = self.customers_df.iloc[j], self.customers_df.iloc[i]
            self.customers_df.reset_index(drop=True)
        else:
            self.customers_df.reset_index(drop=True)

            print(f"route:{self.route_no} mutated")
