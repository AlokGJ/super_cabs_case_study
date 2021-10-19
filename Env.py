# Import routines

import numpy as np
import math
import random
from itertools import permutations

# Defining hyperparameters
m = 5 # number of cities, ranges from 1 ..... m
t = 24 # number of hours, ranges from 0 .... t-1
d = 7  # number of days, ranges from 0 ... d-1
C = 5 # Per hour fuel and other costs
R = 9 # per hour revenue from a passenger


class CabDriver():

    def __init__(self):
        """initialise your state and define your action space and state space"""
        self.action_space = list(permutations([i for i in range(m)], 2)) + [(0,0)]
        self.state_space = [[x, y, z] for x in range(m) for y in range(t) for z in range(d)]
        self.state_init = random.choice(self.state_space)
        self.action_init = random.choice(self.action_space)

        # Start the first round
        self.reset()

    ## Encoding state (or state-action) for NN input
    def state_encod_arch1(self, state):
        """convert the state into a vector so that it can be fed to the NN. This method converts a given state into a vector format. Hint: The vector is of size m + t + d."""
        pass

    # Use this function if you are using architecture-2 
    def state_encod_arch2(self, state, action):
        """convert the (state-action) into a vector so that it can be fed to the NN. This method converts a given state-action pair into a vector format. Hint: The vector is of size m + t + d + m + m."""
        range_value = t + d + (3 * m)
        state_encode = [0 for n in range(range_value)]
        state_encode[state[0]] = 1
        state_encode[m + state[1]] = 1
        state_encode[m + t + state[2]] = 1
        if (action[1] != 0):
            state_encode[(2 * m) + t + d + action[1]] = 1
        if (action[0] != 0):
            state_encode[m + t + d + action[0]] = 1
        return state_encode


    ## Getting number of requests
    def requests(self, state):
        """Determining the number of requests basis the location. 
        Use the table specified in the MDP and complete for rest of the locations"""
        location = state[0]
        if location == 0:
            requests = np.random.poisson(2)
        if location == 1:
            requests = np.random.poisson(12)
        if location == 2:
            requests = np.random.poisson(4)
        if location == 3:
            requests = np.random.poisson(7)
        if location == 4:
            requests = np.random.poisson(8)

        if requests > 15:
            requests = 15

        actions_index = random.sample(range(1, (m - 1) * m + 1), requests) # (0,0) is not considered as customer request
        actions = [self.action_space[i] for i in actions_index]

        actions.append([0,0])
        actions_index.append(self.action_space.index((0,0)))

        return actions_index, actions   

    def reward_func(self, state, action, Time_matrix):
        """Takes in state, action and Time-matrix and returns the reward"""
        next_state, time_wait, time_transit, time_ride = self.next_state_func(state, action, Time_matrix)
        idle = time_wait + time_transit
        reward = (R * time_ride) - (C * (time_ride + idle))
        return reward

    def next_state_func(self, state, action, Time_matrix):
        """Takes state and action as input and returns next state"""
        time_total = 0
        time_wait = 0
        time_ride = 0
        time_transit = 0
        location = state[0]
        time = state[1]
        day = state[2]
        pickup = action[0]
        drop = action[1]

        if (pickup) == 0 and (drop == 0):
            time_wait = 1
            location_next = location
        elif pickup == location:
            time_ride = Time_matrix[location][drop][time][day]
            location_next = drop
        else:
            time_transit = Time_matrix[location][pickup][time][day]
            time, day = self.get_updated_day_time(time, day, time_transit)
            time_ride =  Time_matrix[pickup][drop][time][day]
            location_next  = drop

        time_total = time_ride + time_wait
        updated_time, updated_day = self.get_updated_day_time(time, day, time_total)
        next_state = [location_next, updated_time, updated_day]

        return next_state, time_wait, time_transit, time_ride

    def get_updated_day_time(self, time, day, duration):
        period = int(duration)
        if time + period < 24:
            next_time = time + period
            next_day = day
        else:
            days = (time + period) // 24
            next_day = (days + day ) % 7
            next_time = (time + period) % 24
        return next_time, next_day

    def reset(self):
        return self.action_space, self.state_space, self.state_init
