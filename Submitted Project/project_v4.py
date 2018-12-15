# Code Written by Pascal Sauerborn, 13/12/2018, Student ID: 4313894, University of Nottingham, School of Physical Sciences.

# The following code defines the 2D model; this is simply an extension of the basic model, with similar basic structure. See README files for details.


from matplotlib.animation import FuncAnimation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style


class Car():

    """Car object stored on the road array

    Parameters
    ----------
    car_number: int
        label for the car; useful for tracking progress of individual car
    position: int
        the initial position in the road array
    speed: int
        cars original speed
    lane: int
        lane number the car is in
    p_slow: float, range(0,1)
        probability of random slowdown
    vmax: int
        maximum velocity allowed
    profile: str
        optional string argument that defines the drivers profile; key used for dictionary to determine the probability of slow down and the speed limit excession

    """

    def __init__(self, car_number, position, speed, lane, p_slow=0.2, vmax=10, profile=None):

        self.car_number = car_number
        self.speed = speed
        self.position = position
        self.lane = lane
        self.vmax = vmax

        if profile is None:
            self.p_slow = p_slow

        else:

            # profiles are included in order to model the effect of different drivers. The quality of a driver is quantified by his probability of spontaneous speed change (p_slow) and the amount by which the driver exceeds the speed limit given by the road

            profiles = {'bad': (0.35, self.vmax), 'good': (
                0.2, np.ceil(self.vmax*0.25)), 'excellent': (0.1, np.ceil(self.vmax*0.1)), 'perfect': (0, 0)}

            self.p_slow = profiles[profile][0]
            self.vmax += profiles[profile][1]


    def update_speed(self, road):

        """Function to update the speed of the car

        Parameters
        ----------
        road: array-like, shape=[n_spaces, n_lanes]
            the array giving the various locations of cars
        """

        # the scope is the portion of road that each car effectively sees ahead of it. the distance is the minimum distance to the next car

        self.scope = self.look_ahead(road, lane=self.lane)

        # if the speed is lower than the maximum speed and the the nearest car is further ahead than its velocity, it is accelerated

        # if the scope array (road each car sees) is empty i.e. no cars ahead of it, the car accelerates

        if np.array_equal(self.scope, np.zeros((self.scope.shape[0]))) and self.speed < self.vmax:
            self.speed += 1
        else:

            for dist, space in enumerate(self.scope, start=1):

                if space != 0:
                    nearest_car = space
                    break

            self.speed = (dist - 1)

        if (self.speed > 1):
            rand = np.random.uniform(0, 1)
            if rand < self.p_slow:
                self.speed -= 1

        # cars also randomly and spontaneously slow down. This is defined by the p_slow parameter

    def update_position(self, road):

        """Function to update the position of the car

        Parameters
        ----------
        road: array-like, shape=[n_spaces, n_lanes]
            the array giving the various locations of cars
        """

        # the new position is simply the old position plus the velocity

        new_pos = self.position + self.speed

        # the periodic boundary is applied if the new position is outside the reach of the road

        if new_pos > road.shape[1] - 1:
            new_pos -= road.shape[1]

        self.position = new_pos

    def switch_lane(self, road):

        """Function called to check whether of not the car should switch lane

        Parameters
        ----------
        road: road object
            the current 2 lane road object containing the current configuration of cars

        Returns
        -------
        the function returns a simple True/False value

        """
        scope = self.look_ahead(road, lane=self.lane)

        # first, it must be preferable to switch lanes as opposed to staying in its own lane. Hence if the road ahead is clear, the car will never switch

        if np.array_equal(scope, np.zeros(scope.shape[0])):
            return False

        # the scope for both ahead and behind is then gathered for the opposite lane is

        scope_ahead = self.look_ahead(road, lane=abs(self.lane - 1))
        scope_behind = self.look_behind(road, lane=abs(self.lane - 1))

        # if and only if the scope ahead and behind is clear, then the car will switch

        if np.array_equal(scope_ahead, np.zeros((scope_ahead.shape[0]))):
            if np.array_equal(scope_behind, np.zeros((scope_behind.shape[0]))):
                return True

    def look_ahead(self, road, lane):

        """Function used to determine the road ahead. Note that the 'scope' variable is simple a segment of the road; in order to save time, the car only effectively 'sees' as far ahead as it could travel in this turn Also note that, when looking ahead to see whether or not to switch lanes, a safty cushion constant is added (3 in this case)

        Parameters
        ----------
        road: road object
            the current 2 lane road object containing the current configuration of cars
        lane: int
            the lane which should be checked. Note that abs(self.lane-1) is used to return the lane the car is not currently in.

        Returns
        -------
        scope: Numpy ndarray object, shape = [self.speed + 2, 1]
            the section of road that the car effectively sees ahead of it
        """

        if lane == self.lane:
            vision = self.position + self.speed + 2
        else:
            vision = self.position + self.speed + 5

        if vision < road.shape[1]:
            scope = road[lane, self.position + 1:vision]

        if vision == road.shape[1]:
            scope = np.hstack((road[lane, self.position + 1:], road[lane, 0]))

        if vision > road.shape[1]:
            delta = vision - road.shape[1]
            scope = np.hstack(
                (road[lane, self.position + 1:], road[lane, 0:delta]))

        return scope

    def look_behind(self, road, lane):

        """Function used to determine the road ahead. Note that the 'scope' variable is simple a segment of the road; in order to save time, the car only effectively 'sees' as far ahead as it could travel in this turn Also note that, when looking ahead to see whether or not to switch lanes, a safty cushion constant is added (3 in this case)

        Parameters
        ----------
        road: road object
            the current 2 lane road object containing the current configuration of cars
        lane: int
            the lane which should be checked. Note that abs(self.lane-1) is used to return the lane the car is not currently in.

        Returns
        -------
        scope: Numpy ndarray object, shape = [self.speed + 2, 1]
            the section of road that the car effectively sees behind it
        """

        vision = self.position - self.speed - 5

        if vision >= 0:
            scope = road[lane, vision:self.position + 1]

        if vision < 0:
            delta = road.shape[1] + vision
            scope = np.hstack(
                (road[lane, delta - 1:], road[lane, 0:self.position + 1]))

        return scope

    def __repr__(self):
        return 'Position: {}\nSpeed: {}'.format(self.position, self.speed)


class Road():

    """Road object Automaton. The road object stores the positions of the cars and consequently updates their state

    Parameters
    ----------
    L: int
        length of the road
    random_state: int
        seed used to evaluate random number sequences
    car_count: int
        number of cars on the road

    """

    def __init__(self, L=40, random_state=1, car_count=10, lanes=2, vmax=5, p_slow=0.1, ratios=None):

        self.L = L
        self.road = np.zeros((2, L), dtype=object)
        self.lanes = lanes

        self.rgen = np.random.RandomState(random_state)

        # a list of random indexes is then generated. Each index corresponds to a space on the road which will have a car placed on it

        # Note that, in order to avoid duplicates, one cannot simply use a random number generator. Hence, a list of all possible avaiable spaces is made, then shuffled and a corresponding number of spaces is then chosen.

        available_spaces_lane1 = [i for i in range(self.L)]
        available_spaces_lane2 = [i for i in range(self.L)]

        self.rgen.shuffle(available_spaces_lane1)
        self.rgen.shuffle(available_spaces_lane2)

        lane1_occupation = int(np.ceil(car_count / 2))
        lane2_occupation = car_count - lane1_occupation

        # print(lane1_occupation + lane2_occupation)

        lane1 = available_spaces_lane1[:lane1_occupation]
        lane2 = available_spaces_lane2[:lane2_occupation]

        # a car object is then placed at each random index

        for car, i in enumerate(lane1):
            self.road[0, i] = Car(
                car + 1, i, 0, vmax=vmax, lane=0, p_slow=p_slow)

        for car, j in enumerate(lane2, start=i):
            self.road[1, j] = Car(
                car + 1, j, 0, vmax=vmax, lane=1, p_slow=p_slow)

        # A representation of the road is also created. Simply, the representation of the road contains 0 for empty spaces and 1's for occupied spaces

        self.road_rep = np.where(self.road == 0, 0, 1)

    def update(self, flow_rate=False):

        """Updates the state for the road

        returns
        -------
        data: list
            list containing tuples of form (car.position, car.speed, car.lane) for each car
        avg_speed: float
            average speed of the system
        top_speed: int
            top speed in the system
        overtakes: int
            number of lane switches that occured within the evolution of the system

        """

        overtakes = []
        cars = []
        speeds = []
        data = []

        # first, the system checks whether or not any cars are eligble for a lane switch. If they are, the cars switch lanes before anything else is done

        for car in self.road[self.road != 0]:

            if car.switch_lane(self.road):

                self.road[car.lane, car.position] = 0
                car.lane = abs(car.lane - 1)
                overtakes.append(car)

        # print('\nTotal Overtakes: {}'.format(len(overtakes)))

        for car in overtakes:

            self.road[car.lane, car.position] = car

        # the speeds and positions are the updated the same as in the standard model

        for car in self.road[self.road != 0]:

            car.update_speed(self.road)

        for car in self.road[self.road != 0]:

            car.update_position(self.road)

            if flow_rate:
                data.append((car.position, car.speed, car.lane))
            else:
                speeds.append(car.speed)
                data.append((car.position, car.speed, car.lane))

            cars.append(car)

        # print(len(cars))

        # the road then needs to be updated with the new car positions and speeds

        self.road = np.zeros((self.lanes, self.L), dtype=object)
        #
        for car in cars:
            self.road[car.lane, car.position] = car

        self.road_rep = np.where(self.road == 0, 0, 1)

        if flow_rate:
            return data
        else:
            return data, np.mean(np.array(speeds)), max(speeds), len(overtakes)
