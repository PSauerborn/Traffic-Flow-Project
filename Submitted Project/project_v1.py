# Code Written by Pascal Sauerborn, 13/12/2018, Student ID: 4313894, University of Nottingham, School of Physical Sciences.

# The following code defines the basic model, as defined by Nagel and Schreckenberg in their study.

from matplotlib.animation import FuncAnimation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns


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
    p_slow: float, range(0,1)
        probability of random slowdown
    vmax: int
        maximum velocity allowed as per road speed limit. Note that this is not neccesarily the maximum speed the car reaches. Bad drivers exceed the speed limit be a set amount
    profile: str
        type of driver; key used for dictionary to determine the probability of slow down and the speed limit excession
    """

    def __init__(self, car_number, position, speed, p_slow=0.2, vmax=5, profile=None):

        self.car_number = car_number
        self.speed = speed
        self.position = position

        self.vmax = vmax

        if profile is None:
            self.p_slow = p_slow

        else:

            # profiles are included in order to model the effect of different drivers. The quality of a driver is quantified by his probability of spontaneous speed change (p_slow) and the amount by which the driver exceeds the speed limit given by the road

            profiles = {'bad': (0.35, 20), 'good': (
                0.2, 10), 'excellent': (0.05, 3), 'perfect': (0, 0)}

            self.p_slow = profiles[profile][0]
            self.vmax += profiles[profile][1]

    def update_speed(self, road):

        """Function to update the speed of the car

        Parameters
        ----------
        road: array-like, shape=[n_spaces, n_lanes]
            the array giving the various locations of cars
        """

        # the scope is the portion of road that each car effectively sees ahead of it. 'dist' is the distance to the car ahead

        self.scope, dist = self.locate_nearest(road)

        # if the scope array (road each car sees) is empty i.e. no cars ahead of it, the car accelerates

        if np.array_equal(self.scope, np.zeros(self.scope.shape)) and self.speed < self.vmax:
            self.speed += 1
        else:
            self.speed = (dist - 1)

        # cars also randomly and spontaneously slow down. This is defined by the p_slow parameter

        if self.speed > 0:
            rand = np.random.uniform(0, 1)
            if rand < self.p_slow:
                self.speed -= 1

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

        if new_pos > road.shape[0] - 1:
            new_pos = new_pos - road.shape[0]

        self.position = new_pos

    def locate_nearest(self, road):

        """Function to analyse the road ahead of each car

        Parameters
        ----------
        road: array-like, shape=[n_spaces, n_lanes]
            the array giving the various locations of cars

        Returns
        -------
        scope: array, shape = [velocity of car]
            the road (and cars) that each car 'sees' ahead of it

        dist: int
            distance to nearest car
        """

        # the scope variable is then defined. Essentially, the scope is the road ahead that each car 'sees'. Note that, if the scope extends past the array, then the periodic condition is applied and the resulting array is stacked

        upper_lim = self.position + self.speed + 2

        if upper_lim < road.shape[0]:
            scope = road[self.position + 1:upper_lim]

        # if the car is at the each of the boundary (or going to cross the boundary) the periodic condition of the road is applied

        elif self.position == (road.shape[0] - 1):
            scope = road[0:self.speed + 1]

        else:
            delta = upper_lim - road.shape[0]
            scope = np.hstack((road[self.position + 1:], road[0:delta]))

        # the minimum distance to the next car is then evaluated

        for dist, space in enumerate(scope):
            if space != 0:
                break

        # the scope and the distance are returned

        return scope, (dist + 1)

    def __repr__(self):
        return 'Position: {}\nSpeed: {}'.format(self.position, self.speed)


class Road():

    """Road object Automaton. The road object stores the positions of the cars and updates their state

    Parameters
    ----------
    L: int
        length of the road
    random_state: int
        seed used to evaluate random number sequences
    car_count: int
        number of cars on the road
    vmax: int
        speed limit of road. Note that this can be exceeded by cars
    pslow: float
        probability that a car slows down randomly with each iteration

    """

    def __init__(self, L=40, random_state=1, car_count=10, vmax=5, p_slow=0.01, setup=None):

        self.L = L
        self.road = np.zeros((L), dtype=object)

        self.rgen = np.random.RandomState(random_state)

        # first, the number of cars on the toad initially is randomly chosen

        if setup is None:

            initial_count = car_count

            # a list of random indexes is then generated. Each index corresponds to a space on the road which will have a car placed on it

            # Note that, in order to avoid duplicates, one cannot simply use a random number generator. Hence, a list of all possible avaiable spaces is made, then shuffled and a corresponding number of spaces is then chosen.

            available_spaces = [i for i in range(self.L)]

            self.rgen.shuffle(available_spaces)

            initial_cars = available_spaces[:car_count]

            # a car object is then placed at each random index

            for car, i in enumerate(initial_cars):
                self.road[i] = Car(car + 1, i, 0, vmax=vmax, p_slow=p_slow)

        else:
            for i, position in enumerate(setup):
                self.road[position] = Car(i + 1, position, 0, vmax=vmax, p_slow=p_slow)

        # A representation of the road is also created. Simply, the representation of the road contains 0 for empty spaces and 1's for occupied spaces

        self.road_rep = np.where(self.road == 0, 0, 1)

    def update(self, flow_rate=False):

        """Updates the state for the road

        parameters
        ----------
        flow_rate: boolean, default=False
            when the flow rate is evaluated, certain data (such as the average speed and top speed) is not neccesary. In order to boost computational efficiency, if flow_rate is set to True, uneccesary data is not gathered

        returns
        -------
        the method returns various statistical properties of the road object, such as the average speed of the cars on the road as well as the top speed etc.

        """

        cars = []
        speeds = []
        data = []

        # first, all the cars (their speed and positions) are updated

        for car in self.road[self.road != 0]:

            car.update_speed(self.road)

        for car in self.road[self.road != 0]:

            car.update_position(self.road)

            if flow_rate:
                data.append((car.position, car.speed))
            else:
                speeds.append(car.speed)
                data.append((car.position, car.speed, car.car_number))

            cars.append(car)


        # the road then needs to be updated with the new car positions and speeds

        self.road = np.zeros((self.L), dtype=object)

        for car in cars:
            self.road[car.position] = car

        self.road_rep = np.where(self.road == 0, 0, 1)

        if flow_rate:
            return data
        else:
            return data, np.mean(np.array(speeds)), max(speeds)

    def __repr__(self):
        return '{}'.format(self.road_rep)
