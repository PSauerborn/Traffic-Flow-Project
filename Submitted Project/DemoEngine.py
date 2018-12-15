# Code Written by Pascal Sauerborn, Student ID: 4313894, 13/12/2018, University of Nottingham, School of Physical Sciences.

# The below code defines the DemoEngine() object; this simply provides a convenient and clean way of presenting a demo run of the project.

from matplotlib.animation import FuncAnimation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
from scipy.interpolate import interp1d
import os

import project_v1 as v1
import project_v2 as v2
import project_v4 as v4

import project_functions as pf
import project_functions_2d as func_2d


class DemoEngine():

    """Analyis Engine used for demonstration Purposes

    Parameters
    ----------
    road_length: int (default=100)
        length of road object to be created
    n_iterations: int (default=100)
        number of times the system is eveolved within each simulation
    vmax: int (default=5)
        maximum speed of road
    p_slow: float (default=0.1)
        probability of random slow down
    random_state: int
        seed used for random number generator

    """

    def __init__(self, road_length=100, n_iterations=100, vmax=5, p_slow=0.1, random_state=3):

        self.road_length = road_length
        self.n_iterations = n_iterations
        self.vmax = vmax
        self.p_slow = p_slow
        self.random_state = random_state

    def show_setup(self, single_lane=True):

        """Method that animates the system to demonstrate the initial setup

        Parameters
        ----------
        single_lane: Boolean, default=True
            shows 1-D system if set to True, 2-D system if set to False
        """

        if single_lane:
            pf.get_animation(road_length=self.road_length, car_count=20,
                             vmax=self.vmax, p_slow=self.p_slow, random_state=self.random_state)
        else:
            func_2d.animate_system_2lanes(
                road_length=self.road_length, car_count=10, vmax=self.vmax, p_slow=self.p_slow)

    def analyse_spacetime(self):

        """Method to plot the spacetime graphs, demonstrating smooth flow at low densities and the back-propogation of traffic jams at higher densities"""

        pf.get_space_time(road_length=self.road_length, car_count=3, n_iterations=40,
                          vmax=self.vmax, p_slow=0.05, random_state=self.random_state)

        pf.get_space_time(road_length=self.road_length, car_count=20, n_iterations=40,
                          vmax=self.vmax, p_slow=0.4, random_state=self.random_state)

    def analyse_flow(self, road_lengths=[50, 75, 100], max_speeds=[5, 10, 15], probabilities=[0.05, 0.1, 0.2]):

        """Method that demonstrates the dependence of the flow rate of the traffic system on various parameters. Note that the values the parameters take on are passed down as external arguments in the form of an interator when the method is called. By default, 3 values of each parameter are compared, however this is not neccesary. Any number of parameter values can be used however it should be noted that larger road lengths lead to longer computation times. Additionaly, if more than 4 values for a particular parameter are used, then a list containing the colors for each curve also needs to be passed down.

        Parameters
        ----------
        road_lengths: list
            list containing the road lengths that are compared in the analysis. Note that numbers within the list must be of integer type
        max_speeds: list
            list containing the maximum speeds that are compared in the analysis. Numbers within list must be integer types
        probabilites: list
            list containing the probabilites of random deceleration that are compared in the analysis. Numbers must be floats in range 0 <= p < 1
        """

        # the flow rate is first analysed with respect to road length

        data = []

        for length in road_lengths:

            print('Gathering Data for Road Length {}....'.format(length))

            flow = pf.get_flow_rate(road_length=length, n_iterations=self.n_iterations,
                                    vmax=self.vmax, p_slow=self.p_slow, random_state=self.random_state, prog_bar=False)

            data.append(flow)

        # the data produced is then plotted

        self.plot_flow_rate(
            data, labels=['Road Length {}'.format(i) for i in road_lengths], title='Effect of Road Length on Flow Rate')

        # the flow rate is then analyzed with respect to maximum speed

        data = []

        for speed in max_speeds:

            print('Gathering Data for vmax {}....'.format(speed))

            flow = pf.get_flow_rate(road_length=self.road_length, n_iterations=self.n_iterations,
                                    vmax=speed, p_slow=self.p_slow, random_state=self.random_state, prog_bar=False)

            data.append(flow)

        self.plot_flow_rate(
            data, labels=['vmax {}'.format(i) for i in max_speeds], title='Effect of Speed limit on Flow Rate')

        # finally, the flow rate is analyzed with respect to probabilite of deceleration

        data = []

        for proba in probabilities:

            print('Gathering Data for p_slow = {}....'.format(proba))

            flow = pf.get_flow_rate(road_length=self.road_length, n_iterations=self.n_iterations,
                                    vmax=self.vmax, p_slow=proba, random_state=self.random_state, prog_bar=False)

            data.append(flow)

        self.plot_flow_rate(
            data, labels=['pslow = {}'.format(i) for i in probabilities], title='Effect of Probability or random deceleration on Flow Rate')

    def analyse_profiles(self):

        """Method used to demonstrate the effect that increasingly larger number of bad drivers have on the flow rate"""

        # each car object can be optionally instantiated with a profile; non-perfect profiles have increasingly higher probabilities of random deceleration, and also have a constant added unto their maximum speed to simulate speeding drivers.

        profiles = ['bad', 'good', 'excellent', 'perfect']
        percentages = [[0.2, 0.2, 0.5, 0.1],
                       [0.3, 0.3, 0.3, 0.1],
                       [0.5, 0.2, 0.2, 0.1]]

        # the profiles are passed down to the road object in the form of a dictionary containing the profiles and what proportion of drivers have what profile. For example, a possible configuration could be {'perfect': 0.1, 'excellent': 0.2, 'good': 0.5, 'bad': 0.2}. Note that the ratios must add up to 1.

        perfect = {'bad': 0., 'good': 0., 'excellent': 0., 'perfect': 1.0}
        ratios1 = {profile: perc for profile,
                   perc in zip(profiles, percentages[0])}
        ratios2 = {profile: perc for profile,
                   perc in zip(profiles, percentages[1])}
        ratios3 = {profile: perc for profile,
                   perc in zip(profiles, percentages[2])}

        print('Gathering Data for Select Profile Distributions...')

        perfect_set = pf.get_flow_rate(
            road_length=100, n_iterations=100, vmax=5, ratios=perfect, prog_bar=False)
        data1 = pf.get_flow_rate(
            road_length=100, n_iterations=100, vmax=5, ratios=ratios1, prog_bar=False)
        data2 = pf.get_flow_rate(
            road_length=100, n_iterations=100, vmax=5, ratios=ratios2, prog_bar=False)
        data3 = pf.get_flow_rate(
            road_length=100, n_iterations=100, vmax=5, ratios=ratios3, prog_bar=False)

        data = [perfect_set, data1, data2, data3]
        self.plot_flow_rate(data, labels=[
                            'perfect drivers', 'bad driver ratio: 0.2', 'bad driver ratio: 0.3', 'bad driver ratio: 0.5'])

    def analyse_2d(self):

        """Method that compares the flow rate of the doube lane system to the single lane system"""

        print('Gathering Data for Double Lane Road....')
        data_2d = func_2d.get_flow_rate(road_length=self.road_length, n_iterations=self.n_iterations, vmax=self.vmax, p_slow=self.p_slow, random_state=self.random_state, prog_bar=False)

        print('Gathering Data for Single Lane Road....')
        data_1d = pf.get_flow_rate(road_length=self.road_length, n_iterations=self.n_iterations, vmax=self.vmax, p_slow=self.p_slow, random_state=self.random_state, prog_bar=False)

        data = [data_2d, data_1d]
        self.plot_flow_rate(data, labels=['Double Lane', 'Single Lane'])

        probabilities = [0.1, 0.3, 0.5]

        data = []


        for proba in probabilities:
            print('Gathering Data for pslow = {}....'.format(proba))

            flow = func_2d.get_flow_rate(road_length=self.road_length, n_iterations=self.n_iterations, vmax=self.vmax, p_slow=proba, random_state=self.random_state, prog_bar=False)
            data.append(flow)

        self.plot_flow_rate(data, labels=['p_slow = {}'.format(i) for i in probabilities])

        lane_switches = []

        for proba in probabilities:
            print('Calculating average switches at pslow = {}'.format(proba))
            switches = func_2d.get_switches(road_length=self.road_length, n_iterations=self.n_iterations, p_slow=proba, vmax=self.vmax)
            lane_switches.append(switches)

        fig, ax = plt.subplots()
        colors = ['r', 'limegreen', 'black']

        for set, label, c in zip(lane_switches, probabilities, colors):
            ax.scatter(set.index, set['Avg Number of Switches'], s=3, c=c, label='p_slow = {}'.format(label))

        ax.set_ylabel('Density')
        ax.set_ylabel('Average Number of Switches')
        plt.show()


    def plot_flow_rate(self, data, labels, colors=['r', 'limegreen', 'b', 'black'], title=None):

        """Method to plot the flow rate data

        Parameters
        ----------
        data: iterator of numpy arrays
            the data is passed down as some form of iterator (lists by default) consisting of individual data subsets to be plotted
        labels: list
            list containing the labels for the data
        colors: list
            list containing colors to be used for each curve
        """

        style.use('bmh')

        fig, ax = plt.subplots()

        for subset, label, c in zip(data, labels, colors):
            ax.scatter(subset[:, 0], subset[:, 1], c=c, label=label, s=2)

        if title is not None:
            ax.set_title(title)

        ax.set_xlabel('Density')
        ax.set_ylabel('Flow Rate')
        ax.legend(loc='upper right')
        plt.show()
