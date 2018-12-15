# Code Written by Pascal Sauerborn, 13/12/2018, Student ID: 4313894, University of Nottingham, School of Physical Sciences.

# the following code contains the functions used to gather data and analyse the basic model. See README file for details

from matplotlib.animation import FuncAnimation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
from scipy.interpolate import interp1d
import os

import project_v1 as v1
import project_v2 as v2


def get_data(road, n_iterations=100, flow_rate=False):

    """Convenience function to run the simulation for a set number of iterations and gather the corresponding data

    Parameters
    ----------
    road: road object
        the road object that contains the the cars
    n_iterations: int
        the total number of times the system is evolved
    flow_rate: boolean, default=False
        when the flow rate is evaluated, certain data (such as the average speed and top speed) is not neccesary. In order to boost computational efficiency, if flow_rate is set to True, uneccesary data is not gathered

    Returns
    -------
    data: list, shape=[n_iterations]
        the data is a list; each individual element within the list contains a further list, which contains the data for each iteration. Each individual data point is a tuple consisting of (car.position, car.speed)
    top_speed: list, shape=[n_iterations]
        list containing the top speed at each iteration
    avg_speed: list, shape=[n_iterations]
        list containing the average speed at each iteration
    """

    data = []
    avg_speeds = []
    top_speeds = []

    for i in range(n_iterations):

        if flow_rate:
            data_points = road.update(flow_rate=True)
            data.append(data_points)
        else:
            data_points, avg_speed, top_speed = road.update()

            avg_speeds.append(avg_speed)
            top_speeds.append(top_speed)
            data.append(data_points)

    if flow_rate:
        return data
    else:
        return data, top_speeds, avg_speeds

def plot_setup():

    """Function used to illustrate the model using the imshow function"""

    # a road object is instantiated

    style.use('bmh')

    M1 = v1.Road(L=40, car_count=5, vmax=5,
                 p_slow=0., random_state=2)

    # the system is then evolved 5 times

    data = get_data(M1, n_iterations=5, flow_rate=True)

    # axis and figure objects are instantiated. Note that, in order to keep the graph clean, instead of plotting a 40x5 matrix, each iteration of the road is plotted on a seperate axes. This allows greater control in terms of where verything is placed and annotated.

    fig, ax = plt.subplots(nrows=5, ncols=1, sharex=True, sharey=True)
    plt.subplots_adjust(bottom=0.3)

    # the data is returned in the form of a a series of lists within a list. Each nested list is the data for one iteration of the system, which contains the data in a tuple of the form (car.position, car.speed)

    for count, subset in enumerate(data):
        road = np.zeros((40))
        for car in subset:

            # the cars are then placed on the 'road' array; note that a 'car' in this case is a tuple of form (car.position, car.speed). The value stored in the array is the speed + 1. Note that the +1 ensures that the speed is not stored as 0 so that when the imshow funciton is used to display the road, the cars with speed = 0 can still be seen

            road[car[0]] = car[1] + 1
            ax[count].annotate(car[1], (car[0], 0), ha='center', va='center')

        a = ax[count].imshow(road.reshape(-1, 40), aspect='equal',
                             cmap='Wistia')

    ax[0].set_yticks([])
    ax[0].set_xticks([])
    ax[0].set_xlabel(r'$Space \longrightarrow$', labelpad=20, fontsize=20)
    ax[2].set_ylabel(r'$\longleftarrow Time $', labelpad=20, fontsize=20)
    ax[0].xaxis.set_label_position('top')
    ax[0].xaxis.tick_top()

    plt.show()

def get_animation(road_length=100, car_count=20, vmax=5, p_slow=0.2, random_state=3):

    """Convenience Function used to generate an animated plot of the setup

    Parameters
    ----------
    road_length: int
        length of road object the function instantiates
    car_count: int
        number of cars placed on road object
    vmax: int
        maximum speed of road
    p_slow: float
        probability of random deceleration
    random_state: int
        seed used for random number generator object

    """

    M1 = v1.Road(L=road_length, car_count=car_count, vmax=vmax,
                 p_slow=p_slow, random_state=random_state)


    def animate(i):
        """Animation function called by FuncAnimation"""

        # the road object is updated and the data is gathered

        data_points, avg_speed, top_speed = M1.update()

        # the data is then stored in its respective lists

        avg_speeds.append(avg_speed)
        top_speeds.append(top_speed)
        data.append(data_points)

        # the road representation is retrieved. Note that this is what is actually plotted

        road_rep = M1.road_rep

        ax[0].clear()
        road_plot = ax[0].scatter(x, road_rep, marker='s')

        ax[0].set_ylim(0.25, 1.75)
        ax[0].set_yticklabels([])

        # the average and top speeds are then plotted on a seperate axes

        avg_plot = ax[1].plot(avg_speeds, c='b', lw=0.5)
        top_plot = ax[1].plot(top_speeds, c='r', lw=0.5)

        ax[1].set_ylabel('Speed')
        ax[1].set_xticklabels([])
        ax[1].set_xlim(i - 10, i + 10)

        return avg_plot, top_plot,

    style.use('bmh')

    avg_speeds = []
    top_speeds = []
    data = []

    x = np.arange(0, road_length)
    y = M1.road_rep

    fig, ax = plt.subplots(nrows=2, ncols=1)

    road_plot = ax[0].scatter(x, y, marker='s')
    ax[0].set_yticklabels([])

    top_plot, = ax[1].plot(0, 0, c='r',  label='Top Speed')
    avg_plot, = ax[1].plot(0, 0, c='b', label='Average Speed')
    ax[1].legend(loc='upper right')

    ani = FuncAnimation(fig, animate, interval=200)
    plt.tight_layout()
    plt.show()

def get_space_time(road_length=100, car_count=20, n_iterations=50, vmax=5, p_slow=0.1, random_state=3):

    """Convenience Function used to generate the data for the spacetime maps, demonstrating smooth flow at low densities along with the backpropogation of traffic jams at higher densities.

    Parameters
    ----------
    road_length: int
        length of road object instantiated by the function
    car_count: int
        number of cars on the road object
    n_iterations: int
        number of times the system is evolved
    vmax: int
        maximum speed of cars on road object
    p_slow: float
        probability of random deceleration
    random_state: int
        seed used for random number generator object
    """

    M1 = v1.Road(L=road_length, car_count=car_count, vmax=vmax,
                 p_slow=p_slow, random_state=random_state)

    data, top_speeds, avg_speeds = get_data(M1, n_iterations)

    fig, ax = plt.subplots()

    plot_space_time_map(data, ax, road_length, n_iterations)

    ax.xaxis.set_label_position('top')
    ax.set_xticks([])
    ax.set_yticks([])

    plt.show()

def plot_space_time_map(data, axes, road_length, n_iterations):

    """Convenience function to plot the spacetime map data

    parameters
    ----------
    data: list, shape=[n_iterations]
        the data is a list; each individual element within the list contains a further list, which contains the data for each iteration. Each individual data point is a tuple consisting of (car.position, car.speed)
    axes: matplotlib axes object
        axes on which to plot the data
    road_length: int
        length of the road
    n_iterations: int
        the number of times the simulation was evolved
    """
    # style.use('seaborn')

    x, y = np.meshgrid(np.arange(0, road_length), np.arange(0, n_iterations))

    axes.scatter(x, y, s=1)

    # the results are then plotted. Note that the data is plotted in reverse so that the the first evolution is plotted on the top row, and consequent evolutions are plotted vertically downwards

    for i, set in enumerate(reversed(data)):
        for car in set:
            axes.annotate(car[1], (car[0], i))

    axes.axis('equal')

    axes.set_xlabel(r'$Space \ \longrightarrow$', fontsize=30, labelpad=20)
    axes.set_ylabel(r'$\longleftarrow \ Time $', fontsize=30, labelpad=20)

# the analyze the equilibrium time

def test_equi_time_v2():

    """Function used to investigate the equilibrium time of a system. Note that the system is defined to be in equilibrium when the average speed remains unchanged for 10 iterations"""

    n_iterations = 100
    vmax = 5
    p_slow = 0.
    random_state = 3

    road_lengths = np.arange(100, 1000, 5)

    densities = [0.05, 0.1, 0.2, 0.5]
    equi_times = pd.DataFrame(np.zeros((road_lengths.shape[0], len(
        densities))), index=road_lengths, columns=densities)

    for density in densities:
        for k, length in enumerate(road_lengths):

            car_count = int(np.ceil(density * length))

            avg_speeds = np.zeros(n_iterations)

            M1 = v1.Road(L=length, car_count=car_count, vmax=vmax,
                             p_slow=p_slow, random_state=random_state)

            for count, i in enumerate(range(n_iterations)):

                data_points, avg_speed, top_speed = M1.update()

                avg_speeds[i] = avg_speed

                    # the system has reached equilibrium if the average speed doesnt change within 5 iterations

                if i > 5:

                    if np.array_equal(avg_speeds[i - 5:i], np.array([avg_speed for i in range(5)])):
                        equi_time = count
                        break

            equi_times.loc[length, density] = equi_time


    style.use('bmh')

    fig, ax = plt.subplots()

    colors = ['red', 'limegreen', 'blue', 'black']

    for density, c in zip(densities, colors):

        sns.regplot(x=equi_times.index, y=equi_times[density], color=c, label=r'$\rho = ${}'.format(
            density), scatter_kws={"s": 2}, line_kws={'lw': 1})

    ax.set_xlabel(r'$Road \ Length$', labelpad=10, fontsize=20)
    ax.set_ylabel(r'$Iterations \ to \ Equilibrium$', labelpad=10, fontsize=20)
    ax.set_xlim(100, 1000)
    ax.legend(prop={'size': 20})

    plt.show()

def get_flow_rate(road_length, n_iterations, vmax=5, p_slow=0.1, random_state=1, prog_bar=True, ratios=None):

    """Function used to gather data for the fundamental flow rate plots of the traffic system

    Parameters
    ----------
    road_length: int
        length of the road to be created
    n_iterations: int
        number of times the generated road object should be evolved
    vmax: int (default=5)
        speed limit of road
    p_slow: float (default=0.1)
        probability of spontaneous slowdown
    random_state: int
        number used as seed for numpy RandomGenerator object
    prog_bar: boolean (default=False)
        if set to True, a progress bar is generated using the Pyprind Module. Note that this requires the pyprind module to be installed

    Returns
    -------
    flow_rate: NumPy ndarray object, shape = [number of density values, 2]
        array object that contains the densities and the corresponding average flow rate
    """

    # the function comes with an optional progress bar. However, thus requires the pyprind module to be installed

    if prog_bar:

        import pyprind
        pbar = pyprind.ProgBar(road_length)

    # a particular site to be measured is chosen. In this case, the middle site is chosen.

    site = int(road_length / 2)

    # the simulation is run with a range of different car numbers

    car_count = [i for i in range(1, road_length, 1)]
    densities = [i / road_length for i in car_count]

    flow_rate = np.zeros((len(densities) + 1, 2))

    # the 'i' variable is simply a counter used to store the data in the corresponding arrays cells

    i = 0

    for count, density in zip(car_count, densities):

        # a road object with the specified variables in instantiated

        if ratios is None:

            M1 = v2.Road(L=road_length, car_count=count, vmax=vmax,
                         p_slow=p_slow, random_state=random_state)

        else:
            M1 = v2.Road(L=road_length, car_count=count, vmax=vmax,
                         p_slow=p_slow, random_state=random_state, ratios=ratios)

        # the road object is then evolved a set number of times (given by the n_iterations variable). The 'data' list is made up of a serious of sublists, where each sublist corresponds to one iteration of the system. The data within each sublist consists of tuples which contain data in the form (car.position, car.speed, car.lane)

        data = get_data(M1, n_iterations, flow_rate=True)

        # the traffic flow with respect to a certain site is defined as the number of passes that site has per unit time

        passes = 0

        # a subset refers/corresponds to the data from one evolution of the traffic system

        for subset in data:

            # a 'car' in this case is a tuple of (car.position, car.speed).

            for car in subset:

                # initial position of the car is evaluated. Note that the middle site is chosen so that the boundary conditions can effetively be ignored (since the car will never travel over half the road in a course of one iteration). However, it is included for completeness

                initial_pos = car[0] - car[1]

                if initial_pos < 0:
                    initial_pos += road_length

                # if the chosen site (defined as road_length/2) lies within the intial position i.e. position_initial < site < position_final, then the car has passed said site in that evolution, and the 'passes' variable is increased by 1

                if site in range(initial_pos, car[0]):
                    passes += 1

        # the number of passes and the corresponding density is then stored

        flow_rate[i, 0], flow_rate[i, 1] = density, passes / n_iterations

        i += 1

        if prog_bar:
            pbar.update()

    return flow_rate

def get_flow_v2(road_length, n_iterations, vmax=5, type='road_length', p_slow=0.1, ratios=None):

    """Convenience function used to gather data. Note that this function stores the gathered data in a corresponding data file.  Note that this function merely calls the get_flow_rate function and then saves the data to the corresponding directory. Hence this function does the same thing as the get_flow_rate function, expect that is saves the data as well once finished.

    Parameters
    ----------
    road_length: int
        length of road object used in dataset
    n_iterations: int
        number of times the instantiated road object was evolved
    vmax: int (default=5)
        maximum road speed used
    type: str (default = 'road_length')
        indicates the type of parameter that was varied to produce the dataset. Accepted values are ['road_length', 'speed', 'proba']
    p_slow: float (default=0.1)
        probability of sponateous slow down
    """

    # the path where the data is to be saved is first defined

    path = '.\\data\\flow_rate\\{}\\road_length_{}_n_iterations_{}_vmax_{}.npy'.format(
        type, road_length, n_iterations, vmax)

    if type == 'profiles':
        path = '.\\data\\flow_rate\\{}\\road_length_{}_n_iterations_{}_type_{}.npy'.format(
            type, road_length, n_iterations, ratios['bad'])


    # the data is then gathered

    data = get_flow_rate(road_length=road_length,
                         n_iterations=n_iterations, vmax=vmax, p_slow=p_slow, ratios=ratios)

    # the data is saved

    np.save(path, data)

def load_flow_data(road_length, n_iterations, vmax=5, type='road_length', interpolate=False, profile=None):

    """Convenience Function used to load a saved data set. Note that this function can only be used to load data already gathered. If a dataset does not exist, the funcion shows an error message

    Parameters
    ----------
    road_length: int
        length of road object used in dataset
    n_iterations: int
        number of times the instantiated road object was evolved
    vmax: int (default=5)
        maximum road speed used
    type: str (default = 'road_length')
        indicates the type of parameter that was varied to produce the dataset. Accepted values are ['road_length', 'speed', 'proba']
    interpolate: boolean (default=False)
        if set to True, the data is interpolated using the Scipy interp1d function

    Returns
    -------
    data: Numpy ndarray object, shape = [road_length, 2]
        array containing the densities and corresponding flow rate
    """

    # the path for the desired file is first generated using the given arguments

    if type == 'profiles':
        path = '.\\data\\flow_rate\\{}\\road_length_{}_n_iterations_{}_type_{}.npy'.format(
            type, road_length, n_iterations, profile)


    else:
        path = '.\\data\\flow_rate\\{}\\road_length_{}_n_iterations_{}_vmax_{}.npy'.format(
            type, road_length, n_iterations, vmax)

    # the data is then loaded and prepared

    try:
        data = np.load(path)
        data = data[data[:, 1] != 0]

        if interpolate:

            cubic = interp1d(data[:, 0], data[:, 1], kind='cubic')
            x = np.linspace(0.02, 0.98, 1000)
            data = pd.DataFrame(columns=['density', 'flow_rate'])

            data['density'] = x
            data['flow_rate'] = cubic(x)
            data = data.values

        return data

    except:
        print('Error: Dataset does not exist. Please use different values or generate a new dataset')
