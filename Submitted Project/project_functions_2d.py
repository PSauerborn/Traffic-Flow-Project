# Code Written by Pascal Sauerborn, 13/12/2018, Student ID: 4313894,  University of Nottingham, School of Physical Sciences.

# the following code contains the functions used to gather data and analyse the 2D model. See README file for details

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import style
from scipy.interpolate import interp1d
import project_v4 as v4


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
    switch_count: list
        list containing the number of lane switches that occured at each iteration
    """

    data = []
    avg_speeds = []
    top_speeds = []
    switch_count = []

    for i in range(n_iterations):

        # again, if flow_rate is set to True, then any unneccesary data (such as average and top speed) is not gathered/calculated by the model to boost computational efficiency.

        if flow_rate:
            data_points = road.update(flow_rate=True)
            data.append(data_points)

        else:
            data_points, avg_speed, top_speed, switches = road.update()

            avg_speeds.append(avg_speed)
            top_speeds.append(top_speed)
            data.append(data_points)
            switch_count.append(switches)

    if flow_rate:
        return data
    else:
        return data, top_speeds, avg_speeds, switch_count

def animate_system_2lanes(road_length=40, car_count=10, vmax=5, p_slow=0.2):

    """Function that animates the Double Lane system"""

    def animate(i):
        """Animation function called by FuncAnimation"""

        data, avg_speed, top_speed, switch = M1.update()

        lane1 = M1.road[0, :]
        lane2 = M1.road[1, :]

        lane1 = np.where(lane1 !=0, 1, 0)
        lane2 = np.where(lane2 != 0, 1.25, 0)

        ax[0].clear()

        road_plot1 = ax[0].scatter(x, lane1, marker='s', c='b')
        road_plot2 = ax[0].scatter(x, lane2, marker='s', c='b')

        ax[0].set_yticklabels([])
        ax[0].set_ylim(0.5, 1.5)

        avg_speeds.append(avg_speed)
        top_speeds.append(top_speed)

        avg_plot = ax[1].plot(avg_speeds, c='b', lw=0.5)
        top_plot = ax[1].plot(top_speeds, c='r', lw=0.5)

        ax[1].set_ylabel('Speed')
        ax[1].set_xticklabels([])
        ax[1].set_xlim(i - 10, i + 10)
        ax[1].set_ylim(0, 11)

        return avg_plot, top_plot,

    #  a road object is instantiated

    M1 = v4.Road(L=road_length, car_count=car_count, random_state=3, vmax=vmax, p_slow=p_slow)

    avg_speeds = []
    top_speeds = []

    # and the results are plotted

    x = np.arange(0, M1.L)

    lane1 = M1.road[0, :]
    lane2 = M1.road[1, :]

    lane1 = np.where(lane1 !=0, 1, 0)
    lane2 = np.where(lane2 != 0, 1.25, 0)

    # note that two plots are made; on displays the configuration of the road, and the other displays the average and top speeds of the system

    fig, ax = plt.subplots(nrows=2, ncols=1)

    road_plot1 = ax[0].scatter(x, lane1, marker='s', c='b')
    road_plot2 = ax[0].scatter(x, lane2, marker='s', c='b')

    ax[0].set_yticklabels([])
    ax[0].set_ylim(0.5, 1.5)

    top_plot, = ax[1].plot(0, 0, c='r',  label='Top Speed')
    avg_plot, = ax[1].plot(0, 0, c='b', label='Average Speed')
    ax[1].legend(loc='upper right')

    # the animation function defined above is called at with a framerate of 200 ms

    ani = FuncAnimation(fig, animate, interval=200)
    plt.tight_layout()
    plt.show()

def get_flow_rate(road_length, n_iterations, print_iter=False, vmax=5, p_slow=0.1, random_state=1, prog_bar=True, ratios=None):

    """Function used to gather the flow rate data of the traffic system. An important distinction needs to be made about the 2D system; the 2D system has twice the number of cars on it as the 1D system for any given density value. Additionally, flow is not measured as the number of passes over a single site, but rather as the number of passes over a pair of parallel sites.

    Parameters
    ----------
    road_length: int
        length of the road to be created
    n_iterations: int
        number of times the generated road object should be evolved
    print_iter: boolean (default=False)
        if set to True, the iteration that the system is on is printed to screen
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

        count = count * 2

        # a road object with the specified variables in instantiated

        if ratios is None:

            M1 = v4.Road(L=road_length, car_count=count, vmax=vmax,
                         p_slow=p_slow, random_state=random_state)

        else:
            M1 = v4.Road(L=road_length, car_count=count, vmax=vmax,
                         p_slow=p_slow, random_state=random_state, ratios=ratios)

        # the road object is then evolved a set number of times (given by the n_iterations variable). The 'data' list is made up of a serious of sublists, where each sublist corresponds to one iteration of the system. The data within each sublist consits of tuples which contain data in the form (car.position, car.speed, car.lane)

        data = get_data(M1, n_iterations, flow_rate=True)

        # the traffic flow with respect to a certain sites is defined as the number of passes those sites have per unit time

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

        if print_iter:
            print(i)

        i += 1

        if prog_bar:
            pbar.update()

    return flow_rate

def get_flow_v2(road_length, n_iterations, vmax=5, type='road_length', p_slow=0.1, ratios=None):

    """Convenience function used to gather data. Note that this function merely calls the get_flow_rate function and then saves the data to the corresponding directory. Hence this function does the same thing as the get_flow_rate function, expect that is saves the data as well once finished.

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

    path = '.\\data\\flow_rate_2d\\{}\\road_length_{}_n_iterations_{}_vmax_{}.npy'.format(
        type, road_length, n_iterations, vmax)

    if type == 'profiles':
        path = '.\\data\\flow_rate_2d\\{}\\road_length_{}_n_iterations_{}_type_{}.npy'.format(
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
        path = '.\\data\\flow_rate_2d\\{}\\road_length_{}_n_iterations_{}_type_{}.npy'.format(
            type, road_length, n_iterations, profile)


    else:
        path = '.\\data\\flow_rate_2d\\{}\\road_length_{}_n_iterations_{}_vmax_{}.npy'.format(
            type, road_length, n_iterations, vmax)

    # the data is then loaded and prepared

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

def get_space_time_2d(road_length=100, car_count=20, n_iterations=50, vmax=5, p_slow=0.1, random_state=3):

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

    M1 = v4.Road(L=road_length, car_count=car_count, vmax=vmax,
                 p_slow=p_slow, random_state=random_state)

    data = get_data(M1, n_iterations, flow_rate=True)

    fig, axes = plt.subplots(ncols=1, nrows=2, sharex=True, sharey=True)

    plot_space_time_map_2d(data, axes, road_length, n_iterations, car_count)

    axes[0].xaxis.set_label_position('top')
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    plt.show()

def plot_space_time_map_2d(data, axes, road_length, n_iterations, car_count):

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
    ax1, ax2 = axes[0], axes[1]

    x, y = np.meshgrid(np.arange(0, road_length), np.arange(0, n_iterations))

    # the results are then plotted. Note that the data is plotted in reverse so that the the first evolution is plotted on the top row, and consequent evolutions are plotted vertically downwards. Additionally, each lane is plotted ona  seperate axes

    for i, set in enumerate(reversed(data)):
        for car in set:
            if car[2] == 0:
                ax1.scatter(car[0], i, c='black', s=0.5)

            else:
                ax2.scatter(car[0], i, c='black', s=0.5)

    ax1.set_xlabel(r'$Space \ \longrightarrow$', fontsize=20, labelpad=20)
    ax1.set_ylabel(r'$\longleftarrow \ Time $', fontsize=20, labelpad=20)

def get_switches(road_length=300, n_iterations=250, p_slow=0.1, vmax=5, prog_bar=False):

    """Convenience functon to obtain the average number of lane switches at a given density

    Parameters
    ----------
    road_length: int
        length of road object to be instaited
    n_iterations: int
        number of times the system is evolved
    p_slow: float, 0 <= p_slow < 1
        probability of random deceleration
    vmax: int
        maximum speed of road
    prog_bar: boolean
        if set to True, a progress bar is included. Note that this requires the pyprind module to be installed and hence defaults to False

    """

    car_counts = [i for i in range(1, road_length)]
    densities = [count / road_length for count in car_counts]

    if prog_bar:
        from pyprind import ProgBar
        prog = ProgBar(len(densities))

    # the number of switches is stored in a dataframe

    switches = pd.DataFrame(np.zeros(len(densities)), dtype=float, columns=['Avg Number of Switches'], index=densities)
    switches.index.name = 'Density'

    for density, count in zip(densities, car_counts):

        # a road object is instantiated for each density value

        M1 = v4.Road(L=road_length, car_count=count*2, vmax=vmax, p_slow=p_slow, random_state=3)

        # the data is then gathered via the get_data() function

        data, top_speeds, avg_speeds, switch_count = get_data(M1, n_iterations=n_iterations)

        switch_count = np.array(switch_count)

        # the average number of switches is then stored at the corresponding density value

        switches.loc[density] = np.mean(switch_count)

        if prog_bar:

            prog.update()

    return switches
