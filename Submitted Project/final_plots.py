import matplotlib.pyplot as plt
from matplotlib import style
from project_functions import load_flow_data, get_data

import numpy as np
import project_v1 as v1


def plot_flow_rate_v2_road_length():
    """Convenice function used to plot the data gathered for the road_length test"""

    style.use('bmh')

    data1_road = load_flow_data(road_length=500, n_iterations=10000, vmax=5)
    data2_road = load_flow_data(road_length=200, n_iterations=10000, vmax=5)
    data3_road = load_flow_data(road_length=100, n_iterations=10000, vmax=5)
    data4_road = load_flow_data(road_length=50, n_iterations=10000, vmax=5)

    fig, ax = plt.subplots()

    ax.plot(data1_road[:, 0], data1_road[:, 1], lw=1, c='r', label='500')
    ax.plot(data2_road[:, 0], data2_road[:, 1], lw=1, c='b', label='200')
    ax.plot(data3_road[:, 0], data3_road[:, 1],
            lw=1, c='limegreen', label='100')
    ax.plot(data4_road[:, 0], data4_road[:, 1], lw=1, c='black', label='50')

    ax.set_xlim(0, 0.8)
    ax.set_ylim(0, 1)
    ax.set_ylabel(
        r'$Flow \ Rate \ \frac{passes}{iteration}$', labelpad=10, fontsize=30)
    ax.set_xlabel(r'$Density \ \frac{cars}{sites}$', labelpad=10, fontsize=30)

    from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

    # zoom-factor: 2.5, location: upper-left
    axins = zoomed_inset_axes(ax, 2.5, loc='upper right')
    axins.plot(data1_road[:, 0], data1_road[:, 1], lw=1, c='r', label='500')
    axins.plot(data2_road[:, 0], data2_road[:, 1], lw=1, c='b', label='200')
    axins.plot(data3_road[:, 0], data3_road[:, 1],
               lw=1, c='limegreen', label='100')
    axins.plot(data4_road[:, 0], data4_road[:, 1], lw=1, c='black', label='50')

    x1, x2, y1, y2 = 0.1, 0.25, 0.6, 0.8  # specify the limits
    axins.set_xlim(x1, x2)  # apply the x-limits
    axins.set_ylim(y1, y2)
    axins.legend(loc='upper right', prop={'size': 20})

    ax.minorticks_on()
    axins.minorticks_on()

    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
    # #
    plt.show()

def plot_flow_rate_v2_speed():
    """Convenice function used to plot the data gathered for the maximum speed test"""

    style.use('bmh')

    data1 = load_flow_data(
        road_length=500, n_iterations=10000, vmax=5, type='speed')
    data2 = load_flow_data(
        road_length=500, n_iterations=10000, vmax=10, type='speed')
    data3 = load_flow_data(
        road_length=500, n_iterations=10000, vmax=20, type='speed')
    data4 = load_flow_data(
        road_length=500, n_iterations=10000, vmax=50, type='speed')

    fig, ax = plt.subplots()

    ax.plot(data1[:, 0], data1[:, 1], lw=1, c='r', label='5')
    ax.plot(data2[:, 0], data2[:, 1], lw=1, c='b', label='10')
    ax.plot(data3[:, 0], data3[:, 1], lw=1, c='limegreen', label='20')
    ax.plot(data4[:, 0], data4[:, 1], lw=1, c='black', label='50')

    ax.set_ylabel(
        r'$Flow \ Rate \ \frac{passes}{iteration}$', labelpad=10, fontsize=30)
    ax.set_xlabel(r'$Density \ \frac{cars}{sites}$', labelpad=10, fontsize=30)

    from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

    # zoom-factor: 2.5, location: upper-left
    axins = zoomed_inset_axes(ax, 2, loc='upper right')
    axins.plot(data1[:, 0], data1[:, 1], lw=1, c='r', label='5')
    axins.plot(data2[:, 0], data2[:, 1], lw=1, c='b', label='10')
    axins.plot(data3[:, 0], data3[:, 1], lw=1, c='limegreen', label='20')
    axins.plot(data4[:, 0], data4[:, 1], lw=1, c='black', label='50')

    x1, x2, y1, y2 = 0, 0.3, 0.55, 0.99  # specify the limits
    axins.set_xlim(x1, x2)  # apply the x-limits
    axins.set_ylim(y1, y2)
    axins.legend(loc='upper right', prop={'size': 20})
    ax.minorticks_on()
    axins.minorticks_on()

    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
    plt.show()

def plot_flow_rate_v2_proba():
    """Convenice function used to plot the data gathered for the probability test"""

    style.use('bmh')

    data1_road = load_flow_data(
        road_length=500, n_iterations=9995, vmax=5, type='proba')
    data2_road = load_flow_data(road_length=500, n_iterations=10000, vmax=5)
    data3_road = load_flow_data(
        road_length=500, n_iterations=9998, vmax=5, type='proba')
    data4_road = load_flow_data(
        road_length=500, n_iterations=9997, vmax=5, type='proba')

    fig, ax = plt.subplots()

    ax.plot(data1_road[:, 0], data1_road[:, 1], lw=1, c='r', label='0.05')
    ax.plot(data2_road[:, 0], data2_road[:, 1], lw=1, c='b', label='0.1')
    ax.plot(data3_road[:, 0], data3_road[:, 1],
            lw=1, c='limegreen', label='0.2')
    ax.plot(data4_road[:, 0], data4_road[:, 1], lw=1, c='black', label='0.3')

    ax.set_xlim(0, 0.8)
    ax.set_ylim(0, 1)
    ax.set_ylabel(
        r'$Flow \ Rate \ \frac{passes}{iteration}$', labelpad=10, fontsize=30)
    ax.set_xlabel(r'$Density \ \frac{cars}{sites}$', labelpad=10, fontsize=30)
    ax.legend(loc='upper right', prop={'size': 20})
    ax.minorticks_on()
    plt.show()

def plot_flow_rate_v2_profiles():

    style.use('bmh')

    data_perfect = load_flow_data(road_length=500, n_iterations=10000, vmax=15)
    data1 = load_flow_data(road_length=500, n_iterations=10000, vmax=15, type='profiles', profile='0.2')
    data2 = load_flow_data(road_length=500, n_iterations=10000, vmax=15, type='profiles', profile='0.3')
    data3 = load_flow_data(road_length=500, n_iterations=10000, vmax=15, type='profiles', profile='0.5')

    fig, ax = plt.subplots()

    ax.plot(data_perfect[:, 0], data_perfect[:, 1], c='r', label='Perfect', lw=1)
    ax.plot(data1[:, 0], data1[:, 1], c='black', label='Profile 1', lw=1)
    ax.plot(data2[:, 0], data2[:, 1], c='limegreen', label='Profile 2', lw=1)
    ax.plot(data3[:, 0], data3[:, 1], c='b', label='Profile 3', lw=1)
    ax.legend(loc='upper right', prop={'size': 20})
    ax.set_ylabel(
        r'$Flow \ Rate \ \frac{passes}{iteration}$', labelpad=10, fontsize=30)
    ax.set_xlabel(r'$Density \ \frac{cars}{sites}$', labelpad=10, fontsize=30)
    ax.minorticks_on()
    plt.show()

def plot_2d_flow():

    data1 = load_flow_data(road_length=500, n_iterations=10000, vmax=5)
    data2 = pf.load_flow_data(road_length=500, n_iterations=10000, vmax=5)

    font = {'family': 'serif',
            'color':  'black',
            'weight': 'normal',
            'size': 15}



    style.use('bmh')

    fig, ax = plt.subplots()

    ax.plot(data1[:, 0], data1[:, 1], lw=1, c='r', label='Double Lane')
    ax.plot(data2[:, 0], data2[:, 1], lw=1, c='b', label='Single Lane')
    ax.legend(loc='upper right', prop={'size': 15})
    ax.minorticks_on()
    ax.set_ylabel(
        r'$Flow \ Rate \ \frac{Passes}{Iteration}$', labelpad=10, fontdict=font)
    ax.set_xlabel(r'$Density \ \frac{Cars}{Sites}$', labelpad=10, fontdict=font)

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

    style.use('bmh')

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
    style.use('bmh')

    x, y = np.meshgrid(np.arange(0, road_length), np.arange(0, n_iterations))

    axes.scatter(x, y, s=1)

    # the results are then plotted. Note that the data is plotted in reverse so that the the first evolution is plotted on the top row, and consequent evolutions are plotted vertically downwards

    for i, set in enumerate(reversed(data)):
        for car in set:
            axes.scatter(car[0], i, c='black', s=0.5)
            # axes.annotate(car[1], (car[0], i))

    # axes.axis('equal')

    axes.set_xlabel(r'$Space \ \longrightarrow$', fontsize=30, labelpad=20)
    axes.set_ylabel(r'$\longleftarrow \ Time $', fontsize=30, labelpad=20)

get_space_time(road_length=500, car_count=250, p_slow=0.7, random_state=1)
