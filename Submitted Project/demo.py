# Code Written by Pascal Sauerborn, 13/12/2018, Student ID: 4313894, University of Nottingham, School of Physical Sciences.

from DemoEngine import DemoEngine

# the project is demonstrated by using the DemoEngine() object. Note that this simply provides a clean and efficient way of generating a demo run. The DemoEngine uses the same functions that were used to gather the data, and all of the data gathered is generated from the DemoEngine i.e. not simply loaded from a seperate file and displayed. See README files for details.


# The demon engine is instantiated with the following road properties; n_iterations specifies number of times each road object is evolved when various data is gathered. increasing n_iterations reduces statistical fluctuations in data, but also results in longer computation time. Similarly, longer road lengths provide better data distributions, but also result in higher computational time

# All Objects and all funnctions contain doc strings that detail the parameters taken and what the functions do/return. doc strings for some function 'func' can be viewed from the interpreter as standard using help(func) or func.__doc__, or simply read directly from the code file.

# it should also be noted that much of the data generated depends on random numbers, and hence each run of the demo will produce slightly varying results.

ProjectDemo = DemoEngine(road_length=100, n_iterations=250, vmax=5, p_slow=0.1, random_state=3)

# first, an animation of the basic 1-D model is generated

ProjectDemo.show_setup()

# space time plots are then produced to demostrate smooth flow at low densities and the backpropogation of jams at higher densities

ProjectDemo.analyse_spacetime()

# The fundamental flow rate diagrams are then produced by variying the road length, maximum speed and probablity of random deceleration. Note that the values of the parameters are passed down as lists; any values and any number of values can be passed down, however the below are the recommended settings since they display a nice distribution of data with a low computation time.

ProjectDemo.analyse_flow(road_lengths=[50, 75, 100], max_speeds=[5, 10, 15], probabilities=[0.05, 0.1, 0.2])

# the effect of using various driver profile distributions is then analysed

ProjectDemo.analyse_profiles()

# finally, a Double lane system which allows for lane switching and overtaking is Demonstrated; first, an animation is generated to illustrate the system

ProjectDemo.show_setup(single_lane=False)

# the flow rate of the single lane and the double lane system are then compared

ProjectDemo.analyse_2d()
