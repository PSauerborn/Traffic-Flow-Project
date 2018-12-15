Code Written by Pascal Sauerborn, 13/12/2018,Student ID: 4313894, University of Nottingham, School of Physical Sciences.

Before going into specifics of individual files/models, it should be noted that all functions/objects contain doc strings detailing the parameters taken by the function, how the function operates and what the function returns. doc strings for some function 'func' can be viewed from the interpreter as normal using help(func) or func.__doc__, or simply read directly from the code file. Hence the first point of reference for details of how a particular component operates should be the functions/objects doc string.

The demo was tested on my personal PC and the lab computers, and ran fine on both (total run time of ~ 4 mins); however, if the demo runs too slowly, then the length of the roads used and the number of iterations can simply be reduced by changing their values upon instantiation of the demo engine


Project Contents
--------------------------------------------------------------------------------

The project consists of several files, which are summarized below. It should be noted that the underlying principle and operation of all models are the same; the details naturally vary slightly. The differences are described below.


demo.py
-------

the demo.py file provides a demo run of the project and reproduces all the fundamental data present in the report. This is the file that should be run in order to see the workings of the project. The demo.py file simply instantiates an DemoEngine object and then runs through the various methods/functions to produce the data. Note that the demo run uses less iterations and shorter road lengths than the data present in the report; this is done to ensure that the demo run executes in a reasonable amount of time. The data gathered still shows the same general trends and qualitatively contains the same information as the data gathered from the report.

DemoEngine.py
----------

the DemoEngine.py file defines the DemoEngine object used to demonstrate the project. Note that this file does not need to be run; it is imported by the demo.py file and all functions/methods are also called from the demo.py file


project_v1.py
-------------

defines the basic model as described by Nagel-Schreckenberg. Note that this file does not contain any functions for gathering data; it simply defines the Road and Car objects that are used within the project. Note that the basic model might seem slightly more complicated than need be, but this allows for easy expansion to accommodate more complex behaviour.


project_v2.py
-------------

extension of the basic model defined in project_v1.py; the code is altered in order to allow the user to define a series of driver profile distributions, which then allow the model to explore the effects that an increasingly larger number of bad drivers have on the flow rate. A driver is quantified by his probability of random deceleration and by how much he exceeds the speed limit of the road.

It should be noted that project_v2.py and project_v1.py as essentially the same code; the only difference in the two occurs at lines 188-237, where the car objects are instantiated. Once the road object is built, the model is self-contained and the evolution/data gathering process is exactly the same as with the basic model. Note that, again, this file contains no functions for the analysis/gathering of data. It simply defines the model.


project_v4.py
-------------

Further extension of the basic model to include a second lane of cars, where cars can overtake and switch lanes if preferable. Note that, again, this file contains no functions for the analysis/gathering of data. It simply defines the model.


project_functions.py
--------------------

This file contains all the functions that are used to gather/analyse the data for the models defined in project_v1 and project_v2. The first function (get_data()) is the only function that accesses the model files to gather data; all subsequent functions call on get_data() (or variations of it) to gather data which is then analysed. This function hence forms the bridge between model definition and analysis. Note that the 2D model has a separate file of functions.


project_functions_2d.py
----------------------

This file contains the functions used to gather/analyse data from the 2-lane model. Note that the functions are in principle identical to that of the 1D model. However, there are slight differences and these are specified and explained within the function itself



Flow of Information
--------------------------------------------------------------------------------

the information flow of the demo run is illustrated below

project_v1.py |
              | -->
project_v2.py |
                      | project_functions.py    |
                      |                         |  --> | DemoEngine.py | --> demo.py
                      | project_functions_2d.py |

project_v4.py | -->

Again, the models are bridged with the functions via the get_data() function

Modules and Versions Used
--------------------------------------------------------------------------------

The following modules and corresponding versions were used in the project


Python version 3.6.5, 64-bit

NumPy version 1.15.4
Matplotlib version 3.0.2
Pandas version 0.23.4
SciPy 1.1.0
Pyprind 2.11.2

(Note: the pyrind module should only be needed if certain optional settings are chosen; by default, these are all switched off and would have to be specifically chosen to be used. Hence they shouldnt be imported at all. However, in the case of an unforseen error, version is as given)
