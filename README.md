liljonnystyle.github.io
=======================

Robot that learns to play the 2048 game!

langugages used: Python, javascript, html

python libraries used: numpy, scipy, MySQLdb, PyV8

The 2048 game is cloned from gabrielecirulli and is being modified to accommodate this project. An artificial neural network is implemented in js/robot.py using the backpropagation algorithm. Ultimately, as the user plays the game, the board configuration and user inputs will populate a training dataset. When switched over to robot-mode, the machine will learn from the training set, and then play the game by itself.

Warning: this is a project in development.

Challenges to address: javascript-python interface, test ANN structure (add more nodes or levels?), output training statistics to browser, ...
