# BirdBrain

Neural Network Project Initiated for the CSM Robotics Club Exoskeleton Team's WreckItRalph Project

Matlab code can support Artificial Neural Network with fully connected layers and various adjustable parameters

Python code can support any combination in layers involved in a Convolution Neural Network

The Convolution type layers are designed for N sets of 2D arrays with c channels. Ex: multiple RGB Images at a time



# TODO

No further improvements are planned for the Matlab code

Debugg the python code to ensure the program is fully functioning

Incoprerate different methods of gradient descent such as Momentum and Nesterov accelerated gradient
(Not Neuton's method! It was a hard enough with just the first derivative)

Incorperate S-shaped ReLU activation function
(If I can get this to work then other functions with more than 1 learnable parameter will also work)

Add Recurrent Layers to add RNN functionality
(Used in transient data predictions. Ex: alexa)


Add Evolution Training Method
(Will probably be something like given a set of neural networks and a score array, will produce a new generation)
(Using and getting the score of each Neural Network is up to the user)


Create a new version that uses the GPU to do calculations much faster
(Only if I run out of other improvements to make!)
(The new version would be much faster, but will lose usability on a Rasberry Pi or other device that does not have a GPU)
