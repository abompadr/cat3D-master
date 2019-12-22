# cat3D-master
A falling cat has to use its cat righting reflex to land on its feet! 

1) How to run it:

There are two different versions to model a cat falling to the ground: the "cat3D" and a simpler one with fewer actions, "cat3D_simpler". Each one has two files, cat3D.py or cat3D_simpler.py in the main folder, and the corresponding cat3D.py or the cat3D_simpler.py environment in gym/envs/classic_control. See  https://gym.openai.com/docs/ for a description of the original gym. 

To build this local environment from our source:

cd gym

pip install -e .

To run it, go to the main folder and run either

python cat3D.py

or

python cat3D_simpler.py

2) What the environment cat3D.py (or cat3D_simpler.py) does:

A cat starts upside down at a certain high with zero initial velocity and zero initial angular momentum. Gravity sends the cat to the floor. The cat can do a few things to land on its feet: twist its body one way or another, or tuck or untuck its legs. The simulation (approximately) enforces the zero angular momentum throughout.

The cat is composed of 5 points, P1 to P5. The legs are the segments P1P2 and P4P5. The body is the two segments P2P3 and P3P4.

3) Objective:

The cat has to land on its feet.

4) Machine learning objective:

A neural network should try to learn the sequence of movements to land the cat on its feet. The one of the cat3D_simpler environment is able to learn to do it. 
It'd be great if a neural network in the more complex version (cat3D) were able lo learn how to land the cat on its feet (ie the cat righting reflex). The one of the main program (cat3D.py) hasn't yet...
For now, the method "cyclic_action" of the main program is able to do it under the current parameters.

Bibliography

1)Wikipedia article: https://en.wikipedia.org/wiki/Cat_righting_reflex

2)R. Mongomery, Gauge Theory of the Falling Cat (1993) 
