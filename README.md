# cat3D-master
A falling cat has to use its cat righting reflex to land on its feet! 

1) How to run it:

There are two original files, cat3D.py in the main folder, and the cat3D.py environment in gym/envs/classic_control. See  https://gym.openai.com/docs/ for a description of the original gym. To build this local environment from our source:

cd gym

pip install -e .

To run it, go to the main folder and run

python cat3D.py

2) What the environment cat3D.py does:

A cat starts upside down with zero initial momentum and zero initial angular momentum. Gravity sends the cat to the floor. The cat can do a few things to land on its feet: twist its body, tuck or untuck its legs. The simulation (approximately) enforces the zero angular momentum throughout.

The cat is composed of 5 points, P1 to P5. The legs are the segments P1P2 and P4P5. The body is the two segments P2P3 and P3P4.

3) Objective:

It'd be great if a neural network were able lo learn how to land the cat on its feet (ie the cat righting reflex). The one of the main program (cat3D.py) hasn't yet...
For now, the method "cyclic_action" of the main program is able to do it under the current parameters.

Bibliography

1)Wikipedia article: https://en.wikipedia.org/wiki/Cat_righting_reflex

2)R. Mongomery, Gauge Theory of the Falling Cat (1993) 
