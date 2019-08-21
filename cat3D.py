import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


from scores.score_logger import ScoreLogger

ENV_NAME = "cat3D-v1"

GAMMA = 0.95
LEARNING_RATE = 0.001

MEMORY_SIZE = 1000000
BATCH_SIZE = 20

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995


class DQNSolver:

    def __init__(self, observation_space, action_space):
        self.exploration_rate = EXPLORATION_MAX

        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)

        self.model = Sequential()
        self.model.add(Dense(24, input_shape=(observation_space,), activation="relu"))
        self.model.add(Dense(24, activation="relu"))
        #new layer?
        self.model.add(Dense(24, activation="relu"))
        self.model.add(Dense(self.action_space, activation="linear"))
        self.model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, state_next, terminal in batch:
            q_update = reward
            if not terminal:
                q_update = (reward + GAMMA * np.amax(self.model.predict(state_next)[0]))
            q_values = self.model.predict(state)
            q_values[0][action] = q_update
            self.model.fit(state, q_values, verbose=0)
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)

    def save_weights(self, name):
        self.model.save_weights(name)

    def load_weights(self, name):
        print('Loading weights')
        self.model.load_weights(name)

#returns an action that repeats according to a cycle
#It implements the steps described in the Cat righting reflex article from Wikipedia (https://en.wikipedia.org/wiki/Cat_righting_reflex)

def cyclic_action(step, env):
   
    _, penalty = env.cost()   
    if penalty < 50:
        action = 0
    else:    
        cycle = 22
        if step % cycle == 0:
            action = 3
        elif (step % cycle >= 1 and step % cycle <= 9 ):
            action = 1
        elif step % cycle == 10:
            action = 4
        elif step % cycle == 11:
            action = 5
        elif (step % cycle >=12 and step % cycle <= 20):
            action = 2
        elif step % cycle == 21:
            action = 6
        else: # should not enter here
            action = 0
    return action
         
def runCat3D():
    env = gym.make(ENV_NAME)
    score_logger = ScoreLogger(ENV_NAME)
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
        
    '''
    method = 1: use DQNSolver
    method = 2: use cyclic_action, which makes the cat land on its feet (under the current initial conditions and parameters)
    method = 3: user inputs the actions
    '''
    method = 2
    
    if method == 1:
        dqn_solver = DQNSolver(observation_space, action_space)
        fileWeights = "weights3D.h5"
        #uncomment to start off with saved weights
        dqn_solver.load_weights(fileWeights)
    
    run = 0
    while run < 10: #True:
        run += 1
        state = env.reset()
        state = np.reshape(state, [1, observation_space])
        step = 0
        while True:            
            env.render()
         
            if method == 1:            
                action = dqn_solver.act(state)
            elif method == 2:
                action = cyclic_action(step, env)
            else:
                action = input("Enter action")       


            action = int(action)
            state_next, reward, terminal, info = env.step(action)
            reward = reward if not terminal else -reward
            state_next = np.reshape(state_next, [1, observation_space])
            if method == 1:
                dqn_solver.remember(state, action, reward, state_next, terminal)

            state = state_next
            
            if terminal:
                if method == 1:
                    print ("Run: " + str(run) + ", exploration: " + str(round(dqn_solver.exploration_rate,4)) + ", score: " + str(round(reward,2)))
                else:    
                    print ("Run: " + str(run) + ", score: " + str(round(reward,2)))
                #score_logger.add_score(int(reward), run)
                break
            
            if method == 1:
                dqn_solver.experience_replay()
                if run % 100 == 0:
                    dqn_solver.save_weights(fileWeights)

            step += 1    

    input("End. Press any key")
    env.close()

if __name__ == "__main__":
    runCat3D()
