"""
Cat in 3D
Loosely follows the cat righting reflex description of Wikipedia: see https://en.wikipedia.org/wiki/Cat_righting_reflex

The original code came from cartpole.py:
Based on classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from scipy.optimize import minimize

class cat3D_simpler(gym.Env):
    """
    Description:
        A 3D cat is falling due to gravity. The agent can increase or decrease the angle formed by the legs of the cat, tuck or untuck the legs. 
        The objective is for the cat to land on its paws.
        The cat is described by 5 points P1, P2, ..., P5. The legs are P1P2 and P4P5. The body is P2P3P4.
    Source:
        No source
    Observation: 
        Type: Box(6)
        Num	Observation                 Min         Max
        0       x-position of P1
        1       x-position of P2
        2       x-position of P3
        3       x-position of P4
        4       x-position of P5
        5       y-position of P1
        6       y-position of P2
        7       y-position of P3
        8       y-position of P4
        9       y-position of P5
        10      z-position of P1
        11      z-position of P2
        12      z-position of P3
        13      z-position of P4
        14      z-position of P5        
       (       x-position of the center of mass         (constant 0))
        15       y-position of the center of mass         0          400
        (       z-position of the center of mass        (constant 0))      
        16       y velocity                               -Inf       Inf
        17       (TB deprecated) angle 1 between negative x axis and P1P2            0 deg      180 deg
        18       angle 2 between P1P2 and P2P3                       0 deg      180 deg
        19       angle 3 between P2P3 and P3P4                       0 deg      180 deg
        20       angle 4 between P3P4 and P4P5                       0 deg      180 deg
        21       angle between P1P2 and the plane P2P3P4             0 deg      360 deg
        22       angle of the rotation of the cat on axis x          0 deg      360 deg
        23       angle of the rotation of the cat on axis y          0 deg      360 deg
        24       angle of the rotation of the cat on axis z          0 deg      360 deg
        
    Actions:
        Type: Discrete(13)
        Num	Action
        0	do nothing
(        --	Increase angle 2)
(        --	decrease angle 2)        
(        --	Increase angle 3)
(        --	decrease angle 3)        
(        --	Increase angle 4)
(        --	decrease angle 4)
        1	Increase angle between P1P2 and P4P5
        2	decrease angle between P1P2 and P4P5        
        3	tuck leg 1 (P1P2)
        4	untuck leg 1 (P1P2)        
        5	tuck leg 2 (P4P5)
        6	untuck leg 2 (P4P5)        
        
        
    Reward:
        The cat is penalized if it doesn't fall with feet first
    Starting State:
        The cat starts upside down. A perturbation is added on the Y-angle.
    Episode Termination:
        any part of the cat reaches the floor (min of y-coordinates = 0)
        Solved Requirements
        Considered solved when the average reward is greater than or equal to -50 over 100 consecutive trials.
    """
    
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):
        #parameters
        #self.gravity = 9.8
        self.gravity = 7
    
        self.numberPoints = 5

        self.masses = np.array(np.zeros(self.numberPoints))
        self.lengths = np.array(np.zeros(self.numberPoints - 1))
        
        self.masses[ 0 ] = 1.0 # mass of the front paw
        self.masses[ 1 ] = 2.0 # mass of the head
        self.masses[ 2 ] = 1.0/2.0 # mass of the middle body
        self.masses[ 3 ] = 1.0/2.0 # mass of the back body
        self.masses[ 4 ] = 1.0 # mass of the back paw

        self.total_mass = sum(self.masses)


        self.lengthLegs = 50.0
        self.lengthBody = 100.0
        
        self.lengths[ 0 ] = self.lengthLegs # length of the front leg
        self.lengths[ 1 ] = self.lengthBody/2 # length of the first 1/2 of the body
        self.lengths[ 2 ] = self.lengthBody/2 # length of the second 1/2 of the body
        self.lengths[ 3 ] = self.lengthLegs # length of the back leg

        # seconds between state updates
        self.tau = 0.1

        # max angle
        self.theta_threshold_radians = math.pi

        #The min angle of the back (allAngles[2]) 
        self.minAngleBack = 5.0*2*math.pi / 360
        #self.minAngleBack = 90.0*2*math.pi / 360
        
        #Max torsion allowed (ie allAngles[4] can be at most this value).
        self.maxAngleTorsion = 45.0*2*math.pi / 360

        self.deltaTheta = 10 * 2 * math.pi / 360

        #initial y-value of the center of mass
        self.y_threshold = 500.0

        #legs tuck/untuck by this factor
        self.tuckFactor = 2.0

        self.extended = 2* self.lengthLegs + self.lengthBody
                
        low = np.array([
            -2*self.extended,
            -2*self.extended,
            -2*self.extended,
            -2*self.extended,
            -2*self.extended,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -2*self.extended,             
            -2*self.extended,
            -2*self.extended,
            -2*self.extended,
            -2*self.extended,
            -2*self.extended,
            -self.y_threshold * 2*self.gravity, 
            0.0, 0.0, 
            self.minAngleBack, 0.0,
            -self.theta_threshold_radians, 0.0,
            0.0, 0.0])

        high = np.array([
            2*self.extended,
            2*self.extended,
            2*self.extended,
            2*self.extended,
            2*self.extended,
            self.y_threshold * 2,
            self.y_threshold * 2,
            self.y_threshold * 2,
            self.y_threshold * 2,
            self.y_threshold * 2,            
            2*self.extended,
            2*self.extended,
            2*self.extended,
            2*self.extended,
            2*self.extended,
            self.y_threshold * 2,
            self.y_threshold * 2*self.gravity,
            self.theta_threshold_radians, self.theta_threshold_radians,
            self.theta_threshold_radians, self.theta_threshold_radians,            
            self.theta_threshold_radians, 2.0*self.theta_threshold_radians, 
            2.0*self.theta_threshold_radians, 2.0*self.theta_threshold_radians])

        self.action_space = spaces.Discrete(3)
        self.inner_action_space = spaces.Discrete(7)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None
        
        # to print out debugging messages
        self.debug_messages = False

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    ''' 
    Simpler step function with three actions only: 1 cycle one way, 1 cycle other way, or do nothing.
    '''
    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))

        if action == 1:
            #tuck front legs
            inner_action = 3
            self.inner_step(inner_action)

            #twist one way
            inner_action = 1
            self.inner_step(inner_action)

            #untuck front legs
            inner_action = 4
            self.inner_step(inner_action)

            #tuck back legs
            inner_action = 5
            self.inner_step(inner_action)

            #twist the other way
            inner_action = 2
            self.inner_step(inner_action)

            #untuck back legs
            inner_action = 6
            self.inner_step(inner_action)

        elif action == 2:
            #tuck front legs
            inner_action = 3
            self.inner_step(inner_action)

            #twist the other way
            inner_action = 2
            self.inner_step(inner_action)

            #untuck front legs
            inner_action = 4
            self.inner_step(inner_action)

            #tuck back legs
            inner_action = 5
            self.inner_step(inner_action)

            #twist one way
            inner_action = 1
            self.inner_step(inner_action)

            #untuck back legs
            inner_action = 6
            self.inner_step(inner_action)

        else: 
            # do nothing
            inner_action = 0
            self.inner_step(inner_action)
            
        done, penalty = self.cost()
        if not done:
            #when it's not done yet, try with a smaller penalty instead of zero
            #penalty = penalty / 10.0
            penalty = 0
        
        #try returning reward instead
        #reward = - penalty        
        return np.array(self.state), penalty, done, {}

    #the original step function. It allows for more actions
    def inner_step(self, action):
        assert self.inner_action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        
        pointsP = np.array((3, self.numberPoints))        
        allAngles = np.array(self.numberPoints - 1 + 1)
        
        pointsP, y, y_dot, allAngles, rotAngleX, rotAngleY, rotAngleZ = self.unpack_state()
        
        if (self.debug_messages):
            print("Before running step:")
            self.printVectors(pointsP[0,:], pointsP[1,:], pointsP[2,:])
            print("action = "+str(action))
        
        num_repeated_actions = 1
        
        # new: less actions:
        if (action == 1 
        and allAngles[4] + self.deltaTheta/2 <= self.maxAngleTorsion):
            num_repeated_actions = min(math.floor((self.maxAngleTorsion)/(self.deltaTheta/2)),
                                       math.floor((self.maxAngleTorsion - allAngles[4])/(self.deltaTheta/2)) )
            allAngles[4] += self.deltaTheta/2
        elif (action == 2 
        and allAngles[4] - self.deltaTheta/2 >= -self.maxAngleTorsion):
            num_repeated_actions = min(math.floor((self.maxAngleTorsion)/(self.deltaTheta/2)), 
                                       math.floor((allAngles[4] + self.maxAngleTorsion)/(self.deltaTheta/2)) )
            allAngles[4] -= self.deltaTheta/2
        elif (action == 3 and self.lengths[ 0 ] >= self.lengthLegs):
            self.lengths[ 0 ] = self.lengths[ 0 ] / self.tuckFactor
        elif (action == 4 and self.lengths[ 0 ] < self.lengthLegs):
            self.lengths[ 0 ] = self.lengthLegs
        elif (action == 5 and self.lengths[ 3 ] >= self.lengthLegs):
            self.lengths[ 3 ] = self.lengths[ 3 ] / self.tuckFactor
        elif (action == 6 and self.lengths[ 3 ] < self.lengthLegs):
            self.lengths[ 3 ] = self.lengthLegs

        ymin = np.amin(pointsP[1,:])
        it = 1
        while it <= num_repeated_actions and ymin > 0:
            #update the y-velocity and the y-value of the center of mass
            yacc  = -self.gravity
            y_old = y        
            y = y + self.tau * y_dot
            y_dot = y_dot + self.tau * yacc

            self.state = np.concatenate((pointsP[0,:], pointsP[1,:], pointsP[2,:], np.array([y, y_dot]), allAngles, np.array([rotAngleX, rotAngleY, rotAngleZ])))

            #update the vertices w.r.t the new angles while keeping zero angular momentum
            pointsP, rotAngleX, rotAngleY, rotAngleZ = self.verticesAndRotation3D()

            self.state = np.concatenate((pointsP[0,:], pointsP[1,:], pointsP[2,:], np.array([y, y_dot]), allAngles, np.array([rotAngleX, rotAngleY, rotAngleZ])))

            # if the cat hit the ground we stop, otherwise we repeat the action if 1 or 2
            ymin = np.amin(pointsP[1,:])
            if ymin <= 0:
                it = num_repeated_actions + 1
            else:    
                # prepare for the next iteration of the repeated action
                if action == 1:
                    allAngles[4] += self.deltaTheta/2
            
                if action == 2:
                    allAngles[4] -= self.deltaTheta/2
                it += 1
            self.render()
                        
        if (self.debug_messages):
            print("After running step:")
            self.printVectors(pointsP[0,:], pointsP[1,:], pointsP[2,:])
            print("old y = "+str(round(y_old,4)) + ", new y = " +str(round(y,4)))
            #input("After running step. Press any key.")

        done, penalty = self.cost()
        if not done:
            #when it's not done yet, try with a smaller penalty instead of zero
            penalty = penalty / 10.0
            #penalty = 0
                
        return np.array(self.state), penalty, done, {}

    #cost of the current state
    def cost(self):
        pointsP = np.array((3, self.numberPoints))        
        allAngles = np.array(self.numberPoints - 1 + 1)        
        pointsP, y, y_dot, allAngles, rotAngleX, rotAngleY, rotAngleZ = self.unpack_state()

        # y-position of the head
        y1 = pointsP[1,1]
        ymin = np.amin(pointsP[1,:])
        done =  ymin <= 0
        done = bool(done)
        
        penalty = 0.0

        #penalize when the head is below the y-center of mass
        penalty += y - y1
        
        return done, penalty

    def reset(self):

        #Start with legs untucked
        self.lengths[ 0 ] = self.lengthLegs
        self.lengths[ 3 ] = self.lengthLegs
        
        allAngles = np.array([
                  0.0, 
                  math.pi / 2 - self.minAngleBack/2,
                  self.minAngleBack,
                  math.pi / 2 - self.minAngleBack/2,
                  0])

        pointsP = np.zeros((3, self.numberPoints))
        
        # this is the cumulative angle w.r.t the negative x-axis
        cumAngle = 0  
        for i in range(1, self.numberPoints):
            cumAngle += allAngles[i - 1]
            pointsP[0, i] = pointsP[0, i - 1] + math.cos(math.pi - cumAngle)*self.lengths[i - 1]
            pointsP[1, i] = pointsP[1, i - 1] + math.sin(math.pi - cumAngle)*self.lengths[i - 1]
        
        #Center-of-Mass (CM) vector
        CM = np.array([0.0, self.y_threshold, 0.0])
        #translate the points to have center of mass equal to CM
        pointsP = self.translate_points_new_CM(pointsP, CM)

        #rotate the cat 90 degrees over the z-axis so it starts upside down
        rotAngleX = 0.0
        rotAngleY = 270.0*2*math.pi / 360 #0.0
        rotAngleZ = math.pi / 2.0 

        #Add a perturbation
        anglePerturbation = 10.0*2*math.pi / 360 #45.0*2*math.pi / 360
        angle1 = self.np_random.uniform(low = -anglePerturbation, high=anglePerturbation)
        rotAngleY += angle1

        #what if it starts closer to the final position?
        rotAngleY +=math.pi / 2.0
        
        if (self.debug_messages):                   
            print("After the perturbation: initial rotAngleY = " + str(round(rotAngleY/(2*math.pi)*360.0, 4)))
        
        rotAngleXYZ = (rotAngleX, rotAngleY, rotAngleZ)
        pointsP = self.rotate3D_v4(pointsP, rotAngleXYZ, CM)

        self.state = np.concatenate((pointsP[0, :], pointsP[1, :], pointsP[2, :], np.array([CM[1], 0.0]), allAngles, 
                    np.array([rotAngleX, rotAngleY, rotAngleZ])))
        
        self.steps_beyond_done = None
        return np.array(self.state)

    '''
    Norm of the angular momentum of M(theta)*pointsP, prev_pointsP, as a function of the rotation matrix M(theta). 
    This function should be zero so as to have zero angular momentum throughout
    '''
    def normAngMom(self, theta, pointsP, prev_pointsP, CM):
        rot2pointsP = self.rotate3D_v4(pointsP, theta, CM)
        angM = self.angularMomentum3D(prev_pointsP, rot2pointsP)
        return np.linalg.norm(angM)

    """
    This function computes the vertices from the internal angles. 
    The vertices are translated to keep the same center of mass,
    and they are rotated w.r.t the center of mass to keep zero angular momentum
    We use scipy minimize function to keep zero angular momentum
    """

    def verticesAndRotation3D(self):
        pointsP = np.zeros((3, self.numberPoints))                
        allAngles = np.array(np.zeros(self.numberPoints - 1 + 1))
        prev_pointsP = np.zeros((3, self.numberPoints))
        XCM = 0.0
        ZCM = 0.0
        prev_pointsP, YCM, y_dot, allAngles, rotAngleX, rotAngleY, rotAngleZ = self.unpack_state()
        
        CM = np.array([XCM, YCM, ZCM])
        
        # this is the cumulative angle w.r.t the negative x-axis
        cumAngle = 0
        for i in range(1, self.numberPoints):
            cumAngle += allAngles[i - 1]
            pointsP[0][i] = pointsP[0][i - 1] + math.cos(math.pi - cumAngle)*self.lengths[i - 1]
            pointsP[1][i] = pointsP[1][i - 1] + math.sin(math.pi - cumAngle)*self.lengths[i - 1]
                   
        #rotate the leg #1 i.e. rotate P0 around the axis P1P2 using allAngles[4] 
        axis = np.array(3)
        axis = pointsP[:, 2] - pointsP[:, 1]
        axis = self.unit_vector(axis)                
        pointsP[:, 0] = self.rotate3D(pointsP[:, 0], axis, allAngles[4], pointsP[:,1])
                        
        #rotate the leg #2, i.e. rotate P4 around the axis P2P3 using -allAngles[4]                
        axis = pointsP[:, 3] - pointsP[:, 2]
        axis = self.unit_vector(axis)
        pointsP[:, 4] = self.rotate3D(pointsP[:, 4], axis, -allAngles[4], pointsP[:,3])

        #translate the points to have center of mass equal to CM
        pointsP = self.translate_points_new_CM(pointsP, CM)
                    
        x0 = np.array([rotAngleX, rotAngleY, rotAngleZ])
        
        #scipy minimization magic here...
        res = minimize(self.normAngMom, x0, args= (pointsP, prev_pointsP, CM), method='nelder-mead', options={'xtol': 1e-8, 'disp': False})

        if (self.debug_messages):
            print("new rot angleX = " + str(round(res.x[0]/(2*math.pi)*360.0, 4))
            + ", new rot angleY = " + str(round(res.x[1]/(2*math.pi)*360.0, 4))
            + ", new rot angleZ = " + str(round(res.x[2]/(2*math.pi)*360.0, 4)))

        rot2pointsP = self.rotate3D_v4(pointsP, res.x, CM)
        return rot2pointsP, res.x[0], res.x[1], res.x[2] 

    #returns the rotation of the points (XpointsP, YpointsP, ZpointsP) using the rot matrix defined by tryThetaX, Y, Z around point CM
    def rotate3D_v4(self, pointsP, tryThetaXYZ, CM):
        rot_pointsP = np.zeros(pointsP.shape)        
        rot = rot3D()
        M = rot.rotationMatrix3D_v2(tryThetaXYZ)
        
        for i in range(pointsP.shape[1]):    
            rot_pointsP[:, i] = M @ pointsP[:, i] - M @ CM + CM

        return rot_pointsP

    #returns the rotation of the point pointP around an axis by an angle tryTheta around point CM
    def rotate3D(self, pointP, axis, tryTheta, CM):        
        rot_pointsP = np.zeros(pointP.size)

        rot = rot3D()
        rot.rot_axis = axis
        rot.rotAngle = tryTheta
        M = rot.rotationMatrix3D()
        rot_pointsP = M @ pointP - M @ CM + CM

        return rot_pointsP
        
    #approximation of the angular momentum for a common center of mass
    #the calculation is done after both sets of points have the same center of mass
    def angularMomentum3D(self, pointsPOld, pointsPNew):
        
        CM = np.zeros(3)
        
        #these points are the old points translated to have zero CM
        pointsP_Old0 = self.translate_points_new_CM(pointsPOld, CM)
        
        #these points are the new points translated to have zero CM
        pointsP_New0 = self.translate_points_new_CM(pointsPNew, CM)

        angularMomentum = np.zeros(3)
        for i in range(self.numberPoints):    
            r = pointsP_New0[:, i]
            #velocity without dividing by time (note: since the center of mass is the same, this is also the velocity relative to the center of mass)
            v = pointsP_New0[:, i] - pointsP_Old0[:, i]
            crossProd3D =self.masses[i]*np.cross(r, v)
            angularMomentum += crossProd3D
            
        return angularMomentum


    #returns the state as a collection of arrays instead of just one array
    def unpack_state(self):
        pointsP = np.zeros((3, self.numberPoints))
        allAngles = np.array(np.zeros(self.numberPoints - 1 + 1))
        pointsP[0,0], pointsP[0,1], pointsP[0,2], pointsP[0,3], pointsP[0,4], pointsP[1,0], pointsP[1,1], pointsP[1,2], pointsP[1,3], pointsP[1,4], pointsP[2,0], pointsP[2,1], pointsP[2,2], pointsP[2,3], pointsP[2,4], y, y_dot, allAngles[0], allAngles[1], allAngles[2], allAngles[3], allAngles[4], rotAngleX, rotAngleY, rotAngleZ = self.state
        return pointsP, y, y_dot, allAngles, rotAngleX, rotAngleY, rotAngleZ
    
    def unit_vector(self, vector):
        """ Returns the unit vector of the vector.  """
        return vector / np.linalg.norm(vector)

    '''
    Render displays two views of the cat, one in the XY-plane ("cat viewed from the side") and the other in the YZ-plane ("cat viewed from the front")
    '''
    def render(self, mode='human'):
        floory = 50 # floor
        screen_width = 600
        screen_height = int(self.y_threshold) + 2*floory
        polewidth = 10.0
        state = self.state
        x1, x2, x3, x4, x5, y1, y2, y3, y4, y5, z1, z2, z3, z4, z5, YCM, yd, _, _, _, _, _, rotAngleX, rotAngleY, rotAngleZ = state        
        
        from gym.envs.classic_control import rendering        
        if self.viewer is None:
            
            self.viewer = rendering.Viewer(screen_width, screen_height)
            self.body_trans = rendering.Transform(translation=(screen_width/3, floory))
            self.body_transzy = rendering.Transform(translation=(2*screen_width/3, floory))
            self.head_trans = rendering.Transform(translation=(screen_width/3, floory))
            self.head_transzy = rendering.Transform(translation=(2*screen_width/2, floory))
            
            # back leg
            deltax = polewidth/2*(y2-y1)/self.lengths[0]
            deltay = -polewidth/2*(x2-x1)/self.lengths[0]
            #deltaz = -polewidth/2*(z2-z1)/self.lengths[0]
            leg1 = rendering.FilledPolygon([(x1 - deltax/2,y1 - deltay/2), (x2 - deltax,y2 - deltay), (x2 + deltax,y2 + deltay), (x1 + deltax/2,y1 + deltay/2)])
            leg1zy = rendering.FilledPolygon([(z1 - deltax/2, y1 - deltay/2), (z2 - deltax, y2 - deltay), (z2 + deltax, y2 + deltay), (z1 + deltax/2, y1 + deltay/2)])
            
            # back of the body
            deltax = polewidth/2*(y3-y2)/self.lengths[1]
            deltay = -polewidth/2*(x3-x2)/self.lengths[1]
            frontBody = rendering.FilledPolygon([(x2 -deltax,y2 - deltay), (x3 - deltax,y3 - deltay), (x3 + deltax,y3 + deltay), (x2 + deltax,y2 + deltay)])
            frontBodyzy = rendering.FilledPolygon([(z2 -deltax,y2 - deltay), (z3 - deltax, y3 - deltay), (z3 + deltax, y3 + deltay), (z2 + deltax, y2 + deltay)])

            # front of the body
            deltax = polewidth/2*(y4-y3)/self.lengths[2]
            deltay = -polewidth/2*(x4-x3)/self.lengths[2]
            backBody = rendering.FilledPolygon([(x3 -deltax, y3 - deltay), (x4 - deltax, y4 - deltay), (x4 + deltax, y4 + deltay), (x3 + deltax, y3 + deltay)])
            backBodyzy = rendering.FilledPolygon([(z3 -deltax, y3 - deltay), (z4 - deltax, y4 - deltay), (z4 + deltax, y4 + deltay), (z3 + deltax, y3 + deltay)])

            # leg 2
            deltax = polewidth/2*(y5-y4)/self.lengths[3]
            deltay = -polewidth/2*(x5-x4)/self.lengths[3]
            leg2 = rendering.FilledPolygon([(x4 -deltax, y4 - deltay), (x5 -deltax/2, y5 - deltay/2), (x5 + deltax/2, y5 + deltay/2), (x4 + deltax, y4 + deltay)])
            leg2zy = rendering.FilledPolygon([(z4 -deltax, y4 - deltay), (z5 -deltax/2, y5 - deltay/2), (z5 + deltax/2, y5 + deltay/2), (z4 + deltax, y4 + deltay)])

            leg1.set_color(.8,.6,.4)
            leg1.add_attr(self.body_trans)
            self.viewer.add_geom(leg1)

            leg1zy.set_color(.8,.6,.4)
            leg1zy.add_attr(self.body_transzy)
            self.viewer.add_geom(leg1zy)

            frontBody.set_color(.8,.6,.4)
            frontBody.add_attr(self.body_trans)
            self.viewer.add_geom(frontBody)

            frontBodyzy.set_color(.8,.6,.4)
            frontBodyzy.add_attr(self.body_transzy)
            self.viewer.add_geom(frontBodyzy)
            
            backBody.set_color(.8,.6,.4)
            backBody.add_attr(self.body_trans)
            self.viewer.add_geom(backBody)

            backBodyzy.set_color(.8,.6,.4)
            backBodyzy.add_attr(self.body_transzy)
            self.viewer.add_geom(backBodyzy)

            leg2.set_color(.8,.6,.4)
            leg2.add_attr(self.body_trans)
            self.viewer.add_geom(leg2)

            leg2zy.set_color(.8,.6,.4)
            leg2zy.add_attr(self.body_transzy)
            self.viewer.add_geom(leg2zy)

            #cat head
            head = rendering.make_circle(polewidth)
            head.add_attr(self.body_trans)
            head.add_attr(self.head_trans)
            head.set_color(.8,.6,.4)
            self.viewer.add_geom(head)
            
            headzy = rendering.make_circle(polewidth)
            headzy.add_attr(self.head_transzy)
            headzy.set_color(.8,.6,.4)
            self.viewer.add_geom(headzy)

            #add horizon line corresponding to the floor
            self.track = rendering.Line((0,floory), (screen_width, floory))
            self.track.set_color(0,0,0)
            self.viewer.add_geom(self.track)
            
            self._cat3D_geom1 = leg1
            self._cat3D_geom1zy = leg1zy
            self._cat3D_geom2 = frontBody
            self._cat3D_geom2zy = frontBodyzy
            self._cat3D_geom3 = backBody
            self._cat3D_geom3zy = backBodyzy
            self._cat3D_geom4 = leg2
            self._cat3D_geom4zy = leg2zy
            self._cat3D_geom5 = head
            self._cat3D_geom6 = headzy
            
        if self.state is None: return None

        # leg 1
        leg1 = self._cat3D_geom1
        delta_vector = self.unit_vector(np.array([(y2 - y1), -(x2 - x1)]))
        deltax = polewidth/2*delta_vector[0]
        deltay = polewidth/2*delta_vector[1]        
        leg1.v = [(x1 - deltax/2,y1 - deltay/2), (x2 - deltax,y2 - deltay), (x2 + deltax,y2 + deltay), (x1 + deltax/2,y1 + deltay/2)]

        leg1zy = self._cat3D_geom1zy
        delta_vector = self.unit_vector(np.array([(y2 - y1), -(z2 - z1)]))
        deltax = polewidth/2*delta_vector[0]
        deltay = polewidth/2*delta_vector[1]
        leg1zy.v = [(z1 - deltax/2,y1 - deltay/2), (z2 - deltax,y2 - deltay), (z2 + deltax,y2 + deltay), (z1 + deltax/2,y1 + deltay/2)]
                
        # front of the body
        frontBody = self._cat3D_geom2
        delta_vector = self.unit_vector(np.array([(y3 - y2), -(x3 - x2)]))
        deltax = polewidth/2*delta_vector[0]
        deltay = polewidth/2*delta_vector[1]        
        frontBody.v = [(x2 - deltax,y2 - deltay), (x3 - deltax,y3 - deltay), (x3 + deltax,y3 + deltay), (x2 + deltax,y2 + deltay)]

        frontBodyzy = self._cat3D_geom2zy
        delta_vector = self.unit_vector(np.array([(y3 - y2), -(z3 - z2)]))
        deltax = polewidth/2*delta_vector[0]
        deltay = polewidth/2*delta_vector[1]        
        frontBodyzy.v = [(z2 - deltax, y2 - deltay), (z3 - deltax, y3 - deltay), (z3 + deltax, y3 + deltay), (z2 + deltax, y2 + deltay)]

        # back of the body        
        backBody = self._cat3D_geom3
        delta_vector = self.unit_vector(np.array([(y4 - y3), -(x4 - x3)]))
        deltax = polewidth/2*delta_vector[0]
        deltay = polewidth/2*delta_vector[1]                
        backBody.v = [(x3 -deltax, y3 - deltay), (x4 - deltax, y4 - deltay), (x4 + deltax, y4 + deltay), (x3 + deltax, y3 + deltay)]

        backBodyzy = self._cat3D_geom3zy
        delta_vector = self.unit_vector(np.array([(y4 - y3), -(z4 - z3)]))
        deltax = polewidth/2*delta_vector[0]
        deltay = polewidth/2*delta_vector[1]
        backBodyzy.v = [(z3 -deltax, y3 - deltay), (z4 - deltax, y4 - deltay), (z4 + deltax, y4 + deltay), (z3 + deltax, y3 + deltay)]

        # leg 2
        leg2 = self._cat3D_geom4
        delta_vector = self.unit_vector(np.array([(y5 - y4), -(x5 - x4)]))
        deltax = polewidth/2*delta_vector[0]
        deltay = polewidth/2*delta_vector[1]        
        leg2.v = [(x4 -deltax, y4 - deltay), (x5 -deltax/2, y5 - deltay/2), (x5 + deltax/2, y5 + deltay/2), (x4 + deltax, y4 + deltay) ]

        leg2zy = self._cat3D_geom4zy
        delta_vector = self.unit_vector(np.array([(y5 - y4), -(z5 - z4)]))
        deltax = polewidth/2*delta_vector[0]
        deltay = polewidth/2*delta_vector[1]                
        leg2zy.v = [(z4 -deltax, y4 - deltay), (z5 -deltax/2, y5 - deltay/2), (z5 + deltax/2, y5 + deltay/2), (z4 + deltax, y4 + deltay) ]
        
        # head
        head = self._cat3D_geom5
        self.head_trans.set_translation(x2, y2)
        
        headzy = self._cat3D_geom6
        self.head_transzy.set_translation(2*screen_width/3 + z2, floory + y2)
        
        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    #translates the points P so they have a new center of mass CM = (XCM,YCM, ZCM)
    def translate_points_new_CM(self, P, CM):
        pointsP_new = np.zeros((3, self.numberPoints))
        
        #compute the current center of mass (CM) of P
        current_CM = np.zeros(3)
        for i in range(self.numberPoints):
            current_CM += P[:, i] *self.masses[i] / self.total_mass

        #translate the points by (CM - modifCM) so that the translated points have center of mass equal to CM
        for i in range(self.numberPoints):
            pointsP_new[ :, i ] = P[ :, i ] + (CM - current_CM)

        return pointsP_new
        
    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    # ********************* Debugging functions *************
    #prints the (x, y, z)-coordinates 
    def printVectors(self, Px, Py, Pz):        
        for i in range(self.numberPoints):
            print("i ="+ str(i) +", (" + str(round(Px[i], 2)) + ", " + str(round(Py[i], 2)) + ", " + str(round(Pz[i], 2)) + ")")
        
    def printAngles(self, name, v):
        for i in range(len(v)):
            print(name +"["+ str(i) + "] = "+ str(round(360.0*v[i]/(2*math.pi), 4)))

class rot3D():
    '''
    Class that stores a rotation in 3D
    '''
    def __init__(self):
        #parameters
        self.rotAngle = 0.0
        self.rot_axis = np.array([1, 1, 1])

    # returns the 3D-rotation matrix defined by the axis self.rot_axis and the angle self.rotAngle
    # Taken from https://en.wikipedia.org/wiki/Rotation_matrix#In_three_dimensions
    def rotationMatrix3D(self):
        M = np.zeros((3, 3))
        
        sintheta = math.sin(self.rotAngle)
        costheta = math.cos(self.rotAngle)
        M[0, 0] = costheta + self.rot_axis[0]*self.rot_axis[0]*(1 - costheta)
        M[0, 1] = self.rot_axis[0]*self.rot_axis[1]*(1 - costheta) - self.rot_axis[2]*sintheta
        M[0, 2] = self.rot_axis[0]*self.rot_axis[2]*(1 - costheta) + self.rot_axis[1]*sintheta
        
        M[1, 0] = self.rot_axis[1]*self.rot_axis[0]*(1 - costheta) + self.rot_axis[2]*sintheta
        M[1, 1] = costheta + self.rot_axis[1]*self.rot_axis[1]*(1 - costheta)
        M[1, 2] = self.rot_axis[1]*self.rot_axis[2]*(1 - costheta) - self.rot_axis[0]*sintheta
        
        M[2, 0] = self.rot_axis[0]*self.rot_axis[2]*(1 - costheta) - self.rot_axis[1]*sintheta
        M[2, 1] = self.rot_axis[1]*self.rot_axis[2]*(1 - costheta) + self.rot_axis[0]*sintheta
        M[2, 2] = costheta + self.rot_axis[2]*self.rot_axis[2]*(1 - costheta)
        return M           

    # returns the 3D-rotation matrix from 3 rotating angles angleXYZ
    # Taken from https://en.wikipedia.org/wiki/Rotation_matrix#In_three_dimensions
        
    def rotationMatrix3D_v2(self, angleXYZ):
        M = np.zeros((3, 3))
        M2 = np.zeros((3, 3))
        M2[0,0] = 1.0
        M2[1,1] = 1.0
        M2[2,2] = 1.0
        
        sinthetaX = math.sin(angleXYZ[0])
        costhetaX = math.cos(angleXYZ[0])
        
        M[0, 0] = 1.0
        M[0, 1] = 0.0        
        M[0, 2] = 0.0
        
        M[1, 0] = 0.0
        M[1, 1] = costhetaX
        M[1, 2] = -sinthetaX
        
        M[2, 0] = 0.0
        M[2, 1] = sinthetaX
        M[2, 2] = costhetaX

        M2 = M @ M2

        sinthetaY = math.sin(angleXYZ[1])
        costhetaY = math.cos(angleXYZ[1])
        
        M[0, 0] = costhetaY
        M[0, 1] = 0.0        
        M[0, 2] = sinthetaY
        
        M[1, 0] = 0.0
        M[1, 1] = 1.0
        M[1, 2] = 0
        
        M[2, 0] = -sinthetaY
        M[2, 1] = 0.0
        M[2, 2] = costhetaY        

        M2 = M @ M2
        
        sinthetaZ = math.sin(angleXYZ[2])
        costhetaZ = math.cos(angleXYZ[2])
        
        M[0, 0] = costhetaZ
        M[0, 1] = -sinthetaZ        
        M[0, 2] = 0.0
        
        M[1, 0] = sinthetaZ
        M[1, 1] = costhetaZ
        M[1, 2] = 0.0
        
        M[2, 0] = 0.0
        M[2, 1] = 0.0
        M[2, 2] = 1.0        

        M2 = M @ M2
        return M2         