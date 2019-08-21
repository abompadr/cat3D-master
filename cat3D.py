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

class cat3D(gym.Env):
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
        any part of the cat reaches the floor (y = 0)
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
        
        self.masses[ 0 ] = 1.0
        self.masses[ 1 ] = 1.0/2.0
        self.masses[ 2 ] = 1.0/2.0
        self.masses[ 3 ] = 1.0/2.0
        self.masses[ 4 ] = 1.0

        self.lengthLegs = 50.0
        self.lengthBody = 100.0
        
        self.lengths[ 0 ] = self.lengthLegs
        self.lengths[ 1 ] = self.lengthBody/2
        self.lengths[ 2 ] = self.lengthBody/2
        self.lengths[ 3 ] = self.lengthLegs

        # seconds between state updates
        self.tau = 0.1

        # max angle
        self.theta_threshold_radians = math.pi

        #The min angle of the back (angle[2]) 
        self.minAngleBack = 5.0*2*math.pi / 360

        #Max torsion allowed (ie 2*allAngles[4] can be at most this value).
        self.maxAngleTorsion = 90.0*2*math.pi / 360

        self.deltaTheta = 10 * 2 * math.pi / 360

        #initial y-value of the center of mass
        self.y_threshold = 500.0

        #legs tuck/untuck by this factor
        self.tuckFactor = 2.0

        self.extended = 2* self.lengthLegs + self.lengthBody
        
        self.total_mass = 0.0
        i = 0
        while (i < self.numberPoints):
            self.total_mass += self.masses[ i ]
            i+=1
        
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
            0.0, 0.0,
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
            2.0*self.theta_threshold_radians, 2.0*self.theta_threshold_radians, 
            2.0*self.theta_threshold_radians, 2.0*self.theta_threshold_radians])

        self.action_space = spaces.Discrete(7)
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

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        
        pointsP = np.array((3, self.numberPoints))        
        allAngles = np.array(self.numberPoints - 1 + 1)
        
        pointsP, y, y_dot, allAngles, rotAngleX, rotAngleY, rotAngleZ = self.unpack_state()
        
        if (self.debug_messages):
            print("Before running step:")
            self.printVectors(pointsP[0,:], pointsP[1,:], pointsP[2,:])
            print("action = "+str(action))
        
        # new: less actions:
        if (action == 1 
        and allAngles[4] + self.deltaTheta/2 <= self.theta_threshold_radians
        and 2*allAngles[4] + self.deltaTheta <= self.maxAngleTorsion):
            allAngles[4] += self.deltaTheta/2
        elif (action == 2 
        and allAngles[4] - self.deltaTheta/2 >= 0):
            allAngles[4] -= self.deltaTheta/2
        elif (action == 3 and self.lengths[ 0 ] >= self.lengthLegs):
            self.lengths[ 0 ] = self.lengths[ 0 ] / self.tuckFactor
        elif (action == 4 and self.lengths[ 0 ] < self.lengthLegs):
            self.lengths[ 0 ] = self.lengthLegs
        elif (action == 5 and self.lengths[ 3 ] >= self.lengthLegs):
            self.lengths[ 3 ] = self.lengths[ 3 ] / self.tuckFactor
        elif (action == 6 and self.lengths[ 3 ] < self.lengthLegs):
            self.lengths[ 3 ] = self.lengthLegs

        #update the y-velocity and the y-value of the center of mass
        yacc  = -self.gravity
        y_old = y        
        y  = y + self.tau * y_dot
        y_dot = y_dot + self.tau * yacc

        self.state = np.concatenate((pointsP[0,:], pointsP[1,:], pointsP[2,:], np.array([y, y_dot]), allAngles, np.array([rotAngleX, rotAngleY, rotAngleZ])))

        #update the vertices w.r.t the new angles
        pointsP, rotAngleX, rotAngleY, rotAngleZ = self.verticesAndRotation3D()

        self.state = np.concatenate((pointsP[0,:], pointsP[1,:], pointsP[2,:], np.array([y, y_dot]), allAngles, np.array([rotAngleX, rotAngleY, rotAngleZ])))

        if (self.debug_messages):
            print("After running step:")
            self.printVectors(pointsP[0,:], pointsP[1,:], pointsP[2,:])
            print("old y = "+str(round(y_old,4)) + ", new y = " +str(round(y,4)))
            #input("After running step. Press any key.")

        done, penalty = self.cost()
        if not done:
            penalty = 0
                
        return np.array(self.state), -penalty, done, {}

    #cost of the current state
    def cost(self):
        pointsP = np.array((3, self.numberPoints))        
        allAngles = np.array(self.numberPoints - 1 + 1)        
        pointsP, y, y_dot, allAngles, rotAngleX, rotAngleY, rotAngleZ = self.unpack_state()

        x1 = pointsP[0,0]
        y1 = pointsP[1,0]
        x2 = pointsP[0,1]
        y2 = pointsP[1,1]
        x3 = pointsP[0,2]
        y3 = pointsP[1,2]
        x4 = pointsP[0,3]
        y4 = pointsP[1,3]
        x5 = pointsP[0,4]
        y5 = pointsP[1,4]

        ymin = np.amin(pointsP[1,:])
        done =  ymin <= 0
        done = bool(done)
        
        penalty = 0.0
        #penalize the delta between the y values of the paws
        r = y1 - y5
        penalty += abs(r)
        #penalize the delta between the y values of the body
        r = y2 - y4
        penalty += abs(r)

        #penalize if the paws are not the points that touch the ground
        if (y1 >= y2 or y1 >= y3 or y1 >= y4):
            penalty += 100
        if (y5 >= y2 or y5 >= y3 or y5 >= y4):
            penalty += 100

        #penalize if the center of mass is not within the legs
        if (x1*x5 > 0):
            penalty += 100

        #penalize if the legs are not extended
        if (self.lengths[ 0 ] < self.lengthLegs):
            penalty += 50

        if (self.lengths[ 3 ] < self.lengthLegs):
            penalty += 50

        #penalize if z-coordinate is different from zero
        i = 0
        while (i < self.numberPoints):
            penalty += abs(pointsP[2,i])
            i += 1

        return done, penalty

    def reset(self):

        self.lengths[ 0 ] = self.lengthLegs
        self.lengths[ 3 ] = self.lengthLegs
        
        allAngles = np.array([
                  0.0, 
                  math.pi / 2 - self.minAngleBack/2,
                  self.minAngleBack,
                  math.pi / 2 - self.minAngleBack/2,
                  0])

        pointsP = np.zeros((3, self.numberPoints))
        
        i = 1
        # this is the angle w.r.t the negative x-axis
        cumAngle = 0  
        while (i < self.numberPoints):
            cumAngle += allAngles[i - 1]
            pointsP[0, i] = pointsP[0, i - 1] + math.cos(math.pi - cumAngle)*self.lengths[i - 1]
            pointsP[1, i] = pointsP[1, i - 1] + math.sin(math.pi - cumAngle)*self.lengths[i - 1]
            i += 1
        
        #Center-of-Mass (CM) vector
        CM = np.array([0.0, self.y_threshold, 0.0])
        #translate the points to have center of mass equal to CM
        pointsP = self.translate_points_new_CM(pointsP, CM)

        #rotate the cat 90 degrees over the z-axis so it starts upside down
        rotAngleX = 0.0
        rotAngleY = 0.0
        rotAngleZ = math.pi / 2.0 

        #Add a perturbation
        anglePerturbation = 30.0*2*math.pi / 360 #180.0*2*math.pi / 360
        angle1 = self.np_random.uniform(low = -anglePerturbation, high=anglePerturbation)
        rotAngleY += angle1
                           
        print("reset: rotAngleY = " + str(round(rotAngleY/(2*math.pi)*360.0, 4)))
        pointsP = self.rotate3D_v4(pointsP, rotAngleX, rotAngleY, rotAngleZ, CM)

        self.state = np.concatenate((pointsP[0, :], pointsP[1, :], pointsP[2, :], np.array([CM[1], 0.0]), allAngles, 
                    np.array([rotAngleX, rotAngleY, rotAngleZ])))
        
        self.steps_beyond_done = None
        return np.array(self.state)

    '''
    Norm of the angular momentum of M(x)*pointsP, prev_pointsP, as a function of the rotation matrix M(x). This is the function to keep zero
    '''
    def normAngMom(self, x, pointsP, prev_pointsP, CM):
        tryThetaX = x[0]
        tryThetaY = x[1] 
        tryThetaZ = x[2]
                
        rot2pointsP = self.rotate3D_v4(pointsP, tryThetaX, tryThetaY, tryThetaZ, CM)
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
        
        i = 1
        # this is the angle w.r.t the negative x-axis
        cumAngle = 0  
        while (i < self.numberPoints):
            cumAngle += allAngles[i - 1]
            pointsP[0][i] = pointsP[0][i - 1] + math.cos(math.pi - cumAngle)*self.lengths[i - 1]
            pointsP[1][i] = pointsP[1][i - 1] + math.sin(math.pi - cumAngle)*self.lengths[i - 1]
            i += 1
                   
        #rotate the leg #1 i.e. rotate P0 around the axis P1P2                
        axis = np.array(3)
        axis = pointsP[:, 2] - pointsP[:, 1]
        axis = self.unit_vector(axis)                
        pointsP[:, 0] = self.rotate3D(pointsP[:, 0], axis, allAngles[4], pointsP[:,1])
                        
        #rotate the leg #2, i.e. rotate P4 around the axis P2P3                
        axis = pointsP[:, 3] - pointsP[:, 2]
        axis = self.unit_vector(axis)
        pointsP[:, 4] = self.rotate3D(pointsP[:, 4], axis, -allAngles[4], pointsP[:,3])

        #translate the points to have center of mass equal to CM
        pointsP = self.translate_points_new_CM(pointsP, CM)
                    
        x0 = np.array([rotAngleX, rotAngleY, rotAngleZ])
        #scipy minimization magic here...
        res = minimize(self.normAngMom, x0, args= (pointsP, prev_pointsP, CM), method='nelder-mead', options={'xtol': 1e-8, 'disp': False})

        print("new rot angleX = " + str(round(res.x[0]/(2*math.pi)*360.0, 4))
        + ", new rot angleY = " + str(round(res.x[1]/(2*math.pi)*360.0, 4))
        +", new rot angleZ = " + str(round(res.x[2]/(2*math.pi)*360.0, 4)))

        rot2pointsP = self.rotate3D_v4(pointsP, res.x[0], res.x[1], res.x[2], CM)
        return rot2pointsP, res.x[0], res.x[1], res.x[2] 

    #returns the rotation of the points (XpointsP, YpointsP, ZpointsP) using the rot matrix defined by tryThetaX, Y, Z around point CM
    def rotate3D_v4(self, pointsP, tryThetaX, tryThetaY, tryThetaZ, CM):
        rot_pointsP = np.zeros(pointsP.shape)
        rot = rot3D()
        M = rot.rotationMatrix3D_v2(tryThetaX, tryThetaY, tryThetaZ)
        
        i = 0
        while (i < pointsP.shape[1]):
            rot_pointsP[:, i] = M @ pointsP[:, i] - M @ CM + CM
            i += 1

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
        
    #approximation of the angular momentum for constant center of mass
    #the calculation is done after both sets of points have the same center of mass
    def angularMomentum3D(self, pointsPOld, pointsPNew):
        
        CM = np.array([0.0, 0.0, 0.0])
        
        #these points are the old points translated to have zero CM
        pointsP_Old0 = self.translate_points_new_CM(pointsPOld, CM)
        
        #these points are the new points translated to have zero CM
        pointsP_New0 = self.translate_points_new_CM(pointsPNew, CM)

        i = 0        
        angularMomentum = 0
        r = np.zeros(3)
        v = np.zeros(3)
        while (i < self.numberPoints):            
            r = pointsP_New0[:, i]
            #velocity (since the center of mass is the same, this is also the velocity relative to the center of mass)
            v = pointsP_New0[:, i] - pointsP_Old0[:, i]
            crossProd3D =self.masses[i]*np.cross(r, v)
            angularMomentum += crossProd3D
            i += 1

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
            self.poletrans = rendering.Transform(translation=(screen_width/2, floory))
            self.poletranszy = rendering.Transform(translation=(screen_width/2 + 200, floory))
            self.headtrans = rendering.Transform(translation=(screen_width/2, floory))
            self.headtranszy = rendering.Transform(translation=(screen_width/2 + 200, floory))
            
            # back leg
            deltax = polewidth/2*(y2-y1)/self.lengths[0]
            deltay = -polewidth/2*(x2-x1)/self.lengths[0]
            #deltaz = -polewidth/2*(z2-z1)/self.lengths[0]
            leg1 = rendering.FilledPolygon([(x1 - deltax/2,y1 - deltay/2), (x2 - deltax,y2 - deltay), (x2 + deltax,y2 + deltay), (x1 + deltax/2,y1 + deltay/2)])
            leg1zy = rendering.FilledPolygon([(z1 - deltax/2, y1 - deltay/2), (z2 - deltax, y2 - deltay), (z2 + deltax, y2 + deltay), (z1 + deltax/2, y1 + deltay/2)])
            
            # back of the body
            deltax = polewidth/2*(y3-y2)/self.lengths[1]
            deltay = -polewidth/2*(x3-x2)/self.lengths[1]
            backBody = rendering.FilledPolygon([(x2 -deltax,y2 - deltay), (x3 - deltax,y3 - deltay), (x3 + deltax,y3 + deltay), (x2 + deltax,y2 + deltay)])
            backBodyzy = rendering.FilledPolygon([(z2 -deltax,y2 - deltay), (z3 - deltax, y3 - deltay), (z3 + deltax, y3 + deltay), (z2 + deltax, y2 + deltay)])

            # front of the body
            deltax = polewidth/2*(y4-y3)/self.lengths[2]
            deltay = -polewidth/2*(x4-x3)/self.lengths[2]
            frontBody = rendering.FilledPolygon([(x3 -deltax, y3 - deltay), (x4 - deltax, y4 - deltay), (x4 + deltax, y4 + deltay), (x3 + deltax, y3 + deltay)])

            frontBodyzy = rendering.FilledPolygon([(z3 -deltax, y3 - deltay), (z4 - deltax, y4 - deltay), (z4 + deltax, y4 + deltay), (z3 + deltax, y3 + deltay)])

            # leg 2
            deltax = polewidth/2*(y5-y4)/self.lengths[3]
            deltay = -polewidth/2*(x5-x4)/self.lengths[3]
            leg2 = rendering.FilledPolygon([(x4 -deltax, y4 - deltay), (x5 -deltax/2, y5 - deltay/2), (x5 + deltax/2, y5 + deltay/2), (x4 + deltax, y4 + deltay)])
            leg2zy = rendering.FilledPolygon([(z4 -deltax, y4 - deltay), (z5 -deltax/2, y5 - deltay/2), (z5 + deltax/2, y5 + deltay/2), (z4 + deltax, y4 + deltay)])

            leg1.set_color(.8,.6,.4)
            leg1.add_attr(self.poletrans)
            self.viewer.add_geom(leg1)

            leg1zy.set_color(.8,.6,.4)
            leg1zy.add_attr(self.poletranszy)
            self.viewer.add_geom(leg1zy)
            
            backBody.set_color(.8,.6,.4)
            backBody.add_attr(self.poletrans)
            self.viewer.add_geom(backBody)

            backBodyzy.set_color(.8,.6,.4)
            backBodyzy.add_attr(self.poletranszy)
            self.viewer.add_geom(backBodyzy)

            frontBody.set_color(.8,.6,.4)
            frontBody.add_attr(self.poletrans)
            self.viewer.add_geom(frontBody)

            frontBodyzy.set_color(.8,.6,.4)
            frontBodyzy.add_attr(self.poletranszy)
            self.viewer.add_geom(frontBodyzy)

            leg2.set_color(.8,.6,.4)
            leg2.add_attr(self.poletrans)
            self.viewer.add_geom(leg2)

            leg2zy.set_color(.8,.6,.4)
            leg2zy.add_attr(self.poletranszy)
            self.viewer.add_geom(leg2zy)

            #cat head
            head = rendering.make_circle(polewidth)
            head.add_attr(self.poletrans)
            head.add_attr(self.headtrans)
            head.set_color(.8,.6,.4)
            self.viewer.add_geom(head)
            
            headzy = rendering.make_circle(polewidth)
            headzy.add_attr(self.headtranszy)
            headzy.set_color(.8,.6,.4)
            self.viewer.add_geom(headzy)

            #add horizon line corresponding to the floor
            self.track = rendering.Line((0,floory), (screen_width, floory))
            self.track.set_color(0,0,0)
            self.viewer.add_geom(self.track)
            
            self._cat3D_geom1 = leg1
            self._cat3D_geom1zy = leg1zy
            self._cat3D_geom2 = backBody
            self._cat3D_geom2zy = backBodyzy
            self._cat3D_geom3 = frontBody
            self._cat3D_geom3zy = frontBodyzy
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
        
        # back of the body
        backBody = self._cat3D_geom2
        delta_vector = self.unit_vector(np.array([(y3 - y2), -(x3 - x2)]))
        deltax = polewidth/2*delta_vector[0]
        deltay = polewidth/2*delta_vector[1]        
        backBody.v = [(x2 - deltax,y2 - deltay), (x3 - deltax,y3 - deltay), (x3 + deltax,y3 + deltay), (x2 + deltax,y2 + deltay)]

        backBodyzy = self._cat3D_geom2zy
        delta_vector = self.unit_vector(np.array([(y3 - y2), -(z3 - z2)]))
        deltax = polewidth/2*delta_vector[0]
        deltay = polewidth/2*delta_vector[1]        
        backBodyzy.v = [(z2 - deltax, y2 - deltay), (z3 - deltax, y3 - deltay), (z3 + deltax, y3 + deltay), (z2 + deltax, y2 + deltay)]

        # front of the body
        frontBody = self._cat3D_geom3
        delta_vector = self.unit_vector(np.array([(y4 - y3), -(x4 - x3)]))
        deltax = polewidth/2*delta_vector[0]
        deltay = polewidth/2*delta_vector[1]                
        frontBody.v = [(x3 -deltax, y3 - deltay), (x4 - deltax, y4 - deltay), (x4 + deltax, y4 + deltay), (x3 + deltax, y3 + deltay)]

        frontBodyzy = self._cat3D_geom3zy
        delta_vector = self.unit_vector(np.array([(y4 - y3), -(z4 - z3)]))
        deltax = polewidth/2*delta_vector[0]
        deltay = polewidth/2*delta_vector[1]
        frontBodyzy.v = [(z3 -deltax, y3 - deltay), (z4 - deltax, y4 - deltay), (z4 + deltax, y4 + deltay), (z3 + deltax, y3 + deltay)]

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
        
        #head
        head = self._cat3D_geom5
        self.headtrans.set_translation(x4, y4)
        
        headzy = self._cat3D_geom6
        self.headtranszy.set_translation(screen_width/2 + 200 + z4, floory + y4)
        
        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    #translates the points P so they have a new center of mass CM = (XCM,YCM, ZCM)
    def translate_points_new_CM(self, P, CM):
        pointsP_new = np.zeros((3, self.numberPoints))
        current_CM = np.zeros(3)
        
        #compute the current center of mass (CM) of P
        i = 0
        while (i < self.numberPoints):
            current_CM += P[:, i] *self.masses[i] / self.total_mass
            i += 1

        #translate the points by (CM - modifCM) so that the translated points have center of mass equal to CM
        i = 0
        while (i < self.numberPoints):
            pointsP_new[ :, i ] = P[ :, i ] + (CM - current_CM)
            i += 1

        return pointsP_new
        
    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    # ********************* Debugging functions *************
    #prints the (x, y, z)-coordinates 
    def printVectors(self, Px, Py, Pz):
        i = 0        
        while (i < self.numberPoints):
            print("i ="+ str(i) +", (" + str(round(Px[i], 2)) + ", " + str(round(Py[i], 2)) + ", " + str(round(Pz[i], 2)) + ")")
            i += 1
        
    def printAngles(self, name, v):
        i = 0        
        while (i < len(v)):
            print(name +"["+ str(i) + "] = "+ str(round(360.0*v[i]/(2*math.pi), 4)))
            i += 1


class rot3D():
    '''
    Class that stores a rotation in 3D
    '''
    def __init__(self):
        #parameters
        self.rotAngle = 0.0
        self.rot_axis = np.array([1, 1, 1])

    # returns the 3D-rotation matrix from axis and angle
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

    # returns the 3D-rotation matrix from 3 rotating angles
    # Taken from https://en.wikipedia.org/wiki/Rotation_matrix#In_three_dimensions
        
    def rotationMatrix3D_v2(self, angleX, angleY, angleZ):
        M = np.zeros((3, 3))
        M2 = np.zeros((3, 3))
        M2[0,0] = 1.0
        M2[1,1] = 1.0
        M2[2,2] = 1.0
        
        sinthetaX = math.sin(angleX)
        costhetaX = math.cos(angleX)
        
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

        sinthetaY = math.sin(angleY)
        costhetaY = math.cos(angleY)
        
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
        
        sinthetaZ = math.sin(angleZ)
        costhetaZ = math.cos(angleZ)
        
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