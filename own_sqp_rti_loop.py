#
# Copyright 2019 Gianluca Frison, Dimitris Kouzoupis, Robin Verschueren,
# Andrea Zanelli, Niels van Duijkeren, Jonathan Frey, Tommaso Sartor,
# Branimir Novoselnik, Rien Quirynen, Rezart Qelibari, Dang Doan,
# Jonas Koenemann, Yutao Chen, Tobias Schöls, Jonas Schlagenhauf, Moritz Diehl
#
# This file is part of acados.
#
# The 2-Clause BSD License
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.;
#



from statistics import mode
import sys
from tracemalloc import take_snapshot

from cvxpy import hstack, vstack
# sys.path.insert(0, 'common')

import time
from acados_template import AcadosOcp, AcadosOcpSolver
# from unicycle_model import export_unicycle_ode_model_with_LocalConstraints
import numpy as np
import scipy.linalg
import math
import matplotlib
import matplotlib.pyplot as plt

from acados_template import AcadosModel
from casadi import SX, vertcat, sin, cos, Function

class Pose2D:
    x = np.array([0.]) 
    y = np.array([0.]) 
    theta = np.array([0.]) 

def export_unicycle_ode_model_with_LocalConstraints():

    model_name = 'unicycle_ode'

    # set up states & controls
    x_pos       = SX.sym('x_pos')
    y_pos       = SX.sym('y_pos')
    theta_orient   = SX.sym('theta_orient')
    
    x = vertcat(x_pos, y_pos, theta_orient)

    v      = SX.sym('v')
    omega  = SX.sym('omega')
    
    u = vertcat(v , omega)

    # xdot
    x_pos_dot      = SX.sym('x_pos_dot')
    y_pos_dot      = SX.sym('y_pos_dot')
    theta_orient_dot   = SX.sym('theta_orient_dot')
    v_dot      = SX.sym('v_dot')
    omega_dot  = SX.sym('omega_dot')

    xdot = vertcat(x_pos_dot, y_pos_dot, theta_orient_dot)

    # algebraic variables


    # parameters
    x_ref = SX.sym('x_ref') 
    y_ref = SX.sym('y_ref')
    p = vertcat(x_ref, y_ref)
    
    # dynamics
    f_expl = vertcat(v * cos(theta_orient), v * sin(theta_orient), omega)
    f_impl = xdot - f_expl

    #nonlinear constraint
    con_h_expr = (x_pos - x_ref) ** 2 + (y_pos - y_ref) ** 2

    model = AcadosModel()

    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.con_h_expr = con_h_expr  
    model.con_h_expr_e = con_h_expr  
    model.x = x
    model.xdot = xdot
    model.u = u
    # model.z = z
    model.p = p  
    model.name = model_name

    return model



def get_reference_circle(t,Ts,N):
    ref_rad = 0.5
    ref_T = 10
    t_vec = t + np.linspace(0,N * Ts, N + 1)
    x_pos_ref = ref_rad * np.cos(2 * math.pi / ref_T * t_vec)
    y_pos_ref = ref_rad * np.sin(2 * math.pi / ref_T * t_vec)
    v_ref = 2 * math.pi * ref_rad / ref_T * np.ones(N)
    omega_ref = 2 * math.pi / ref_T * np.ones(N)
    theta_ref = math.pi / 2 + t_vec * 2 * math.pi / ref_T 
    state_ref = np.vstack((x_pos_ref.reshape(1,N + 1), y_pos_ref.reshape(1,N + 1), theta_ref.reshape(1,N + 1)))
    input_ref = np.vstack((v_ref.reshape(1,N), omega_ref.reshape(1,N)))
    return state_ref, input_ref

# def get_reference_pointsintersection(p0,t,Ts,N):
#     p_seq = np.zeros((2,4))
#     p_seq[0,0] = np.array([0.5])
#     p_seq[1,0] = np.array([0.5])
#     p_seq[0,1] = np.array([-0.5])
#     p_seq[1,1] = np.array([0.5])
#     p_seq[0,2] = np.array([-0.5])
#     p_seq[1,2] = np.array([-0.5])
#     p_seq[0,3] = np.array([0.5])
#     p_seq[1,3] = np.array([-0.5])

#     speed = 0.2

#     t_vec = t + np.lindef get_reference_pointsintersection(p0,t,Ts,N):
#     p_seq = np.zeros((2,4))
#     p_seq[0,0] = np.array([0.5])
#     p_seq[1,0] = np.array([0.5])
#     p_seq[0,1] = np.array([-0.5])
#     p_seq[1,1] = np.array([0.5])
#     p_seq[0,2] = np.array([-0.5])
#     p_seq[1,2] = np.array([-0.5])
#     p_seq[0,3] = np.array([0.5])
#     p_seq[1,3] = np.array([-0.5])

#     speed = 0.2

#     t_vec = t + np.linspace(0,N * Ts, N + 1)

#     x_pos_ref = 0.5 * np.ones(N + 1)
#     y_pos_ref = np.zeros(N  + 1)
#     # theta_ref = (np.arctan2(np.sin(math.pi / 2 - p0[2]),np.cos(math.pi / 2 - p0[2])) + p0[2]) * np.ones(N + 1)
#     theta_ref = math.pi / 2 * np.ones(N + 1)
#     v_ref = np.zeros(N)
#     omega_ref = np.zeros(N)
    
#     if (t_vec[0] < 0.5 / speed):
#         v_ref[0] = speed
#         omega_ref[0]  = 0.0
#         # theta_ref[0]  = np.arctan2(np.sin(math.pi / 2 - p0[2]),np.cos(math.pi / 2 - p0[2])) + p0[2]
#         theta_ref[0]  = math.pi / 2
#         x_pos_ref[0]  = 0.5
#         y_pos_ref[0]  = t_vec[0] * speed
#     if ((t_vec[0] >= 0.5 / speed) and (t_vec[0] < 1.5 / speed)):
#         v_ref[0]  = speed
#         omega_ref[0]  = 0.0
#         # theta_ref[0]  = np.arctan2(np.sin(math.pi - p0[2]),np.cos(math.pi - p0[2])) + p0[2]
#         theta_ref[0]  = math.pi
#         x_pos_ref[0]  = 0.5 - (t_vec[0] - 0.5 / speed) * speed 
#         y_pos_ref[0]  = 0.5
#     if ((t_vec[0] >= 1.5 / speed) and (t_vec[0] < 2.5 / speed)):
#         v_ref[0]  = speed
#         omega_ref[0]  = 0.0        # theta_ref[0]  = np.arctan2(np.sin(-math.pi / 2 - p0[2]),np.cos(-math.pi / 2 - p0[2])) + p0[2]

#         # theta_ref[0]  = np.arctan2(np.sin(-math.pi / 2 - p0[2]),np.cos(-math.pi / 2 - p0[2])) + p0[2]
#         theta_ref[0]  = -math.pi / 2
#         x_pos_ref[0]  = -0.5
#         y_pos_ref[0]  = 0.5 - (t_vec[0] - 1.5 / speed) * speed
#     if ((t_vec[0] >= 2.5 / speed) and (t_vec[0] < 3.5 / speed)):
#         v_ref[0]  = speed
#         omega_ref[0] = 0.0
#         # theta_ref[0] = np.arctan2(np.sin(0.0 - p0[2]),np.cos(0.0 - p0[2])) + p0[2]
#         theta_ref[0] = 0.0 
#         x_pos_ref[0]  = -0.5 + (t_vec[0] - 2.5 / speed) * speed
#         y_pos_ref[0]  = -0.5
#     if ((t_vec[0] >= 3.5 / speed) and (t_vec[0] < 4.0 / speed)):
#         v_ref[0]  = speed 
#         omega_ref[0]  = 0.
#         # theta_ref[0]  = np.arctan2(np.sin(math.pi / 2 - p0[2]),np.cos(math.pi / 2 - p0[2])) + p0[2]
#         theta_ref[0]  = math.pi / 2
#         x_pos_ref[0]  =  0.5
#         y_pos_ref[0]  = -0.5 + (t_vec[0] - 3.5 / speed) * speed

#     for kp in range(N):
#         k = 1 + kp
#         if (t_vec[k] < 0.5 / speed):
#             if k < N:
#                 v_ref[k] = speed
#                 omega_ref[k]  = 0.0
#             # theta_ref[k]  = np.arctan2(np.sin(math.pi / 2 - theta_ref[kp]),np.cos(math.pi / 2 - theta_ref[kp])) + theta_ref[kp]
#             theta_ref[k]  = math.pi / 2
#             x_pos_ref[k]  = 0.5
#             y_pos_ref[k]  = t_vec[k] * speed
#         if ((t_vec[k] >= 0.5 / speed) and (t_vec[k] < 1.5 / speed)):
#             if k < N:
#                 v_ref[k]  = speed
#                 omega_ref[k]  = 0.0
#             # theta_ref[k]  = np.arctan2(np.sin(math.pi - theta_ref[kp]),np.cos(math.pi - theta_ref[kp])) + theta_ref[kp]
#             theta_ref[k]  = math.pi
#             x_pos_ref[k]  = 0.5 - (t_vec[k] - 0.5 / speed) * speed 
#             y_pos_ref[k]  = 0.5
#         if ((t_vec[k] >= 1.5 / speed) and (t_vec[k] < 2.5 / speed)):
#             if k < N:
#                 v_ref[k]  = speed
#                 omega_ref[k]  = 0.0
#             # theta_ref[k]  = np.arctan2(np.sin(-math.pi / 2 - theta_ref[kp]),np.cos(-math.pi / 2 - theta_ref[kp])) + theta_ref[kp]
#             theta_ref[k]  = -math.pi / 2
#             x_pos_ref[k]  = -0.5
#             y_pos_ref[k]  = 0.5 - (t_vec[k] - 1.5 / speed) * speed
#         if ((t_vec[k] >= 2.5 / speed) and (t_vec[k] < 3.5 / speed)):
#             if k < N:
#                 v_ref[k]  = speed
#                 omega_ref[k] = 0.0
#             # theta_ref[k] = np.arctan2(np.sin(0theta_ref[k] = np.arctan2(np.sin(0.0.0 - theta_ref[kp]),np.cos(0.0 - theta_ref[kp])) + theta_ref[kp]
#             theta_ref[k] = 0.0
#             x_pos_ref[k]  = -0.5 + (t_vec[k] - 2.5 / speed) * speed
#             y_pos_ref[k]  = -0.5
#         if ((t_vec[k] >= 3.5 / speed) and (t_vec[k] < 4.0 / speed)):
#             if k < N:
#                 v_ref[k]  = speed 
#                 omega_ref[k]  = 0.
#             # theta_ref[k]  = np.arctan2(np.sin(math.pi / 2 - theta_ref[kp]),np.cos(math.pi / 2 - theta_ref[kp])) + theta_ref[kp]
#             theta_ref[k]  = math.pi / 2
#             x_pos_ref[k]  = 0.5
#             y_pos_ref[k]  = -0.5 + (t_vec[k] - 3.5 / speed) * speed
        
#     state_ref = np.vstack((x_pos_ref.reshape(1,N + 1), y_pos_ref.reshape(1,N + 1), theta_ref.reshape(1,N + 1)))
#     input_ref = np.vstack((v_ref.reshape(1,N), omega_ref.reshape(1,N)))
#     return state_ref, input_refspace(0,N * Ts, N + 1)

#     x_pos_ref = 0.5 * np.ones(N + 1)
#     y_pos_ref = np.zeros(N  + 1)
#     # theta_ref = (np.arctan2(np.sin(math.pi / 2 - p0[2]),np.cos(math.pi / 2 - p0[2])) + p0[2]) * np.ones(N + 1)
#     theta_ref = math.pi / 2 * np.ones(N + 1)
#     v_ref = np.zeros(N)
#     omega_ref = np.zeros(N)
    
#     if (t_vec[0] < 0.5 / speed):
#         v_ref[0] = speed
#         omega_ref[0]  = 0.0
#         # theta_ref[0]  = np.arctan2(np.sin(math.pi / 2 - p0[2]),np.cos(math.pi / 2 - p0[2])) + p0[2]
#         theta_ref[0]  = math.pi / 2
#         x_pos_ref[0]  = 0.5
#         y_pos_ref[0]  = t_vec[0] * speed
#     if ((t_vec[0] >= 0.5 / speed) and (t_vec[0] < 1.5 / speed)):
#         v_ref[0]  = speed
#         omega_ref[0]  = 0.0
#         # theta_ref[0]  = np.arctan2(np.sin(math.pi - p0[2]),np.cos(math.pi - p0[2])) + p0[2]
#         theta_ref[0]  = math.pi
#         x_pos_ref[0]  = 0.5 - (t_vec[0] - 0.5 / speed) * speed 
#         y_pos_ref[0]  = 0.5
#     if ((t_vec[0] >= 1.5 / speed) and (t_vec[0] < 2.5 / speed)):
#         v_ref[0]  = speed
#         omega_ref[0]  = 0.0        # theta_ref[0]  = np.arctan2(np.sin(-math.pi / 2 - p0[2]),np.cos(-math.pi / 2 - p0[2])) + p0[2]

#         # theta_ref[0]  = np.arctan2(np.sin(-math.pi / 2 - p0[2]),np.cos(-math.pi / 2 - p0[2])) + p0[2]
#         theta_ref[0]  = -math.pi / 2
#         x_pos_ref[0]  = -0.5
#         y_pos_ref[0]  = 0.5 - (t_vec[0] - 1.5 / speed) * speed
#     if ((t_vec[0] >= 2.5 / speed) and (t_vec[0] < 3.5 / speed)):
#         v_ref[0]  = speed
#         omega_ref[0] = 0.0
#         # theta_ref[0] = np.arctan2(np.sin(0.0 - p0[2]),np.cos(0.0 - p0[2])) + p0[2]
#         theta_ref[0] = 0.0 
#         x_pos_ref[0]  = -0.5 + (t_vec[0] - 2.5 / speed) * speed
#         y_pos_ref[0]  = -0.5
#     if ((t_vec[0] >= 3.5 / speed) and (t_vec[0] < 4.0 / speed)):
#         v_ref[0]  = speed 
#         omega_ref[0]  = 0.
#         # theta_ref[0]  = np.arctan2(np.sin(math.pi / 2 - p0[2]),np.cos(math.pi / 2 - p0[2])) + p0[2]
#         theta_ref[0]  = math.pi / 2
#         x_pos_ref[0]  =  0.5
#         y_pos_ref[0]  = -0.5 + (t_vec[0] - 3.5 / speed) * speed

#     for kp in range(N):
#         k = 1 + kp
#         if (t_vec[k] < 0.5 / speed):
#             if k < N:
#                 v_ref[k] = speed
#                 omega_ref[k]  = 0.0
#             # theta_ref[k]  = np.arctan2(np.sin(math.pi / 2 - theta_ref[kp]),np.cos(math.pi / 2 - theta_ref[kp])) + theta_ref[kp]
#             theta_ref[k]  = math.pi / 2
#             x_pos_ref[k]  = 0.5
#             y_pos_ref[k]  = t_vec[k] * speed
#         if ((t_vec[k] >= 0.5 / speed) and (t_vec[k] < 1.5 / speed)):
#             if k < N:
#                 v_ref[k]  = speed
#                 omega_ref[k]  = 0.0
#             # theta_ref[k]  = np.arctan2(np.sin(math.pi - theta_ref[kp]),np.cos(math.pi - theta_ref[kp])) + theta_ref[kp]
#             theta_ref[k]  = math.pi
#             x_pos_ref[k]  = 0.5 - (t_vec[k] - 0.5 / speed) * speed 
#             y_pos_ref[k]  = 0.5
#         if ((t_vec[k] >= 1.5 / speed) and (t_vec[k] < 2.5 / speed)):
#             if k < N:
#                 v_ref[k]  = speed
#                 omega_ref[k]  = 0.0
#             # theta_ref[k]  = np.arctan2(np.sin(-math.pi / 2 - theta_ref[kp]),np.cos(-math.pi / 2 - theta_ref[kp])) + theta_ref[kp]
#             theta_ref[k]  = -math.pi / 2
#             x_pos_ref[k]  = -0.5
#             y_pos_ref[k]  = 0.5 - (t_vec[k] - 1.5 / speed) * speed
#         if ((t_vec[k] >= 2.5 / speed) and (t_vec[k] < 3.5 / speed)):
#             if k < N:
#                 v_ref[k]  = speed
#                 omega_ref[k] = 0.0
#             # theta_ref[k] = np.arctan2(np.sin(0theta_ref[k] = np.arctan2(np.sin(0.0.0 - theta_ref[kp]),np.cos(0.0 - theta_ref[kp])) + theta_ref[kp]
#             theta_ref[k] = 0.0
#             x_pos_ref[k]  = -0.5 + (t_vec[k] - 2.5 / speed) * speed
#             y_pos_ref[k]  = -0.5
#         if ((t_vec[k] >= 3.5 / speed) and (t_vec[k] < 4.0 / speed)):
#             if k < N:
#                 v_ref[k]  = speed 
#                 omega_ref[k]  = 0.
#             # theta_ref[k]  = np.arctan2(np.sin(math.pi / 2 - theta_ref[kp]),np.cos(math.pi / 2 - theta_ref[kp])) + theta_ref[kp]
#             theta_ref[k]  = math.pi / 2
#             x_pos_ref[k]  = 0.5
#             y_pos_ref[k]  = -0.5 + (t_vec[k] - 3.5 / speed) * speed
        
#     state_ref = np.vstack((x_pos_ref.reshape(1,N + 1), y_pos_ref.reshape(1,N + 1), theta_ref.reshape(1,N + 1)))
#     input_ref = np.vstack((v_ref.reshape(1,N), omega_ref.reshape(1,N)))
#     return state_ref, input_ref

def create_tajectory_randpoints(points):
    poses = []
    # points = [[-1.352, -0.840], [-0.088,1.409],[1.306,-0.948],[0.869,2.150],[-1.155,2.208],[-0.067,-1.547],[0.,0.],[0.,0.4],[0.3,0.]]
    for p in points :
        pose = Pose2D()
        pose.x = p[0]
        pose.y = p[1]
        poses.append(pose)
    return poses

def add_time_to_wayposes(poses,t0,desired_speed,mode = 'ignore_corners'):
    LargeTime = 1000
    
    W = len(poses)
    timed_poses = np.zeros((4,W))
    if mode == 'ignore_corners':
        for i in range(W):
            timed_poses[0,i] = poses[i].x
            timed_poses[1,i] = poses[i].y
            # timed_poses[2,i] = poses[i].theta
            if i > 0:
                timed_poses[2,i] = np.arctan2(poses[i].y - poses[i - 1].y, poses[i].x - poses[i - 1].x)
                timed_poses[3,i] = timed_poses[3,i - 1] + 1 / desired_speed * np.sqrt((poses[i].y - poses[i - 1].y) ** 2 + (poses[i].x - poses[i - 1].x) ** 2)
            else:
                timed_poses[2,i] = 0
                timed_poses[3,i] = t0
    if mode == 'stop_in_corners':
        timed_poses = np.zeros((4,2 * W))
        for i in range(W):
            timed_poses[0,i * 2] = poses[i].x
            timed_poses[1,i * 2] = poses[i].y
            if i > 0:
                timed_poses[2,i  * 2] = np.arctan2(poses[i].y - poses[i - 1].y, poses[i].x - poses[i - 1].x)
                timed_poses[3,i * 2] = timed_poses[3,i * 2 - 1] + 1 / desired_speed * np.sqrt((poses[i].y - poses[i - 1].y) ** 2 + (poses[i].x - poses[i - 1].x) ** 2)
            else:
                timed_poses[2,0] = poses[0].theta
                timed_poses[3,0] = t0
            timed_poses[0,i  * 2 + 1] = poses[i].x
            timed_poses[1,i * 2 + 1] = poses[i].y
            if i < W - 1:
                timed_poses[2,i  * 2 + 1] = np.arctan2(poses[i + 1].y - poses[i].y, poses[i + 1].x - poses[i].x)
                timed_poses[3,i  * 2 + 1] = timed_poses[3,i * 2] + 5 * 0.11 / (2 * desired_speed) * np.absolute(np.arctan2(np.sin(timed_poses[2,i  * 2 + 1] - timed_poses[2,i  * 2 ]),np.cos(timed_poses[2,i  * 2 + 1] - timed_poses[2,i  * 2 ])))
            else:
                timed_poses[2,i  * 2 + 1] = timed_poses[2,i  * 2]
                timed_poses[3,i  * 2 + 1] = t0 + LargeTime             
    return timed_poses

def generate_reference_trajectory_from_timed_wayposes(timed_poses,t,Ts,N,mode = 'ignore_corners'):
    x_pos_ref = np.zeros(N + 1)
    y_pos_ref = np.zeros(N  + 1)
    theta_ref = np.zeros(N + 1)
    v_ref = np.zeros(N + 1)
    omega_ref = np.zeros(N + 1)
    
    if mode == 'ignore_corners':
        t_vec = t + np.linspace(0,N * Ts, N + 1)
        for k in range(N + 1):
            waypose_times = timed_poses[3,:]
            idx_poses_after_t = np.argwhere(waypose_times > t_vec[k])
            if idx_poses_after_t.size > 0:
                idx_k = idx_poses_after_t[0]
                if idx_k > 0:
                    v_ref[k] = np.sqrt((timed_poses[1,idx_k] - timed_poses[1,idx_k - 1]) ** 2 + (timed_poses[0,idx_k] - timed_poses[0,idx_k - 1]) ** 2) / (timed_poses[3,idx_k] - timed_poses[3,idx_k - 1])
                    theta_ref[k] = np.arctan2(timed_poses[1,idx_k] - timed_poses[1,idx_k - 1], timed_poses[0,idx_k] - timed_poses[0,idx_k - 1])
                    l = (t_vec[k] - timed_poses[3,idx_k - 1]) / (timed_poses[3,idx_k] - timed_poses[3,idx_k - 1])
                    x_pos_ref[k] = l * timed_poses[0,idx_k] + (1 - l) * timed_poses[0,idx_k - 1]
                    y_pos_ref[k] = l * timed_poses[1,idx_k] + (1 - l) * timed_poses[1,idx_k - 1]
    
    if mode == 'stop_in_corners':
        t_vec = t + np.linspace(0,N * Ts, N + 1)
        for k in range(N + 1):
            waypose_times = timed_poses[3,:]
            idx_poses_after_t = np.argwhere(waypose_times > t_vec[k])
            if idx_poses_after_t.size > 0:
                idx_k = idx_poses_after_t[0]
                if idx_k > 0:
                    v_ref[k] = np.sqrt((timed_poses[1,idx_k] - timed_poses[1,idx_k - 1]) ** 2 + (timed_poses[0,idx_k] - timed_poses[0,idx_k - 1]) ** 2) / (timed_poses[3,idx_k] - timed_poses[3,idx_k - 1])
                    if np.remainder(idx_k,2) == 0:
                        l = (t_vec[k] - timed_poses[3,idx_k - 1]) / (timed_poses[3,idx_k] - timed_poses[3,idx_k - 1])
                        theta_ref[k] = np.arctan2(timed_poses[1,idx_k] - timed_poses[1,idx_k - 1], timed_poses[0,idx_k] - timed_poses[0,idx_k - 1])
                        x_pos_ref[k] = l * timed_poses[0,idx_k] + (1 - l) * timed_poses[0,idx_k - 1]
                        y_pos_ref[k] = l * timed_poses[1,idx_k] + (1 - l) * timed_poses[1,idx_k - 1]
                    else:
                        x_pos_ref[k] = timed_poses[0,idx_k - 1]
                        y_pos_ref[k] = timed_poses[1,idx_k - 1]
                        l_rot = (t_vec[k] - timed_poses[3,idx_k - 1]) / (timed_poses[3,idx_k] - timed_poses[3,idx_k - 1])
                        # print(l_rot)
                        theta_ref[k]  = timed_poses[2,idx_k - 1] + l_rot *  np.arctan2(np.sin(timed_poses[2,idx_k] - timed_poses[2,idx_k - 1]),np.cos(timed_poses[2,idx_k] - timed_poses[2,idx_k - 1]))
                        omega_ref[k] = np.arctan2(np.sin(timed_poses[2,idx_k] - timed_poses[2,idx_k - 1]),np.cos(timed_poses[2,idx_k] - timed_poses[2,idx_k - 1])) / (timed_poses[3,idx_k] - timed_poses[3,idx_k - 1])
                else:
                    v_ref[k] = np.sqrt((timed_poses[1,idx_k] - timed_poses[1,idx_k - 1]) ** 2 + (timed_poses[0,idx_k] - timed_poses[0,idx_k - 1]) ** 2) / (timed_poses[3,idx_k] - timed_poses[3,idx_k - 1])

    state_ref = np.vstack((x_pos_ref.reshape(1,N + 1), y_pos_ref.reshape(1,N + 1), theta_ref.reshape(1,N + 1)))
    input_ref = np.vstack((v_ref[:-1].reshape(1,N), omega_ref[:-1].reshape(1,N)))
    return state_ref, input_ref

x0 = 1 * np.random.randn(3)
points = [x0[:2].tolist(),[0.,0.],[-1.352, -0.840], [-0.088,1.409],[1.306,-0.948],[0.869,2.150],[-1.155,2.208],[-0.067,-1.547],[0.,0.4],[0.3,0.],[0.,0.]]
speed_des = 0.25

# mode = 'ignore_corners'
mode = 'stop_in_corners' #buggy

wayposes = create_tajectory_randpoints(points)
wayposes[0].theta = x0[2]
timed_poses = add_time_to_wayposes(wayposes,0,speed_des,mode)
print(timed_poses)
[state_plot, input_plot] = generate_reference_trajectory_from_timed_wayposes(timed_poses,0,0.01,12000,mode)
# [state_plot, input_plot] = get_reference_circle(0,0.1,200)
# [state_plot, input_plot] = get_reference_pointsintersection(x0,0,0.1,200)

plt.figure()
plt.plot(state_plot[0,:],state_plot[1,:],'k')
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.axis('square')
plt.show()

plt.figure()
plt.plot(np.linspace(0,12000,12001),np.remainder(state_plot[2,:] + math.pi,2 * math.pi) - math.pi,'k')
plt.xlabel("$k$")
plt.ylabel("$\\theta$")
plt.axis([0,11999,-4,4])
plt.show()

t_sim = 0



# parameters for imput constraints 
WR = 0.03
WS = 0.11
r_a = 35.
l_a = 35.
r_input_max = 0.3
r_input_min = -0.3
l_input_max = 0.3
l_input_min = -0.3

# radius of safe set
sr = 0.25 

# create ocp object to formulate the OCP
ocp = AcadosOcp()

ocp.constraints.x0 = x0

# set model
model = export_unicycle_ode_model_with_LocalConstraints()
ocp.model = model

Tf = 2.0
nx = model.x.size()[0]
nu = model.u.size()[0]
nparam = model.p.size()[0]
ny = nx + nu
ny_e = nx
nsh = 1
N = 20

# set dimensions
ocp.dims.N = N

# set cost
Q = np.diag([1.0, 1.0, 0.1])
R = np.diag([0.001, 0.001])

W_e = N * Q
W = scipy.linalg.block_diag(Q, R)
ocp.cost.W_e = W_e
ocp.cost.W = W

# print(W)

ocp.cost.cost_type = 'LINEAR_LS'
ocp.cost.cost_type_e = 'LINEAR_LS'

Vx = np.zeros((ny, nx))
Vx[:nx,:nx] = np.eye(nx)
ocp.cost.Vx = Vx


Vx_e = np.zeros((ny_e, nx))
Vx_e[:nx, :nx] = np.eye(nx)
ocp.cost.Vx_e = Vx_e

Vu = np.zeros((ny, nu))
Vu[nx:,:] = np.eye(2)
ocp.cost.Vu = Vu
ocp.cost.Vu_0 = Vu


# set intial referencesplt.figure()
# plt.plot(state_plot[0,:],state_plot[1,:],'k')
# plt.xlabel("$x$")
# plt.ylabel("$y$")
# plt.axis('square')
# plt.show()
ocp.cost.yref  = np.zeros(ny)
ocp.cost.yref_e = np.zeros(ny_e)
# ocp.cost.yref[0] = x_ref[0]
# ocp.cost.yref[1] = y_ref[0]
# ocp.cost.yref[2] = theta_ref[0]
# ocp.cost.yref[3] = v_ref[0]
# ocp.cost.yref[4] = omega_ref[0]
# ocp.cost.yref_e[0] = x_ref[0]
# ocp.cost.yref_e[1] = y_ref[0]
# ocp.cost.yref_e[2] = theta_ref[0]

# setting input constraints constraints
D = np.array([[1 / (r_a * WR), WS / (2 * r_a * WR)], [1 / (l_a * WR), -WS / (2 * l_a * WR)]])
ocp.constraints.D = D
ocp.constraints.C = np.zeros((2,nx))
ocp.constraints.lg = np.array([r_input_min, l_input_min])
ocp.constraints.ug = np.array([r_input_max, l_input_max])

# set soft contraint penatly for local safe set
M = 10 * np.amax(W)
ocp.cost.zl = np.array([0.])
ocp.cost.zl_e = np.array([0.])
ocp.cost.Zl = np.array([0.])
ocp.cost.Zl_e = np.array([0.])
ocp.cost.zu = M * np.ones((nsh,))
ocp.cost.zu_e = M * np.ones((nsh,))
ocp.cost.Zu = np.array([0.])
ocp.cost.Zu_e = np.array([0.])
ocp.constraints.ush = np.zeros(nsh)
ocp.constraints.uh = np.array([sr ** 2])
ocp.constraints.ush_e = np.zeros(nsh)
ocp.constraints.uh_e = np.array([sr ** 2])
ocp.constraints.lsh = np.zeros(nsh)
ocp.constraints.lh = np.array([0.])
ocp.constraints.lsh_e = np.zeros(nsh)
ocp.constraints.lh_e = np.array([0.])
ocp.constraints.idxsh = np.array([0])
ocp.constraints.idxsh_e = np.array([0])

ocp.parameter_values = np.array([0., 0.])


# set options
ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' # FULL_CONDENSING_QPOASES
ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
ocp.solver_options.integrator_type = 'ERK'
# ocp.solver_options.print_level = 0
ocp.solver_options.nlp_solver_type = 'SQP_RTI' # SQP_RTI, SQP

# set prediction horizon
ocp.solver_options.tf = Tf

ocp_solver = AcadosOcpSolver(ocp, json_file = 'acados_ocp.json')

xnext = x0
t_sim = 0.0
Ts = Tf / N

Nsim = math.ceil(timed_poses[3,-2] / Ts) + N
print(Nsim)
simX = np.ndarray((nx, Nsim + 1))
simU = np.ndarray((nu, Nsim))
simT = np.ndarray((1, Nsim + 1))




simT[0,0] = t_sim
for j in range(nx):
    simX[j,0] = xnext[j]

for i in range(Nsim):
    print("Time ",t_sim)

    # update initial condition
    x0[0] = xnext[0]
    x0[1] = xnext[1]
    thetanext_wind = np.arctan2(np.sin(xnext[2] - x0[2]), np.cos(xnext[2] - x0[2])) + xnext[2]
    x0[2] = thetanext_wind
    # x0[2] = np.remainder(xnext[2] + math.pi, 2 * math.pi) - math.pi

    ocp_solver.set(0, "lbx", x0)
    ocp_solver.set(0, "ubx", x0)

    # reference
    # [state_ref, input_ref] = get_reference_pointsintersection(xnext,t_sim, Ts, N)
    # [state_ref, input_ref] = get_reference_circle(t_sim, Ts, N)
    [state_ref, input_ref] = generate_reference_trajectory_from_timed_wayposes(timed_poses,t_sim,Ts,N,mode)


    for k in range(N):
        if k == 0:
            thetaref_k = np.arctan2(np.sin(state_ref[2,k] - xnext[2]), np.cos(state_ref[2,k] - xnext[2])) + xnext[2]
        else:
            thetaref_k = np.arctan2(np.sin(state_ref[2,k] - thetaref_k), np.cos(state_ref[2,k] - thetaref_k)) + thetaref_k        
        ocp_solver.set(k, "yref", np.array([state_ref[0,k], state_ref[1,k],thetaref_k, input_ref[0,k], input_ref[1,k]]))
        ocp_solver.set(k, "p", np.array([state_ref[0,k], state_ref[1,k]]))
    thetaref_N = np.arctan2(np.sin(state_ref[2,N] - thetaref_k), np.cos(state_ref[2,N] - thetaref_k)) + thetaref_k 
    ocp_solver.set(N, "yref", np.array([state_ref[0,N], state_ref[1,N], thetaref_k]))
    ocp_solver.set(N, "p", np.array([state_ref[0,N], state_ref[1,N]]))

    # preparation rti_phase
    ocp_solver.options_set('rti_phase', 0)
    t = time.time()
    status = ocp_solver.solve()
    phase0_time = time.time() - t
    print("Solver time is",phase0_time,"s")

    if status != 0:
        raise Exception(f'acados returned status {status}.')

    # get solution
    # x0 = ocp_solver.get(0, "x")
    u0 = ocp_solver.get(0, "u")

    for j in range(nu):
        simU[j,i] = u0[j]

    upred = np.zeros((nu,N))
    xpred  = np.zeros((nx,N + 1))
    for k in range(N):
        upred[:,k] = ocp_solver.get(k,"u")
        xpred[:,k] = ocp_solver.get(k,"x")
    xpred[:,N] = ocp_solver.get(N,"x")
    
    ur_ref = 1.0 / (r_a * WR) * input_ref[0,:] + WS / (2 * r_a * WR) * input_ref[1,:]
    ul_ref = 1.0 / (l_a * WR) * input_ref[0,:] - WS / (2 * l_a * WR) * input_ref[1,:]
    ur_pred = 1.0 / (r_a * WR) * upred[0,:] + WS / (2 * r_a * WR) * upred[1,:]
    ul_pred = 1.0 / (l_a * WR) * upred[0,:] - WS / (2 * l_a * WR) * upred[1,:]

    plt.figure(1)
    plt.cla()
    plt.plot(state_plot[0,:],state_plot[1,:],'k')
    plt.plot(state_ref[0,:],state_ref[1,:],'r--')
    plt.plot(xpred[0,:],xpred[1,:],'g:')
    plt.plot(simX[0,:i],simX[1,:i],'b')
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.axis('square')
    # plt.show()
    plt.pause(Ts/100)

    plt.figure(2)
    plt.subplot(321)
    plt.cla()
    # plt.plot(np.linspace(0,N,N + 1),state_ref[2,:],'r--')
    # plt.plot(np.linspace(0,N,N + 1),xpred[2,:],'g:')
    # plt.axis([0, N, -7, 7])
    plt.plot(np.linspace(0,N,N + 1),np.remainder(state_ref[2,:] + math.pi,2 * math.pi) - math.pi,'r--')
    plt.plot(np.linspace(0,N,N + 1),np.remainder(xpred[2,:] + math.pi,2 * math.pi) - math.pi,'g:')
    plt.axis([0, N, -3.5, 3.5])
    plt.ylabel("$\\theta$")
    plt.xlabel("$k$")

    plt.subplot(323)
    plt.cla()
    plt.plot(np.linspace(0,N - 1,N),input_ref[0,:],'r--')
    plt.plot(np.linspace(0,N - 1,N),upred[0,:],'g:')
    plt.axis([0, N - 1, -0.5, 0.5])
    plt.ylabel("$v$")
    plt.xlabel("$k$")

    plt.subplot(324)
    plt.cla()
    plt.plot(np.linspace(0,N - 1,N),input_ref[1,:],'r--')
    plt.plot(np.linspace(0,N - 1,N),upred[1,:],'g:')
    plt.axis([0, N - 1, -8, 8])
    plt.ylabel("$\\omega$")
    plt.xlabel("$k$")


    plt.subplot(325)
    plt.cla()
    plt.plot(np.linspace(0,N - 1,N),ur_ref,'r--')
    plt.plot(np.linspace(0,N - 1,N),ur_pred,'g:')
    plt.axis([0, N - 1, -0.5, 0.5])
    plt.ylabel("$u^r$")
    plt.xlabel("$k$")

    plt.subplot(326)
    plt.cla()
    plt.plot(np.linspace(0,N,N),ul_ref,'r--')
    plt.plot(np.linspace(0,N,N),ul_pred,'g:')
    plt.axis([0, N - 1, -0.5, 0.5])
    plt.ylabel("$u^l$")
    plt.xlabel("$k$")

    # plt.show()
    plt.pause(Ts / 100)


    # get next state
    xnext = ocp_solver.get(1, "x") + 0.005 * np.random.randn(3)
    # print(xnext)
    t_sim += Ts

    simT[0,i + 1] = t_sim
    for j in range(nx):
        simX[j,i + 1] = xnext[j]



plt.figure(3)
plt.plot(state_plot[0,:],state_plot[1,:],'k--')
plt.plot(simX[0,:],simX[1,:],'b')
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.axis('square')
plt.show()

