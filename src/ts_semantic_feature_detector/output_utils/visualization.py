"""
Jose's code to visualize the landmarks in the global frame.
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation

#landmarks in global frame
path = '/home/daslab/Documents/dev/catkin_ws/src/ts_semantic_feature_detector/output/'

raw_emerging_points = np.loadtxt(path+'emerging_points.txt', delimiter=',')
raw_poses = np.loadtxt(path+'odometry_factors.txt', delimiter=',')

# Rotation from robot imu (body) to left camera (zed)
R_lb = np.array([[-0.11805188474014247, -0.9917158860146585, -0.05062957569901266],
                            [0.04489899166227829, 0.04560316209867554, -0.9979501150630294],
                            [0.9919918513057513, -0.12008310885700718, 0.03914350406166214]])
t_lb = np.array([[0.06], [0.24], [-0.18]])
Tlb = np.block([[R_lb,t_lb],[np.zeros((1,3)), 1.0]])
Tbl = np.linalg.inv(Tlb)


landmarks = [] # landmarks in the global frame (used for initialization in the optimization step)
points_list = []
num_items = []
for i in range(raw_emerging_points.shape[0]):
    frame_id, point_id, x, y, z = raw_emerging_points[i,:].tolist() # point w.r.t. camera frame
    
    if point_id in points_list:
        idx_list = points_list.index(point_id)
        indx =  np.where(raw_poses[:,0]==frame_id)[0][0]
        t = raw_poses[indx,1:4].reshape(3,1)
        q = raw_poses[indx,4:8]
        R = Rotation.from_quat(q).as_matrix()
        Twb = np.block([[R, t],[np.zeros((1,3)), 1.0]]) # body to world frame
        Tbl = np.linalg.inv(Tlb) # left to body frame
        Twl = Twb@Tbl # left camera to world frame
        pw = Twl[0:3, 0:3]@np.array([[x],[y],[z]]) + Twl[0:3,3].reshape(3,1)    # point w.r.t. the world frame
        
        landmarks[idx_list] = (num_items[idx_list]*landmarks[idx_list] + pw.T)/(num_items[idx_list]+1)
        num_items[idx_list] = num_items[idx_list] + 1
        

    else:
        points_list.append(point_id)
        indx =  np.where(raw_poses[:,0]==frame_id)[0][0]
        #print("point id: ", point_id, "frame_id: ", frame_id, "indx: ", indx)
        t = raw_poses[indx,1:4].reshape(3,1)
        q = raw_poses[indx,4:8]
        R = Rotation.from_quat(q).as_matrix()
        Twb = np.block([[R, t],[np.zeros((1,3)), 1.0]]) # body to world frame
        Tbl = np.linalg.inv(Tlb) # left to body frame
        Twl = Twb@Tbl # left camera to world frame

        pw = Twl[0:3, 0:3]@np.array([[x],[y],[z]]) + Twl[0:3,3].reshape(3,1)    # point w.r.t. the world frame
        landmarks.append(pw.T)
        num_items.append(1)
landmarks = np.array(landmarks).reshape(-1,3)

emerging_points = [] # all the observations of emerging points (used for visualization)
for i in range(raw_emerging_points.shape[0]):
    frame_id, point_id, x, y, z = raw_emerging_points[i,:].tolist() # point w.r.t. camera frame
    points_list.append(point_id)
    indx =  np.where(raw_poses[:,0]==frame_id)[0][0]
    t = raw_poses[indx,1:4].reshape(3,1)
    q = raw_poses[indx,4:8]
    R = Rotation.from_quat(q).as_matrix()
    Twb = np.block([[R, t],[np.zeros((1,3)), 1.0]]) # body to world frame
    Tbl = np.linalg.inv(Tlb) # left to body frame
    Twl = Twb@Tbl # left camera to world frame
    pw = Twl[0:3, 0:3]@np.array([[x],[y],[z]]) + Twl[0:3,3].reshape(3,1)    # point w.r.t. the world frame
    em_point = np.block([[point_id, pw.T]])
    emerging_points.append(em_point.T)
emerging_points = np.array(emerging_points).reshape(-1,4)


def random_color():
    levels = range(32,256,32)
    return tuple(np.random.choice(levels)/255 for _ in range(3))
plt.figure(1)

color_list = [random_color() for i in range (emerging_points.shape[0])]
markers_list = ['o', 'd', 'v', '^', 's', 'p', 'P', '*','+', 'x', '_', '1', 'H', '4', '>']
for i in range (emerging_points.shape[0]):
    plt.plot(emerging_points[i,1], emerging_points[i,2], marker = markers_list[int(emerging_points[i,0])%15], color=color_list[int(emerging_points[i,0])])
    plt.text(emerging_points[i,1], emerging_points[i,2], str(int(emerging_points[i,0])), color=color_list[int(emerging_points[i,0])], fontsize=10)

# plt.xlim([-70, -40])
# plt.ylim([-4, 0])
plt.show()