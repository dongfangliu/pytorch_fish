import  numpy as np
import  matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
print(os.getcwd())
random_pos_radius =(0.4,0,0.4)
pos_cov = np.reshape([random_pos_radius[0],0,0,
                      0,random_pos_radius[1],0,
                      0,0,random_pos_radius[2]],
                     (3,3))
random_vel_mean  =  0
random_vel_std = 2
random_orien_mean = 0
random_orien_std = np.pi/12
random_sample_num=100
delta_orientations = np.random.normal(random_orien_mean, random_orien_std, random_sample_num)

# vels = np.random.normal(random_vel_mean,random_vel_std,random_sample_num)
# plt.figure()
# plt.hist(vels)
# plt.show()

# orientations  =[ y_rotation(np.array([1,0,0]), ori) for ori in delta_orientations]
#
# orientations = np.array(orientations)
# fig =plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.quiver(0,0,0,orientations[:,0],orientations[:,1],orientations[:,2], length=0.1, normalize=True)
#
# ax.quiver(0,0,0,1,0,0, length=0.3)
# ax.set_xlabel('variable X')
# ax.set_ylabel('variable Y')
# ax.set_zlabel('variable Z')
# plt.show()
import  py_util.np_util
# py_util.np_util.all_npz_to_one("/home/liuwj/PycharmProjects/testPylib/pydata/train_data/sampled_files","collected_oa")
aa = np.load("/pydata/train_data/collected_oa.npz")
print(aa['observations'].shape[0]==aa['actions'].shape[0])