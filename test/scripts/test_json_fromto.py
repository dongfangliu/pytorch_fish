
from scripts.test.fish_env import *
import py_util
from matplotlib import pyplot as plt


# Fluid
# param = fl.simParam()
# param.width = 4
# param.height = 4
# param.depth = 6
# param.setup_mode = fl.SETUP_MODE.MANUAL
# param.l0p = 4
# param.N = 50
# param.u0p = 5
# param.visp = 1e-3
#
# fluid_param = flare_util.fluid_param()
# fluid_param.from_json("/home/liuwj/PycharmProjects/testPylib/pydata/jsons/fluid_param_short.json")
#
# path = flare_util.path_data()
# path.from_json("/home/liuwj/PycharmProjects/testPylib/pydata/jsons/path.json")
#
# rigids = flare_util.rigid_data()
# rigids.from_json("/home/liuwj/PycharmProjects/testPylib/pydata/jsons/rigids.json")

path_radiusx=0.5
path_radiusz_range = (0.0, 2.0)
path_points = []
path_center_pos = [2, 2, 3]
for  radiusz in np.arange(path_radiusz_range[0], path_radiusz_range[1], 0.1):
    if radiusz!=0:
        path_points.append(py_util.trajectory_util.trajectoryPoints_circle(path_center_pos, path_radiusx, radiusz, 300,
                                                                           angle=180,inverse=False))
        path_points.append(py_util.trajectory_util.trajectoryPoints_circle(path_center_pos, path_radiusx, radiusz, 300,
                                                                           angle=180,inverse=True))
    else:
        path_points.append(
            py_util.trajectory_util.trajectoryPoints_circle(path_center_pos, path_radiusx, radiusz, 100, angle=180,
                                                            inverse=False))

for i in range(len(path_points)):
    path_config = flare_util.path_param()
    path_config.setPoints(path_points[i])
    path_config.to_json("/home/liuwj/PycharmProjects/testPylib/pydata/jsons/paths/path_{0}.json".format(i))



