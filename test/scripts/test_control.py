import math
import numpy as np
import pyflare as fl
import py_util.trajectory_util
skeleton = fl.skeletonFromJson('fish_simple.json')

trajectory = fl.poseTrajectory()
points = py_util.trajectory_util.trajectoryPoints_file("../../pydata/splines/splinePoints")
# points = py_util.trajectory_util.trajectoryPoints_circle([6,2,3],3,2)
trajectory.setPoints(points)
trajectory.fit()
trajectory.sample(100)

fl.VTKWriter.writeTrajectory(trajectory,"data/vis_data/Trajectory/trajectory_ref.vtk")

skeleton_dynamics = fl.SkeletonDynamics(skeleton,5000,[points[0].data[0]-0.6, points[0].data[1], points[0].data[2]+0.2])

world= fl.DartWorld([0,0,0])
world.addSkeleton(skeleton_dynamics)
param = fl.simParam()
param.width = 12
param.height = 4
param.depth = 6
param.setup_mode =fl.SETUP_MODE.MANUAL
param.l0p = 4
param.N = 50
param.u0p = 5
param.visp = 1e-3
param.timestep = param.get_tp()

simulator = fl.simulator3D(param)
simulator.attachWorld(world)
simulator.commitInit()
simulator.log()

pd_controller = fl.PDControl1D(10,2,world.time)

maxSpeed = 0
tailfreq = 2
simTime = 0.005
commonForce = 20
lines=[]

simulator.step(fl.COUPLE_MODE.TWO_WAY)
iterPerSave = simulator.getIterPerSave(30)
poseIdx = 0
for i in range(int(simTime / param.timestep)):
    link = skeleton_dynamics.getBaseLink()
    pos = link.getPosition()
    pose = trajectory.getTargetPose(pos, poseIdx,5)
    poseIdx = trajectory.getLastIdx()
    fwd = skeleton_dynamics.getBaseLinkFwd()
    fwd = fwd/np.linalg.norm(fwd)
    right = skeleton_dynamics.getBaseLinkRight()
    right = right/np.linalg.norm(right)
    expected_fwd =pose.getPosition() - pos
    expected_fwd = expected_fwd/np.linalg.norm( expected_fwd)
    vec_fwd2expected = expected_fwd - (fwd.dot(expected_fwd)) * fwd
    vec_error = vec_fwd2expected.dot(right)
    pdforce = pd_controller.feedback(vec_error, world.time)
    joint = skeleton_dynamics.getJoint("joint2")
    joint.setCommand(0, commonForce * math.cos(world.time / (1.0 / tailfreq / 4.0) * (3.14 / 2)) - pdforce)
    simulator.step(fl.COUPLE_MODE.TWO_WAY)
    # if simulator.getIterNum() % iterPerSave == 0 :
    #     ref_line=fl.debugLine()
    #     ref_line.vertices = [pos,pose.getPosition()]
    #     lines.append(ref_line)
    #     line_name="./data/vis_data/Trajectory/debug_lines_%05d.vtk"%(simulator.getIterNum() / iterPerSave)
    #     fluid_name = "fluid%05d"%(simulator.getIterNum() / iterPerSave)
    #     object_name = "object%05d"%(simulator.getIterNum() / iterPerSave)
    #     # simulator.saveFluidData(fluid_name)
    #     simulator.saveObjectsData(object_name)
    #     fl.VTKWriter.writeLines(lines, line_name)

