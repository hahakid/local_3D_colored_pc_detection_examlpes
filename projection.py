import os
import numpy as np
import open3d as o3d

def showpc(points,param,name):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points[:, :3])
    points[:,[4,5,6]]=points[:,[6,5,4]]/255.0
    pc.colors=o3d.utility.Vector3dVector(points[:,4:7])
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1024, height=512,)
    opt = vis.get_render_option()
    opt.point_size = 4
    vis.update_renderer()
    vis.add_geometry(pc)
    vis_control = vis.get_view_control()
    vis_control.convert_from_pinhole_camera_parameters(param)
    vis.run()
    vis.capture_screen_image(os.path.join("./",name+'.png'))
    vis.close()
    vis.destroy_window()
    return 0

list=os.listdir("./RGBPoint")
for l in list:
    param = o3d.io.read_pinhole_camera_parameters('./ScreenCamera_2021-01-22-22-13-01.json')
    pc=np.load(os.path.join("./RGBPoint",l))
    '''
    pc_copy = pc.copy()
    pc[:,3] = pc_copy[:,-1]
    pc[:,4] = pc_copy[:,-1]
    pc[:,3] = pc_copy[:,-1]
    pc[:,3] = pc_copy[:,-1]
    '''
    showpc(pc,param,l.split(".")[0])
