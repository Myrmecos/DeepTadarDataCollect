# TODO: 
# 1. accepts a point cloud data
# 2. accept an image data
# 3. given the coordinate on the image, report the corresponding points near the position in the point cloud
import open3d as o3d 
import numpy as np 

def readPcd(path):
    pcd = o3d.io.read_point_cloud(path)
    if pcd.is_empty():
        print("readPcd received empty point cloud")
        return
    return pcd

def readExtrinsic(path):
    output = np.genfromtxt(path, delimiter=',', dtype=float)
    return output

class ImageLidarAligner:
    def __init__(self):
        pass

    '''
    @param position: the (x, y) pixel coordinate in the image
    @return correspondingPts: points near the pixel
    '''
    def reportPoints(position):
        pass

    
    '''
    @param pts: a list of points
    @param percentile: the upper and lower percentiles to filter out. for example, percentile=0.25 indicates we select 25% to 75% points and clear other outliers.
    @return clearedPts: a list of points that had outliers removed
    '''
    def clearOutliers(pts, percentile):
        pass

if __name__=="__main__":
    extrinsic = readExtrinsic("/home/astar/dart_ws/calib/calibresult.txt")
    print(extrinsic)
    pcd = readPcd("/home/astar/dart_ws/single_scene_calibration/0.pcd")
    o3d.visualization.draw_geometries([pcd], window_name="Point Cloud Visualization", width=800, height=600)