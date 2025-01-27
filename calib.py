import numpy as np
import yaml
import matplotlib
import matplotlib.pyplot as plt
import cv2 as cv
from scipy.interpolate import griddata
from scipy.ndimage import map_coordinates
from matplotlib.widgets import Slider
matplotlib.use('TkAgg')
# import mplcursors
import os
import math
import copy

# calibration for one sensor to map to depth camera, at different distance (one distance gives one set of (R, T, S))

# TODO: seek/1m.yaml; 2m.yaml; ... 7m.yaml;
# senxor_m08/...; senxor_m16/...; mlx/...;


cursor1 = None
cursor11 = None
cursor2 = None
cursor21 = None
im = None
#transform_image_ori = None
R = None
T = None
s = 0

# Load the YAML file
def load_yaml(filename):
    with open(filename, 'r') as file:
        config = yaml.safe_load(file)

    # Extract points
    transform_points = config['transform_points']
    reference_points = config['reference_points']

    return (transform_points, reference_points)

# transform image give the rotation matrix R, translation matrix T and scale s
def transform_img(transform_image, R, T, scale):
    print("starting normal calib")
    # Load the reference image
    # transform = np.load("color.npy")
    transform_image = transform_image.astype(np.float32)
    # Define the padding size (top, bottom, left, right)
    print("image shape:", transform_image.shape)
    if (transform_image.shape[1]-transform_image.shape[0]>=0):
        top = 0 #abs(transform_image.shape[1]-transform_image.shape[0])
        bottom = abs(transform_image.shape[1]-transform_image.shape[0])
        left = 0
        right = 0
    else: 
        # we will check later how to deal with a tall image
        print("it's a tall image")
        #exit(1)
        top = 0 #abs(transform_image.shape[1]-transform_image.shape[0])
        bottom = 0#abs(transform_image.shape[1]-transform_image.shape[0])
        left = 0
        right = 0 #abs(transform_image.shape[1]-transform_image.shape[0])
    #Zero-pad the image using cv2.copyMakeBorder
    transform_image = cv.copyMakeBorder(transform_image, top, bottom, left, right, cv.BORDER_CONSTANT, value=-1)
    #plt.imshow(transform_image)
    #plt.show()
    # Get the shape of the reference image
    transform = transform_image
    height, width = transform.shape

    # Create a grid of coordinates for the reference image
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    coords = np.vstack((x.flatten(), y.flatten()))  # Shape: (2, N), where N = height * width

    # Step 1: Apply scaling
    coords_scaled = coords

    # Step 2: Apply rotation and translation
    coords_transformed = np.dot(R, coords_scaled) + T[:, np.newaxis]

    # Reshape the transformed coordinates back to the image shape
    x_transformed = coords_transformed[0, :].reshape(height, width)
    y_transformed = coords_transformed[1, :].reshape(height, width)

    # Resize the distance image using interpolation
    #resized_distance = cv.resize(transformed_reference, (new_width, new_height), interpolation=cv.INTER_LINEAR)
    transformed_reference = map_coordinates(transform_image, [y_transformed, x_transformed], order=1, mode='constant', cval=np.nan)
    transformed_reference = cv.resize(transformed_reference, (int(transform.shape[0]/scale), int(transform.shape[1]/scale)), interpolation=cv.INTER_AREA)

    return transformed_reference

def transform_image_layered(base_dir, max_dist, depth_ori, padding = True):
    print("starting multi-layer calib")
    Rd, Td, sd = read_yaml(base_dir + max_dist + ".yaml")
    background = transform_img(depth_ori, Rd, Td, sd)
    if not padding:
        background[(background<6.5) | (background>6.5)] = np.nan
    # calib for each distance range
    for i in range(int(max_dist), 0, -1):
    #for i in range(0):
        #print(i, "th")
        ind = i
        depth_ori1 = copy.copy(depth_ori)
        #1.1. make all areas outside the range nan
        min_ind = depth_ori<(ind-0.5)*1000
        max_ind = depth_ori>=(ind+0.5)*1000
        if ind==1: 
            min_ind = 0
        if ind==max_dist:
            max_ind = 100000
        depth_ori1[min_ind]=np.nan
        depth_ori1[max_ind]=np.nan
        # plt.imshow(depth_ori1)
        # plt.title("should see this as bg")
        # plt.show()
        # 2. apply transformation (RTS) to the image
        R2, T2, s2 = read_yaml(base_dir+f"{ind}.yaml")
        transformed_image1 = transform_img(depth_ori1, R2, T2, s2)

        mask = ~np.isnan(transformed_image1)
        background[mask] = transformed_image1[mask]

    # plt.imshow(background)
    # plt.title("and see this actually")
    # plt.show()
    #plt.imshow(transformed_image1, alpha=1)
    #background[background==-1]=np.nan
    return background

# calculates rotation matrix R, transformation matrix T and Scaling factor s
# caution: relative distance of reference points must already be scaled to same scale 
# (i.e. only rotation and translation, no scaling is needed)
def calc_RT(reference_points, transform_points):
     # Step 1: Center the points
    transform_centroid = np.mean(transform_points, axis=0)
    reference_centroid = np.mean(reference_points, axis=0)

    transform_centered = transform_points - transform_centroid
    reference_centered = reference_points - reference_centroid

    # Step 2: Compute the covariance matrix
    H = reference_centered.T @ transform_centered

    # Step 3: Perform Singular Value Decomposition (SVD)
    U, S, Vt = np.linalg.svd(H)

    # Step 4: Compute the rotation matrix R
    R = Vt.T @ U.T

    # Handle reflection case (ensure determinant of R is 1)
    if np.linalg.det(R) < 0:
        print("determinant of R is negative, flip it.")
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Step 5: Compute the translation vector T
    T = transform_centroid - R @ reference_centroid

    # Print the results
    print("Rotation Matrix (R):")
    print(R)
    print("Translation Vector (T):")
    print(T)
    return (R, T)


# calculate scale give reference (thermal) and transform (depth) points
def calc_scale(reference_points, transform_points):
    P_d = np.array(reference_points)  # reference image points
    P_t = np.array(transform_points)  # transform image points
    # Compute scaling factor
    dist_d = 0
    dist_t = 0
    for i in range(1, len(P_d)):
        dist_d += np.linalg.norm(P_d[i] - P_d[0])  # Distance between two points in reference image
        dist_t += np.linalg.norm(P_t[i] - P_t[0])  # Distance between two points in transform image
    
    s = dist_t / dist_d  # Scaling factor
    return s

# callback function, for moving cursor at corresponding positions
def on_mouse_move(event):
    global ax1, ax2, cursor11, cursor21
    if event.inaxes == ax1:
        cursor11.set_data(event.xdata, event.ydata)
        cursor2.set_data(event.xdata, event.ydata)
    elif event.inaxes == ax2:
        cursor21.set_data(event.xdata, event.ydata)
        cursor1.set_data(event.xdata, event.ydata)
    plt.draw()

#draw a black margin inside image without affecting image shape.
def draw_black_margin(image, margin_width):
    height, width = image.shape
    image[:margin_width, :] = 0  # Top margin
    image[-margin_width:, :] = 0  # Bottom margin
    image[:, :margin_width] = 0  # Left margin
    image[:, -margin_width:] = 0  # Right margin
    return image

# update the transform image according to adjusted R, T and S
def update(val):
    global R, T, scale, transform_dir, max_dir, depth_ori
    # Get current slider values
    angle = angle_slider.val
    xshift = xshift_slider.val
    yshift = yshift_slider.val
    scale = scale_slider.val

    # Apply transformations
    R = np.array([[math.cos(angle), -math.sin(angle)],[math.sin(angle), math.cos(angle)]])
    T = np.array([xshift, yshift])
    print(R, T, scale)
    # Update the displayed image
    if mode != "mlc":
        layer1 = transform_img(transform_image, R, T, scale)
    else:
        layer1 = transform_image_layered(basedir1, maxlen1, depth_ori1)
    layer2 = reference_image
    im.set_data(layer1)
    fig.canvas.draw_idle()

def MLX_process(MLX_temperature_map):
    MLX_temperature_map = MLX_temperature_map.astype(np.uint8)
    print("MLX: ", MLX_temperature_map)
    MLX_temperature_map = cv.normalize(MLX_temperature_map, None, 0, 255, cv.NORM_MINMAX)
    #MLX_temperature_map = cv.resize(MLX_temperature_map, (320, 240), interpolation=cv.INTER_NEAREST)
    MLX_temperature_map = cv.applyColorMap(MLX_temperature_map, cv.COLORMAP_JET)
    return MLX_temperature_map

def dump_yaml(R, T, s, filename):
    R = R.tolist()
    T = T.tolist()
    s = float(s)
    
    data = {
        "R": R, "T": T, "s": s
    }
    with open(filename, 'w') as file:
        yaml.dump(data, file, default_flow_style=True)

def read_yaml(filename):
    with open(filename) as file:
        data = yaml.safe_load(file)
    return np.array(data["R"]), np.array(data["T"]), np.float64(data["s"])

def read_RTS(src_distance, reference_dir, mode):
    pointsfile = "calibpoints/"+reference_dir[:-1]+".yaml"
    RTSfileSrc = "calibresults/"+reference_dir+src_distance+".yaml"
    if mode != "mlc":
        transform_points, reference_points = load_yaml(pointsfile)

    if mode == "pointcalib":
        # calculate scale
        scale = calc_scale(reference_points, transform_points)

        # rescale refernce points
        reference_points = np.array(reference_points)*scale

        # calculate rotation and transformation
        R, T = calc_RT(reference_points, transform_points)

    if mode == "mlc":
        R, T, scale = read_yaml(RTSfileSrc)
        print("read: ", R, T, scale)
    return R, T, scale

def load_image(baseDir, transform_dir, transform_files, reference_dir, reference_files):
    transform_image=np.load(baseDir+transform_dir+transform_files[ind])
    print("maximum value of the image:", np.max(transform_image))
    #transform_image = cv.normalize(transform_image, None, 0, 255, cv.NORM_MINMAX)
    #transform_image= cv.cvtColor(transform_image, cv.COLOR_BGR2GRAY)
    # transform_image = cv.normalize(transform_image.astype('float'), None, 0.0, 1.0, cv.NORM_MINMAX)
    #for wrong color-mapped depth image only ================= use correct format next time
    #transform_image = map_color_bk(transform_image)

    
    # plt.imshow(transform_image)
    # plt.show()

    reference_image = np.load(baseDir+reference_dir+reference_files[ind])
    reference_image = reference_image.astype(np.float32)
    reference_image = cv.normalize(reference_image, None, 0, 255, cv.NORM_MINMAX)

    return transform_image, reference_image


def map_color_bk(colored_depth_image):
    colormap_jet = cv.applyColorMap(np.arange(256, dtype=np.uint8).reshape(1, 256), cv.COLORMAP_JET)
    colormap_jet = colormap_jet.squeeze()  # Shape: (256, 3)

    # Step 2: Create a reverse mapping from RGB to depth
    # We'll use a dictionary for fast lookup
    rgb_to_depth = {tuple(colormap_jet[i]): i for i in range(256)}

    # Step 3: Load the colored depth image
    #colored_depth_image = np.load("dep.npy")  # Replace with your image path

    # Step 4: Recover the depth image
    depth_image = np.zeros((colored_depth_image.shape[0], colored_depth_image.shape[1]), dtype=np.uint8)

    for i in range(colored_depth_image.shape[0]):
        for j in range(colored_depth_image.shape[1]):
            pixel_color = tuple(colored_depth_image[i, j])
            if pixel_color in rgb_to_depth:
                depth_image[i, j] = rgb_to_depth[pixel_color]
            else:
                # Handle unknown colors (e.g., use nearest neighbor or default value)
                depth_image[i, j] = 0  # Default to 0 (or handle differently)

    # Step 5: Scale the depth image back to original depth range
    # Assuming the depth was scaled by alpha=0.03 during the original mapping
    original_depth_image = depth_image / 0.03/1000
    return original_depth_image


# add margin to an image, with a scale. margin is number of pixels before scaling
def add_margin(reference_image, margin, scale):
    rmargin = round(margin/scale)
    new_h = reference_image.shape[0]+round(2*rmargin)
    new_w = reference_image.shape[1]+round(2*rmargin)
    padded_reference = np.full((new_h, new_w), 255, dtype=np.float32)
    padded_reference[rmargin:rmargin + reference_image.shape[0], rmargin:rmargin + reference_image.shape[1]] = reference_image
    reference_image = padded_reference
    return reference_image

def visualize_calib_result(transform_image, reference_image, mode, R, T, scale):
    global angle_slider, xshift_slider, yshift_slider, scale_slider, fig, ax1, ax2, cursor1, cursor2, cursor11, cursor21
    #visualize the transformed image
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    print("================init subplots done!=============================")

    transform_image_ori = copy.copy(transform_image)
    # plt.imshow(transform_image_ori)
    # plt.show()
    if (mode == "mlc"): #multi-layer calib
        print("multi-layer calib!")
        transform_image = transform_image_layered(basedir1, maxlen1, depth_ori1)
    else:
        print("normal calib!")
        transform_image = transform_img(transform_image, R, T, scale)
    
    trnasform_image = cv.applyColorMap(cv.convertScaleAbs(transform_image, alpha=0.03), cv.COLORMAP_JET)
    im = ax1.imshow(transform_image)
    ax1.set_title('transform Image')

    reference_image_ori = reference_image
    reference_image = reference_image #MLX_process(reference_image)
    ax2.imshow(reference_image)
    ax2.set_title('reference Image')

    print("==========done initializing two axes=================")

    # Create axes for the sliders=====================
    ax_angle = plt.axes([0.25, 0.25, 0.65, 0.03])
    ax_xshift = plt.axes([0.25, 0.20, 0.65, 0.03])
    ax_yshift = plt.axes([0.25, 0.15, 0.65, 0.03])
    ax_scale = plt.axes([0.25, 0.10, 0.65, 0.03])

    # Create sliders
    angle_slider = Slider(ax_angle, 'Angle', -math.pi/2, math.pi/2, valinit=math.asin(R[1][0]))
    xshift_slider = Slider(ax_xshift, 'X Shift', T[0]-20, T[0]+20, valinit=T[0])
    yshift_slider = Slider(ax_yshift, 'Y Shift', T[1]-20, T[1]+20, valinit=T[1])
    scale_slider = Slider(ax_scale, 'Scale', scale-0.2, scale+0.2, valinit=scale)


    # Attach the update function to the sliders
    angle_slider.on_changed(update)
    xshift_slider.on_changed(update)
    yshift_slider.on_changed(update)
    scale_slider.on_changed(update)

    # sliders done==============================================


    #curser===================================================
    #1. add cursor to show corresponding points
    cursor1, = ax1.plot([], [], 'r+')
    cursor2, = ax2.plot([], [], 'r+')

    cursor11, = ax1.plot([], [], 'r+')
    cursor21, = ax2.plot([], [], 'r+')
    fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)
    #2. layout and show
    plt.tight_layout()
    plt.title("Transformed reference Image")
    plt.show()
    # cursor done=================================================

    #cv.imshow("transformed:", transform_image_ori)
    print(transform_image_ori.shape)
    #plt.clear()
    # show overlapping image
    if mode != "mlc":
        transform_image = transform_img(transform_image_ori, R, T, scale)
    else:
        #transform_image = transform_image_layered(basedir1, maxlen1, depth_ori1)
        pass
    print(transform_image.shape)
    plt.imshow(transform_image, cmap='gray', alpha=0.5)
    plt.imshow(reference_image, cmap='gray', alpha=0.5)
    plt.xlim(-10, 250)
    plt.ylim(200, -10)
    plt.show()



if __name__=="__main__":
    global mode
    margin = 0
    # prepare arguments =====================================================================
    
    src_distance = "4" #the distance where R, T, scale come from
    dest_distance = "4" #which distance we want to adjust our RTS to(e.g. we can read calib result at 7m, transform it to use at 1m)
    baseDir = "/media/zx/zx-data/RawData/exp06/"
    transform_dir = "realsense_depth/"
    reference_dir = "MLX/"
    ind = 300 #index of the image we want to visualize. 1 means 2nd valid image
    # mode = "adjust" # adjust previous R, T, S
    # mode = "pointcalib"
    mode = "mlc" #multi-layer calib

    # read data, get names of files =============================================================
    R, T, scale = read_RTS(src_distance, reference_dir, mode)
    RTSfileDst = "calibresults/"+reference_dir+dest_distance+".yaml"
    transform_files = os.listdir(baseDir+transform_dir)
    reference_files = os.listdir(baseDir+reference_dir)

    print("==========starting image transform===========")
    # map the transform array to reference image
    #0. load the to-be-conferted image and reference image================================================================
    transform_image, reference_image = load_image(baseDir,transform_dir,transform_files,reference_dir,reference_files)
    print("==========image transform done==========")
    # print(transform_image)
    # plt.imshow(transform_image)
    # plt.show()

    # 0. add margin for transform ==========================================================
    reference_image = add_margin(reference_image, margin, scale)
    transform_image = add_margin(transform_image, margin, 1)
    transform_image_ori = copy.copy(transform_image)

    # plt.imshow(transform_image)
    # plt.show()

    #1. transform the transform image ==============================================================================
    basedir1 = "calibresults/"+reference_dir
    maxlen1 = "4"
    depth_ori1 = transform_image
    #transform_image = transform_img(transform_image, R, T, scale) #for debugging
    print("==========prepare to show the transformed image===========")
    print(transform_image)
    # cv.imshow("transformed:", transform_image)
    # cv.waitKey(1000)
    # plt.imshow(transform_image)
    # plt.show()

    visualize_calib_result(transform_image, reference_image, mode, R, T, scale)
    if mode != "mlc":
        to_save = input("save R, T and s? y/n")
        if to_save=="y":
            dump_yaml(R, T, scale, RTSfileDst)

    
