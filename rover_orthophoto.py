from pathlib import Path
import argparse
import sys
from support.test_docker import test_docker

#----- ADD YOUR IMPORTS HERE IF NEEDED -----
import cv2 
import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from pathlib import Path
import os
from numpy.linalg import inv
from scipy.interpolate import griddata
import math
import imageio as io

def loadInfo():
    """
    Loads the provided intrinsic and extrinsic camera information provided in the .txt
    files in the run data.
    
    Parameters:
    -----------

    Returns:
    --------
    allOmni     - an array of 10 elements, each being an array itself
                  of the camera intrinsics of camera n[1...10]
    allT_OrefP  - an array of 10 elements, each being the transform of the 
                  individual omni camera to the OmniSensor frame
    """
    
    """
    INTRINSIC CAMERA CALIBRATION PARAMETERS
    The coefficients fx and fy are the focal lengths and the coefficients cx and cy are the camera centers
    The coefficients k1, k2, p1, p2 and k3 indicate radial (k's) and tangential (p's) distorsion
    fx [px], fy [px], cx [px], cy [px], k1, k2, p1, p2, k3
    """
    mono = np.array([904.04572636,907.01811462,645.74398382,512.14951996,-0.3329137,0.10161043,0.00123166,-0.00096204])
    omni0 = np.array([482.047,485.211,373.237,211.02,-0.332506,0.154213,-9.5973e-05,-0.000236179,-0.0416498])
    omni1 = np.array([479.429,482.666,367.111,230.626,-0.334792,0.161382,4.29188e-05,-0.000324466,-0.0476611])
    omni2 = np.array([483.259,486.027,340.948,204.701,-0.334384,0.15543,0.000171604,0.000300507,-0.0439626])
    omni3 = np.array([483.895,486.584,375.161,220.184,-0.337111,0.160611,0.000146382,0.000406074,-0.0464726])
    omni4 = np.array([473.571,477.53,378.17,212.577,-0.333605,0.159377,6.11251e-05,4.90177e-05,-0.0460505])
    omni5 = np.array([473.368,477.558,371.65,204.79,-0.3355,0.162877,4.34759e-05,2.72184e-05,-0.0472616])
    omni6 = np.array([476.784,479.991,381.798,205.64,-0.334747,0.162797,-0.000305541,0.000163014,-0.0517717])
    omni7 = np.array([480.086,483.581,361.268,221.179,-0.348515,0.199388,-0.000381909,8.83314e-05,-0.0801161])
    omni8 = np.array([478.614,481.574,377.363,194.839,-0.333512,0.157163,-8.2852e-06,0.000265461,-0.0447446])
    omni9 = np.array([480.918,484.086,386.897,206.923,-0.33305,0.156207,-5.95668e-05,0.000376887,-0.0438085])
    
    """
    ROVER TRANSFORMS
    trans_x [m], trans_y [m], trans_z [m], quat_x, quat_y, quat_z, quat_w
    """  
    #Omnidirectional camera 0 relative to Omnidirectional sensor
    T_OrefP0 = np.array([0.000,0.004,0.056,0.002,0.001,-0.006,1.000])
    #Omnidirectional camera 1 relative to Omnidirectional sensor
    T_OrefP1 = np.array([-0.001,0.127,0.054,0.005,0.002,-0.002,1.000])
    #Omnidirectional camera 2 relative to Omnidirectional sensor
    T_OrefP2 = np.array([0.060,0.005,0.023,-0.000,0.585,-0.010,0.811])
    #Omnidirectional camera 3 relative to Omnidirectional sensor
    T_OrefP3 = np.array([0.059,0.128,0.020,-0.006,0.586,-0.007,0.810])
    #Omnidirectional camera 4 relative to Omnidirectional sensor
    T_OrefP4 = np.array([0.030,0.013,-0.046,0.006,0.950,-0.002,0.311])
    #Omnidirectional camera 5 relative to Omnidirectional sensor
    T_OrefP5 = np.array([0.032,0.134,-0.047,0.019,0.951,-0.011,0.309])
    #Omnidirectional camera 6 relative to Omnidirectional sensor
    T_OrefP6 = np.array([-0.033,0.009,-0.048,-0.006,0.951,0.002,-0.310])
    #Omnidirectional camera 7 relative to Omnidirectional sensor
    T_OrefP7 = np.array([-0.034,0.131,-0.048,-0.012,0.951,0.001,-0.310])
    #Omnidirectional camera 8 relative to Omnidirectional sensor 
    T_OrefP8 = np.array([-0.056,0.005,0.017,0.004,-0.587,0.005,0.809])
    #Omnidirectional camera 9 relative to Omnidirectional sensor 
    T_OrefP9 = np.array([-0.057,0.128,0.015,0.002,-0.586,0.008,0.810])
    # Front left wheel relative to Rover, T_RWfl,0.256,0.285,0.033,-,-,-,-
    # Front right wheel relative to Rover, T_RWfr,0.256,-0.285,0.033,-,-,-,-
    # Rear left wheel relative to Rover, T_RWrl,-0.256,0.285,0.033,-,-,-,-
    # Rear right wheel relative to Rover, T_RWrr,-0.256,-0.285,0.033,-,-,-,-
    
    # all omnidirectional camera intrinsics
    allOmni = np.array([omni0, omni1, omni2, omni3, omni4, omni5, omni6, omni7, omni8, omni9]) 
    # all omnidirectional camera transforms
    allT_OrefP = np.array([T_OrefP0, T_OrefP1, T_OrefP2, T_OrefP3, T_OrefP4, T_OrefP5, T_OrefP6, T_OrefP7, T_OrefP8, T_OrefP9]) 

    return allOmni, allT_OrefP

def quat2rot(quat):
    """
    Converts rotations specified by quaternions to the rotation matrix form.
    
    Parameters:
    -----------
    quat        - a 4 element numpy array which each element as a quaternion element

    Returns:
    --------
    C           - a 3x3 element numpy array that is the rotation matrix
    """
    
    q0 = quat[0]
    q1 = quat[1]
    q2 = quat[2]
    q3 = quat[3]
    C = np.array([[1-2*q2**2-2*q3**2, 2*q1*q2-2*q0*q3, 2*q0*q2+2*q1*q3],
                  [2*q0*q3+2*q1*q2, 1-2*q1**2-2*q3**2, 2*q2*q3-2*q0*q1], 
                  [2*q1*q3-2*q0*q2, 2*q0*q1+2*q2*q3, 1-2*q1**2-2*q2**2]])
    return C
    
def getIntrinsicMat(f_x, f_y, c_x, c_y):
    """
    Returns the Camera intrinsics matrix.
    
    Parameters:
    -----------
    f_x     - focal length (x)
    f_y     - focal length (y)
    c_x     - camera center (x)
    c_y     - camera center (y)

    Returns:
    --------
    The camera intrinsics matrix
    """
    K = np.array([[f_x,0,0],[0,f_y,0],[c_x,c_y,1]]).T
    return K

def getOMNI2GPS(sens2omnicam):
    """
    Returns the transform from the omnidirectional camera to the GPS frame.
    
    Parameters:
    -----------
    sens2omnicam  - the transform from the OmniSensor frame to the 
                      omnidirectional camera frame

    Returns:
    --------
    H               -The transform from the omnidirectional camera to the GPS frame
    """
    
    '''
    rover transforms are in this form: 
        trans_x [m], trans_y [m], trans_z [m], quat_x, quat_y, quat_z, quat_w
    '''
    # GPS relative to Rover, gps2rov = T_RG, taken from the .txt files from this
    # repository
    gps2rov = np.array([-0.260,0.000,0.340,0,0,0,0])
    # Omnidirectional sensor relative to Rover, rov2sensor = T_ROref 
    # taken from the .txt files from this repository
    rov2sensor = np.array([0.236,-0.129,0.845,-0.630,-0.321,0.321,0.630])
    
    H1 = getHomog(gps2rov)
    H2 = getHomog(rov2sensor)
    H3 = getHomog(sens2omnicam)
    
    H = H1*H2*H3
    
    return H

def getOMNI2Rover(sens2omnicam):
    """
    Returns the transform from the omnidirectional camera frame to the rover frame.
    
    Parameters:
    -----------
    sens2omnicam    - the transform from the OmniSensor frame to the 
                      omnidirectional camera frame

    Returns:
    --------
    H               - The transform from the omnidirectional camera to the rover frame
    """
    
    '''
    rover transforms are in this form: 
        trans_x [m], trans_y [m], trans_z [m], quat_x, quat_y, quat_z, quat_w
    '''
    # Omnidirectional sensor relative to Rover, rov2sensor = T_ROref 
    rov2sensor = np.array([0.236,-0.129,0.845,-0.630,-0.321,0.321,0.630])
    
    H1 = getHomog(rov2sensor)
    H2 = getHomog(sens2omnicam)
    H = H1*H2
    return H

def getHomog(pos):
    """
    Get the Homogenous transform specified by elements in the pose array.
    
    Parameters:
    -----------
    pos  - a 7 element numpy array, the first 3 elements specify a translation, 
           the last 4 elements specific a rotation in the form of quaternions

    Returns:
    --------
    H    - The homogenous transform specificed by the translation and rotation
           from the 'pos' input parameter
    """
    
    R = quat2rot(pos[3:]) # get the rotation matrix
    T = pos[:3] # extract the translation element
    lstrow = np.array([0,0,0,1]) # specify the last row of the homogenous tranformation
    H = np.insert(np.insert(R, 3, T, axis=1), 3, lstrow, axis=0)
    return H

def bilinear_interp_vectorized(I, pt):
    """
    Performs bilinear interpolation for a given set of image points.

    Given the (x, y) location of a point in an input image, use the surrounding
    4 pixels to compute the bilinearly-interpolated output pixel intensity.

    This function is for a *single* image band only - for RGB images, you will 
    need to call the function once for each colour channel.

    Parameters:
    -----------
    I   - Single-band (greyscale) intensity image, 8-bit np.array (i.e., uint8).
    pt  - 2xn np.array of point in input image (x, y), with subpixel precision.

    Returns:
    --------
    b  - Interpolated brightness or intensity value (whole number >= 0).
    """

    if pt.shape != (2, 1):
        # if the input is a vector of multiple points with the first col being
        # all the x coordinate values, and the second col being all the y 
        # coordinate values
        x_col = pt[:, 0] # grab the raw x values from the input (potentially a decimal)
        y_row = pt[:, 1] # grab the raw y values from the input (potentially a decimal)
        pt = np.floor(pt).astype(int) # round down to get valid pixel coordinates
        col = pt[:, 0] # grab all the rounded x values
        row = pt[:, 1] # grab all the rounded y values
    
        # Start with iterpolation in the x-direction:
        f_1 = (col+1-x_col)/(col+1-col)*I[row, col] + (x_col-col)/(col+1-col)*I[row, col+1]
        f_2 = (col+1-x_col)/(col+1-col)*I[row+1, col] + (x_col-col)/(col+1-col)*I[row+1, col+1]
        # Then interpolate in the y-direction to get the desired estimate:
        b = np.round((row+1-y_row)/(row+1-row)*f_1 + (y_row-row)/(row+1-row)*f_2)

    else:
        # if the input is a single point, i.e. a 2x1 np.array with (x, y) coords
        x_col = pt[0].item() # grab the raw x value from the input (potentially a decimal)
        y_row = pt[1].item() # grab the raw y value from the input (potentially a decimal)
        pt = np.floor(pt).astype(int) # round down to get a valid pixel coordinate
        col = pt[0].item() # grab the rounded x value
        row = pt[1].item() # grab the rounded y value
    
        # Start with iterpolation in the x-direction:
        f_1 = (col+1-x_col)/(col+1-col)*I[row, col] + (x_col-col)/(col+1-col)*I[row, col+1]
        f_2 = (col+1-x_col)/(col+1-col)*I[row+1, col] + (x_col-col)/(col+1-col)*I[row+1, col+1]
        # Then interpolate in the y-direction to get the desired estimate:
        b = round((row+1-y_row)/(row+1-row)*f_1 + (y_row-row)/(row+1-row)*f_2)

    return b

def get3d22d(rawpts, allOmni, n):
    """
    Transforms 3D point cloud data to 2D points on a plane.

    Parameters:
    -----------
    rawpts   - raw 3D point cloud data from omnidirectional camera 'n'
    allOmni  - the array of all 'n' omnicamera intrinsics
    n        - the nth omni-directional camera [0-9] as an (int)

    Returns:
    --------
    X,Y      - X, Y coordinates of the 3D point cloud points projected 
               into the image plane of the omnicamera 'n'
    """
    
    # extract the x, y, and z
    x = rawpts[:, 0]
    y = rawpts[:, 1]
    z = rawpts[:, 2]
    
    # get the focal lengths and centers for the omnicamera in question
    f_x = allOmni[n][0]
    f_y = allOmni[n][1]
    c_x = allOmni[n][2]
    c_y = allOmni[n][3] 
    
    # backproject the points in the point cloud to the image plane of the 
    # omnidirectional camera in question
    proj = np.array([f_x*x/z + c_x, f_y*y/z+ c_y, z/z]).T 
    X = proj[0:, 0] 
    Y = proj[0:, 1] 
    
    return X, Y

def getOmniCamData(n, fileid, frame, input_dir, main):
    """
    Returns the nth omni-directional camera image and the corresponding point cloud.

    Parameters:
    -----------
    n         - the nth omni-directional camera [0-9] as an (int)
    fileid    - the number component of the filename in a string ex. 
    frame       the files 
                omnicam img:            "frame000978_2018_09_04_18_18_15_891197.png"
                corresponding pt cloud: "cloud000978_2018_09_04_18_18_15_891197.txt"
                would have fileid:      "2018_09_04_18_18_15_891197"
                and frame:              "000978"
    input_dir - the input directory name as a string

    Returns:
    --------
    X,Y       - X, Y coordinates of the 3D point cloud points projected 
                into the image plane of the omnicamera 'n'
    """

    # Print Statment for Debugging Image reading with Docker
    
    # print("This Directory: {}".format(THIS_FOLDER))
    # print("This Directory CONTENTS: {}".format(os.listdir(THIS_FOLDER)))

    # print("One Directory UP: {}".format(main))
    # print("One Directory UP CONTENTS: {}".format(os.listdir(main)))
    
    # print("This the input folder directory: {}".format(input_folder))
    # print("This is the contents of the input folder: {}".format(os.listdir(input_folder)))
    
    # print("The omni cam folder: {}".format(input_folder+"/run4_base_hr/run4_base_hr/omni_image{}".format(n)))
    # print("The contents of the omni cam folder: {}".format(os.listdir(input_folder+"/run4_base_hr/run4_base_hr/omni_image{}".format(n))))

    # get the image
    folder = input_dir + "/run4_base_hr/omni_image{}".format(n)
    file_to_open = folder + "/frame{}_{}.png".format(frame, fileid)
    
    # define the intrinsics, extrinsics, and distortion coefficients for this camera
    img = imread(file_to_open) # Read the test img
    
    # Get the point cloud
    folder = input_dir + "/run4_clouds_only_hr/omni_cloud{}/".format(int(np.floor(n/2)))
    file_to_open = folder + "/cloud{}_{}.txt".format(frame, fileid)
    #print(file_to_open.read_text())
    
    # read in the 3D point cloud 
    rawpts = np.loadtxt(file_to_open, comments="#", delimiter=",", unpack=False)
    
    return img, rawpts

def transformGrid(transform, grid_x, grid_y):
    """
    Transform the grid for the orthophoto to the another frame (specified
    by 'transform').
    We need this because the orthophotos grids are in the frame of their 
    respective stereo cameras. By transforming the grid you with this
    function, you have the coordinates in meters of each pixel in the 
    orthophoto with respect to the a common frame (which you get to by
    applying the 'transform' which is given as an input to this function). 
    
    Parameters:
    -----------
    transform   - a 4x4 numpy matrix which specifies a homogenous transform
                  to a desired frame
    grid_x      - x elements of (x,y) coordinate pairs which are the positions
                  of each RGB pixel in the 'rgb' orthophoto image 
                  in the frame of the respective omni camera frame
    grid_y      - same as grid_x but for y elements

    Returns:
    --------
    ptcld_rover - uniform grid point cloud transformed to the specifed frame
    """
    # reshape the 2D grid into a 1D list of all the points
    # the reshape function does this in the order
    # left to right, top to bottom
    grid_x = grid_x.reshape((grid_x.size, 1))
    grid_y = grid_y.reshape((grid_y.size, 1))
    grid_z = np.zeros((grid_x.size, 1))
    
    # this is the uniform grid I generated of points in the image, with repect to the omni camera
    ptsInOmniCam = np.hstack([grid_x, -grid_z, grid_y])
    ones = np.ones((len(ptsInOmniCam), 1))
    ptsInOmniCam_aug = np.hstack([ptsInOmniCam, ones])
    
    # the uniform grid point cloud transformed to the rover frame
    # obtained by multuplying by the transformation
    ptcld_rover = transform.dot(ptsInOmniCam_aug.T)
    
    return ptcld_rover.T

def transformPose(transform, x, y, C):
    """
    Transform a point orientation and position the orthophoto to the another 
    frame (specified by 'transform'). Similar to transformGrid.
    
    Parameters:
    -----------
    transform   - a 4x4 numpy matrix which specifies a homogenous transform
                  to a desired frame
    x           - x element of (x,y) coordinate pair, which is the position
                  of each RGB pixel in the 'rgb' orthophoto image
                  in the frame/the frame itself the transform is starting from 
    y           - same as x but for y element
    C           - a matrix with columns being x, y, z vectors representing
                  the orientation of the frame

    Returns:
    --------
    transpt     - 2x1 numpy array of the (x,y) point in the transformed frame
    """
    ptInOmniCam = np.hstack([x, -0, y]).reshape(3,1)
    tmp = np.hstack([C, ptInOmniCam])
    pose = np.vstack([tmp, [0,0,0,1]])
    # the uniform grid point cloud transformed to the rover frame
    # obtained by multuplying by the transformation
    transpt = transform.dot(pose)
    
    return transpt.T

def compress(grid_x, grid_y):
    """
    Given a grid_x and grid_y of length 'p', we can compress then into a 2xp
    numpy array. 
    
    Parameters:
    -----------
    grid_x      - x elements of (x,y) coordinate pairs which are the positions
                  of each RGB pixel in the 'rgb' orthophoto image 
                  in the frame of the respective omni camera frame
    grid_y      - same as grid_x but for y elements

    Returns:
    --------
    comp        - the combined 'columnized' grid_x and grid_y
    """
    # reshape the 2D grid into a 1D list of all the points
    # the reshape function does this in the order
    # left to right, top to bottom
    grid_x = grid_x.reshape((1, grid_x.size))
    grid_y = grid_y.reshape((1, grid_y.size))
  
    comp = np.vstack([grid_x, grid_y]).T
    
    return comp

def getOrthophoto(n, fileid, frame, input_dir, allOmni, main):
    """
    Gets the Orthophoto/BEV for omni camera 'n' at a specific timeframe.
    
    Parameters:
    -----------
    n        - the nth omni-directional camera [0-9] as an (int).
    fileid   - the number component of the filename in a string ex. 
    frame      the files 
               omnicam img:            "frame000978_2018_09_04_18_18_15_891197.png"
               corresponding pt cloud: "cloud000978_2018_09_04_18_18_15_891197.txt"
               would have fileid:      "2018_09_04_18_18_15_891197"
               and frame:              "000978"
    allOmni  - the array of all 'n' omnicamera intrinsics

    Returns:
    --------
    img      - the original omnicamera image
    rgb      - the RGB orthophoto image   
    grid_x   - x elements of (x,y) coordinate pairs which are the positions
               of each RGB pixel in the 'rgb' orthophoto image 
               in the frame of the respective omni camera frame
    grid_y   - same as grid_x but for y elements
    """
    ###########################################################################
    # get the omnicamera image and the 3D point cloud for the 'n'th camera in question
    imgr, rawpts = getOmniCamData(n, fileid, frame, input_dir, main)
    img  = (sunCorrection(imgr)) # correct for the sun's effects, i.e. preprocess
    # plt.imshow(img) # plot the raw image
  
    # get the camera intrinsics
    intrin = getIntrinsicMat(allOmni[n][0], allOmni[n][1], allOmni[n][2], allOmni[n][3]) 
    # get the distortion coefficients
    distCoeff = allOmni[n][4:] 
    # undistort the image, i.e. correct for lens effects
    dst = cv2.undistort(img,intrin,distCoeff) 
    # project the 3D point cloud into the image plane of the 
    # omnicamera
    X, Y = get3d22d(rawpts, allOmni, n) 
    # X, Y are the x, y position of each 3D point in the point cloud 
    # (with the X, Y plane being the ground)
    # NOTE: the X, Y corresponsed to the x,z directions in the omnicamera frame
    # careful this is confusing, refer to report figures for clarification***
    
    # plt.scatter(X, Y, marker=',')
    #v = pptk.viewer(np.array([rawpts[:, 0], rawpts[:, 1], rawpts[:, 2]]).T)
    #v = pptk.viewer(proj)
    ###########################################################################
    # remove all points that aren't in the range of the 2D 
    # omnidirectional camera image plane
    h = dst.shape[0] - 10 # get height of the omni image
    w = dst.shape[1] - 10 # get width of the omni image
    zero = 10
    # Can eliminate points from the point cloud list of points if:
    # 1) X is less than the defined 'zero'
    # 2) X is greater than the width of the omnidirectional image we are 
    #    interpolating from
    # 3) Y is less than the defined 'zero'
    # 4) Y is greater than the height of the omnidirectional image we are 
    #    interpolating from 
    mask = (X>zero)*(X<w)*(Y>zero)*(Y<h) # use a boolean mask to do this
    xin = mask*X
    yin = mask*Y
    xin = xin[xin != 0]
    yin = yin[yin != 0]
    
    # plt.imshow(dst) 
    # plt.scatter(xin, yin, marker=',')
        
    # Bilinearly interpolate the R, G, B pixel value for each of the 
    # 3D point cloud points (which have been transformd to the 2D image place)
    c = np.vstack([xin, yin]).T # all the 2d transformed 3D points in the point cloud
    r_interp = (bilinear_interp_vectorized(dst[:, :, 0], c))#.reshape(totpx_bg, 1)*b)
    g_interp = (bilinear_interp_vectorized(dst[:, :, 1], c))#.reshape(totpx_bg, 1)*b)
    b_interp = (bilinear_interp_vectorized(dst[:, :, 2], c))#.reshape(totpx_bg, 1)*b)
    
    # at this point you have the RGB value of each point in the point cloud that 
    # is in the actual image taken by the omnidirectional camera in question
    ##########################################################################
    # Now do a similar thing but for the raw 3d points (above we did it for 
    # the 2D transformed points in the omnidirectional camera image plane!)
    mask_r = np.array([mask*1, mask*1, mask*1]).T # make a mask that can be multplied by the raw points
    # this mutiplication will eliminate (make zero) any points that 
    # aren't in the stereo photo
    rawpts_in = np.multiply(mask_r, rawpts)  
    x = rawpts_in[:, 0] 
    y = rawpts_in[:, 1]
    z = rawpts_in[:, 2]
    # get rid of the zero entries
    x = x[x != 0] 
    y = y[y != 0]
    z = z[z != 0]
    # plt.scatter(x, z, marker=',')
    
    # get the following quantities:
    # x min
    xmin = np.min(x)
    # x max
    xmax = np.max(x)
    # y min
    ymin = np.min(z)
    #ymin = 0.2
    # y max
    ymax = np.max(z)
    # x min
    zmin = np.min(-y)
    # x max
    zmax = np.max(-y)
    
    # plt.plot(xmin, ymin, marker='o', color='r')
    # plt.plot(xmax, ymax, marker='o', color='r')
    
    
    ###########################################################################
    
    
    # spacing factor for generating the orthogonal pixel grid of the 
    # resulting orthophoto
    sp_fc = 100 
    rangex = int((xmax - xmin)*sp_fc)
    rangey = int((ymax - ymin)*sp_fc)
    rangez = int((zmax - zmin)*sp_fc)
    # generate points to interpolate, these quantities are in meters 
    # in real 3D coordinates, x, z with respect to the omnicamera frame
    # x, y plane if you consider the ground to be the x-y plane
    allx = np.linspace(xmin, xmax, rangex)
    ally = np.linspace(ymin, ymax, rangey)
    allz = np.linspace(zmin, zmax, rangez)

    grid_x, grid_y = np.meshgrid(allx, ally)
    # the 3D point cloud points that are in the iamge
    # NOTE: we are using x and x, and z as y, *** this is because of the
    # difference between the camera frame (x,z frame is the physical ground)
    # and the orthophoto frame (x,y plane being the physical ground)
    points = np.array([x, z]).T 


    # use the griddata function to interpolate the r, g, b channels 
    # respectively for the image
    # point are the (x, y) points and [r,g,b]_interp are the corresponding 
    # pixel value of each of those points
    r = griddata(points, r_interp, (grid_x, grid_y), method='linear')
    g = griddata(points, g_interp, (grid_x, grid_y), method='linear')
    b = griddata(points, b_interp, (grid_x, grid_y), method='linear')
    
    # get rid of all nan values, these nan values are ones
    # that could not be interpolated because they are not in the point cloud
    # (due to view of the omnicameras to generate the 3D point cloud)
    r[np.isnan(r)]=0
    g[np.isnan(g)]=0
    b[np.isnan(b)]=0
    
    # make sure all the pixel values are integers
    r = r.astype(int)
    g = g.astype(int)
    b = b.astype(int)
    
    # pack up the points
    points = np.array([x, y, z]).T
    
    # pack up the channels to make the RGB image orthophoto
    rgb = np.dstack([r, g, b])
    
    return imgr, rgb, grid_x, grid_y

def surroundOrthophoto(rgb0, transGrid0, rgb1, transGrid1, rgb2, transGrid2, rgb3, transGrid3, rgb4, transGrid4):
    """
    Gets the stiched surrounding orthophoto given all the individual orthophotos at a given time.
    
    Parameters:
    -----------
    rgb[0,1,2,3,4]          - 5 RGB images of the orthophoto from each stero omnicamera pair.
    transGrid[0,1,2,3,4]    - a 4xp array (p being the number of individual pixels in each 
                              rgb[0,1,2,3,4] orthophoto respectively) with the first 3 rows being all the 
                              x, y, z coordinate pairs representing the position (in meters) of each pixel
                              in the orthophoto, in the omniSensor frame                          
                              i.e. these quantities tells you where each pixel of the 5 orthophotos 
                              are with respect to the same frame, such that you know where they should be 
                              placed in the final combined birds eye view all around the rover
    Returns:
    --------
    ALLPXrgb               - the birds eye view surrounding the rover (combined all the orthophotos together)
    grid_x                 - x elements of (x,y) coordinate pairs which are the positions
                             of each RGB pixel in the 'rgb' STICHED orthophoto image 
                             in the shared frame of the omnisensor frame
    grid_y                 - same as grid_x but for y elements
    """
    
    rgb0_c = rgb0
    rgb1_c = rgb1
    rgb2_c = rgb2
    rgb3_c = rgb3
    rgb4_c = rgb4
    
    # plot the transformed grids transGrid[n] 
    # the transGrid[n] contains the x,y information in the frame of the OmniSensor
    # of each individual *** pixel *** in the orthophoto rgb[n] for the nth 
    # omnidirectional camera
    
    # plt.plot(transGrid1[:, 2], -transGrid1[:, 0], color='green')
    # plt.plot(transGrid2[:, 2], -transGrid2[:, 0], color='blue')
    # plt.plot(transGrid0[:, 2], -transGrid0[:, 0], color='red')
    # plt.plot(transGrid3[:, 2], -transGrid3[:, 0], color='magenta')
    # plt.plot(transGrid4[:, 2], -transGrid4[:, 0], color='yellow')
    
    # reshape the images into a list of their component pixels
    rgb0_lstpts = rgb0_c.reshape(rgb0_c.shape[0]*rgb0_c.shape[1], rgb0_c.shape[2])
    rgb1_lstpts = rgb1_c.reshape(rgb1_c.shape[0]*rgb1_c.shape[1], rgb1_c.shape[2])
    rgb2_lstpts = rgb2_c.reshape(rgb2_c.shape[0]*rgb2_c.shape[1], rgb2_c.shape[2])
    rgb3_lstpts = rgb3_c.reshape(rgb3_c.shape[0]*rgb3_c.shape[1], rgb3_c.shape[2])
    rgb4_lstpts = rgb4_c.reshape(rgb4_c.shape[0]*rgb4_c.shape[1], rgb4_c.shape[2])
    
    # get rid of all 0 values 
    res0 = (rgb0_lstpts[:,0] != 0)*(rgb0_lstpts[:,1] != 0)*(rgb0_lstpts[:,2] != 0)
    res1 = (rgb1_lstpts[:,0] != 0)*(rgb1_lstpts[:,1] != 0)*(rgb1_lstpts[:,2] != 0)
    res2 = (rgb2_lstpts[:,0] != 0)*(rgb2_lstpts[:,1] != 0)*(rgb2_lstpts[:,2] != 0)
    res3 = (rgb3_lstpts[:,0] != 0)*(rgb3_lstpts[:,1] != 0)*(rgb3_lstpts[:,2] != 0)
    res4 = (rgb4_lstpts[:,0] != 0)*(rgb4_lstpts[:,1] != 0)*(rgb4_lstpts[:,2] != 0)
    
    # get rid of all the grid values that don't matter (i.e. correspond to zero pixels)
    transGrid0_cut = transGrid0[res0]
    transGrid1_cut = transGrid1[res1]
    transGrid2_cut = transGrid2[res2]
    transGrid3_cut = transGrid3[res3]
    transGrid4_cut = transGrid4[res4]
    
    # get rid of all the pixel values that don't matter (i.e. are zero)
    rgb0_lstpts_cut = rgb0_lstpts[res0]
    rgb1_lstpts_cut = rgb1_lstpts[res1]
    rgb2_lstpts_cut = rgb2_lstpts[res2]
    rgb3_lstpts_cut = rgb3_lstpts[res3]
    rgb4_lstpts_cut = rgb4_lstpts[res4]
    
    # display the alignment
    # plt.plot(transGrid0_cut[:, 0], transGrid0_cut[:, 2], color='red')
    # plt.plot(transGrid1_cut[:, 0], transGrid1_cut[:, 2], color='green')
    # plt.plot(transGrid2_cut[:, 0], transGrid2_cut[:, 2], color='blue')
    # plt.plot(transGrid3_cut[:, 0], transGrid3_cut[:, 2], color='magenta')
    # plt.plot(transGrid4_cut[:, 0], transGrid4_cut[:, 2], color='yellow')
    
    # display in image coordinates!!!!!
    # plt.plot(transGrid0_cut[:, 2], -transGrid0_cut[:, 0], color='red')
    # plt.plot(transGrid1_cut[:, 2], -transGrid1_cut[:, 0], color='green')
    # plt.plot(transGrid2_cut[:, 2], -transGrid2_cut[:, 0], color='blue')
    # plt.plot(transGrid3_cut[:, 2], -transGrid3_cut[:, 0], color='magenta')
    # plt.plot(transGrid4_cut[:, 2], -transGrid4_cut[:, 0], color='yellow')

    # Concatenate all quantities into a sigle array, preparation for the interpolation step
    allGrid = np.vstack([transGrid0_cut, transGrid1_cut, transGrid2_cut, transGrid3_cut, transGrid4_cut])
    x_pos = (np.vstack([transGrid0_cut, transGrid1_cut, transGrid2_cut, transGrid3_cut, transGrid4_cut])[:, 0]).reshape((len(allGrid), 1))
    y_pos = (np.vstack([transGrid0_cut, transGrid1_cut, transGrid2_cut, transGrid3_cut, transGrid4_cut])[:, 2]).reshape((len(allGrid), 1))    
    
    allImgPx = np.vstack([rgb0_lstpts_cut, rgb1_lstpts_cut, rgb2_lstpts_cut, rgb3_lstpts_cut, rgb4_lstpts_cut])

    # x,y,z RGB
    ALLPX = np.hstack([x_pos, y_pos, allImgPx])
    
    ###########################################################################
    # Extract and rename quantities for ease of use
    x = ALLPX[:, 0]
    y = ALLPX[:, 1]
    R = ALLPX[:, 2]
    G = ALLPX[:, 3]
    B = ALLPX[:, 4]
    
    # x min
    xmin = np.min(x)
    # x max
    xmax = np.max(x)
    # y min
    ymin = np.min(y)
    # y max
    ymax = np.max(y)
    
    # generate the grid
    sp_fc = 100# spacing factor
    rangex = int((xmax - xmin)*sp_fc)
    rangey = int((ymax - ymin)*sp_fc)
    # generate points to interpolate, these quantities are in m in real 3D coordinates, x, z with respect to the omni camera
    # x, y plane if you consider the ground to be the x-y plane
    allx = np.linspace(ymin, ymax, rangey)
    ally = np.linspace(xmin, xmax, rangex)
    
    #generate the mesh grid
    grid_x, grid_y = np.meshgrid(allx, ally)
    points = np.array([y,x]).T

    # interpolate the pixel values on the grid
    r = griddata(points, R, (grid_x, grid_y), method='linear')
    g = griddata(points, G, (grid_x, grid_y), method='linear')
    b = griddata(points, B, (grid_x, grid_y), method='linear')
    
    # in the interpolation some values will be nans, remove them.
    r[np.isnan(r)]=0
    g[np.isnan(g)]=0
    b[np.isnan(b)]=0
    
    # make sure all quantities are integers to make it a proper image
    r = r.astype(int)
    g = g.astype(int)
    b = b.astype(int)
    
    # form the full RGB image
    ALLPXrgb = np.dstack([r,g,b])
    
    return grid_x, grid_y, ALLPXrgb

def sunCorrection(i):
    """
    Correct for the effect of the auto-exposure of the omnidirectional 
    camera so that when stiching the images together there is some consistency.
    
    Parameters:
    -----------
    i      - input image to be sun corrected (i.e. gamma correction)

    Returns:
    --------
    comp        - the combined 'columnized' grid_x and grid_y
    """
    # take only the bottom pixels (i.e. the ground pixels that will be relevant for the orthophoto/bird's eye view image)
    img = i[200:, :, :]     
    # convert img to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # compute gamma = log(mid*255)/log(mean)
    mid = 0.5
    mean = np.mean(gray)
    gamma = math.log(mid*255)/math.log(mean)
    # do gamma correction
    img_gammacorr = np.power(i, gamma).clip(0,255).astype(np.uint8)
    return img_gammacorr
    
def main(input_dir, output_dir):   
    print("running get_orthophoto...")
    allOmni, allT_OrefP = loadInfo()    
    
    # all the representative time points used to generate orthophotos 
    # shown in the final report 
    # 1) frame000000_2018_09_04_18_14_33_618401 (start pt)
    # 2) frame000111_2018_09_04_18_14_58_760015
    # 3) frame000233_2018_09_04_18_15_26_674852
    # 4) frame000811_2018_09_04_18_17_38_6072
    # 5) frame000978_2018_09_04_18_18_15_891197
    # 6) frame001445_2018_09_04_18_20_01_817314
    # 7) frame001579_2018_09_04_18_20_32_275748
    # 8) frame002010_2018_09_04_18_22_10_193947 (end pt)
    
    ###########################################################################
    # CHANGE ME TO GENERATE ORTHOPHOTOS FOR DIFFERNENT FRAMES!!!!
    # Can copy the frame and time point data from the selected 
    # representative 1) 2) ... 8) points from above!
    fileid = "2018_09_04_18_18_15_891197"
    frame = "000978"
    # define all the different camera views as n, we use these ones because those
    # are the ones the omni cameras point clouds were defined in
    alln = [0,2,4,6,8] 
    ###########################################################################
    # Get the individual orthophotos for each stereo camera image
    THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
    main = THIS_FOLDER
    #os.chdir(main) # Change the current working directory to be the file you are currently in
    for n in alln:
        print("getting orthophoto for omni-camera {}...".format(n))
        omni2sensor = getHomog(allT_OrefP[n]) #getOMNI2Rover(allT_OrefP[n])
        #omni2rover = getOMNI2Rover(allT_OrefP[n])

        if n == 0:
            img0, rgb0, grid_x, grid_y = getOrthophoto(n, fileid, frame, input_dir, allOmni, main)
            c0 = compress(grid_x, grid_y)
            rgb0 = np.flip(rgb0, axis=1)
            transGrid0 = transformGrid(omni2sensor, grid_x, grid_y) 
            #plt.plot(transGrid0[:, 0], transGrid0[:, 2], color='red')
            #plt.plot(c0[:, 0], c0[:, 1], color='red')
            #plt.imshow(cv2.rotate(rgb, cv2.cv2.ROTATE_180))
        if n == 2:
            img1, rgb1, grid_x, grid_y = getOrthophoto(n, fileid, frame, input_dir, allOmni, main)
            c1 = compress(grid_x, grid_y)
            rgb1 = np.flip(rgb1, axis=1)
            transGrid1 = transformGrid(omni2sensor, grid_x, grid_y) 
            #plt.plot(transGrid1[:, 0], transGrid1[:, 2], color='green')
        if n == 4:
            img2, rgb2, grid_x, grid_y = getOrthophoto(n, fileid, frame, input_dir, allOmni, main)
            c2 = compress(grid_x, grid_y)
            rgb2 = np.flip(rgb2, axis=1)
            transGrid2 = transformGrid(omni2sensor, grid_x, grid_y) 
            #plt.plot(transGrid2[:, 0], transGrid2[:, 2], color='blue')
        if n == 6:
            img3, rgb3, grid_x, grid_y = getOrthophoto(n, fileid, frame, input_dir, allOmni,main)
            c3 = compress(grid_x, grid_y)
            rgb3 = np.flip(rgb3, axis=1)
            transGrid3 = transformGrid(omni2sensor, grid_x, grid_y) 
            #plt.plot(transGrid3[:, 0], transGrid3[:, 2], color='magenta')
        if n == 8:
            img4, rgb4, grid_x, grid_y = getOrthophoto(n, fileid, frame, input_dir, allOmni, main)
            c4 = compress(grid_x, grid_y)
            rgb4 = np.flip(rgb4, axis=1)
            transGrid4 = transformGrid(omni2sensor, grid_x, grid_y) 
            #plt.plot(transGrid4[:, 0], transGrid4[:, 2], color='yellow')
    
    ###########################################################################
    # All the individual Orthophotos and their transformed grids
    indivOrthophotos = np.array([rgb0, rgb1, rgb2, rgb3, rgb4])
    allTransGrids = np.array([rgb0, rgb1, rgb2, rgb3, rgb4])
    # Get the stiched Orthophoto 
    allGrid_x, allGrid_y, ALLPXrgb = surroundOrthophoto(rgb0, transGrid0, rgb1, transGrid1, rgb2, transGrid2, rgb3, transGrid3, rgb4, transGrid4)
    
    print("The output directory where images are saved is: {}".format(output_dir))
    
    # print("This the input folder directory: {}".format(input_folder))
    # print("This is the contents of the input folder: {}".format(os.listdir(input_folder)))
    # plt.figure(1)
    # plt.imshow(img0)
    # plt.savefig('img0.png')
    cv2.imwrite(output_dir + "/img0.png", img0)
    # plt.figure(2)
    # plt.imshow(rgb0)
    # plt.savefig('rgb0.png')
    cv2.imwrite(output_dir + "/rgb0.png", rgb0)
    # plt.figure(3)
    # plt.imshow(img1)
    # plt.savefig('img1.png')
    cv2.imwrite(output_dir + "/img1.png", img1)
    # plt.figure(4)
    # plt.imshow(rgb1)
    # plt.savefig('rgb1.png')
    cv2.imwrite(output_dir + "/rgb1.png", rgb1)
    # plt.figure(5)
    # plt.imshow(img2)
    # plt.savefig('img2.png')
    cv2.imwrite(output_dir + "/img2.png", img2)
    # plt.figure(6)
    # plt.imshow(rgb2)
    # plt.savefig('rgb2.png')
    cv2.imwrite(output_dir + "/rgb2.png", rgb2)
    # plt.figure(7)
    # plt.imshow(img3)
    # plt.savefig('img3.png')
    cv2.imwrite(output_dir + "/img3.png", img3)
    # plt.figure(8)
    # plt.imshow(rgb3)
    # plt.savefig('rgb3.png')
    cv2.imwrite(output_dir + "/rgb3.png", rgb3)
    # plt.figure(9)
    # plt.imshow(img4)
    # plt.savefig('img4.png')
    cv2.imwrite(output_dir + "/img4.png", img4)
    # plt.figure(10)
    # plt.imshow(rgb4)
    # plt.savefig('rgb4.png')
    cv2.imwrite(output_dir + "/rgb4.png", rgb4)
    
    # plt.figure(11)
    # plt.imshow(ALLPXrgb)
    cv2.imwrite(output_dir + "/ALLPXrgb.png", ALLPXrgb)
    
    print("SAVED all images!!!")
    ###########################################################################
    
    # Get the global position

    # get the image    
    folder = input_dir + "/run4_base_hr/"
    file_to_open = folder + "global-pose-utm.txt"
   
    globalPos = np.loadtxt(file_to_open, comments="#", delimiter=",", unpack=False)
    
    # NOTE: The "global-pose-utm.txt" file has every sensor measurement
    # which were timestamped according to the rover's onboard computer clock 
    # upon arrival at the CPU, I didn't know how to convert this to correspond to 
    # frame data collected, so I did this conversion/approximation:
    
    # RUN4: 2010 frames of data, 996 Timestamped measurements
    # 2010/996 ~ 2 frames per time stamped measurements
    # so I make this assumtion when trying to find the global pose that 
    # matches the frame I'm looking at 
    fptsm = 2010/995 # frames per timestamped measurement
    FOI = int(995*int(frame)/2010) # frame of interest
    globCurrPose = getHomog(globalPos[FOI, 1:])
    position = globalPos[FOI, 1:][0:3]
    rotation = globalPos[FOI, 1:][3:]
    
    # print("The center of this orthophoto is at [easting(m), northing(m), altitude(m)]: ")
    # print(position)
    # print("The orientation of the rover at the point in time this orthophoto represents, as 4 quaternions: ")
    # print(rotation)
    text_file = open(output_dir + "/orthophoto_pose.txt", "w")
    n = text_file.write('The center of this orthophoto is at [easting(m), northing(m), altitude(m)]: \n{}'.format(position))
    n = text_file.write('\nThe orientation of the rover at the point in time this orthophoto represents, as 4 quaternions: \n{}'.format(rotation))
    text_file.close()
    
    print("SAVED rover pose!!!")
    
    # ###########################################################################
    # VALIDATION, check to see if distances in the real world match those 
    # in the orthophoto world
    
    # only relevant to : 
    #   fileid = "2018_09_04_18_18_15_891197"
    #   frame = "000978"
    
    # Check the distance between the rover tracks
    # plt.figure(12)
    # plt.imshow(rgb0)
    # plt.plot(720, 200, marker='+', color ='cyan')
    # plt.plot(665, 200, marker='+', color ='magenta')
    
    # plt.figure(13)
    # plt.imshow(ALLPXrgb)
    # plt.plot(1250, 665, marker='+', color ='cyan')
    # plt.plot(1250, 720, marker='+', color ='magenta')
    
    # the grid_x and grid_y are generated to be equally spaced 
    xspacing = allGrid_x[0, 1]-allGrid_x[0, 2]
    yspacing = allGrid_y[1, 0]-allGrid_y[2, 0]
  
    ###########################################################################
    
    # I defined some transforms that can be used for further work on this problem
    # I acutally didn't end up using these functions in the implementation of
    # my code but this would be have been used for the orthophoto map of 
    # multiple orthophotos
    
    # sensor2gps = getOMNI2GPS(allT_OrefP[n]) 
    # gps2sensor = np.linalg.inv(omni2gps)  
    # sensor2rover = getOMNI2Rover(allT_OrefP[n])
    
    # # GPS relative to Rover, gps2rov = T_RG, taken from the .txt files from this
    # # repository
    # gps2rov = np.array([-0.260,0.000,0.340,0,0,0,0])
    # H1 = getHomog(gps2rov)
    # # Omnidirectional sensor relative to Rover, rov2sensor = T_ROref 
    # # taken from the .txt files from this repository
    # rov2sensor = np.array([0.236,-0.129,0.845,-0.630,-0.321,0.321,0.630])
    # H2 = getHomog(rov2sensor)
    
    # gps2sensor = H1*H2
    # sensor2gps = np.linalg.inv(gps2sensor)
    # sensorInGPS = transformPose(sensor2gps, 0,0, np.eye(3))
    
    ###########################################################################
    
    # Code to Display the Rover Moving, used to understand the images :D
    
    # n=0
    # folder = "run4_base_hr/run4_base_hr/omni_image{}/".format(n)
    # # data_folder = Path("run4_base_hr/run4_base_hr/omni_image{}".format(n))
    # # file_to_open = data_folder / "frame{}_{}.png".format(frame, fileid)
    
    # #Display all images
    # scale=0.5
    # for file in os.listdir(folder):
    #     for camera in range(0,10):
    #         folder = "run4_base_hr/run4_base_hr/omni_image{}/".format(camera)
    #         img = cv2.imread(folder+file)
    #         dim = (int(img.shape[1]*scale), int(img.shape[0]*scale))
    #         img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    #         cv2.imread(folder+file)
    #         if camera == 0:  # stereo A
    #             stereoA = img
    #             continue
    #         if camera == 1: # stereo B
    #             stereoB = img
    #             continue
    #         elif camera%2==1: # odd number
    #             stereoA = np.concatenate((stereoA, img), axis=1) 
    #         else: 
    #             stereoB = np.concatenate((stereoB, img), axis=1)
    #         # cv2.imshow('image',img)
    #         # cv2.waitKey()
    #     img = cv2.imread(folder+file)
    #     stereo = np.concatenate((stereoA, stereoB), axis=0)
    #     cv2.imshow('image',stereo)
    #     cv2.waitKey(10)
       
    # Helper code that displays all the omni-directional camera images at once
    # #Display all images
    # scale=0.5
    # for file in os.listdir(folder):
    #     for camera in range(0,10):
    #         folder = "run4_base_hr/run4_base_hr/omni_image{}/".format(camera)
    #         img = cv2.imread(folder+file)
    #         dim = (int(img.shape[1]*scale), int(img.shape[0]*scale))
    #         img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    #         cv2.imread(folder+file)
    #         if camera == 0:  # stereo A
    #             stereoA = img
    #             continue
    #         if camera == 1: # stereo B
    #             stereoB = img
    #             continue
    #         elif camera%2==1: # odd number
    #             stereoA = np.concatenate((stereoA, img), axis=1) 
    #         else: 
    #             stereoB = np.concatenate((stereoB, img), axis=1)
    #         # cv2.imshow('image',img)
    #         # cv2.waitKey()
    #     img = cv2.imread(folder+file)
    #     stereo = np.concatenate((stereoA, stereoB), axis=0)
    #     cv2.imshow('image',stereo)
    #     cv2.waitKey(10)
    
def run_project(input_dir, output_dir):
    """
    Main entry point for your project code. 
    
    DO NOT MODIFY THE SIGNATURE OF THIS FUNCTION.
    """
    #---- FILL ME IN ----

    # Add your code here...
    main(input_dir, output_dir)
    #--------------------


# Command Line Arguments
parser = argparse.ArgumentParser(description='ROB501 Final Project.')
parser.add_argument('--input_dir', dest='input_dir', type=str, default="./input",
                    help='Input Directory that contains all required rover data')
parser.add_argument('--output_dir', dest='output_dir', type=str, default="./output",
                    help='Output directory where all outputs will be stored.')


if __name__ == "__main__":
    
    # Parse command line arguments
    args = parser.parse_args()

    # Uncomment this line if you wish to test your docker setup
    #test_docker(args.input_dir, args.output_dir)

    # Run the project code
    run_project(args.input_dir, args.output_dir)