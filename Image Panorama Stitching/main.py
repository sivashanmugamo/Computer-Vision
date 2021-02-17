'''
Importing required libraries
'''
import os

import sys
user_arg = sys.argv

import cv2
import itertools as iter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from random import randint
from scipy.spatial.distance import cdist

def img_files(img_location):
    '''
    This function returns a list of paths of the images in the given directory

    INPUT: Directory where the images or stored ("./data" or "./ubdata")
    OUTPUT: List of paths of all the images in the given directory
    '''
    print("Getting paths of images")

    img_paths = []

    for root, directory, files in os.walk(img_location):
        if(len(files) == 0):
            print("There are no images in the given directory")
            FileNotFoundError

        else:
            for file_name in files:
                if(file_name[-4:] == ".jpg" or file_name[-4:] == ".JPG"):
                    img_paths.append(os.path.join(root, file_name))

    return img_paths

def get_image(img_path):
    '''
    This function returns a list of dictionary with "color" and "gray" data of the image in the given path

    INPUT: List of paths of the images for which the data should be obtained
    OUTPUT: List of dictionary containing the image's path and its associated data in "color" and "gray" color space
    '''
    print("Getting data of images")

    img_data_list = []

    for each_path in img_path:
        img_data = {}
        img_data[each_path] = {"color" : cv2.imread(each_path, 1), "gray" : cv2.imread(each_path, 0)}
        img_data_list.append(img_data)

    return img_data_list

def img_detect(img_data_list):
    '''
    This function returns a list of disctionaries with keypoints ("kps") and descriptions ("desc") data of the images

    INPUT: List of dictionaires with image's path, color data & gray data
    OUTPUT: List of dictionaries with image's path, color data, gray data, keypoints & descriptors
    '''
    print("Detecting image")

    temp_list = img_data_list
    img_data_list = []

    sift_desc = cv2.xfeatures2d.SIFT_create()

    for each_data in temp_list:
        for i in each_data:
            keypoints, descriptions = sift_desc.detectAndCompute(each_data[i]["gray"], None)
            each_data[i]["kps"] = keypoints
            each_data[i]["desc"] = descriptions
        img_data_list.append(each_data)

    return img_data_list

def cal_euclidean_dist(img_data_1, img_data_2):
    '''
    This function returns the calculated squared euclidean distance between matching pair

    INPUT: Image data (color data, gray data, keypoint data & descriptions) of the image pair
    OUTPUT: Matching points among the image pair
    '''
    print("Squared Euclidean distance calculation")

    threshold = 7000

    for i in img_data_1:
        for j in img_data_2:

            distance = cdist(img_data_1[i]["desc"], img_data_2[j]["desc"], "sqeuclidean")
            
            coordinates_1 = np.array([img_data_1[i]["kps"][pnt].pt for pnt in np.where(distance < threshold)[0]])
            coordinates_2 = np.array([img_data_2[j]["kps"][pnt].pt for pnt in np.where(distance < threshold)[1]])
            
            euclidean_distance = np.concatenate((coordinates_1, coordinates_2), axis = 1)

    return euclidean_distance

def ransac_stitch(distance_data):
    '''
    This function finds the homography using the distances and the matching points between the image pair

    INPUT: Matching points among the image pair with the image data
    OUTPUT: Most fitting homography matrix
    '''
    print("Inside RANSAC")

    highest_inlier = 0
    h_matrix = []

    count = 0

    while count < 1000:

        temp_list = []
        for i in range(4):
            temp_list.append(distance_data[randint(0, distance_data.shape[0]-1)])
        
        matched_matrix = np.array(temp_list)

        points_1 = np.float32(matched_matrix[:, 0:2])
        points_2 = np.float32(matched_matrix[:, 2:4])

        H = cv2.getPerspectiveTransform(points_1, points_2)

        img_pts_1 = np.concatenate((distance_data[:, 0:2], np.ones((len(distance_data), 1))), axis = 1)
        img_pts_2 = distance_data[:, 2:4]

        corresponding_points = np.zeros((len(distance_data), 2))

        for i in range(len(distance_data)):
            temp = np.matmul(H, img_pts_1[i])
            corresponding_points[i] = (temp/temp[2])[0:2]
        
        pt_error = np.linalg.norm(img_pts_2 - corresponding_points, axis= 1)**2

        indices = np.where(pt_error < 0.5)[0]
        inliers = distance_data[indices]

        inlier_count = len(inliers)

        if inlier_count > highest_inlier:
            highest_inlier = inlier_count
            h_matrix = H.copy()

        count = count + 1

    return h_matrix

def img_stitch(detected_data):
    '''
    This function stiches and saves the images

    INPUT: Image data (color, gray, keypoints and descriptors)
    OUTPUT: Stiched image in the working directory
    '''
    print("Inside stich")

    if(len(detected_data) > 2): # For more than 2 images

        temp_img_list = []

        for each_permutation_1 in list(iter.permutations(detected_data, len(detected_data))):
            img_list = list(each_permutation_1)

            color_data = []
            for each_img in img_list:
                for i in each_img:
                    color_data.append(each_img[i]["color"])
            
            temp = 0
            while len(color_data) > 1:
                euclidean_distance = cal_euclidean_dist(img_list[temp], img_list[temp+1])
                img_h = ransac_stitch(distance_data= euclidean_distance)

                for img_1 in img_list[temp]:
                    for img_2 in img_list[temp+1]:
                        intermediate = cv2.warpPerspective(img_list[temp][img_1]["color"], img_h, (int(img_list[temp][img_1]["color"].shape[1] + img_list[temp+1][img_2]["color"].shape[1]*0.8), int(img_list[temp][img_1]["color"].shape[0] + img_list[temp+1][img_2]["color"].shape[0]*0.4)))

                        intermediate[0:img_list[temp+1][img_2]["color"].shape[0], 0:img_list[temp+1][img_2]["color"].shape[1]] = img_list[temp+1][img_2]["color"]

                color_data.pop(0)
                color_data[0] = intermediate
            
            panorama = remove_black_area(color_data[0])
            temp_img_list.append(panorama)

        blk_count = 0
        blk_count_dict = {}
        blk = np.zeros(3)

        i = 0
        for each_pano in temp_img_list:
            for x in range(each_pano.shape[0]):
                for y in range(each_pano.shape[1]):
                    if np.array_equal(each_pano[x, y, :], blk):
                        blk_count = blk_count + 1

            blk_count_dict[i] = blk_count
            i = i+1
        
        temp_blk = min(blk_count_dict.values())
        pano_index = [key for key in blk_count_dict if blk_count_dict[key] == temp_blk]

        write_img(temp_img_list[pano_index[0]])

    else: # For 2 images
        euclidean_distance = cal_euclidean_dist(detected_data[0], detected_data[1])
        img_h = ransac_stitch(distance_data= euclidean_distance)

        for img_1 in detected_data[0]:
            for img_2 in detected_data[1]:
                output = cv2.warpPerspective(detected_data[0][img_1]["color"], img_h ,(int(detected_data[0][img_1]["color"].shape[1] + detected_data[1][img_2]["color"].shape[1]*0.8), int(detected_data[0][img_1]["color"].shape[0] + detected_data[1][img_2]["color"].shape[0]*0.4)))

                output[0:detected_data[1][img_2]["color"].shape[0], 0:detected_data[1][img_2]["color"].shape[1]] = detected_data[1][img_2]["color"]
                
        panorama = remove_black_area(output)
        write_img(panorama)

def remove_black_area(crude_pan):
    '''
    This function removes the black pixel area that results from warping, by resizing to the borders of color image

    INPUT: Warped color panorama data
    OUTPUT: Resized panorama image
    '''
    print("Inside remove black")
    
    black = np.zeros(3) # Reference - To c
    x = 0
    y = 0

    for i in range(crude_pan.shape[0]):
        for j in range(crude_pan.shape[1]):
            if not np.array_equal(crude_pan[i, j, :], black):
                if i > y:
                    y = i
                if j > x:
                    x = j
    
    clean_pan = crude_pan[0:y, 0:x, :]
    return clean_pan

def write_img(out_img_data):
    '''
    This function is to write the image in the working directory

    INPUT: Color image data
    OUTPUT: Image written to the working directory
    '''
    print("Write image")

    cv2.imwrite(str(user_arg[1][1:-2])+'/'+'panorama.jpg', out_img_data)

def main():
    # Gets the user input of the local directory where the images are stored
    img_location = user_arg[1][1:-2]

    # Gets the paths of all the images in the given directory
    img_paths = img_files(img_location)

    if(len(img_paths) == 1):
        print("Cannot create a panorama with 1 image")
    elif(len(img_paths) > 1):
        # Gets the list of dictionaries containing "color" and "gray" data of the images
        img_data_list = get_image(img_path= img_paths)

        # Gets the list of dictionaries containing keypoint ("kps") and descriptions ("desc") data of the images
        img_detect_data = img_detect(img_data_list= img_data_list)

        # Stiches images by finding squared euclidean and then finding homography to stich the images over multiple iterations
        img_stitch(detected_data= img_detect_data)
    else:
        print("There are no images in the directory to create a Panorama")

if __name__ == "__main__":
    main()