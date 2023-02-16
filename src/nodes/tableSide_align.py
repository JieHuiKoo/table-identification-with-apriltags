#! /usr/bin/python3
import rospy
import sys
import math
import numpy as np
import cv2
from pupil_apriltags import Detector

from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from std_msgs.msg import Float32
from std_msgs.msg import Int32
from std_msgs.msg import String

print("Python Version: " + str(sys.version_info[0]) + '.' + str(sys.version_info[1]))
print("OpenCV Version: " + str(cv2.__version__))

apriltag_detector = Detector(
    families="tag36h11",
    nthreads=1,
    quad_decimate=1.0,
    quad_sigma=0.0,
    refine_edges=1,
    decode_sharpening=0.25,
    debug=0
    )

def imgmsg_to_cv2(img_msg):
    rgb8_flag = 0
    if img_msg.encoding != "bgr8":
        rgb8_flag = 1
    
    dtype = np.dtype("uint8") # Hardcode to 8 bits...
    dtype = dtype.newbyteorder('>' if img_msg.is_bigendian else '<')
    image_opencv = np.ndarray(shape=(img_msg.height, img_msg.width, 3), # and three channels of data. Since OpenCV works with bgr natively, we don't need to reorder the channels.
                    dtype=dtype, buffer=img_msg.data)
    # If the byt order is different between the message and the system.
    if img_msg.is_bigendian == (sys.byteorder == 'little'):
        image_opencv = image_opencv.byteswap().newbyteorder()

    if rgb8_flag:
        image_opencv = cv2.cvtColor(image_opencv, cv2.COLOR_RGB2BGR)

    return image_opencv

def cv2_to_imgmsg(cv_image):
    img_msg = Image()
    img_msg.height = cv_image.shape[0]
    img_msg.width = cv_image.shape[1]
    img_msg.encoding = "bgr8"
    img_msg.is_bigendian = 0
    img_msg.data = cv_image.tobytes()
    img_msg.step = len(img_msg.data) // img_msg.height # That double line is actually integer division, not a comment
    return img_msg

def draw_bounding_box(image, marker):
    top_left = marker.points[0]
    bottom_right = marker.points[1]
    image = cv2.rectangle(image, (top_left.x, top_left.y), (bottom_right.x, bottom_right.y), (255, 0, 0), 3)
    return image

def showImage(img):
    cv2.imshow('image', img)
    cv2.waitKey(1)

def get_apriltags(img, camera_params):
    global apriltag_detector

    results = apriltag_detector.detect( img, 
                                        estimate_tag_pose=True,
                                        tag_size=0.1,
                                        camera_params = (camera_params))

    return results

def visualise_apriltags(results, image):

    # loop over the AprilTag detection results
    for r in results:
        # extract the bounding box (x, y)-coordinates for the AprilTag
        # and convert each of the (x, y)-coordinate pairs to integers
        (ptA, ptB, ptC, ptD) = r.corners
        ptB = (int(ptB[0]), int(ptB[1]))
        ptC = (int(ptC[0]), int(ptC[1]))
        ptD = (int(ptD[0]), int(ptD[1]))
        ptA = (int(ptA[0]), int(ptA[1]))
        # draw the bounding box of the AprilTag detection
        cv2.line(image, ptA, ptB, (0, 255, 0), 2)
        cv2.line(image, ptB, ptC, (0, 255, 0), 2)
        cv2.line(image, ptC, ptD, (0, 255, 0), 2)
        cv2.line(image, ptD, ptA, (0, 255, 0), 2)
        # draw the center (x, y)-coordinates of the AprilTag
        (cX, cY) = (int(r.center[0]), int(r.center[1]))
        cv2.circle(image, (cX, cY), 5, (0, 0, 255), -1)
        # draw the tag family on the image
        tagFamily = r.tag_family.decode("utf-8")
        tagID = r.tag_id
        cv2.putText(image, "Tag ID: " + str(tagID), (ptA[0], ptA[1]),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image

class objectTagID:
    topLeftID = int(-1)
    topRightID = int(-1)
    bottomRightID = int(-1)
    bottomLeftID = int(-1)

    def __init__(self, topLeftID, topRightID, bottomRightID, bottomLeftID):
        self.topLeftID = topLeftID
        self.topRightID = topRightID
        self.bottomRightID = bottomRightID
        self.bottomLeftID = bottomLeftID
        
tableSide_tagIDs = objectTagID(7, 4, 5, 6) 
tableTop_tagIDs = objectTagID(3, 0, 1, 2)

armCamera_Intrinsics = [431.09375, 430.4695739746094, 418.5455017089844, 235.63516235351562]
frontCamera_Intrinsics = [644.6495361328125, 643.85009765625, 654.1922607421875, 353.6024475097656]

def rad_to_deg(rad):
    return rad * 180/math.pi

def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

def get_euler_angles_from_rotation_matrix(R):

    assert(isRotationMatrix(R))
 
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
 
    singular = sy < 1e-6
 
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
 
    return Point(x, y, z)

def get_identified_corners(apriltag_locations, tagIDs):
    found_apriltag_idx_array = [-1, -1, -1, -1]

    for i in range(0, len(apriltag_locations)):
        apriltag_location = apriltag_locations[i]
        
        if apriltag_location.tag_id == tagIDs.topLeftID:
            found_apriltag_idx_array[0] = i
        elif apriltag_location.tag_id == tagIDs.topRightID:
            found_apriltag_idx_array[1] = i
        elif apriltag_location.tag_id == tagIDs.bottomRightID:
            found_apriltag_idx_array[2] = i
        elif apriltag_location.tag_id == tagIDs.bottomLeftID:
            found_apriltag_idx_array[3] = i

    return found_apriltag_idx_array

def able_to_identify_center(found_apriltag_idx_array):

    # We can identify the center if we identify at least 1 apriltag on right and left side
    if found_apriltag_idx_array[0] != -1 and found_apriltag_idx_array[1] != -1\
    or found_apriltag_idx_array[3] != -1 and found_apriltag_idx_array[2] != -1\
    or found_apriltag_idx_array[0] != -1 and found_apriltag_idx_array[2] != -1\
    or found_apriltag_idx_array[3] != -1 and found_apriltag_idx_array[1] != -1:
        return True
    else:
        return False

def able_to_identify_skew_of_left_side(found_apriltag_idx_array):
    
    # We can identify the center if we identify at least 2 apriltag on left side
    if found_apriltag_idx_array[0] != -1 and found_apriltag_idx_array[3] != -1:
        return True
    else:
        return False

def able_to_identify_skew_of_right_side(found_apriltag_idx_array):
    
    # We can identify the center if we identify at least 2 apriltag on right side
    if found_apriltag_idx_array[1] != -1 and found_apriltag_idx_array[2] != -1:
        return True
    else:
        return False

def get_skew_of_tableSide(apriltag_locations, found_apriltag_idx_array):

    print("===")

    # Always return left side first. If cannot return left skew, we will only return right skew.
    if able_to_identify_skew_of_left_side(found_apriltag_idx_array):
        topleft = apriltag_locations[found_apriltag_idx_array[0]]
        bottomleft = apriltag_locations[found_apriltag_idx_array[3]]
        avg_left_skew = (get_euler_angles_from_rotation_matrix(topleft.pose_R).y + get_euler_angles_from_rotation_matrix(bottomleft.pose_R).y)/2
        avg_left_skew = rad_to_deg(avg_left_skew)
        print("Left Skew: " + str(avg_left_skew))

        return avg_left_skew

    if able_to_identify_skew_of_right_side(found_apriltag_idx_array):
        topright = apriltag_locations[found_apriltag_idx_array[1]]
        bottomright = apriltag_locations[found_apriltag_idx_array[2]]
        avg_right_skew = (get_euler_angles_from_rotation_matrix(topright.pose_R).y + get_euler_angles_from_rotation_matrix(bottomright.pose_R).y)/2
        avg_right_skew = rad_to_deg(avg_right_skew)
        print("Right Skew: " + str(avg_right_skew))

        return avg_right_skew
    
    return np.nan

def find_center_x_coord(apriltag_locations, found_apriltag_idx_array):
    
    topleft = 0
    topright = 0
    bottomleft = 0
    bottomright = 0

    if found_apriltag_idx_array[0] != -1:
        topleft = apriltag_locations[found_apriltag_idx_array[0]]
    if found_apriltag_idx_array[1] != -1:
        topright = apriltag_locations[found_apriltag_idx_array[1]]
    if found_apriltag_idx_array[2] != -1:
        bottomright = apriltag_locations[found_apriltag_idx_array[2]]
    if found_apriltag_idx_array[3] != -1:
        bottomleft = apriltag_locations[found_apriltag_idx_array[3]]

    if topleft and topright:
        x_center = (topleft.center[0] + topright.center[0])/2
    elif topleft and bottomright:
        x_center = (topleft.center[0] + bottomright.center[0])/2
    elif bottomleft and bottomright:
        x_center = (bottomleft.center[0] + bottomright.center[0])/2
    elif bottomleft and topright:
        x_center = (bottomleft.center[0] + topright.center[0])/2

    return int(x_center)
    

def get_center_horizontal_offset_of_apriltags(apriltag_locations, found_apriltag_idx_array, image_x_center):
       
    if able_to_identify_center(found_apriltag_idx_array):
        center_x_coord = find_center_x_coord(apriltag_locations, found_apriltag_idx_array)
        return image_x_center - center_x_coord, center_x_coord        
    else:
        return -1, -1


def process_image(image_msg):

    # Declare the cvBridge object
    proc_image = imgmsg_to_cv2(image_msg)

    # Convert to grayscale
    gray = cv2.cvtColor(proc_image, cv2.COLOR_BGR2GRAY)

    # Detect the apriltags
    apriltag_locations = get_apriltags(gray, frontCamera_Intrinsics)

    # Visualise apriltags
    proc_image = visualise_apriltags(apriltag_locations, proc_image)

    found_apriltag_idx_array = get_identified_corners(apriltag_locations, tableSide_tagIDs)

    # Find the angle of the apriltags
    skewness_of_table_side = get_skew_of_tableSide(apriltag_locations, found_apriltag_idx_array)
    
    # Find the center of the apriltags
    center_offset, apriltags_center_x = get_center_horizontal_offset_of_apriltags(apriltag_locations, found_apriltag_idx_array, int(proc_image.shape[1]/2))
    
    apriltags_success = ""
    
    if skewness_of_table_side != -1:
        apriltags_success += "Skew "
        cv2.putText(proc_image, "Skew: " + str(round(skewness_of_table_side,2)) + " deg", (int(proc_image.shape[1]//2)-60, int(proc_image.shape[0]*0.1)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    else:
        skewness_of_table_side = 0
    
    if center_offset != -1:
        apriltags_success += "Center Found"
        cv2.line(proc_image, (apriltags_center_x, 0), (apriltags_center_x, proc_image.shape[0]), (0, 255, 0), 3)
        cv2.putText(proc_image, "Offset: " + str(center_offset) + " pix", 
                    (int(proc_image.shape[1]//2)+10, int(proc_image.shape[0]*0.15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    else:
        center_offset = 0
   
    if center_offset == 0 and skewness_of_table_side == 0:
        apriltags_success = "Center Skew NotFound"

    cv2.line(proc_image, (int(proc_image.shape[1]//2), int(proc_image.shape[0] * 0.11)), (int(proc_image.shape[1]//2), int(proc_image.shape[0])), (255,0, 0), 3) 

    skewness_of_table_side_pub = rospy.Publisher('frontCamera/tableSide_Skew', Float32, queue_size=1)
    skewness_of_table_side_pub.publish(skewness_of_table_side)

    center_offset_pub = rospy.Publisher('frontCamera/tableSide_CenterOffset', Int32, queue_size=1)
    center_offset_pub.publish(center_offset)

    image_pub = cv2_to_imgmsg(proc_image)
    tableside_image_pub = rospy.Publisher('frontCamera/tableSide_AnnotatedImage', Image, queue_size=1)
    tableside_image_pub.publish(image_pub)

    success_pub = rospy.Publisher('frontCamera/tableSide_FoundSuccess', String, queue_size=1)
    success_pub.publish(apriltags_success)

def start_node():
    rospy.init_node('tableSide_align')
    rospy.loginfo('tableSide_align node started')

    rospy.Subscriber("/frontCamera/color/image_raw", Image, process_image)
    
    rospy.spin()

if __name__ == '__main__':
    
    try:
        start_node()
    except rospy.ROSInterruptException:
        pass

