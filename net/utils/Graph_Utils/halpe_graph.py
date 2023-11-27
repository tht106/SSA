
halpe_joint_d = {
    0:  "Nose",
    1:  "LEye",
    2:  "REye",
    3:  "LEar",
    4:  "REar",
    5:  "LShoulder",
    6:  "RShoulder",
    7:  "LElbow",
    8:  "RElbow",
    9:  "LWrist",
    10: "RWrist",
    11: "LHip",
    12: "RHip",
    13: "LKnee",
    14: "Rknee",
    15: "LAnkle",
    16: "RAnkle",
    17:  "Head",
    18:  "Neck",
    19:  "Hip",
    20: "LBigToe",
    21: "RBigToe",
    22: "LSmallToe",
    23: "RSmallToe",
    24: "LHeel",
    25: "RHeel",
    # {26-93, 68 Face Keypoints}
    #//left hand
    #{94-114, 21 Left Hand Keypoints}
    #//right hand
    #{115-135, 21 Right Hand Keypoints}
    }

# facing inwards, 0_indexed
# without special hand or face links
simple_halpe_graph_links = [(0, 18), (1, 0), (2, 0), (3, 0), (4, 0), (5, 18), (6, 18),
                          (7, 5), (8, 6), (9, 7), (10, 8), (11, 19), (12, 19), (13, 11),
                          (14, 12), (15, 13), (16, 14), (17, 0), (18, 19), (19, 18),
                          (20, 15), (21, 16), (22, 15), (23, 16), (24, 15), (25, 16)]

# with two hand links per hand
simple_halpe_graph_links_wh = [(0, 18), (1, 0), (2, 0), (3, 0), (4, 0), (5, 18), (6, 18),
                          (7, 5), (8, 6), (9, 7), (10, 8), (11, 19), (12, 19), (13, 11),
                          (14, 12), (15, 13), (16, 14), (17, 0), (18, 19), (19, 18),
                          (20, 15), (21, 16), (22, 15), (23, 16), (24, 15), (25, 16), (26, 9),
                          (27, 26), (28, 26), (29, 10), (30, 29), (31, 29)]
lh_idx = (94, 98, 106)
rh_idx = (115, 119, 127)

# a simple link structure consisting of adjacent joints numbers followed by a loop
# this is for the face keypoints
halpe_face_links = [(i, i+1) for i in range(26, 93)]
halpe_face_links.append((93, 26))

# before shifting by the correct indices, according to halpe_hand_keypoints.png
halpe_hand_links = [(1, 0), (2, 1), (3, 2), (4, 3), (5, 0), (6, 5), (7, 6), (8, 7),
                         (9, 0), (10, 9), (11, 10), (12, 11), (13, 0), (14, 13), (15, 14), (16, 15),
                         (17, 0), (18, 17), (19, 18), (20, 19)]

# left hand offset
lh_i = 94
rh_i = 115
alpha_pose_left_hand_links = [(i+lh_i, j+lh_i) for (i, j) in halpe_hand_links]
alpha_pose_left_hand_links.insert(0, (lh_i, 9))
alpha_pose_right_hand_links = [(i+rh_i, j+rh_i) for (i, j) in halpe_hand_links]
alpha_pose_right_hand_links.insert(0, (rh_i, 10))
alpha_pose_hand_links = alpha_pose_left_hand_links + alpha_pose_right_hand_links

# all links
all_halpe_links = simple_halpe_graph_links + halpe_face_links + alpha_pose_hand_links

# hand + simple
face_offset = 68
shifted_hand_joints = []
# reshift by the number of face links
for (i, j) in alpha_pose_hand_links:
    if i >= 93:
        a = i - face_offset
    else:
        a = i
    if j >= 93:
        b = j - face_offset
    else:
        b = j
    shifted_hand_joints.append((a, b))

medium_halpe_graph_links = simple_halpe_graph_links + shifted_hand_joints