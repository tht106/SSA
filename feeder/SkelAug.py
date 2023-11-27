import numpy as np
import feeder.Graph_Utils.Graph as g

import torch.nn.functional as F
import torch
import random
import numpy as np


def remove_diag(x):
    x_no_diag = np.ndarray.flatten(x)
    x_no_diag = np.delete(x_no_diag, range(0, len(x_no_diag), len(x) + 1), 0)
    x_no_diag = x_no_diag.reshape(len(x), len(x) - 1)
    return x_no_diag

def get_frame_angles(frame):
    """
    get the angles between one vertex and all other vertices
    :param frame: a tensor of shape (num_vertices, num_coords)
    :return: a tensor of shape (num_vertices, num_vertices) where entry (i,j)
    is the angle between vertex i and j
    """
    num_vertices, num_coords = frame.shape
    norms = np.linalg.norm(frame, axis=-1)
    norms[norms == 0.0] = 0.00001
    norms = np.expand_dims(norms, axis=-1)
    norms = np.tile(norms, (1, num_coords))
    frame_n = frame/norms
    A = np.expand_dims(frame_n, axis=1)
    A = np.tile(A, (1, num_vertices, 1))
    B = np.tile(np.expand_dims(frame_n, axis=0), (num_vertices, 1, 1))
    C = A*B
    dps = np.sum(C, axis=-1)
    dps = np.clip(dps, a_min=-1.0, a_max=1.0)
    angles = np.arccos(dps)
    angles[angles == np.nan] = 0.0
    angles = remove_diag(angles)
    return angles


def get_angles(joints):
    #joints of shape (num_frames, num_people, num_joints, coord)
    num_frames, num_people, num_joints, _ = joints.shape
    angles = np.zeros((num_frames, num_people, num_joints, num_joints-1))
    for frame in range(num_frames):
        for p_n in range(num_people):
            if np.any(joints[frame, p_n, :, :]):
                angles[frame, p_n, :] = get_frame_angles(joints[frame, p_n, :, :])
    return angles


"""
joints: np.ndarray of shape (M, T, V, C)
theta: float in range [0, 2*pi]
Only supports 3 dimensions per modality
"""
def random_rot3D(joints, num_dimensions=3, theta_range=0.3):
    num_modalities = joints.shape[-1]//num_dimensions
    theta = np.random.uniform(-theta_range, theta_range, 3) # 3 rotations matrices
    cos, sin = np.cos(theta), np.sin(theta)
    rx = np.array([[1, 0, 0], [0, cos[0], sin[0]], [0, -sin[0], cos[0]]])
    ry = np.array([[cos[1], 0, -sin[1]], [0, 1, 0], [sin[1], 0, cos[1]]])
    rz = np.array([[cos[2], sin[2], 0], [-sin[2], cos[2], 0], [0, 0, 1]])
    rot_mat = np.matmul(rz, np.matmul(ry, rx))
    rotated_vals = []
    for i in range(num_modalities):
        rotated_vals.append(np.einsum('ab,mtvb->mtva', rot_mat, joints[:, :, :, i*num_dimensions:(i+1)*num_dimensions]))
    rotated_joints = np.concatenate([rotated_val for rotated_val in rotated_vals], axis=-1)
    return rotated_joints


"""
joints: np.ndarray of shape (M, T, V, C)
scale: small positive float
"""
def random_scale(joints, scale=0.2):
    scale = (scale,) * joints.shape[-1]
    # a different small random scaling factor in each dimension
    scale = 1 + np.random.uniform(-1, 1, size=len(scale)) * np.array(scale)
    scaled_joints = joints*scale
    return scaled_joints


def unit_vector(vector):
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """Returns the angle in radians between vectors 'v1' and 'v2'. """
    if np.abs(v1).sum() < 1e-6 or np.abs(v2).sum() < 1e-6:
        return 0
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def get_rodrigues_rotation_matr(axis, theta):
    if np.abs(axis).sum() < 1e-6 or np.abs(theta) < 1e-6:
        return np.eye(3)
    axis = np.asarray(axis)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def PreNormalize3D(joints, zaxis=[0, 1], xaxis=[8, 4], align_spine=True,
                   align_center=True, is_halpe=False, conf_vals=None):
    M, T, V, C = joints.shape
    if joints.sum() == 0:
        return joints
    index0 = [i for i in range(T) if not np.all(np.isclose(joints[0, i], 0))]

    assert M in [1, 2]
    if M == 2:
        index1 = [i for i in range(T) if not np.all(np.isclose(joints[1, i], 0))]
        if len(index0) < len(index1):
            joints = joints[:, np.array(index1)]
            joints = joints[[1, 0]]
            if conf_vals is not None:
                conf_vals = conf_vals[:, np.array(index1)]
                conf_vals = conf_vals[[1, 0]]
        else:
            joints = joints[:, np.array(index0)]
            if conf_vals is not None:
                conf_vals = conf_vals[:, np.array(index0)]
    else:
        joints = joints[:, np.array(index0)]
        if conf_vals is not None:
            conf_vals = conf_vals[:, np.array(index0)]

    if align_center:
        if joints.shape[2] == 25 and not is_halpe:
            main_body_center = joints[0, 0, 1].copy()
        elif joints.shape[2] == 32 and not is_halpe: # azure kinect
            main_body_center = joints[0, 0, 1].copy()
        elif is_halpe: # alphapose form
            main_body_center = joints[0, 0, 19].copy()
        else:
            main_body_center = joints[0, 0, -1].copy()
        mask = ((joints != 0).sum(-1) > 0)[..., None]
        joints = (joints - main_body_center) * mask

    if align_spine:
        # align with z axis
        joint_bottom = joints[0, 0, zaxis[0]]
        joint_top = joints[0, 0, zaxis[1]]
        axis = np.cross(joint_top - joint_bottom, [0, 0, 1])
        angle = angle_between(joint_top - joint_bottom, [0, 0, 1])
        matrix_z = get_rodrigues_rotation_matr(axis, angle)
        joints = np.einsum('abcd,kd->abck', joints, matrix_z)

        # align with left shoulder to right shoulder
        joint_rshoulder = joints[0, 0, xaxis[0]]
        joint_lshoulder = joints[0, 0, xaxis[1]]
        axis = np.cross(joint_rshoulder - joint_lshoulder, [1, 0, 0])
        angle = angle_between(joint_rshoulder - joint_lshoulder, [1, 0, 0])
        matrix_x = get_rodrigues_rotation_matr(axis, angle)
        joints = np.einsum('abcd,kd->abck', joints, matrix_x)
    if conf_vals is None:
        return joints
    else:
        return joints, conf_vals


def uniform_random_sample(joints, clip_len=100, test_mode=False, with_indices=False):
    if test_mode:
        np.random.seed(255)
    original_num_frames = joints.shape[1]
    unsampled_indices = np.linspace(0, original_num_frames-1, clip_len+1, dtype=np.int)
    indices = []
    for i in range(clip_len):
        lb = unsampled_indices[i]
        ub = unsampled_indices[i+1]
        if lb == ub:
            indices.append(lb)
        else:
            rand_int = np.random.randint(low=lb, high=ub+1, size=(1))[0]
            indices.append(rand_int)
    joints = joints[:, indices, :, :]
    if with_indices:
        return joints, indices
    return joints


def pad_two_person(joints, mode="zeros"):
    assert mode in ["zeros", "duplicate"]
    if mode == "zeros":
        _, T, V, C = joints.shape
        tp_joints = np.zeros((2, T, V, C), dtype=float)
        tp_joints[0] = joints
    if mode == "duplicate":
        _, T, V, C = joints.shape
        tp_joints = np.zeros((2, T, V, C), dtype=float)
        tp_joints[0] = joints
        tp_joints[1] = joints


def crop_end_beginning(pct, joints):
    _, T, _, _ = joints.shape
    total_num_joints = int(pct*T)
    to_crop = total_num_joints//2
    joints = joints[:, to_crop:T-to_crop, :, :]
    return joints


ntu_bone_links = [(0, 1), (1, 20), (2, 20), (3, 2), (4, 20), (5, 4), (6, 5), (7, 6), (8, 20), (9, 8),
                          (10, 9), (11, 10), (12, 0), (13, 12), (14, 13), (15, 14), (16, 0), (17, 16), (18, 17),
                          (19, 18), (21, 22), (20, 20), (22, 7), (23, 24), (24, 11)]


# joints of shape M, T, V, C
def get_bones(joints, graph_type="ntu"):
    bones = np.zeros_like(joints, dtype=np.float32)
    if graph_type == "ntu":
        bone_links = ntu_bone_links
    elif graph_type == "azure":
        bone_links = g.azure_links
    for (a, b) in bone_links:
        bones[:, :, a, :] = joints[:, :, a, :] - joints[:, :, b, :]
    return bones


# joints of shape M, T, V, C
def get_motion(joints):
    motion_vals = np.zeros_like(joints, dtype=np.float32)
    num_frames = joints.shape[1]
    motion_vals[:, 0:num_frames-1:, :, :] = joints[:, 1:num_frames, :, :] - joints[:, 0:num_frames-1, :, :]
    # duplicate the last value to preserve the original number of frames
    motion_vals[:, num_frames-1, :, :] = motion_vals[:, num_frames-2, :, :]
    return motion_vals


# multi-data types will be concatenated along the channel dimension
def get_representation(joints, repr_type, graph_type):
    if repr_type == "joints":
        return joints
    elif repr_type == "bones":
        return get_bones(joints, graph_type=graph_type)
    elif repr_type == "joint_motion":
        return get_motion(joints)
    elif repr_type == "bones_motion":
        bones = get_bones(joints, graph_type=graph_type)
        return get_motion(bones)
    elif repr_type == "angles":
        return get_angles(joints)
    elif repr_type == "angles_motion":
        return get_motion(get_angles(joints))
    elif repr_type == "joints_and_bones_motion":
        bones = get_bones(joints, graph_type=graph_type)
        joint_motion = get_motion(joints)
        bone_motion = get_motion(bones)
        return np.concatenate([joint_motion, bone_motion], axis=-1)
    elif repr_type == "joints_and_bones":
        bones = get_bones(joints, graph_type=graph_type)
        return np.concatenate([joints, bones], axis=-1)
    elif repr_type == "all": # joints, bones, joint_motion, bones_motion
        bones = get_bones(joints, graph_type=graph_type)
        joint_motion = get_motion(joints)
        bone_motion = get_motion(bones)
        return np.concatenate([joints, bones, joint_motion, bone_motion], axis=-1)
    else: #repr not supported
        exit()


def get_halpe_format(joints, conf_vals, format):
    #joints of shape (num_frames, num_people, num_joints, coord)
    c_vals = None
    if format == "halpe_all":
        return joints, conf_vals
    elif format == "halpe_medium":
        del_vals = list(range(26, 94))
        joints = np.delete(joints, obj=del_vals, axis=2)
        if conf_vals is not None:
            c_vals = np.delete(conf_vals, obj=del_vals, axis=-1)
        return joints, c_vals
    elif format == "halpe_simple":
        j = joints[:, :, :26, :]
        if conf_vals is not None:
            c_vals = conf_vals[:, :, :26]
        return j, c_vals
    elif format == "halpe_simple_wh":
        j = joints[:, :, :26, :]
        lh_idx = (94, 98, 106)  # middle finger-tip and thumb tip
        rh_idx = (115, 119, 127)
        lh_val1 = joints[:, :, lh_idx[0]:lh_idx[0]+1, :]
        lh_val2 = joints[:, :, lh_idx[1]:lh_idx[1]+1, :]
        lh_val3 = joints[:, :, lh_idx[2]:lh_idx[2]+1, :]
        rh_val1 = joints[:, :, rh_idx[0]:rh_idx[0]+1, :]
        rh_val2 = joints[:, :, rh_idx[1]:rh_idx[1]+1, :]
        rh_val3 = joints[:, :, rh_idx[2]:rh_idx[2]+1, :]
        j = np.concatenate([j, lh_val1, lh_val2, lh_val3, rh_val1, rh_val2, rh_val3], axis=2)
        if conf_vals is not None:
            c_vals = conf_vals[:, :, :26]
            lh_c_val1 = conf_vals[:, :, lh_idx[0]:lh_idx[0] + 1]
            lh_c_val2 = conf_vals[:, :, lh_idx[1]:lh_idx[1] + 1]
            lh_c_val3 = conf_vals[:, :, lh_idx[2]:lh_idx[2] + 1]
            rh_c_val1 = conf_vals[:, :, rh_idx[0]:rh_idx[0] + 1]
            rh_c_val2 = conf_vals[:, :, rh_idx[1]:rh_idx[1] + 1]
            rh_c_val3 = conf_vals[:, :, rh_idx[2]:rh_idx[2] + 1]
            c_vals = np.concatenate([c_vals, lh_c_val1, lh_c_val2, lh_c_val3,
                                     rh_c_val1, rh_c_val2, rh_c_val3], axis=2)
        return j, c_vals


def threshold_filter(joints, conf_vals, threshold=0.2):
    #joints of shape (num_frames, num_people, num_joints, coord)
    #conf_vals of shape (num_frames, num_people, num_joints)
    c_vals = np.expand_dims(conf_vals, axis=-1)
    c_vals = np.repeat(c_vals, repeats=2, axis=-1)
    mask = np.where(c_vals > threshold, c_vals, 0.0)
    j = mask*joints
    return j


def random_2D_rot(joints, theta_range=0.3):
    # joints of shape (num_frames, num_people, num_joints, coord)
    theta = np.random.uniform(-theta_range, theta_range)
    random_2d_rot_matr = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    num_modalities = joints.shape[-1] // 2
    rotated_vals = []
    for i in range(num_modalities):
        rotated_vals.append(
            np.einsum('ab,mtvb->mtva',  random_2d_rot_matr, joints[:, :, :, i * 2:(i + 1) * 2]))
    rotated_joints = np.concatenate([rotated_val for rotated_val in rotated_vals], axis=-1)
    return rotated_joints


def random_2d_scale(joints, alpha=0.3):
    # joints of shape (num_frames, num_people, num_joints, coord)
    scale_factor = float(np.random.uniform(1-alpha, 1+alpha, 1))
    joints = joints * scale_factor
    return joints



def joint_courruption(input_data):
    out = input_data.copy()

    flip_prob = random.random()

    if flip_prob < 0.5:

        # joint_indicies = np.random.choice(25, random.randint(5, 10), replace=False)
        joint_indicies = np.random.choice(25, 15, replace=False)
        out[:, :, joint_indicies, :] = 0
        return out

    else:
        # joint_indicies = np.random.choice(25, random.randint(5, 10), replace=False)
        joint_indicies = np.random.choice(25, 15, replace=False)

        temp = out[:, :, joint_indicies, :]
        Corruption = np.array([
            [random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)],
            [random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)],
            [random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)]])
        temp = np.dot(temp.transpose([1, 2, 3, 0]), Corruption)
        temp = temp.transpose(3, 0, 1, 2)
        out[:, :, joint_indicies, :] = temp
        return out


def joint_courruption_for_spec(input_data):
    out = input_data.copy()

    flip_prob = random.random()

    if flip_prob < 0.5:

        # joint_indicies = np.random.choice(25, random.randint(5, 10), replace=False)
        joint_indicies = np.random.choice(25, 5, replace=False)
        out[:, :, joint_indicies, :] = 0
        return out

    else:
        # joint_indicies = np.random.choice(25, random.randint(5, 10), replace=False)
        joint_indicies = np.random.choice(25, 5, replace=False)

        temp = out[:, :, joint_indicies, :]
        Corruption = np.array([
            [random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)],
            [random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)],
            [random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)]])
        temp = np.dot(temp.transpose([1, 2, 3, 0]), Corruption / 5)
        temp = temp.transpose(3, 0, 1, 2)
        out[:, :, joint_indicies, :] = temp
        return out



def pose_augmentation(input_data):
    Shear = np.array([
        [1, random.uniform(-1, 1), random.uniform(-1, 1)],
        [random.uniform(-1, 1), 1, random.uniform(-1, 1)],
        [random.uniform(-1, 1), random.uniform(-1, 1), 1]
    ])

    temp_data = input_data.copy()
    result = np.dot(temp_data.transpose([1, 2, 3, 0]), Shear.transpose())
    output = result.transpose(3, 0, 1, 2)

    return output

def random_transformation(input_data): # 3, 64, 32, 1

    x = random.uniform(-1, 1)
    y = random.uniform(-1, 1)
    z = random.uniform(-1, 1)
    x_theta = random.uniform(-30, 30)
    y_theta = random.uniform(-30, 30)
    z_theta = random.uniform(-5, 5)
    input_data = input_data.transpose([1,2,0,3])
    output = transform_joints(input_data[:,:,:,0], x, y, z, x_theta, y_theta, z_theta)
    input_data[:, :, :, 0] = output
    input_data = input_data.transpose([2,0,1,3])
    return input_data


def pose_augmentation_for_spec(input_data):

    input_data = random_transformation(input_data)

    Shear = np.array([
        [1, random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)],
        [random.uniform(-0.5, 0.5), 1, random.uniform(-0.5, 0.5)],
        [random.uniform(-0.5, 0.5), random.uniform(-0.5,0.5), 1]
    ])

    temp_data = input_data.copy()
    result = np.dot(temp_data.transpose([1, 2, 3, 0]), Shear.transpose())
    output = result.transpose(3, 0, 1, 2)

    return output



def temporal_cropresize(input_data, num_of_frames, l_ratio, output_size):
    C, T, V, M = input_data.shape

    # Temporal crop
    min_crop_length = 64

    scale = np.random.rand(1) * (l_ratio[1] - l_ratio[0]) + l_ratio[0]
    temporal_crop_length = np.minimum(np.maximum(int(np.floor(num_of_frames * scale)), min_crop_length), num_of_frames)

    start = np.random.randint(0, num_of_frames - temporal_crop_length + 1)
    temporal_context = input_data[:, start:start + temporal_crop_length, :, :]

    # interpolate
    temporal_context = torch.tensor(temporal_context, dtype=torch.float)
    temporal_context = temporal_context.permute(0, 2, 3, 1).contiguous().view(C * V * M, temporal_crop_length)
    temporal_context = temporal_context[None, :, :, None]
    temporal_context = F.interpolate(temporal_context, size=(output_size, 1), mode='bilinear', align_corners=False)
    temporal_context = temporal_context.squeeze(dim=3).squeeze(dim=0)
    temporal_context = temporal_context.contiguous().view(C, V, M, output_size).permute(0, 3, 1, 2).contiguous().numpy()

    return temporal_context


def crop_subsequence(input_data, num_of_frames, l_ratio, output_size):
    C, T, V, M = input_data.shape

    if l_ratio[0] == 0.5:
        # if training , sample a random crop

        min_crop_length = 64
        scale = np.random.rand(1) * (l_ratio[1] - l_ratio[0]) + l_ratio[0]
        temporal_crop_length = np.minimum(np.maximum(int(np.floor(num_of_frames * scale)), min_crop_length),
                                          num_of_frames)

        start = np.random.randint(0, num_of_frames - temporal_crop_length + 1)
        temporal_crop = input_data[:, start:start + temporal_crop_length, :, :]

        temporal_crop = torch.tensor(temporal_crop, dtype=torch.float)
        temporal_crop = temporal_crop.permute(0, 2, 3, 1).contiguous().view(C * V * M, temporal_crop_length)
        temporal_crop = temporal_crop[None, :, :, None]
        temporal_crop = F.interpolate(temporal_crop, size=(output_size, 1), mode='bilinear', align_corners=False)
        temporal_crop = temporal_crop.squeeze(dim=3).squeeze(dim=0)
        temporal_crop = temporal_crop.contiguous().view(C, V, M, output_size).permute(0, 3, 1, 2).contiguous().numpy()

        return temporal_crop

    else:
        # if testing , sample a center crop

        start = int((1 - l_ratio[0]) * num_of_frames / 2)
        data = input_data[:, start:num_of_frames - start, :, :]
        temporal_crop_length = data.shape[1]

        temporal_crop = torch.tensor(data, dtype=torch.float)
        temporal_crop = temporal_crop.permute(0, 2, 3, 1).contiguous().view(C * V * M, temporal_crop_length)
        temporal_crop = temporal_crop[None, :, :, None]
        temporal_crop = F.interpolate(temporal_crop, size=(output_size, 1), mode='bilinear', align_corners=False)
        temporal_crop = temporal_crop.squeeze(dim=3).squeeze(dim=0)
        temporal_crop = temporal_crop.contiguous().view(C, V, M, output_size).permute(0, 3, 1, 2).contiguous().numpy()

        return temporal_crop

def transform_translation(x,y,z):
    return np.array([[1,0,0,x],
                     [0,1,0,y],
                     [0,0,1,z],
                     [0,0,0,1]])

def transform_z_rotation(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0, 0],
              [np.sin(theta), np.cos(theta), 0, 0],
              [0, 0, 1, 0],
              [0,0,0,1]])

def transform_y_rotation(theta):
    return np.array([[np.cos(theta), 0, np.sin(theta), 0],
                     [0, 1, 0, 0],
                     [-np.sin(theta),  0, np.cos(theta), 0],
                     [0, 0, 0, 1]])

def transform_x_rotation(theta):
    return np.array([[1, 0, 0, 0],
                     [0, np.cos(theta), -np.sin(theta), 0],
                     [0, np.sin(theta), np.cos(theta), 0],
                     [0, 0, 0, 1]])

def transform_joints(joints, x=0, y=0, z=0, x_theta=0, y_theta=0, z_theta=0):
    #joints of shape
    num_frames, num_joints, num_channels = joints.shape
    x_theta = np.pi / 180 * x_theta
    y_theta = np.pi / 180 * y_theta
    z_theta = np.pi / 180 * z_theta
    t_vec = transform_translation(x, y, z)[:, -1][0:-1]
    t_vec = t_vec.reshape(1, 1, 3)
    t_vec = np.tile(t_vec, (num_frames, num_joints, 1))
    joints_wst = joints#-t_vec
    joints_wst = np.einsum('abc,kc->abk', joints_wst, transform_z_rotation(z_theta)[:3, :3])
    joints_wst = np.einsum('abc,kc->abk', joints_wst, transform_y_rotation(y_theta)[:3, :3])
    joints_wst = np.einsum('abc,kc->abk', joints_wst, transform_x_rotation(x_theta)[:3, :3])
    joints_wst = joints_wst + t_vec  # use - t_vec to get the best performance for ntu-spc

    return joints_wst
