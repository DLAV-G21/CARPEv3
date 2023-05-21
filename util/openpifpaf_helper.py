# copyed from openpifpaf and PE-former
# https://github.com/openpifpaf/openpifpaf/blob/main/src/openpifpaf/plugins/apollocar3d/constants.py
# https://github.com/openpifpaf/openpifpaf/blob/main/src/openpifpaf/plugins/apollocar3d/transforms.py
from collections import defaultdict
from typing import List, Tuple


def skeleton_mapping(kps_mapping):
    """Map the subset of keypoints from 0 to n-1"""
    map_sk = defaultdict(lambda: 100)  # map to 100 the keypoints not used
    for i, j in zip(kps_mapping, range(len(kps_mapping))):
        map_sk[i] = j
    return map_sk


def transform_skeleton(skeleton_orig: List[Tuple[int, int]], kps_mapping) -> List[Tuple[int, int]]:
    """
    Transform the original apollo skeleton of 66 joints into a skeleton from 1 to n
    """
    map_sk = skeleton_mapping(kps_mapping)
    # skeleton = [[dic_sk[i], dic_sk[j]] for i, j in SKELETON]  # TODO
    skeleton = []
    for i, j in skeleton_orig:
        skeleton.append((map_sk[i] + 1, map_sk[j] + 1))   # skeleton starts from 1
    return skeleton


CAR_KEYPOINTS_24 = [
    'front_up_right',       # 1
    'front_up_left',        # 2
    'front_light_right',    # 3
    'front_light_left',     # 4
    'front_low_right',      # 5
    'front_low_left',       # 6
    'central_up_left',      # 7
    'front_wheel_left',     # 8
    'rear_wheel_left',      # 9
    'rear_corner_left',     # 10
    'rear_up_left',         # 11
    'rear_up_right',        # 12
    'rear_light_left',      # 13
    'rear_light_right',     # 14
    'rear_low_left',        # 15
    'rear_low_right',       # 16
    'central_up_right',     # 17
    'rear_corner_right',    # 18
    'rear_wheel_right',     # 19
    'front_wheel_right',    # 20
    'rear_plate_left',      # 21
    'rear_plate_right',     # 22
    'mirror_edge_left',     # 23
    'mirror_edge_right',    # 24
]

SKELETON_ORIG = [
    (49, 46), (49, 8), (49, 57), (8, 0), (8, 11), (57, 0),
    (57, 52), (0, 5), (52, 5), (5, 7),  # frontal
    (7, 20), (11, 23), (20, 23), (23, 25), (34, 32),
    (9, 11), (9, 7), (9, 20), (7, 0), (9, 0), (9, 8),  # L-lat
    (24, 33), (24, 25), (24, 11), (25, 32), (25, 28),
    (33, 32), (33, 46), (32, 29), (28, 29),  # rear
    (65, 64), (65, 25), (65, 28), (65, 20), (64, 29),
    (64, 32), (64, 37), (29, 37), (28, 20),  # new rear
    (34, 37), (34, 46), (37, 50), (50, 52), (46, 48), (48, 37),
    (48, 49), (50, 57), (48, 57), (48, 50)
]

KPS_MAPPING = [49, 8, 57, 0, 52, 5, 11, 7, 20, 23, 24, 33, 25, 32, 28,
               29, 46, 34, 37, 50, 65, 64, 9, 48]

CAR_SKELETON_24 = transform_skeleton(SKELETON_ORIG, KPS_MAPPING)

CAR_KEYPOINTS_66 = [
    "top_left_c_left_front_car_light",      # 0
    "bottom_left_c_left_front_car_light",   # 1
    "top_right_c_left_front_car_light",     # 2
    "bottom_right_c_left_front_car_light",  # 3
    "top_right_c_left_front_fog_light",     # 4
    "bottom_right_c_left_front_fog_light",  # 5
    "front_section_left_front_wheel",       # 6
    "center_left_front_wheel",              # 7
    "top_right_c_front_glass",              # 8
    "top_left_c_left_front_door",           # 9
    "bottom_left_c_left_front_door",        # 10
    "top_right_c_left_front_door",          # 11
    "middle_c_left_front_door",             # 12
    "front_c_car_handle_left_front_door",   # 13
    "rear_c_car_handle_left_front_door",    # 14
    "bottom_right_c_left_front_door",       # 15
    "top_right_c_left_rear_door",           # 16
    "front_c_car_handle_left_rear_door",    # 17
    "rear_c_car_handle_left_rear_door",     # 18
    "bottom_right_c_left_rear_door",        # 19
    "center_left_rear_wheel",               # 20
    "rear_section_left_rear_wheel",         # 21
    "top_left_c_left_rear_car_light",       # 22
    "bottom_left_c_left_rear_car_light",    # 23
    "top_left_c_rear_glass",                # 24
    "top_right_c_left_rear_car_light",      # 25
    "bottom_right_c_left_rear_car_light",   # 26
    "bottom_left_c_trunk",                  # 27
    "Left_c_rear_bumper",                   # 28
    "Right_c_rear_bumper",                  # 29
    "bottom_right_c_trunk",                 # 30
    "bottom_left_c_right_rear_car_light",   # 31
    "top_left_c_right_rear_car_light",      # 32
    "top_right_c_rear_glass",               # 33
    "bottom_right_c_right_rear_car_light",  # 34
    "top_right_c_right_rear_car_light",     # 35
    "rear_section_right_rear_wheel",        # 36
    "center_right_rear_wheel",              # 37
    "bottom_left_c_right_rear_car_door",    # 38
    "rear_c_car_handle_right_rear_car_door",    # 39
    "front_c_car_handle_right_rear_car_door",   # 40
    "top_left_c_right_rear_car_door",       # 41
    "bottom_left_c_right_front_car_door",   # 42
    "rear_c_car_handle_right_front_car_door",   # 43
    "front_c_car_handle_right_front_car_door",  # 44
    "middle_c_right_front_car_door",        # 45
    "top_left_c_right_front_car_door",      # 46
    "bottom_right_c_right_front_car_door",  # 47
    "top_right_c_right_front_car_door",     # 48
    "top_left_c_front_glass",               # 49
    "center_right_front_wheel",             # 50
    "front_section_right_front_wheel",      # 51
    "bottom_left_c_right_fog_light",        # 52
    "top_left_c_right_fog_light",           # 53
    "bottom_left_c_right_front_car_light",  # 54
    "top_left_c_right_front_car_light",     # 55
    "bottom_right_c_right_front_car_light",  # 56
    "top_right_c_right_front_car_light",     # 57
    "top_right_c_front_lplate",             # 58
    "top_left_c_front_lplate",              # 59
    "bottom_right_c_front_lplate",           # 60
    "bottom_left_c_front_lplate",          # 61
    "top_left_c_rear_lplate",               # 62
    "top_right_c_rear_lplate",              # 63
    "bottom_right_c_rear_lplate",           # 64
    "bottom_left_c_rear_lplate", ]            # 65

HFLIP_ids = {
    0: 57,
    1: 56,
    2: 55,
    3: 54,
    4: 53,
    5: 52,
    6: 51,
    7: 50,
    8: 49,
    9: 48,
    10: 47,
    11: 46,
    12: 45,
    13: 44,
    14: 43,
    15: 42,
    16: 41,
    17: 40,
    18: 39,
    19: 38,
    20: 37,
    21: 36,
    22: 35,
    23: 34,
    24: 33,
    25: 32,
    26: 31,
    27: 30,
    28: 29,
    59: 58,
    61: 60,
    62: 63,
    65: 64
}

HFLIP_66 = {}
checklist = []
for ind in HFLIP_ids:
    HFLIP_66[CAR_KEYPOINTS_66[ind]] = CAR_KEYPOINTS_66[HFLIP_ids[ind]]
    HFLIP_66[CAR_KEYPOINTS_66[HFLIP_ids[ind]]] = CAR_KEYPOINTS_66[ind]
    checklist.append(ind)
    checklist.append(HFLIP_ids[ind])
assert sorted(checklist) == list(range(len(CAR_KEYPOINTS_66)))
assert len(HFLIP_66) == len(CAR_KEYPOINTS_66)

CAR_CATEGORIES_66 = ['car']

SKELETON_LEFT = [
    [59, 61], [59, 1], [61, 5], [0, 1], [0, 2], [2, 3], [3, 1], [3, 4], [4, 5],  # front
    [5, 6], [6, 7], [4, 7], [2, 9], [9, 8], [8, 11], [7, 10], [6, 10], [9, 10],  # side front part
    [11, 12], [11, 24], [9, 12], [10, 15], [12, 15],
    [9, 13], [13, 14], [14, 12], [14, 15],  # side middle part
    [24, 16], [12, 16], [12, 17], [17, 18], [18, 16],
    [15, 19], [19, 20], [19, 18], [20, 21], [16, 21],  # side back part
    [16, 22], [21, 28], [22, 23], [23, 28], [22, 25], [25, 26],
    [23, 26], [26, 27], [25, 62], [27, 65], [62, 65], [28, 65]]

SKELETON_RIGHT = [[HFLIP_ids[bone[0]], HFLIP_ids[bone[1]]] for bone in SKELETON_LEFT]

SKELETON_CONNECT = [
    [28, 29], [62, 63], [65, 64], [24, 33], [46, 11],
    [48, 9], [59, 58], [60, 61], [0, 57], [49, 8]]

SKELETON_ALL = SKELETON_LEFT + SKELETON_RIGHT + SKELETON_CONNECT

CAR_SKELETON_66 = [(bone[0] + 1, bone[1] + 1) for bone in SKELETON_ALL]  # COCO style skeleton
