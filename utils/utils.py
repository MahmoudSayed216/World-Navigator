import math
import yaml

def euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x2-x1)**2 + (y2-y1)**2)



def do_collide(x1, y1, w1, h1, x2, y2, w2, h2, margin=0.0): #IoU > 0
    #Bigger Rectangle 
    new_x1 = x1 - margin
    new_y1 = y1 - margin
    new_w1 = w1 + 2*margin
    new_h1 = h1 + 2*margin

    #intersection region coordinates:
    x_left = max(new_x1, x2)
    y_top = max(new_y1, y2)
    x_right = min(new_x1+new_w1, x2+w2)
    y_bottom = min(new_y1+new_h1, y2+h2)

    intersection_w = x_right - x_left
    intersection_h = y_bottom - y_top
    
    return intersection_w >= 0 and intersection_h >= 0



def calculate_allowed_range(s1, s2) -> tuple:
    if s1 > s2:
        return int(s1/2), 'x'
    else:
        return int(s2/2), 'y'



# def world_to_ui_coordinates(x_world, y_world, x_obj, y_obj):
#     pass


def get_configs(world_configs_path):
    with open(world_configs_path) as f:
        configs = yaml.safe_load(f)

    return configs