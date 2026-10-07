def pos_wh_to_min_max(pos, width, height):
	
	min_coords = pos[0], pos[1]
	max_coords = pos[0] + width, pos[1] + height

	return (min_coords, max_coords)



def euclidean_distance(el1, el2):
	return ((el1[0] - el2[0])**2 + (el1[1] - el2[1])**2)**0.5


def distance_between_point_and_square(square_coords, point_coords): ##* square_coords -> xywh

	between = lambda v, lb, ub : v >= lb and v <= ub


	if between(point_coords[0], square_coords[0], square_coords[0] + square_coords[2]):
		pass


def compute_distance_to_walls(agent_pos: tuple[int], world_dims: tuple[int]):
	pass


def compute_distance_to_square(agent_pos, obstacles):
	dos = []
	
	for obstacle in obstacles:
		obs_pos_min_max = pos_wh_to_min_max(obstacles) 