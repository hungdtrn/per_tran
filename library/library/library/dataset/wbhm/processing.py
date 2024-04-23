
import numpy as np

def create_object_representation(raw_object):
    """Create the representation of the object described in the paper

    Args:
        raw_object ([np.array]): (num_frames, 12), The 3D coordinate of the bounding box
    Returns:
        processed_object ([np.array]): (num_frames, 12): Concatenation of 8 3D vectors
    """    
    output = []
    for i in range(len(raw_object)):
        current_frame = raw_object[i]
        tmp = current_frame.reshape(-1, 3)
        
        min_x, max_x = np.min(tmp[:, 0]), np.max(tmp[:, 0])
        min_y, max_y = np.min(tmp[:, 1]), np.max(tmp[:, 1])
        min_z, max_z = np.min(tmp[:, 2]), np.max(tmp[:, 2])

        output.append([min_x, min_y, min_z, 
                       max_x, min_y, min_z, 
                       max_x, max_y, min_z, 
                       min_x, max_y, min_z, 
                       min_x, min_y, max_z, 
                       max_x, min_y, max_z, 
                       max_x, max_y, max_z, 
                       min_x, max_y, max_z])
        
    return np.array(output)

def create_4vec_obj_representation(object_8vec):
    original_shape = object_8vec.shape
    object_8vec = object_8vec.reshape(-1, 8, 3)

    object_4vec = np.stack([object_8vec[:, 0], object_8vec[:, 1], object_8vec[:, 3], object_8vec[:, 4]], 1)
    
    new_shape = list(original_shape)
    new_shape[-1] = 12
    
    return object_4vec.reshape(new_shape)