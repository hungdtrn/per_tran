from calendar import c
import torch

from src.shared.data.utils import CoordinateHandler as BaseCoordinateHandler, normalization_stats
from library.utils import unsqueeze_and_repeat
from library.dataset.bimanual.processing import skeleton_to_box_torch, get_hands_from_arm_pytorch, get_box_center_torch, get_arm_and_hand_numpy, iou, get_arm_from_arm_pytorch

def sliding_windows(data, size, stride=1):
    seq_len = data.shape[1]
    num_seq = (seq_len - size) // stride + 1
    out = []
    
    for i in range(num_seq):
        start = i * stride
        end = i * stride + size
        
        if end > seq_len:
            break
        
        out.append(data[:, start:end])
    return out

class CoordinateHandler(BaseCoordinateHandler):
    def distance_from_human_to_obj(self, human, obj):
        original_shape = human.shape
        human = human.reshape(-1, human.size(-1))
        obj = obj.reshape(-1, obj.size(-1))
        
        hands = get_hands_from_arm_pytorch(human)
        hands_center = get_box_center_torch(hands)
        
        # batch, 2
        box_center = get_box_center_torch(obj)
        
        distance = torch.linalg.norm(hands_center - box_center, axis=-1)
        distance = distance.reshape(tuple(list(original_shape)[:-1]))
        return distance
    
    def get_arm_and_hands(self, human):
        return get_arm_from_arm_pytorch(human), get_hands_from_arm_pytorch(human)

    def iou_from_human_to_obj(self, human, obj):
        original_shape = human.shape
        human = human.reshape(-1, human.size(-1))
        obj = obj.reshape(-1, obj.size(-1))
        
        hands = get_hands_from_arm_pytorch(human)
        out = iou(hands, obj)
        return out.reshape(list(original_shape)[:-1])

    def get_arm_center(self, arm):
        assert len(arm.shape) == 2
        assert arm.shape[-1] == 10
        return arm[:, :2]

    def compose_coordinate(self, coordinate: torch.Tensor, 
                        global_component: torch.Tensor):
        assert len(coordinate) == len(global_component)

        original_coordinate_shape = coordinate.shape
        original_global_shape = global_component.shape

        assert original_coordinate_shape[-1] % 2 == 0
        assert original_global_shape[-1] % 2 ==0
        assert global_component.shape[-1] == 10
        
        coordinate = coordinate.reshape(-1, original_coordinate_shape[-1])
        num_coordinate_points = original_coordinate_shape[-1] // 2

        global_component = global_component.reshape(-1, original_global_shape[-1])

        center = self.get_arm_center(global_component)
        center = unsqueeze_and_repeat(center, 1, num_coordinate_points).view(-1, num_coordinate_points * 2)

        coordinate = coordinate + center
        return coordinate.reshape(original_coordinate_shape)

    def decompose_coordinate(self, coordinate, global_component=None):
        """
        Args:
            coordinate: (*, dim)
            global_component: (*, dim)
        """
        
        origin_shape = coordinate.shape
        num_points = origin_shape[-1] // 2
        
        coordinate = coordinate.reshape(-1, origin_shape[-1])
        center = None
        
        if global_component is None:
            center = self.get_arm_center(coordinate)
            global_component = coordinate.clone()
        else:
            center = self.get_arm_center(global_component.reshape(-1, global_component.shape[-1]))

        assert global_component.shape[-1] == 10

        global_component_shape = list(origin_shape)
        global_component_shape[-1] = global_component.shape[-1]

        # (len, 2)
        center = unsqueeze_and_repeat(center, 1, num_points).view(-1, num_points * 2)
        coordinate = (coordinate - center).float()

        
        return coordinate.reshape(origin_shape), global_component.reshape(global_component_shape)    
