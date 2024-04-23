import os

import cv2
import dgl
import torch
import numpy as np

def rand_color():
    # colors = []
    # colors = np.random.randint(0, 255, (100, 3))
    # colors[:, 0] = 0
    colors = np.array([[  0.,  99., 116.],
       [184., 133.,  10.],
       [ 60.,  60.,  60.],
       [ 89.,  47.,  13.],
       [ 89.,  30., 113.],
       [140.,   8.,   0.],
       [ 18., 113.,  28.],
       [177.,  64.,  13.],
       [  0.,  28., 127.]], dtype=int)
    
    colors = np.concatenate([np.array([[170, 68, 0],
                                       [0, 128, 0],
                                       [0, 0, 170],
                                       [0, 85, 212]]), colors])
    additional_colors = []
    for value in range(0, 150, 50):        
        for i in range(3):
            color = [0, 0, 0]
            
            color[i] = 255 - value
            additional_colors.append(np.array(color))

    # colors = colors + blend(0, 0, 0, 0)
    # colors = np.stack(colors, axis=0)
    colors = np.concatenate([colors, additional_colors])

    return colors

def get_key_from_frame():
    code = 0
    key_pressed = cv2.waitKey(0) & 0xFF
    if key_pressed == ord('q'):
        code = 2
    elif key_pressed == ord('b'):
        code = 1

    return code 

colors = rand_color()
colors = [x.tolist() for x in colors]


def draw_graph(draw_obj_fn, draw_human_fn, 
               oh_graph, obj_type_name, data_name, 
               obs_len, length=None, 
               time_window=0, frame_paths=None, frame_name="", start_idx=0):
    human_nodes = oh_graph.nodes["human"]
    obj_nodes = oh_graph.nodes["obj"]
    
    human_loc = human_nodes.data[data_name]
    obj_loc = obj_nodes.data[data_name]
    
    if length is not None:
        human_loc = human_loc[:, :length]
        obj_loc = obj_loc[:, :length]

    obj_type = torch.argmax(obj_nodes.data["type"], dim=1)
    obj_type = [obj_type_name[int(i)] for i in obj_type]

    seq_len = human_loc.shape[1]

    prev_human, prev_obj = [], []
    for seq_idx in range(seq_len):
        if seq_idx < obs_len:
            stage = "Observation"
            # continue
        else:
            stage = "Prediction"
        
        if frame_name != "":
            stage = "{}, Stage: {}".format(frame_name, stage)
            
        if frame_paths is not None:
            current_frame = cv2.imread(frame_paths[seq_idx])
            if current_frame is None:
                yield None
                
            current_frame = np.uint8(np.ones_like(current_frame) * 255 * 0.6 + current_frame * 0.4)
        else:
            current_frame = np.ones((480, 640, 3)).astype(np.uint8) * 255
        prev_frame = current_frame.copy()
        
        for prev_h in prev_human:
            for i, h in enumerate(prev_h):
                prev_frame = draw_human_fn(prev_frame, h, type="human", color=colors[i+start_idx], thickness=2)
            
        for prev_o in prev_obj:
            for i, o in enumerate(prev_o):
                # prev_frame = draw_object(prev_frame, o, type=obj_type[i], color=colors[len(human_loc) + i], thickness=2)
                prev_frame = draw_obj_fn(prev_frame, o, type=obj_type[i], color=colors[i+3], thickness=1)

        weight = 0.3
        current_frame = np.uint8(prev_frame * weight + current_frame * (1-weight))

        for i, h in enumerate(human_loc):
            current_frame = draw_human_fn(current_frame, h[seq_idx], type="human", color=colors[i+start_idx], thickness=5)

        for i, o in enumerate(obj_loc):
            current_frame = draw_obj_fn(current_frame, o[seq_idx], type=obj_type[i], color=colors[i+3], thickness=3)

        # new_frame = np.ones((current_frame.shape[0] + 100,
        #                      current_frame.shape[1], 
        #                      3)) * 255
        # new_frame[100:] = current_frame
        # current_frame = new_frame
        # current_frame = cv2.putText(current_frame, stage, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        
        prev_human.append(human_loc[:, seq_idx])
        prev_obj.append(obj_loc[:, seq_idx])
        
        if len(prev_human) > time_window:
            prev_human.pop(0)
            
        if len(prev_obj) > time_window:
            prev_obj.pop(0)

        yield current_frame

    yield None


def draw_switch(switch_data, name, color_list):
    drawers = [SwitchDrawer(data, name, color) for (data, color) in zip(switch_data, color_list)]
    seq_len = switch_data.shape[1]
    for seq_idx in range(seq_len):
        yield np.concatenate([d.draw(seq_idx) for d in drawers], 0)

    yield None


class GeneralFrameDrawer:
    def __init__(self, draw_obj_fn, draw_human_fn, pixel_fn,
                 cfg, dataset, kwargs, model=None, model_name=None) -> None:
        self.cfg = cfg
        self.dataset = dataset
        self.model = model

        self.draw_obj_fn = draw_obj_fn
        self.draw_human_fn = draw_human_fn
        
        # The function converting the coordinates to pixel for visualization
        self.pixel_fn = pixel_fn
        
        self.is_full_seq = kwargs.get("is_full_seq", False)
        self.video_path = kwargs.get("video_path", "")
        
        if model_name is None:
            self.model_name = kwargs.get("model_name", None)
        else:
            self.model_name = model_name
        
        if self.video_path is None:
            self.video_path = ""
            
        self.is_ego = kwargs.get("is_ego", False)
        self.is_draw_switch = kwargs.get("is_draw_switch", False)
        
    def get_data_drawer(self, data_idx): 
        g, _ = self.dataset[data_idx]
        obj_type_name = self.dataset.additional_info["object_types"]
        obs_len = g.nodes["human"].data["x"].size(1)

        if self.is_full_seq:
            seq_len = int(torch.max(torch.sum(g.nodes["human"].data["mask"], 1)).item())
        else:
            seq_len = g.nodes["human"].data["y"].size(1) + obs_len
        
        frame_paths = None
        if self.video_path != "":
            video_idx = self.dataset.additional_info["video_idx"][data_idx]
            
            if self.is_full_seq:
                sampling_rate = self.cfg.sampling_rate
                frame_idx = np.arange(len(os.listdir(os.path.join(self.video_path, video_idx))))
                frame_idx = frame_idx[sampling_rate-1::sampling_rate]
            else:
                frame_idx = self.dataset.additional_info["frame_idx"][data_idx]
                print(frame_idx)
                
            frame_paths = [os.path.join(self.video_path, video_idx,
                                        "frame_{}.png".format(x)) for x in frame_idx]
            
        g.nodes["human"].data["seq"] = torch.cat([g.nodes["human"].data["x"], 
                                                g.nodes["human"].data["y"]], 1)
        g.nodes["obj"].data["seq"] = torch.cat([g.nodes["obj"].data["x"], 
                                                g.nodes["obj"].data["y"]], 1)

        g, params = self.pixel_fn(g)

        drawers = [draw_graph(self.draw_obj_fn, self.draw_human_fn,
                                g, obj_type_name, "seq", obs_len, seq_len, frame_paths=frame_paths)]
        
        ego_drawers = []
        if self.is_ego:
            num_human = g.num_nodes("human")
            num_obj = g.num_nodes("obj")

            center_idx, leaf_idx = g.edges(etype=("center", "interacts", "leaf"))
            drawers = []
            for i in range(num_human):
                with g.local_scope():
                    ego_g = dgl.node_subgraph(g, {
                        "human": [i],
                        "center": [i],
                        "obj": np.arange(num_obj),
                        "leaf": leaf_idx[center_idx == i]
                    })
                    
                    # ego_g.nodes["human"].data["x"] = ego_g.nodes["center"].data["x"]
                    # ego_g.nodes["human"].data["y"] = ego_g.nodes["center"].data["y"]
                    
                    # ego_g.nodes["obj"].data["x"] = ego_g.nodes["obj"].data["x"] 
                    # ego_g.nodes["obj"].data["y"] = ego_g.nodes["leaf"].data["y"]

                    ego_g.nodes["human"].data["seq"] = torch.cat([ego_g.nodes["human"].data["x"], 
                                                            ego_g.nodes["human"].data["y"]], 1)
                    ego_g.nodes["obj"].data["seq"] = torch.cat([ego_g.nodes["obj"].data["x"], 
                                                            ego_g.nodes["obj"].data["y"]], 1) 
                    
                    ego_g.nodes["human"].data["seq"] = ego_g.nodes["human"].data["seq"][:, :-1]
                    ego_g.nodes["obj"].data["seq"] = ego_g.nodes["obj"].data["seq"][:, :-1] * ego_g.nodes["leaf"].data["is_inter"]

                    ego_g = self.pixel_fn(ego_g)

                    ego_drawers.append(draw_graph(self.draw_obj_fn, self.draw_human_fn, 
                                              ego_g, obj_type_name, "seq", obs_len, seq_len, frame_paths=frame_paths, start_idx=i))

        out =  {
            "frame": drawers,
            "pixel_param": params
        }
        
        if self.is_draw_switch:
            switch_score = g.nodes["human"].data["switch_score"]
            segment_score = g.nodes["human"].data["segment_score"]
            out["switch"] = {
                "switch": draw_switch(switch_score[:, :seq_len].cpu().detach().numpy(), 
                                        "switch_score", colors),
                
                "segment": draw_switch(segment_score[:, :seq_len].cpu().detach().numpy(), 
                                        "segment_score", colors),
            }
        
        return out
    
    def get_model_drawer(self, data_idx, device, crnn_format=False, pixel_params=None):
        g, _ = self.dataset[data_idx]
        g = g.to(device)
        obj_type_name = self.dataset.additional_info["object_types"]
        
        if self.is_full_seq:
            seq_len = int(torch.max(torch.sum(g.nodes["human"].data["mask"], 1)).item())
        else:
            seq_len = g.nodes["human"].data["y"].size(1) + g.nodes["human"].data["x"].size(1)

        data_stats = {}
        for k, v in self.dataset.additional_info["data_stats"].items():
            if ("human" in k) or ("obj" in k) or ("point" in k):
                data_stats[k] = v.to(device)
            else:
                data_stats[k] = v

        additional_info = {}
        additional_info["post_process"] = True
        additional_info["data_stats"] = data_stats

        pred_len = g.nodes["human"].data["y"].size(1)
        _,  (y_pred_human, y_pred_obj), additional_output = self.model(g, pred_len, 
                                                                    additional_info, is_training=False,
                                                                    is_visualize=True)

        if crnn_format:
            y_pred_obj = torch.cat([y_pred_obj, torch.zeros(len(y_pred_human), 
                                                            y_pred_obj.size(1), 
                                                            y_pred_obj.size(-1)).to(y_pred_obj.device)])

        # num_pred = int(torch.max(torch.sum(g.nodes["human"].data["mask"], 1)).item())
        
        frame_paths = None
        # print(self.dataset.additional_info["seg_info"][data_idx])
        if self.video_path != "":
            video_idx = self.dataset.additional_info["video_idx"][data_idx]
            
            sampling_rate = self.cfg.sampling_rate
            if self.is_full_seq:
                sampling_rate = self.cfg.sampling_rate
                frame_idx = np.arange(len(os.listdir(os.path.join(self.video_path, video_idx))))
                frame_idx = frame_idx[sampling_rate-1::sampling_rate]
            else:
                frame_idx = self.dataset.additional_info["frame_idx"][data_idx]
                print(frame_idx)
            frame_paths = [os.path.join(self.video_path, video_idx,
                                        "frame_{}.png".format(x)) for x in frame_idx]

        obs_len = g.nodes["human"].data["x"].size(1)
        for node_name in ["human", "obj"]:
            g.nodes[node_name].data["seq"] = torch.cat([g.nodes[node_name].data["x"],
                                                        g.nodes[node_name].data["y"]], dim=1)

        g.nodes['human'].data["pred_seq"] = torch.cat([g.nodes['human'].data["x"],
                                                        y_pred_human], dim=1)

        if len(y_pred_obj) < len(g.nodes['obj'].data["y"]):
            new_y_pred_obj = torch.zeros_like(g.nodes['obj'].data["y"])
            new_y_pred_obj[:len(y_pred_obj)] = y_pred_obj
            y_pred_obj = new_y_pred_obj

        g.nodes['obj'].data["pred_seq"] = torch.cat([g.nodes['obj'].data["x"],
                                                    y_pred_obj], dim=1)

        g, pixel_params = self.pixel_fn(g, pixel_params)

        gt_drawers = [draw_graph(self.draw_obj_fn, self.draw_human_fn, 
                                 g, obj_type_name, "seq", obs_len, seq_len, frame_paths=frame_paths, frame_name="Ground truth")]
        pred_drawers = [draw_graph(self.draw_obj_fn, self.draw_human_fn,
                                   g, obj_type_name, "pred_seq", obs_len, seq_len, frame_paths=frame_paths, frame_name="Our Prediction")]
        
        out =  {
            "frame": gt_drawers,
            "pred": pred_drawers,
            "pixel_params": pixel_params
        }
        
        ego_drawers = []
        if self.is_ego:
            num_human = g.num_nodes("human")
            num_obj = g.num_nodes("obj")

            center_idx, leaf_idx = g.edges(etype=("center", "interacts", "leaf"))
            for i in range(num_human):
                with g.local_scope():
                    ego_g = dgl.node_subgraph(g, {
                        "human": [i],
                        "center": [i],
                        "obj": np.arange(num_obj),
                        "leaf": leaf_idx[center_idx == i]
                    })
                    
                    # ego_g.nodes["human"].data["x"] = ego_g.nodes["center"].data["x"]
                    # ego_g.nodes["human"].data["y"] = ego_g.nodes["center"].data["y"]
                    
                    # ego_g.nodes["obj"].data["x"] = ego_g.nodes["obj"].data["x"] 
                    # ego_g.nodes["obj"].data["y"] = ego_g.nodes["leaf"].data["y"]

                    ego_g.nodes["human"].data["pred_seq"] = ego_g.nodes["human"].data["pred_seq"][:, :-1]
                    ego_g.nodes["obj"].data["pred_seq"] = ego_g.nodes["obj"].data["pred_seq"][:, :-1] * ego_g.nodes["leaf"].data["is_inter_seq"].unsqueeze(-1)

                    ego_g = self.pixel_fn(ego_g)
                    ego_drawers.append(draw_graph(self.draw_obj_fn, self.draw_human_fn, 
                                              ego_g, obj_type_name, "pred_seq", obs_len, seq_len, frame_paths=frame_paths, start_idx=i, frame_name="ego_{}".format(i)))
        
        if len(ego_drawers) > 0:
            out["ego"] = ego_drawers    
        
        if self.is_draw_switch and "pred_switch_score" in additional_output:
            out["switch"] = {
                "pred_switch": draw_switch(additional_output["pred_switch_score"][:, :seq_len].cpu().detach().numpy(), 
                                    "pred_switch_score", colors),
                # "gt_switch": draw_switch(additional_output["gt_switch_score"][:, :seq_len].cpu().detach().numpy(), 
                #                             "gt_switch_score", colors)
            }
            
            if "pred_segment_score" in additional_output:
                out["switch"].update({
                    "pred_segment": draw_switch(additional_output["pred_segment_score"][:, :seq_len].cpu().detach().numpy(), 
                                    "pred_segment_score", colors),
                    # "gt_segment": draw_switch(additional_output["gt_segment_score"][:, :seq_len].cpu().detach().numpy(), 
                    #                             "gt_segment_score", colors)
                })
            
            # for k, v in self.model.visualize_data.items():
            #     if "score" in k:
            #         print(k, v.shape)
            #         out["switch"][k] = draw_switch(v[:, :seq_len].cpu().detach().numpy(), k, colors)
        additional_output["object_type"] = torch.argmax(g.nodes["obj"].data["type"], dim=1)
        out["colors"] = colors
        out["additional_output"] = additional_output
        return out   


class SwitchDrawer:
    def __init__(self, switch_seq, switch_name, color, width=1800, heigh=300) -> None:
        frame = np.ones((heigh, width, 3)).astype(np.uint8) * 255
        pred_len = switch_seq.shape[-1]
        
        start_x = 90
        start_y = heigh - 55
        
        font_size = 1.25
        font_thickness = 2
        font = cv2.FONT_HERSHEY_SIMPLEX

        # frame = cv2.putText(frame, "time", (frame.shape[1] // 2, heigh - 5), font,
        #                     font_size, (0, 0, 0),
        #                     font_thickness, lineType=cv2.LINE_AA)
        
        signal_height = start_y - 20
        yoffset = 10
        
        xoffset = (frame.shape[1] - start_x) / pred_len
        
        y = start_y

        frame = cv2.line(frame, (int(start_x), int(y)), (int(start_x + xoffset * (pred_len - 1)), int(y)), (0, 0, 0), 2)
        frame = cv2.line(frame, (int(start_x), int(y - signal_height // 2 + 5)), (int(start_x + xoffset * (pred_len - 1)), int(y - signal_height // 2 + 5)), (0, 0, 0), 2)
        frame = cv2.arrowedLine(frame, (int(start_x), int(y)), 
                                (int(start_x), int(y - signal_height)), (0, 0, 0), 2)
        frame = cv2.arrowedLine(frame, (int(start_x + xoffset * (pred_len - 2)), int(y)), 
                                (int(start_x + xoffset * (pred_len - 1)), int(y)), (0, 0, 0), 2, tipLength=0.5)

        # text_img = np.zeros_like(frame)
        # text_img = cv2.putText(text_img, "switch_score", (4, y - signal_height // 3), font,
        #                     font_size, color,
        #                     font_thickness, lineType=cv2.LINE_AA)
        # M = cv2.getRotationMatrix2D((frame.shape[1]//2, frame.shape[0]//2), 
        #                             90, 1)
        # text_img = cv2.warpAffine(text_img, M, (text_img.shape[1], text_img.shape[0]))
        # frame = frame + text_img
        # frame_with_name = np.ones((heigh + 100, width, 3)).astype(np.uint8) * 255
        # frame_with_name[100:] = frame
        # frame = frame_with_name
        # frame = cv2.putText(frame, "Switch score", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)


        self.switch_seq = switch_seq
        self.frame = frame
        self.start_x = start_x
        self.start_y = start_y
        
        self.xoffset = xoffset
        self.yoffset = yoffset
        
        self.signal_height = signal_height
        self.color = color
        self.y = y
                
    def draw(self, t):
        x = self.start_x + self.xoffset * t
        flag = self.switch_seq[t]
        flagx = x
        flagy = flag * -self.signal_height + self.y

        self.frame = cv2.circle(self.frame, (int(flagx), int(flagy)), 8, self.color, -1)
        
        return self.frame

def id_to_file_name(id, num_digit):
    file_name = str(id)
    for i in range(num_digit - len(file_name)):
        file_name = "0" + file_name
        
    return file_name