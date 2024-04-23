import os
import dgl
import cv2
import pickle
import argparse
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

add_size = 100
font_scale = 0.8
font_weight = 2
def vertical_name(frame, name, pos):
    ver_frame = np.ones((add_size, frame.shape[0], 3)).astype(np.uint8) * 255
    ver_frame = cv2.putText(ver_frame, name, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_weight, cv2.LINE_AA)
    ver_frame = cv2.flip(ver_frame.transpose(1, 0, 2), 0)
    
    return ver_frame

def pad(frame, pad_size, dim='x', reverse=False):
    if dim == "x":
        new_frame = np.ones((frame.shape[0], frame.shape[1] + pad_size, 3)).astype(np.uint8) * 255
        if not reverse:
            new_frame[:, pad_size:] = frame
        else:
            new_frame[:, :-pad_size] = frame
    elif dim == "y":
        new_frame = np.ones((frame.shape[0] + pad_size, frame.shape[1], 3)).astype(np.uint8) * 255
        if not reverse:
            new_frame[pad_size:] = frame
        else:
            new_frame[:-pad_size] = frame
    return new_frame
    
def append_to_add_name_vertical(frame, name, pos=(5, 40)):
    vertical_name_frame = vertical_name(frame, name, pos=pos)
    new_frame = np.ones((frame.shape[0], frame.shape[1]+add_size, 3)).astype(np.uint8) * 255
    new_frame[:, add_size:] = frame
    new_frame[:, :add_size] = vertical_name_frame

    return new_frame

def append_to_add_name_horizontal(frame, name, pos=(100, 40), color=(0, 0, 0)):
    if len(name) == 1:
        name = name[0]
        new_frame = np.ones((frame.shape[0] + add_size, frame.shape[1], 3)).astype(np.uint8) * 255
        new_frame[add_size:] = frame
        new_frame = cv2.putText(new_frame, name, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_weight, cv2.LINE_AA)
    else:
        new_frame = np.ones((frame.shape[0] + add_size, frame.shape[1], 3)).astype(np.uint8) * 255
        new_frame[add_size:] = frame
        left_size = 300
        left_frame = np.ones((add_size, left_size, 3)).astype(np.uint8) * 255
        left_frame = cv2.putText(left_frame, name[0], pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_weight, cv2.LINE_AA)
        right_frame = np.ones((add_size, frame.shape[1] - left_size, 3)).astype(np.uint8) * 255
        right_frame = cv2.putText(right_frame, name[1], pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_weight, cv2.LINE_AA)
        
        new_frame[:add_size, :left_size] = left_frame
        new_frame[:add_size, left_size:] = right_frame

    return new_frame


def append_per_tran_name(frame):
    ver_frame = np.ones((add_size, frame.shape[0], 3)).astype(np.uint8) * 255
    ver_frame = cv2.putText(ver_frame, "Factual Switch", (1000, 40), cv2.FONT_HERSHEY_SIMPLEX, font_scale  , (0, 0, 0), font_weight, cv2.LINE_AA)
    ver_frame = cv2.putText(ver_frame, "CounterFactual Switch", (200, 40), cv2.FONT_HERSHEY_SIMPLEX, font_scale , (0, 0, 0), font_weight, cv2.LINE_AA)
    ver_frame = cv2.flip(ver_frame.transpose(1, 0, 2), 0)


    new_frame = np.ones((frame.shape[0], frame.shape[1]+add_size, 3)).astype(np.uint8) * 255
    new_frame[:, add_size:] = frame
    new_frame[:, :add_size] = ver_frame

    return new_frame

def covert_color_format(c):
    return '#%02x%02x%02x' % (c[2],c[1],c[0])

def resize(target_dim, current, dim='x'):
    if dim == "x":
        int_dim = 1
    else:
        int_dim = 0
    ratio = target_dim / current.shape[int_dim]
    if dim == "x":
        target_dim = (target_dim, int(current.shape[0] * ratio))
    else:
        target_dim = (int(current.shape[1] * ratio), target_dim)
    
    return cv2.resize(current, tuple(target_dim))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    # parser.add_argument("graph_id", type=int)
    args = parser.parse_args()
    writer = None
    
    graphs = {}
    num_obs = 10
    is_break = False
    target_idx = [695, 805]
    for gidx in sorted([int(x) for x in os.listdir(args.path) if os.path.isdir(os.path.join(args.path, x))]):
        if gidx not in target_idx:
            continue
        
        gidx = str(gidx)
        print(gidx)
        # if int(gidx) not in [1057, 1058, 2421, 2420, 1059, 2419, 534, 
        #         2773, 2774, 546, 2245, 1487, 540,
        #         2702, 415, 1484, 539, 2248, 2251, 363]:
        #     continue
        # print(gidx)
        frames = []
        for frame_idx in range(num_obs):
            factual_path = os.path.join(args.path, gidx, f"obs_{frame_idx}.png")
            factual_switch_path = os.path.join(args.path, gidx, f"obs_switch_{frame_idx}.png")
            factual_architecture_path = os.path.join(args.path, gidx, f"obs_{frame_idx}_both.png")

            factual_frame = cv2.imread(factual_path)
            factual_switch_frame = cv2.imread(factual_switch_path)
            factual_architecture_frame = cv2.imread(factual_architecture_path)

            frame = factual_frame
            factual_switch_frame = resize(frame.shape[1], factual_switch_frame, dim='x')
            factual_switch_frame = pad(factual_switch_frame, 50, dim='y')
            frame = np.concatenate([frame, factual_switch_frame], axis=0)

            line_frame_ver = np.zeros((len(frame), 1, 3)).astype(np.uint8)
            factual_architecture_frame = resize(frame.shape[0], factual_architecture_frame, dim='y')
            frame = np.concatenate([frame, line_frame_ver, factual_architecture_frame], axis=1)  
            obs_frame = frame
        
        for frame_idx in range(38):
            # print(frame_idx)
            if frame_idx < num_obs:
                stage_color = (0, 0, 0)
                stage = f"Observation {frame_idx}"
                obs_path = os.path.join(args.path, gidx, f"obs_{frame_idx}.png")
                obs_switch_path = os.path.join(args.path, gidx, f"obs_switch_{frame_idx}.png")
                obs_architecture_path = os.path.join(args.path, gidx, f"obs_{frame_idx}_both.png")
                
                obs_frame = cv2.imread(obs_path)
                obs_switch_frame = cv2.imread(obs_switch_path)
                obs_architecture_frame = cv2.imread(factual_architecture_path)
                
                factual_frame = np.ones_like(obs_frame) * 255
                factual_switch_frame = np.ones_like(obs_switch_frame) * 255
                factual_architecture_frame = np.ones_like(obs_architecture_frame) * 255
                count_frame = np.ones_like(obs_frame) * 255
                count_switch_frame = np.ones_like(obs_switch_frame) * 255
                count_architecture_frame = np.ones_like(obs_architecture_frame) * 255
                

                obs_switch_frame = resize(obs_frame.shape[1], obs_switch_frame, dim='x')
                obs_switch_frame = pad(obs_switch_frame, 50, dim='y')
                obs_frame = np.concatenate([obs_frame, obs_switch_frame], axis=0)

                line_frame_ver = np.zeros((len(obs_frame), 1, 3)).astype(np.uint8)
                obs_architecture_frame = resize(obs_frame.shape[0], obs_architecture_frame, dim='y')
                obs_frame = np.concatenate([obs_frame, line_frame_ver, obs_architecture_frame], axis=1)  
                obs_frame = append_to_add_name_horizontal(obs_frame, [f"Stage: {stage}"], color=stage_color)
            else:
                stage_color = (255, 0, 0)
                stage = f"Prediction {frame_idx-num_obs}"
                factual_path = os.path.join(args.path, gidx, f"pred_{frame_idx-num_obs}.png")
                factual_switch_path = os.path.join(args.path, gidx, f"pred_switch_{frame_idx-num_obs}.png")
                factual_architecture_path = os.path.join(args.path, gidx, f"pred_{frame_idx-num_obs}_both.png")

                counterfactual_path = os.path.join(args.path, gidx, f"pred_count_{frame_idx-num_obs}.png")
                counterfactual_switch_path = os.path.join(args.path, gidx, f"pred_count_switch_{frame_idx-num_obs}.png")
                counterfactual_architecture_path = os.path.join(args.path, gidx, f"pred_{frame_idx-num_obs}_count_both.png")

    
                assert os.path.exists(factual_path), factual_path
                assert os.path.exists(factual_switch_path), factual_switch_path
                assert os.path.exists(factual_architecture_path), factual_architecture_path
                factual_frame = cv2.imread(factual_path)
                factual_switch_frame = cv2.imread(factual_switch_path)
                factual_architecture_frame = cv2.imread(factual_architecture_path)
                # factual_architecture_frame = pad(pad(factual_architecture_frame, 1000, dim='y'),
                #                                  50, dim='y', reverse=True)
                
                assert os.path.exists(counterfactual_path), counterfactual_path
                assert os.path.exists(counterfactual_switch_path), counterfactual_switch_path
                assert os.path.exists(counterfactual_architecture_path), counterfactual_architecture_path
                count_frame = cv2.imread(counterfactual_path)
                count_switch_frame = cv2.imread(counterfactual_switch_path)
                count_architecture_frame = cv2.imread(counterfactual_architecture_path)
                # count_architecture_frame = pad(pad(count_architecture_frame, 1000, dim='y'),
                #                                  50, dim='y', reverse=True)

            frame = factual_frame
            factual_switch_frame = resize(frame.shape[1], factual_switch_frame, dim='x')
            factual_switch_frame = pad(factual_switch_frame, 50, dim='y')
            frame = np.concatenate([frame, factual_switch_frame], axis=0)

            line_frame_ver = np.zeros((len(frame), 1, 3)).astype(np.uint8)
            factual_architecture_frame = resize(frame.shape[0], factual_architecture_frame, dim='y')
            frame = np.concatenate([frame, line_frame_ver, factual_architecture_frame], axis=1)  

            count_switch_frame = resize(count_frame.shape[1], count_switch_frame, dim='x')
            count_switch_frame = pad(count_switch_frame, 50, dim='y')
            count_frame = np.concatenate([count_frame, count_switch_frame], axis=0)
            
            line_frame_ver = np.zeros((len(count_frame), 1, 3)).astype(np.uint8)
            count_architecture_frame = resize(count_frame.shape[0], count_architecture_frame, dim='y')
            count_frame = np.concatenate([count_frame, line_frame_ver, count_architecture_frame], axis=1)

            line_frame_hor = np.zeros((1, frame.shape[1], 3)).astype(np.uint8)
            frame = pad(frame, 50, dim='y', reverse=True)
            count_frame = pad(count_frame, 50, dim='y')
            frame = np.concatenate([frame, line_frame_hor, count_frame], axis=0)
            
            print(frame.shape)
            # line_frame_hor = np.zeros((1, factual_frame.shape[1], 3)).astype(np.uint8)
            # frame = np.concatenate([factual_frame, line_frame_hor, count_frame], axis=0)
            # frame = append_to_add_name_horizontal(frame, f"Stage: {stage}", color=stage_color)

            # line_frame_hor = np.zeros((1, factual_switch_frame.shape[1], 3)).astype(np.uint8)
            # switch_frame = np.concatenate([factual_switch_frame, count_switch_frame], axis=0)
            # switch_frame = resize(frame.shape[0], switch_frame, dim='y')

            # line_frame_hor = np.zeros((1, factual_architecture_frame.shape[1], 3)).astype(np.uint8)
            # arcitecture_frame = np.concatenate([factual_architecture_frame, count_architecture_frame], axis=0)
            # arcitecture_frame = resize(frame.shape[0], arcitecture_frame, dim='y')

            # line_frame_ver = np.zeros((len(frame), 1, 3)).astype(np.uint8)
            # frame = np.concatenate([frame, line_frame_ver, switch_frame, line_frame_ver, arcitecture_frame], axis=1)                        
            # frame =  cv2.resize(frame, (1280 - add_size, 960 - add_size))
            frame = append_per_tran_name(frame)
            if frame_idx < num_obs:
                frame = append_to_add_name_horizontal(frame, [""])
            else:
                frame = append_to_add_name_horizontal(frame, [f"{stage}"], color=stage_color)
            # frame = append_to_add_name_horizontal(frame, [f"{stage}"], color=stage_color)
            pad_size = (frame.shape[0] - obs_frame.shape[0]) // 2
            new_obs_frame = np.ones((frame.shape[0], obs_frame.shape[1], 3)).astype(np.uint8) * 255
            new_obs_frame[pad_size:pad_size+obs_frame.shape[0]] = obs_frame.copy()
            
            line_frame_ver = np.zeros((len(frame), 1, 3)).astype(np.uint8)
            new_obs_frame = pad(new_obs_frame, 50, dim='x', reverse=True)
            frame = pad(frame, 50, dim='x')
            frame = np.concatenate([new_obs_frame, line_frame_ver, frame], axis=1)
            
            print(frame.shape)
            frame = cv2.resize(frame, (1280, 720))
            
            # cv2.imshow("frame", frame)
            # cv2.waitKey(0)
            if writer is None:
                writer = cv2.VideoWriter("counterfactual.avi", fourcc=cv2.VideoWriter_fourcc(*'MJPG'), fps=2, frameSize=(frame.shape[1], frame.shape[0]))
            writer.write(frame)
            
    
            
    writer.release()
        # break

    # n_nodes = 4
    # colors = [[0, 0, 0] for i in range(n_nodes + 3)]
    # nodes = [i for i in range(n_nodes)]
    # pos = nx.spring_layout(nodes)
    # edges = []
    # nodes_with_color = [(0, {"color": colors[0]})] + [(i+1, {"color": colors[i+3]}) for i in range(n_nodes - 1)]
    # converted_colors = []
    # G = nx.MultiDiGraph()
    # G.add_nodes_from(nodes_with_color)
    # plt.figure()
    # width = []
    # for i in range(n_nodes):
    #     weight = np.random.randint(1, 10, (n_nodes))
    #     weight = np.exp(weight) / np.sum(np.exp(weight))
    #     for j in range(n_nodes):
    #         if i != j:
    #             G.add_edge(j, i)
    #             edges.append((j, i))
    #             width.append(weight[j] * 2)
                
    #     converted_colors.append(covert_color_format(nodes_with_color[i][1]["color"]))
                
    # nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=converted_colors)
    # arc_rad = 0.25
    # nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=covert_color_format(colors[0]), width=width, connectionstyle=f'arc3, rad = {arc_rad}')
    
    
    
    # plt.axis("off")
    # plt.tight_layout()
    # plt.show()



        # print(is_switch, switch_score)
    #     cv2.imshow("pred", pred)
    #     cv2.waitKey(0)
    cv2.destroyAllWindows()
