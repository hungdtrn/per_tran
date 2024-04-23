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

def append_to_add_name_horizontal(frame, name, pos=(20, 40), color=(0, 0, 0)):
    add_size = 150
    if len(name) == 1:
        name = name[0]
        new_frame = np.ones((frame.shape[0] + add_size, frame.shape[1], 3)).astype(np.uint8) * 255
        new_frame[add_size:] = frame
        new_frame = cv2.putText(new_frame, name, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_weight, cv2.LINE_AA)
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
    ver_frame = cv2.putText(ver_frame, "Transient Graph", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, font_scale  , (0, 0, 0), font_weight, cv2.LINE_AA)
    ver_frame = cv2.putText(ver_frame, "Persistent Graph", (540, 50), cv2.FONT_HERSHEY_SIMPLEX, font_scale , (0, 0, 0), font_weight, cv2.LINE_AA)
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
    target_idx = [2227, 2239, 2255, 2700, 2765, 2780]
    for gidx in sorted([int(x) for x in os.listdir(args.path)]):
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
        for frame_idx in range(29):
            if frame_idx < num_obs:
                stage_color = (0, 0, 0)
                stage = f"Observation {frame_idx}"
                continue 
            else:
                stage_color = (255, 0, 0)
                stage = f"Prediction {frame_idx - num_obs}"
            baseline_frame = cv2.imread(os.path.join(args.path.replace("viz_demo", "viz_demo2"), gidx, f"{frame_idx}_pred.png"))
            # if int(gidx) in [2700, 2765, 2780]:
            #     tmp = np.ones_like(baseline_frame) * 255
            #     tmp[5:] = baseline_frame[:-5]
            #     baseline_frame = tmp
            if baseline_frame is None:
                continue

            baseline_frame = append_to_add_name_horizontal(baseline_frame, ["CRNN", f"Stage: {stage}"], color=stage_color)
            gt_frame = cv2.imread(os.path.join(args.path, gidx, f"{frame_idx}_gt.png"))
            gt_frame = append_to_add_name_horizontal(gt_frame, ["Ground Truth", f"Stage: {stage}"], color=stage_color)
            pred_frame = cv2.imread(os.path.join(args.path, gidx, f"{frame_idx}_pred.png"))
            if pred_frame is None:
                continue

            pred_frame = append_to_add_name_horizontal(pred_frame, ["PTD (Ours)", f"Stage: {stage}"], color=stage_color)

            switch_frame = cv2.imread(os.path.join(args.path, gidx, f"{frame_idx}_switch.png"))
            both_frame = cv2.imread(os.path.join(args.path, gidx, f"{frame_idx}_both.png"))

            if both_frame is None:
                continue
            
            # baseline_frame = pad(baseline_frame, 100, dim='y', reverse=True)
            # gt_frame = pad(gt_frame, 100, dim='y', reverse=True)
            # pred_frame = pad(pred_frame, 100, dim='y', reverse=True)

            # transient_frame = cv2.imread(os.path.join(args.path, gidx, f"{frame_idx}_transient.png"))
            # persistent_frame = np.ones_like(transient_frame) * 255
            # persistent_frame = append_to_add_name(persistent_frame, "Baseline prediction (not draw yet)")
            # transient_frame = append_to_add_name(transient_frame, "Our Transient Graph")
            line_frame_ver = np.zeros((len(baseline_frame), 1, 3)).astype(np.uint8)
            print(baseline_frame.shape, pred_frame.shape)
            frame = np.concatenate([gt_frame, line_frame_ver.copy(), baseline_frame, line_frame_ver, pred_frame], 1)
            line_frame_hor = np.zeros((1, frame.shape[1], 3)).astype(np.uint8)

            switch_frame = resize(frame.shape[1] - add_size, switch_frame, dim='x')
            switch_frame = pad(switch_frame, 100, dim='y')
            switch_frame = append_to_add_name_vertical(switch_frame, "Switch Score", (75, 75))
            switch_frame = pad(switch_frame, 50, dim='y', reverse=True)

            
            frame = np.concatenate([frame, line_frame_hor, switch_frame])

            both_frame = resize(frame.shape[0]-150, both_frame, dim='y')
            both_frame = append_to_add_name_horizontal(both_frame, ["PTD's architecture"], pos=(80, 40))
            both_frame = append_per_tran_name(both_frame)
            # both_frame = append_per_tran_name(both_frame)
            frame = np.concatenate([frame, np.zeros((frame.shape[0], 1, 3)).astype(np.uint8), both_frame], axis=1)        
            print(frame.shape)
            # frame = append_to_add_name_horizontal(frame, [f"Graph {gidx}"])
            # # frame1 = np.concatenate([persistent_frame, line_frame_ver, transient_frame], 1)
            # frame = np.concatenate([frame, line_frame_hor, frame1])
            
            # ratio = frame.shape[1] / switch_frame.shape[1]
            # switch_frame = cv2.resize(switch_frame, (frame.shape[1], int(switch_frame.shape[0] * ratio)))

            # ratio = 640 * 2 / frame.shape[1]
            # frame = cv2.resize(frame, (640*2, int(frame.shape[0] * ratio)))
            # frame = resize(640*2, frame, dim='x')
            
            
            frame =  cv2.resize(frame, (1280, 720))
            # cv2.imshow("frame", frame)
            # key = cv2.waitKey(0) & 0xff
            # if key == ord("q"):
            #     is_break = True
            #     break
                        
            if writer is None:
                writer = cv2.VideoWriter("test1.avi", fourcc=cv2.VideoWriter_fourcc(*'MJPG'), fps=2, frameSize=(frame.shape[1], frame.shape[0]))
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
