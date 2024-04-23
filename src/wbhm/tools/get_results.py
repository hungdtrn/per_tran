import pandas as pd
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    args = parser.parse_args()
    
    csv = pd.read_csv(args.csv_path, index_col=0)

    print("Switch_acc")
    current_k = -1
    for i in range(5, len(csv)-9, 5):
        indexes = csv.index[i:i+5]
        k, type, metric, _ = indexes[1].split("_")
        
        current_switch =csv[i:i+5].to_numpy()[:, 0]
        if k != current_k:
            print("---------------")
            current_k = k
        print("{}: {:.2f} +- {:.2f}".format("_".join([k, type, metric]),
                                            current_switch.mean(),
                                            current_switch.std()))
        
    human_loss = csv[:5].to_numpy()
    obj_loss = csv[:-5].to_numpy()
    
    print("Human loss")
    print("MPJPE (ADE): {:.2f} +- {:.1f}".format(human_loss.mean(1).mean(), human_loss.mean(1).std()))
    
    print("Obj loss")    
    print("MPJPE (ADE): {:.2f} +- {:.1f}".format(obj_loss.mean(1).mean(), obj_loss.mean(1).std()))

        