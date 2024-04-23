import cv2
import click

from src.wbhm.data import build_dataset
from src.shared.visualization.dataset import visualize
from .utils import draw_human, draw_object, to_pixel

@click.command()
@click.option("--data_cfg_path", default="src/wbhm/config/dataset/wbhm.yaml")
@click.option("--part", default="test")
@click.option("--seed", default=0)
@click.option("--shuffle", type=bool, default=True)
@click.option("--batch_size", default=128)
@click.option("--crnn_format", default=False)
@click.option("--duality_format", default=False)
@click.option("--is_ego", default=False)
@click.option("--is_show", default=True, type=bool)
@click.option("--is_save", default=False, type=bool)
@click.option("--save_path", default="reports/wbhm_dataset_visualization")

def visualize_dataset(**kwargs):
    visualize(kwargs, build_dataset,
              draw_human_fn=draw_human,
              draw_obj_fn=draw_object,
              pixel_fn=to_pixel)
    
    
if __name__ == "__main__":
    visualize_dataset()