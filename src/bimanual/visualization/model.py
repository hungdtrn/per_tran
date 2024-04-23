import cv2
import click

from src.bimanual.data import build_dataset
from src.bimanual.architectures import build_model

from src.shared.visualization.model import visualize
from .utils import draw_human, draw_object, to_pixel

@click.command()
@click.option("--data_cfg_path", default="src/bimanual/config/dataset/bimanual_clip.yaml")
@click.option("--checkpoint_path")
@click.option("--video_path")
@click.option("--part", default="test")
@click.option("--seed", default=0)
@click.option("--shuffle", type=bool, default=True)
@click.option("--batch_size", default=128)
@click.option("--crnn_format", default=False)
@click.option("--duality_format", default=False)
@click.option("--is_ego", default=False, type=bool)
@click.option("--is_save", type=bool, default=False)
@click.option("--is_show", type=bool, default=True)
@click.option("--crnn_format", default=False)
@click.option("--gpu", type=int, default=0)
@click.option("--is_draw_switch", type=bool, default=False)
@click.option("--model_name", type=str)
@click.option("--save_path")
def visualize_dataset(**kwargs):
    visualize(kwargs, build_dataset, build_model,
              draw_human_fn=draw_human,
              draw_obj_fn=draw_object,
              pixel_fn=to_pixel)
    
    
if __name__ == "__main__":
    visualize_dataset()
