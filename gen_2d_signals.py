import logging

import hydra
from omegaconf import DictConfig, OmegaConf
from joblib import Parallel, delayed
from mrc_insar_common.data import data_reader
from siminsar.sim2d import Signal_2d
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
logger = logging.getLogger(__name__)

def generate_2d_signals(index, height, width, output_prefix):
    np.random.seed(np.random.randint(1,1000000+1) + index)
    signal_2d = Signal_2d(height, width, rayleigh_scale=120) 
    signal_2d.add_n_gauss_bubbles(sigma_range=[20,300], amp_range=[-10,10], nps=20)
    signal_2d.add_n_ellipses(z_range=[0.,150.], x_range=[0.1,100.], y_range=[0.1,10.], nps=10) # these are randomly positive or negative
    signal_2d.add_n_polygons(z_range=[0.,140.], r_range=[0.,100.], n_range=[3,10], nps=20)
    signal_2d.add_n_buildings(width_range=[3.,30.], height_range=[3.,50.], depth_factor=2., nps=30)
    signal_2d.add_n_amp_stripes(thickness_range=[1.,10.], nps=100)
    signal_2d.compile()
    output_path = f'{output_prefix}_{index}'
    plt.figure(); plt.imshow(signal_2d.signal, interpolation="None", cmap='rainbow'); plt.colorbar(); plt.savefig(f'{output_path}.png')
    data_reader.writeBin(output_path, signal_2d.signal, 'float')
    logger.info(f'save to {output_path}')
    return 

@hydra.main(config_path='config', config_name='gen_2d_signals')
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    output_prefix = 'sim_2d'
    samples = cfg.samples
    workers = cfg.workers
    Parallel(n_jobs=workers)(delayed(generate_2d_signals)(i, cfg.height, cfg.width, output_prefix) for i in tqdm(range(samples)))

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.error(e)
        exit(1)
