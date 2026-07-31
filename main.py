from config.config import opt
from use_model import get_cluster
from plot.plot_clustered_result import plot_clustered

def test(**kwargs):
    opt._parse(kwargs)
    get_cluster(**vars(opt))
    plot_clustered(**vars(opt))
if __name__=='__main__':
    import fire
    fire.Fire()
