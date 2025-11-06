from config.config import opt
from use_model import get_cluster
#from DSP.get_ReSignal import get_signals
from plot.plot_clustered_result import plot_clustered

def test(**kwargs):
    opt._parse(kwargs)
    get_cluster(**vars(opt))
    #get_signals(**vars(opt))
    plot_clustered(**vars(opt))
if __name__=='__main__':
    import fire
    fire.Fire()
