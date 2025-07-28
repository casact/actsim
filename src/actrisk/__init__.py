from .core.actfitter import DistributionFitter
from .core.actsimulator import StochasticSimulator 
from .utils.utils import Config

def load_config(file_path: str = None):
    """
    unction to load the default config or a custom config file.
    
    Args:
        file_path: Path to a custom config file. If None, uses the default config.
    
    Returns:
        Config object with the loaded configuration.
    """
    return Config(file_path)

__all__ = ["DistributionFitter", "StochasticSimulator", "Config", "load_config"]