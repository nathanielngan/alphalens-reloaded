from . import performance
from . import plotting
from . import tears
from . import tears_plotly
from . import utils
from . import plotting_plotly

try:
    from ._version import version as __version__
    from ._version import version_tuple
except ImportError:
    __version__ = "unknown version"
    version_tuple = (0, 0, "unknown version")


__all__ = ["performance", "plotting", "tears", "tears_plotly", "plotting_plotly","utils"]
