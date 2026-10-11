"""Show a Figure that was built without pyplot, through pyplot's backend."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib import _pylab_helpers
from matplotlib.figure import Figure


def show_figure(fig: Figure) -> None:
    """Make `fig` the current pyplot figure, display it with plt.show() and close it."""
    number = max(plt.get_fignums(), default=0) + 1
    manager = plt._get_backend_mod().new_figure_manager_given_figure(number, fig)
    _pylab_helpers.Gcf._set_new_active_manager(manager)
    plt.show()
    plt.close(fig)
