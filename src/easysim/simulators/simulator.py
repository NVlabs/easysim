import abc


class Simulator(abc.ABC):
    """Simulator."""

    def __init__(self, cfg):
        """ """
        self._cfg = cfg

    @abc.abstractmethod
    def reset(self, bodies, env_ids):
        """ """

    @abc.abstractmethod
    def step(self, bodies):
        """ """

    @property
    @abc.abstractmethod
    def contact(self):
        """ """

    @abc.abstractmethod
    def close(self):
        """ """
