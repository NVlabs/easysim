import importlib


def load(entry_point):
    """ """
    mod_name, attr_name = entry_point.split(":")
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, attr_name)
    return fn


class SimulatorSpec:
    """ """

    def __init__(self, name, entry_point, kwargs=None):
        """ """
        self._name = name
        self._entry_point = entry_point
        self._kwargs = {} if kwargs is None else kwargs

    def make(self, **kwargs):
        """ """
        _kwargs = self._kwargs.copy()
        _kwargs.update(kwargs)

        fn = load(self._entry_point)
        simulator = fn(**_kwargs)

        return simulator


class SimulatorRegistry:
    """ """

    def __init__(self):
        """ """
        self._specs = {}

    def register(self, name, **kwargs):
        """ """
        if name in self._specs:
            raise Exception(f"Cannot re-register name: '{name}'")
        self._specs[name] = SimulatorSpec(name, **kwargs)

    def make(self, name, **kwargs):
        """ """
        if name not in self._specs:
            raise KeyError(f"No registered simulator with name: '{name}'")
        spec = self._specs[name]

        simulator = spec.make(**kwargs)

        return simulator


registry = SimulatorRegistry()


def register(name, **kwargs):
    """ """
    registry.register(name, **kwargs)


def make(name, **kwargs):
    """ """
    return registry.make(name, **kwargs)


register(
    name="bullet",
    entry_point="easysim.simulators.bullet:Bullet",
)

register(
    name="isaac_gym",
    entry_point="easysim.simulators.isaac_gym:IsaacGym",
)
