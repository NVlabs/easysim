import abc
import numpy as np

from contextlib import contextmanager


class Attrs(abc.ABC):
    """ """

    _created = False

    def __init__(self, **kwargs):
        """ """
        self._init(**kwargs)

        self._created = True

    @abc.abstractmethod
    def _init(self):
        """ """

    def __setattr__(self, key, value):
        """ """
        # Exclude keys in self._SETATTR_WHILELIST to prevent infinite recursion in property calls.
        if self._created and key not in self._SETATTR_WHITELIST and not hasattr(self, key):
            raise TypeError(f"Unrecognized {self.__class__.__name__} attribute '{key}'")
        object.__setattr__(self, key, value)


class AttrsArrayTensor(Attrs):
    """ """

    def __init__(self, **kwargs):
        """ """
        self._init_attr_array_pipeline()
        self._init_device()

        super().__init__(**kwargs)

    def _init_attr_array_pipeline(self):
        """ """
        self._attr_array_locked = {}
        self._attr_array_dirty_flag = {}
        self._attr_array_dirty_mask = {}
        self._attr_array_default_flag = {}
        for attr in self._ATTR_ARRAY_NDIM:
            self._attr_array_locked[attr] = False
            self._attr_array_dirty_flag[attr] = False
            self._attr_array_default_flag[attr] = False

    @property
    def attr_array_locked(self):
        """ """
        return self._attr_array_locked

    @property
    def attr_array_dirty_flag(self):
        """ """
        return self._attr_array_dirty_flag

    @property
    def attr_array_dirty_mask(self):
        """ """
        return self._attr_array_dirty_mask

    @property
    def attr_array_default_flag(self):
        """ """
        return self._attr_array_default_flag

    def _init_device(self):
        """ """
        self._device = None

    @property
    def device(self):
        """ """
        return self._device

    def get_attr_array(self, attr, idx):
        """ """
        return self._get_attr(attr, self._ATTR_ARRAY_NDIM[attr], idx)

    def get_attr_tensor(self, attr, idx):
        """ """
        return self._get_attr(attr, self._ATTR_TENSOR_NDIM[attr], idx)

    def _get_attr(self, attr, ndim, idx):
        """ """
        array = getattr(self, attr)
        if array.ndim == ndim:
            return array
        if array.ndim == ndim + 1:
            return array[idx]

    def lock_attr_array(self):
        """ """
        for k in self._attr_array_locked:
            if not self._attr_array_locked[k]:
                self._attr_array_locked[k] = True
            if getattr(self, k) is not None:
                getattr(self, k).flags.writeable = False

    def update_attr_array(self, attr, env_ids, value):
        """ """
        if getattr(self, attr).ndim != self._ATTR_ARRAY_NDIM[attr] + 1:
            raise ValueError(
                f"'{attr}' can only be updated when a per-env specification (ndim: "
                f"{self._ATTR_ARRAY_NDIM[attr] + 1}) is used"
            )
        if len(env_ids) == 0:
            return

        env_ids_np = env_ids.cpu().numpy()

        with self._make_attr_array_writeable(attr):
            getattr(self, attr)[env_ids_np] = value

        if not self._attr_array_dirty_flag[attr]:
            self._attr_array_dirty_flag[attr] = True
        try:
            self._attr_array_dirty_mask[attr][env_ids_np] = True
        except KeyError:
            self._attr_array_dirty_mask[attr] = np.zeros(len(getattr(self, attr)), dtype=bool)
            self._attr_array_dirty_mask[attr][env_ids_np] = True

        if self._attr_array_default_flag[attr]:
            self._attr_array_default_flag[attr] = False

    @contextmanager
    def _make_attr_array_writeable(self, attr):
        """ """
        try:
            getattr(self, attr).flags.writeable = True
            yield
        finally:
            getattr(self, attr).flags.writeable = False

    def set_device(self, device):
        """ """
        self._device = device

        self._set_attr_device(device)

    @abc.abstractmethod
    def _set_attr_device(self):
        """ """
