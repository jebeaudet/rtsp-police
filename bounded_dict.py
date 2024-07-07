from collections import defaultdict, OrderedDict


class BoundedDict(defaultdict):
    def __init__(self, max_size, default_factory=None, *args, **kwargs):
        super().__init__(default_factory, *args, **kwargs)
        self.max_size = max_size
        self._data = OrderedDict()

    def __setitem__(self, key, value):
        if key in self._data:
            del self._data[key]
        elif self.max_size is not None and len(self._data) >= self.max_size:
            self._data.popitem(last=False)
        self._data[key] = value

    def __getitem__(self, key):
        if key not in self._data:
            self._data[key] = self.default_factory()
        return self._data[key]

    def __repr__(self):
        return f"{self.__class__.__name__}({self._data}, max_size={self.max_size})"

    def __len__(self):
        return len(self._data)

    def __delitem__(self, key):
        del self._data[key]

    def __contains__(self, key):
        return key in self._data
