from collections import defaultdict
import json
import marshal
import os
import pickle
import types

import numpy as np

class ReplayMemory:
    def __init__(self, capacity: int, on_retrieve: dict = None):
        self.capacity = int(capacity)
        self.on_retrieve = defaultdict(lambda: (lambda items: items))
        if on_retrieve is not None:
            self.on_retrieve.update(on_retrieve)

        self.reset()

    def reset(self):
        self.fields = set()
        self.memory = []
        self.position = 0

    def push(self, experience: dict):
        e = experience.copy()
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        e['_index_'] = self.position
        self.fields = self.fields.union(set(e.keys()))
        self.memory[self.position] = e
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int, fields: list = None):
        idx = np.random.choice(len(self.memory), batch_size, replace=False)
        return self.retrieve(idx, fields)

    def retrieve(self, idx=None, fields: list = None):
        if idx is None:
            idx = range(len(self.memory))
        try:
            experiences = [self.memory[i] for i in idx]
        except TypeError:
            experiences = [self.memory[idx]]

        if fields is None:
            fields = sorted(list(self.fields))
        if isinstance(fields, str):
            fields = [fields]

        result = []
        for field_name in fields:
            result.append(self._extract_array(experiences, field_name))
        result = tuple(result)
        if len(fields) == 1:
            result = result[0]

        return result

    def __len__(self):
        return len(self.memory)

    def __getstate__(self):
        """Prepare the buffer for pickling

        We need to dump the contents of any lambdas into byte strings
        We also need on_retrieve to be a normal dict, not a defaultdict

        For more info, see: https://stackoverflow.com/a/11878430/2577640
        """
        state = self.__dict__.copy()
        state['on_retrieve'] = {
            key: marshal.dumps(state['on_retrieve'][key].__code__)
            for key in state['on_retrieve']
        }
        return state

    def __setstate__(self, state):
        """Restore the buffer after pickling

        We need to load the code from any lambda byte strings, and construct new lambdas
        We also need on_retrieve to be a defaultdict, not a normal dict

        For more info, see: https://stackoverflow.com/a/11878430/2577640
        """
        for key in state['on_retrieve']:
            new_lambda_code = marshal.loads(state['on_retrieve'][key])
            state['on_retrieve'][key] = types.FunctionType(new_lambda_code, globals())
        state['on_retrieve'] = defaultdict(**state['on_retrieve'])
        self.__dict__.update(state)

    def _extract_array(self, experiences, key):
        items = [experience[key] for experience in experiences]
        if self.on_retrieve:
            if key != '_index_':
                items = self.on_retrieve['*'](items)
            items = self.on_retrieve[key](items)
        return items

    def save(self, directory, filename='replaymemory', extension='.json'):
        filepath = os.path.join(directory, filename + extension)
        if extension == '.json':
            fields = sorted(list(self.fields))

            data = [{key: serialize_np(value)
                     for key, value in experience.items()} for experience in self.memory]
            archive = {
                'capacity': self.capacity,
                'position': self.position,
                'fields': fields,
                'memory': data,
            }
            os.makedirs(directory, exist_ok=True)
            with open(filepath, 'w') as fp:
                json.dump(archive, fp)
        elif extension == '.pkl':
            with open(filepath, 'wb') as fp:
                pickle.dump(self, fp)
        else:
            raise NotImplementedError(f'Unknown replaymemory file type: {extension}')

    def load(self, filepath):
        extension = os.path.splitext(filepath)[-1]
        if extension == '.json':
            with open(filepath, 'r') as fp:
                archive = json.load(fp)
            self.capacity = int(archive['capacity'])
            self.position = int(archive['position'])
            self.fields = set(archive['fields'])
            self.memory = [{
                key: unserialize_np(value, dtype)
                for (key, (value, dtype)) in experience.items()
            } for experience in archive['memory']]
        elif extension == '.pkl':
            with open(filepath, 'rb') as fp:
                return pickle.load(fp)
        else:
            raise NotImplementedError(f'Unknown replaymemory file type: {extension}')

def serialize_np(value):
    output = value
    dtype = None
    try:
        output = value.tolist()
        dtype = str(value.dtype)
    except AttributeError:
        pass
    return output, dtype

def unserialize_np(value, dtype):
    output = value
    if dtype is not None:
        output = np.asarray(value, dtype=dtype)
    return output
