from collections import defaultdict
import json
import marshal
import os
import pickle
import types
from typing import Tuple, Union

import numpy as np
from jax import random, jit
import jax.numpy as jnp

from grl.utils.data import Batch

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

def sample_idx_batch(batch_size: int, length: int, rand_key: random.PRNGKey):
    new_rand_key, sample_rand_key = random.split(rand_key, 2)
    idx = random.randint(sample_rand_key, (batch_size,), minval=0, maxval=length, dtype=int)
    return idx, new_rand_key

def sample_seq_idxes(batch_size: int, capacity: int, seq_len: int, length: int, eligible_idxes: jnp.ndarray, rand_key: random.PRNGKey):
    sampled_ei_idx, new_rand_key = sample_idx_batch(batch_size, length, rand_key)
    sample_idx = eligible_idxes[sampled_ei_idx]
    seq_range = jnp.arange(seq_len, dtype=int)[:, None]
    sample_seq_idx = (sample_idx + seq_range).T % capacity

    return sample_seq_idx, new_rand_key

class ReplayBuffer:
    def __init__(self, capacity: int, rand_key: random.PRNGKey,
                 obs_size: Tuple,
                 obs_dtype: type,
                 state_size: Tuple = None):
        """
        Replay buffer that saves both observation and state.
        :param capacity:
        :param rng:
        """

        self.capacity = capacity
        self.rand_key = rand_key
        self.state_size = state_size
        self.obs_size = obs_size

        # TODO: change these to half precision to save GPU memory.
        if self.state_size is not None:
            self.s = np.zeros((self.capacity, *self.state_size))

        if obs_dtype is not None:
            self.obs = np.zeros((self.capacity, *self.obs_size), dtype=obs_dtype)
        else:
            self.obs = np.zeros((self.capacity, *self.obs_size))

        self.a = np.zeros(self.capacity, dtype=np.int16)
        self.next_a = np.zeros(self.capacity, dtype=np.int16)
        self.r = np.zeros(self.capacity, dtype=np.float)
        self.returns = np.zeros(self.capacity, dtype=np.float)
        self.include_returns = False
        self.d = np.zeros(self.capacity, dtype=bool)

        self._cursor = 0
        self._filled = False

        # We have the -1 here b/c we write next_obs as well.
        self.eligible_idxes = np.zeros(self.capacity - 1, dtype=int)
        self.n_eligible_idxes = 0
        self.ei_cursor = 0
        self.jitted_sampled_idx_batch = jit(sample_idx_batch, static_argnums=0)

    def reset(self):
        if self.state_size is not None:
            self.s = np.zeros((self.capacity, *self.state_size))

        self.obs = np.zeros((self.capacity, *self.obs_size))
        self.a = np.zeros(self.capacity, dtype=np.int16)
        self.next_a = np.zeros(self.capacity, dtype=np.int16)
        self.r = np.zeros(self.capacity, dtype=np.float)
        self.returns = np.zeros(self.capacity, dtype=np.float)
        self.d = np.zeros(self.capacity, dtype=bool)

        self.eligible_idxes = np.zeros(self.capacity - 1, dtype=int)
        self.n_eligible_idxes = 0
        self.ei_cursor = 0
        self._cursor = 0
        self._filled = False

    def push(self, batch: Batch):
        next_cursor = (self._cursor + 1) % self.capacity
        next_ei_cursor = (self.ei_cursor + 1) % (self.capacity - 1)

        self.a[self._cursor] = batch.action
        if self.state_size is not None and batch.state is not None and batch.next_state is not None:
            self.s[self._cursor] = batch.state
            self.s[next_cursor] = batch.next_state

        if batch.returns is not None:
            self.returns[self._cursor] = batch.returns
            self.include_returns = True

        self.obs[self._cursor] = batch.obs
        self.d[self._cursor] = batch.done
        self.r[self._cursor] = batch.reward

        if batch.next_action is not None:
            self.a[next_cursor] = batch.next_action

        self.obs[next_cursor] = batch.next_obs

        self.eligible_idxes[self.ei_cursor] = self._cursor
        self.n_eligible_idxes = min(self.capacity - 1, self.n_eligible_idxes + 1)
        self.ei_cursor = next_ei_cursor
        self._cursor = next_cursor

    def __len__(self):
        return self.n_eligible_idxes

    def sample_eligible_idxes(self, batch_size: int):
        length = self.n_eligible_idxes
        idxes, self.rand_key = self.jitted_sampled_idx_batch(batch_size, length, self.rand_key)
        return self.eligible_idxes[idxes]

    def sample(self, batch_size: int, **kwargs) -> Batch:
        """
        NOTE: If done is True, then the next state returned is either all
        0s or the state from the start of the next episode. Either way
        it shouldn't matter for your target calculation.
        :param batch_size:
        :return:
        """

        sample_idx = self.sample_eligible_idxes(batch_size)
        batch = {}
        if self.state_size is not None:
            batch['state'] = self.s[sample_idx]
            batch['next_state'] = self.s[(sample_idx + 1) % self.capacity]

        batch['obs'] = self.obs[sample_idx]
        batch['next_obs'] = self.obs[(sample_idx + 1) % self.capacity]
        batch['action'] = self.a[sample_idx]
        batch['next_action'] = self.a[(sample_idx + 1) % self.capacity]
        batch['done'] = self.d[sample_idx]
        batch['reward'] = self.r[sample_idx]
        if self.include_returns:
            batch['returns'] = self.returns[sample_idx]
        batch['indices'] = sample_idx

        return Batch(**batch)


class EpisodeBuffer(ReplayBuffer):
    """
    For episode buffer, we return zero-padded batches back.

    We have to save "end" instead of done, to track if either an episode is finished
    or we reach the max number of steps

    How zero-padded batches work in our case is that the "done" tensor
    is essentially a mask for
    """
    def __init__(self, capacity: int, rand_key: random.PRNGKey,
                 obs_size: Tuple, obs_dtype: type, state_size: Tuple = None):
        super(EpisodeBuffer, self).__init__(capacity, rand_key, obs_size, obs_dtype, state_size=state_size)
        self.jitted_sampled_seq_idxes = jit(sample_seq_idxes, static_argnums=(0, 1, 2))
        self.end = np.zeros_like(self.d, dtype=bool)

    def push(self, batch: Batch):
        self.end[self._cursor] = batch.end
        super(EpisodeBuffer, self).push(batch)

    def sample_eligible_idxes(self, batch_size: int, seq_len: int) -> np.ndarray:
        length = self.n_eligible_idxes
        sampled_eligible_idxes, self.rand_key = self.jitted_sampled_seq_idxes(batch_size, self.capacity, seq_len, length,
                                                                              self.eligible_idxes, self.rand_key)
        return sampled_eligible_idxes

    def get_zero_mask(self, ends: jnp.ndarray, sample_idx: np.ndarray):
        # Zero mask is essentially mask where we only learn if we're still within an episode.
        # To do this, we set everything AFTER done == True as 0, and any episode that ends
        # after max steps. The array ends accomplishes this.
        zero_mask = np.ones_like(ends)
        ys, xs = ends.nonzero()

        # if self.ei_cursor == 0, then we can sample anything!
        if self.ei_cursor != 0:
            # Also, we don't want any experience beyond our current cursor.
            ys_cursor, xs_cursor = np.nonzero(np.array(sample_idx) == self.eligible_idxes[self.ei_cursor])

            ys, xs = np.concatenate([ys_cursor, ys]), np.concatenate([xs_cursor, xs])

        if ys.shape[0] > 0:
            for y, x in zip(ys, xs):
                zero_mask[y, x + 1:] = 0
        return zero_mask

    def sample(self, batch_size: int, seq_len: int = 1, as_dict: bool = False) -> Union[Batch, dict]:
        sample_idx = self.sample_eligible_idxes(batch_size, seq_len)
        return self.sample_idx(sample_idx)

    def sample_idx(self, sample_idx: np.ndarray, as_dict: bool = False):
        batch = {}

        # get the last index and bump it
        t_p1_idx = (sample_idx[:, -1:] + 1) % self.capacity
        sample_p1_idx = np.concatenate((sample_idx, t_p1_idx), axis=-1)

        batch['state'] = self.s[sample_p1_idx]
        batch['obs'] = self.obs[sample_p1_idx]
        batch['action'] = self.a[sample_p1_idx]
        batch['done'] = self.d[sample_idx]
        batch['reward'] = self.r[sample_idx]

        if self.include_returns:
            batch['returns'] = self.returns[sample_idx]

        # batch['indices'] = sample_idx
        ends = self.end[sample_idx]

        batch['zero_mask'] = self.get_zero_mask(ends, sample_idx)

        if as_dict:
            return batch

        return Batch(**batch)

    def sample_k(self, batch_size: int, seq_len: int = 1, k: int = 1):
        batch = self.sample(batch_size * k, seq_len=seq_len, as_dict=True)
        for key, arr in batch.items():
            batch[key] = np.stack(np.split(arr, k, axis=0))

        return Batch(**batch)
