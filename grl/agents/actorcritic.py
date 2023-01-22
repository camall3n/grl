import copy
from functools import partial
from itertools import repeat
from multiprocessing import Pool
import os
import shutil
from tqdm import tqdm

# from jax.nn import softmax
from scipy.special import softmax
import numpy as np
import optuna

from grl.utils.math import glorot_init as normal_init
from grl.agents.td_lambda import TDLambdaQFunction
from grl.agents.replaymemory import ReplayMemory

def one_hot(x, n):
    return np.eye(n)[x]

class ActorCritic:
    def __init__(self,
                 n_obs: int,
                 n_actions: int,
                 gamma: float,
                 n_mem_entries: int = 0,
                 n_mem_values: int = 2,
                 learning_rate: float = 0.001,
                 trace_type: str = 'accumulating',
                 policy_epsilon: float = 0.1,
                 replay_buffer_size: int = 1000000) -> None:
        self.n_obs = n_obs
        self.n_actions = n_actions
        self.n_mem_values = n_mem_values
        self.n_mem_entries = n_mem_entries
        self.n_mem_states = n_mem_values**n_mem_entries
        self.policy_epsilon = policy_epsilon

        self.reset_policy()
        self.reset_memory()

        q_fn_kwargs = {
            'n_obs': n_obs,
            'n_actions': n_actions,
            'gamma': gamma,
            'learning_rate': learning_rate,
            'trace_type': trace_type
        }
        self.q_td = TDLambdaQFunction(lambda_=0, **q_fn_kwargs)
        self.q_mc = TDLambdaQFunction(lambda_=0.99, **q_fn_kwargs)
        self.replay = ReplayMemory(capacity=replay_buffer_size)
        self.reset()

    def mem_summary(self, precision=2):
        mem = self.cached_memory_fn.round(precision)
        mem_summary = str(np.concatenate((mem[2, 0], mem[2, 1], mem[2, 2]), axis=-1))
        return mem_summary

    def reset(self):
        self.memory = 0
        self.prev_action = None

    def act(self, obs):
        obs_aug = self.augment_obs(obs)
        action = np.random.choice(self.n_actions, p=self.cached_policy_fn[obs_aug])
        self.step_memory(obs, action)
        return action

    def step_memory(self, obs, action):
        next_memory = np.random.choice(
            self.n_mem_states,
            p=self.cached_memory_fn[action, obs, self.memory],
        )
        self.prev_memory = self.memory
        self.memory = next_memory

    def store(self, experience):
        self.replay.push(experience)

    def update_actor(self):
        best_a = np.argmax(self.q_td.q.transpose(), axis=-1)
        greedy_pi = one_hot(best_a, self.n_actions)
        uniform_pi = np.ones_like(greedy_pi) / self.n_actions
        new_policy = (1 - self.policy_epsilon) * greedy_pi + self.policy_epsilon * uniform_pi
        did_change = self.set_policy(new_policy, logits=False)
        if did_change:
            self.replay.reset()
        return did_change

    def update_critic(self, experience: dict):
        augmented_experience = self.augment_experience(experience)
        self.q_td.update(**augmented_experience)
        self.q_mc.update(**augmented_experience)

    def set_policy(self, params, logits=True):
        if logits:
            did_change = not np.array_equal(params, self.policy_params)
            self.policy_params = params
            self.cached_policy_fn = softmax(self.policy_params, axis=-1)
        else:
            did_change = not np.array_equal(params, self.cached_policy_fn)
            self.cached_policy_fn = params
            self.policy_params = np.log(self.cached_policy_fn + 1e-20)
        return did_change

    def set_memory(self, params, logits=True):
        if logits:
            self.memory_params = params
            self.cached_memory_fn = softmax(self.memory_params, axis=-1)
        else:
            self.cached_memory_fn = params
            self.memory_params = np.log(self.cached_memory_fn + 1e-20)

    def reset_policy(self):
        self.policy_params = None
        self.cached_policy_fn = None
        policy_shape = (self.n_obs * self.n_mem_states, self.n_actions)
        self.set_policy(normal_init(policy_shape, scale=0.2))

    def reset_memory(self):
        mem_shape = (self.n_actions, self.n_obs, self.n_mem_states, self.n_mem_states)
        self.set_memory(normal_init(mem_shape))

    def fill_in_params(self, required_params):
        required_params_shape = self.memory_params.shape[:-1] + (self.n_mem_values - 1, )
        params = np.empty_like(self.memory_params)
        params[:, :, :, :-1] = np.asarray(required_params).reshape(required_params_shape)
        params[:, :, :, -1] = 1 - np.sum(params[:, :, :, :-1], axis=-1)
        return params

    def objective(self, trial: optuna.Trial, study_dir='./results/sample_based/'):
        n_required_params = np.prod(self.memory_params.shape) // self.n_mem_states
        required_params = [
            trial.suggest_float(str(i), low=0.0, high=1.0) for i in range(n_required_params)
        ]
        self.set_memory(self.fill_in_params(required_params), logits=False)
        result = self.evaluate_memory()

        with open(os.path.join(study_dir, 'output.txt'), 'a') as file:
            file.write(f'{trial.number}\n')
            file.write(self.mem_summary() + '\n')
            file.write(f'Discrep: {result}\n\n')
            file.flush()
        return result

    def optimize_memory(
            self,
            study_name=None,
            preamble_str='',
            n_trials=500,
            n_jobs=1,
            sampler=optuna.samplers.TPESampler(),
    ):
        study_dir = f'./results/sample_based/{study_name}'
        # if os.path.exists(study_dir):
        #     shutil.rmtree(study_dir)
        os.makedirs(study_dir, exist_ok=True)
        saved_replaymemory_path = os.path.join(study_dir, 'replaymemory.pkl')
        print(saved_replaymemory_path)
        if os.path.exists(saved_replaymemory_path):
            print("Loading existing replay buffer")
            self.replay = self.replay.load(saved_replaymemory_path)
        else:
            print("Saving replay buffer")
            self.replay.save(study_dir, filename='replaymemory', extension='.pkl')

        study = optuna.create_study(
            study_name=study_name,
            direction='minimize',
            storage=optuna.storages.JournalStorage(
                optuna.storages.JournalFileStorage(os.path.join(study_dir, "study.journal"))),
            sampler=sampler,
            load_if_exists=True,
        )

        with open(os.path.join(study_dir, 'output.txt'), 'w') as file:
            file.write(preamble_str)
            file.flush()

        n_jobs = max(n_jobs, 1)
        n_trials_per_worker = np.ones(n_jobs) * (n_trials // n_jobs)
        n_trials_per_worker[-1] = np.ceil(n_trials / n_jobs)
        print(f'Starting pool with {n_jobs} workers')
        print(f'n_trials_per_worker: {n_trials_per_worker}')
        pool = Pool(n_jobs, maxtasksperchild=1) # Each new tasks gets a fresh worker
        pool.starmap(
            study.optimize,
            zip(repeat(partial(self.objective, study_dir=study_dir)), n_trials_per_worker),
        )
        pool.close()
        pool.join()

        required_params = [
            study.best_trial.params[key]
            for key in sorted(study.best_trial.params.keys(), key=lambda x: int(x))
        ]
        params = self.fill_in_params(required_params)
        self.set_memory(params, logits=False)

        with open(os.path.join(study_dir, 'output.txt'), 'a') as file:
            file.write('--------------------------------------------\n')
            file.write(f'Best trial: {study.best_trial.number}\n')
            file.write(self.mem_summary() + '\n')
            file.write(f'Discrep: {study.best_trial.value}\n\n')
            file.flush()

        return study

    def evaluate_memory(self, n_epochs=1):
        for epoch in range(n_epochs):
            self.reset()
            self.q_mc.reset()
            self.q_td.reset()
            for experience in tqdm(self.replay.memory, desc=f'epoch {epoch}'):
                e = experience.copy()
                del e['_index_']
                self.step_memory(e['obs'], e['action'])
                self.update_critic(e)
                if experience['terminal']:
                    self.reset()
        return np.abs(self.q_mc.q - self.q_td.q).sum()

    def augment_obs(self, obs: int, memory: int = None) -> int:
        if memory is None:
            memory = self.memory
        # augment last dim with mem states
        ob_augmented = self.n_mem_states * obs + memory
        return ob_augmented

    def augment_experience(self, experience: dict) -> dict:
        augmented_experience = copy.deepcopy(experience)
        augmented_experience['obs'] = self.augment_obs(experience['obs'], self.prev_memory)
        augmented_experience['next_obs'] = self.augment_obs(experience['next_obs'], self.memory)
        return augmented_experience

    def augment_policy(self, n_mem_states: int):
        """
        Expand π (O x A) => OM x A to include memory states
        """
        # augment last dim with input mem states
        pi_augmented = np.expand_dims(self.policy_params, 1).repeat(n_mem_states, 1) # O x M x A
        self.set_policy(pi_augmented.reshape(self.n_obs * n_mem_states, self.n_actions)) # OM x A

    def add_memory(self, n_mem_entries=1):
        mem_increase_multiplier = (self.n_mem_values**n_mem_entries)
        self.q_td.augment_with_memory(mem_increase_multiplier)
        self.q_mc.augment_with_memory(mem_increase_multiplier)

        self.n_mem_entries += n_mem_entries
        self.n_mem_states = self.n_mem_states * mem_increase_multiplier
        self.reset_memory()
        self.augment_policy(mem_increase_multiplier)
