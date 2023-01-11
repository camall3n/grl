import copy
from tqdm import tqdm

# from jax.nn import softmax
from scipy.special import softmax
import numpy as np
import optuna

from grl.utils.math import glorot_init
from grl.agents.td_lambda import TDLambdaQFunction
from grl.agents.replaymemory import ReplayMemory

class ActorCritic:
    def __init__(self,
                 n_obs: int,
                 n_actions: int,
                 gamma: float,
                 n_mem_entries: int = 0,
                 n_mem_values: int = 2,
                 learning_rate: float = 0.001,
                 trace_type: str = 'accumulating',
                 replay_buffer_size: int = 1000000) -> None:
        self.n_obs = n_obs
        self.n_actions = n_actions
        self.n_mem_values = n_mem_values
        self.n_mem_entries = n_mem_entries
        self.n_mem_states = n_mem_values**n_mem_entries

        self.set_policy(glorot_init((n_obs * self.n_mem_states, n_actions), scale=0.2))
        self.set_memory(glorot_init((n_actions, n_obs, self.n_mem_states, self.n_mem_states)))

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

    def print_mem_summary(self):
        mem = self.cached_memory_fn.round(1)
        mem_summary = np.concatenate((mem[2, 0], mem[2, 1], mem[2, 2]), axis=-1)
        print(mem_summary)

    def reset(self):
        self.memory = 0
        self.prev_action = None

    def act(self, obs):
        obs_aug = self.augment_obs(obs)
        action = np.random.choice(self.n_actions, p=self.cached_policy[obs_aug])
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

    def update_critic(self, experience: dict):
        augmented_experience = self.augment_experience(experience)
        self.q_td.update(**augmented_experience)
        self.q_mc.update(**augmented_experience)

    def set_policy(self, params):
        self.policy_params = params
        self.cached_policy = softmax(self.policy_params, axis=-1)

    def set_memory(self, params, logits=True):
        if logits:
            self.memory_params = params
            self.cached_memory_fn = softmax(self.memory_params, axis=-1)
        else:
            self.cached_memory_fn = params
            self.memory_params = np.log(self.cached_memory_fn + 1e-20)

    def optimize_memory(
            self,
            n_trials=100,
            n_epochs_per_trial=1,
            sampler=optuna.samplers.CmaEsSampler(),
    ):
        study = optuna.create_study(direction='minimize', sampler=sampler)

        required_params_shape = self.memory_params.shape[:-1] + (self.n_mem_values - 1, )
        n_required_params = np.prod(required_params_shape)

        def fill_in_params(required_params):
            params = np.empty_like(self.memory_params)
            params[:, :, :, :-1] = np.asarray(required_params).reshape(required_params_shape)
            params[:, :, :, -1] = 1 - np.sum(params[:, :, :, :-1], axis=-1)
            return params

        def objective(trial: optuna.Trial):
            print(trial.number)
            required_params = [
                trial.suggest_float(str(i), low=0.0, high=1.0) for i in range(n_required_params)
            ]
            self.set_memory(fill_in_params(required_params), logits=False)
            self.print_mem_summary()
            result = self.evaluate_memory(n_epochs_per_trial)
            print(f'Discrep: {result}\n')
            return result

        study.optimize(objective, n_trials=n_trials)
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
        memory_params = glorot_init(
            (self.n_actions, self.n_obs, self.n_mem_states, self.n_mem_states))
        self.set_memory(memory_params)
        self.augment_policy(mem_increase_multiplier)
