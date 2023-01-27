import copy
from functools import partial
from itertools import repeat
from multiprocessing import Pool, freeze_support
import os
import shutil
from tqdm import tqdm

# from jax.nn import softmax
from scipy.special import softmax
import numpy as np
import optuna

from grl.utils.math import arg_hardmax, arg_mellowmax, arg_boltzman
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
                 policy_epsilon: float = 0.10,
                 mellowmax_beta: float = 10.0,
                 replay_buffer_size: int = 1000000) -> None:
        self.n_obs = n_obs
        self.n_actions = n_actions
        self.n_mem_values = n_mem_values
        self.n_mem_entries = n_mem_entries
        self.n_mem_states = n_mem_values**n_mem_entries
        self.policy_epsilon = policy_epsilon
        self.mellowmax_beta = mellowmax_beta

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
        self.q_mc = TDLambdaQFunction(lambda_=0.9, **q_fn_kwargs)
        self.replay = ReplayMemory(capacity=replay_buffer_size)
        self.reset_memory_state()

    def mem_summary(self, precision=3):
        mem = self.memory_probs.round(precision)
        # mem_summary = str(np.concatenate((mem[2, 0], mem[2, 1], mem[2, 2]), axis=-1))
        mem_summary = mem
        return mem_summary

    def reset_memory_state(self):
        self.memory = 0
        self.prev_action = None

    def reset_value_functions(self):
        self.q_mc.reset()
        self.q_td.reset()

    def act(self, obs):
        obs_aug = self.augment_obs(obs)
        action = np.random.choice(self.n_actions, p=self.policy_probs[obs_aug])
        self.step_memory(obs, action)
        return action

    def step_memory(self, obs, action):
        next_memory = np.random.choice(
            self.n_mem_states,
            p=self.memory_probs[action, obs, self.memory],
        )
        self.prev_memory = self.memory
        self.memory = next_memory

    def store(self, experience):
        self.replay.push(experience)

    def update_actor(self, mode='td', argmax_type='hardmax', eps=None):
        if mode == 'td':
            q_fn = self.q_td.q
        elif mode == 'mc':
            q_fn = self.q_mc.q
        else:
            raise ValueError(f'Invalid mode: {mode}')

        if eps is None:
            eps = self.policy_epsilon

        if argmax_type == 'hardmax':
            greedy_pi = arg_hardmax(q_fn, axis=0)
            uniform_pi = np.ones_like(greedy_pi) / self.n_actions
            new_policy = (1 - eps) * greedy_pi + eps * uniform_pi
        elif argmax_type == 'mellowmax':
            new_policy = arg_mellowmax(q_fn, axis=0, beta=self.mellowmax_beta)
        elif argmax_type == 'boltzman':
            new_policy = arg_boltzman(q_fn, axis=0)
        else:
            raise ValueError(f'Invalid argmax_type: {argmax_type}')

        did_change = self.set_policy(new_policy.T, logits=False)
        if did_change:
            self.replay.reset()
        return did_change

    def update_critic(self, experience: dict):
        augmented_experience = self.augment_experience(experience)
        self.q_td.update(**augmented_experience)
        self.q_mc.update(**augmented_experience)

    def set_policy(self, params, logits=True):
        old_policy_probs = self.policy_probs
        if logits:
            self.policy_logits = params
            self.policy_probs = softmax(self.policy_logits, axis=-1)
        else:
            self.policy_probs = params
            self.policy_logits = np.log(self.policy_probs + 1e-20)

        if old_policy_probs is None:
            did_change = True
        else:
            did_change = not np.allclose(old_policy_probs, self.policy_probs, atol=0.01)
        return did_change

    def set_memory(self, params, logits=True):
        if logits:
            self.memory_logits = params
            self.memory_probs = softmax(self.memory_logits, axis=-1)
        else:
            self.memory_probs = params
            self.memory_logits = np.log(self.memory_probs + 1e-20)

    def reset_policy(self):
        self.policy_logits = None
        self.policy_probs = None
        policy_shape = (self.n_obs * self.n_mem_states, self.n_actions)
        self.set_policy(normal_init(policy_shape, scale=0.2))

    def reset_memory(self):
        mem_shape = (self.n_actions, self.n_obs, self.n_mem_states, self.n_mem_states)
        self.set_memory(normal_init(mem_shape))

    def fill_in_params(self, required_params):
        required_params_shape = self.memory_logits.shape[:-1] + (self.n_mem_values - 1, )
        params = np.empty_like(self.memory_logits)
        params[:, :, :, :-1] = np.asarray(required_params).reshape(required_params_shape)
        params[:, :, :, -1] = 1 - np.sum(params[:, :, :, :-1], axis=-1)
        return params

    def on_trial_stopped_callback(self, study: optuna.study.Study,
                                  trial: optuna.trial.FrozenTrial) -> None:
        # whenever there's a new best trial
        if trial.state == optuna.trial.TrialState.COMPLETE and trial.values[0] < study.best_value:
            print(f"New best trial: {trial.number}")
            # check if rounding params might help
            params = trial.params
            rounded_params = {key: np.round(val) for (key, val) in params.items()}
            if all([params[key] == rounded_params[key] for key in params.keys()]):
                return # already rounded; nothing useful to suggest by rounding

            # also construct a compromise set of "half-rounded" params
            compromise_params = {
                key: np.mean([params[key], rounded_params[key]])
                for key in params.keys()
            }

            # enqueue fully rounded params first, since they're expected to help more
            print(f"Enqueuing rounded version")
            study.enqueue_trial(rounded_params, skip_if_exists=True)

            # then enqueue the compromise as a backup
            print(f"Enqueuing semi-rounded version")
            study.enqueue_trial(compromise_params, skip_if_exists=True)

    def objective(self, trial: optuna.Trial, study_dir='./results/sample_based/'):
        n_required_params = np.prod(self.memory_logits.shape) // self.n_mem_states

        required_params = []
        for i in range(n_required_params):
            n_suggest_float_attempts = 0
            while True:
                n_suggest_float_attempts += 1
                if n_suggest_float_attempts >= 100 and n_suggest_float_attempts % 100 == 0:
                    print(f'Failed to suggest_float {n_suggest_float_attempts} in a row!?')
                try:
                    x = trial.suggest_float(str(i), low=0.0, high=1.0)
                except RuntimeError:
                    continue
                else:
                    break
            required_params.append(x)

        self.set_memory(self.fill_in_params(required_params), logits=False)
        result = self.evaluate_memory()

        with open(os.path.join(study_dir, 'output.txt'), 'a') as file:
            file.write(f'Trial: {trial.number}\n')
            file.write(f'Discrep: {result}\n')
            file.write(f'Memory:\n{self.mem_summary()}\n\n')
            file.flush()
        return result

    def optimize_memory(
            self,
            study_name=None,
            preamble_str='',
            n_trials=500,
            n_jobs=1,
            sampler=optuna.samplers.TPESampler(constant_liar=True),
            new_study=False,
    ):
        study_dir = f'./results/sample_based/{study_name}'
        if new_study and os.path.exists(study_dir):
            shutil.rmtree(study_dir)

        os.makedirs(study_dir, exist_ok=True)
        # saved_replaymemory_path = os.path.join(study_dir, 'replaymemory.pkl')
        # print(saved_replaymemory_path)
        # if os.path.exists(saved_replaymemory_path):
        #     print("Loading existing replay buffer")
        #     self.replay = self.replay.load(saved_replaymemory_path)
        # else:
        #     print("Saving replay buffer")
        #     self.replay.save(study_dir, filename='replaymemory', extension='.pkl')

        study = optuna.create_study(
            study_name=study_name,
            direction='minimize',
            storage=optuna.storages.JournalStorage(
                optuna.storages.JournalFileStorage(os.path.join(study_dir, "study.journal"))),
            sampler=sampler,
            load_if_exists=True,
        )

        with open(os.path.join(study_dir, 'output.txt'), 'a') as file:
            file.write(preamble_str)
            file.flush()

        n_jobs = max(n_jobs, 1)
        n_trials_per_worker = list(map(len, np.array_split(np.arange(n_trials), n_jobs)))
        print(f'Starting pool with {n_jobs} workers')
        print(f'n_trials_per_worker: {n_trials_per_worker}')
        if n_jobs > 1:
            freeze_support()
            pool = Pool(n_jobs, maxtasksperchild=1) # Each new tasks gets a fresh worker
            pool.starmap(
                partial(study.optimize, callbacks=[self.on_trial_stopped_callback]),
                zip(repeat(partial(self.objective, study_dir=study_dir)), n_trials_per_worker),
            )
            pool.close()
            pool.join()
        else:
            study.optimize(
                partial(self.objective, study_dir=study_dir),
                n_trials_per_worker[0],
                callbacks=[self.on_trial_stopped_callback],
            )

        required_params = [
            study.best_trial.params[key]
            for key in sorted(study.best_trial.params.keys(), key=lambda x: int(x))
        ]
        params = self.fill_in_params(required_params)
        self.set_memory(params, logits=False)

        with open(os.path.join(study_dir, 'output.txt'), 'a') as file:
            file.write('--------------------------------------------\n')
            file.write(f'Best trial: {study.best_trial.number}\n')
            file.write(f'Discrep: {study.best_trial.value}\n')
            file.write(f'Memory:\n{self.mem_summary()}\n\n')
            file.flush()

        return study

    def evaluate_memory(self, n_epochs=1):
        for epoch in range(n_epochs):
            assert len(self.replay.memory) > 0
            self.reset_memory_state()
            self.reset_value_functions()

            first_episode = True
            for experience in self.replay.memory:
                e = experience.copy()
                del e['_index_']
                # Skip the first episode since it might be partial
                if first_episode:
                    if e['terminal']:
                        first_episode = False
                    continue
                self.step_memory(e['obs'], e['action'])
                self.update_critic(e)
                if experience['terminal']:
                    self.reset_memory_state()
        abs_lambda_discrepancy = np.abs(self.q_mc.q - self.q_td.q)
        obs, actions = self.replay.retrieve(fields=['obs', 'action'])
        observed_lambda_discrepancies = abs_lambda_discrepancy[actions, obs]
        return observed_lambda_discrepancies.mean()

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
        pi_augmented = np.expand_dims(self.policy_logits, 1).repeat(n_mem_states, 1) # O x M x A
        self.policy_probs = None
        self.set_policy(pi_augmented.reshape(self.n_obs * n_mem_states, self.n_actions)) # OM x A

    def add_memory(self, n_mem_entries=1):
        mem_increase_multiplier = (self.n_mem_values**n_mem_entries)
        self.q_td.augment_with_memory(mem_increase_multiplier)
        self.q_mc.augment_with_memory(mem_increase_multiplier)

        self.n_mem_entries += n_mem_entries
        self.n_mem_states = self.n_mem_states * mem_increase_multiplier
        self.reset_memory()
        self.augment_policy(mem_increase_multiplier)
