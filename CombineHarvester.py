
import numpy as np 
import threading
import random
import scipy.stats as stats
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import device_put, pmap, vmap
from flowjax.bijections import RationalQuadraticSpline
from flowjax.distributions import Normal
from flowjax.flows import masked_autoregressive_flow
import equinox as eqx
from utils import fit_to_data_weight, WeightedMaximumLikelihoodLoss




class Harvest():

    def __init__(self, harvest_path, chain, n_flows, weights=None, random_seed=42):
        self.harvest_path= harvest_path
        self.chain = chain
        self.n_flows = n_flows
        self.weights = weights
        self.random_seed = random_seed

    def _normalize_data(self):
        if self.weights is None:
            self.weights = np.ones_like(self.chain[:,0])
        self.mean = np.average(self.chain, weights=self.weights, axis = 0)
        self.std = (np.average((self.chain - self.mean)**2, weights=self.weights, axis = 0)) ** 0.5
        self.norm_chain = (self.chain - self.mean) / self.std
        return 0.

    def _process_on_device(self, start, end, device):
        def _device_specific_computation(i):
            x = device_put(self.norm_chain, device)
            key = jax.random.PRNGKey(self.random_seed + i)
            key, subkey = jax.random.split(key)
            flow = masked_autoregressive_flow(
                subkey,
                base_dist=Normal(jnp.zeros(x.shape[1])),
                transformer=RationalQuadraticSpline(knots=8, interval=4),
            )

            key, subkey = jax.random.split(key)
            flow, losses = fit_to_data_weight(weights=self.weights, key=subkey, dist=flow, x=x, 
                learning_rate=1e-3, loss_fn=WeightedMaximumLikelihoodLoss())

            #add the model to the list
            self.flow_list[i] = flow

        # Run the loop on the specified device
        for i in range(start, end):
            _device_specific_computation(i)

    def _train_models(self):
        self.flow_list = [None] * self.n_flows

        # Get available GPUs and CPUs
        devices = [device for device in jax.devices()]

        # Total number of iterations
        total_iterations = self.n_flows

        # Number of devices
        num_devices = len(devices)

        # Calculate the number of iterations per device
        iterations_per_device = total_iterations // num_devices
        remainder = total_iterations % num_devices

        threads = []
        start = 0
        for i, device in enumerate(devices):
            end = start + iterations_per_device + (1 if i < remainder else 0)
            thread = threading.Thread(target=self._process_on_device, args=(start, end, device))
            threads.append(thread)
            thread.start()
            start = end

        # Join threads
        for thread in threads:
            thread.join()

        return 0.

    def harvest(self):
        self._normalize_data()
        print ('Training the flows')
        self._train_models()

    def save_models(self):
        np.save(self.harvest_path + '_mean.npy', self.mean)
        np.save(self.harvest_path + '_std.npy', self.std)
        np.save(self.harvest_path + '_weights.npy', self.weights)
        np.save(self.harvest_path + '_norm_chain.npy', self.norm_chain)
        np.save(self.harvest_path + '_chain.npy', self.chain)
        for _ in range(len(self.flow_list)):
            eqx.tree_serialise_leaves(self.harvest_path + f'_flow_{_}.eqx', self.flow_list[_])
        return 0. 

    def load_models(self):
        self.mean = np.load(self.harvest_path + '_mean.npy')
        self.std = np.load(self.harvest_path + '_std.npy')
        self.weights = np.load(self.harvest_path + '_weights.npy')
        self.norm_chain = np.load(self.harvest_path + '_norm_chain.npy')
        self.chain = np.load(self.harvest_path + '_chain.npy')
        self.flow_list = []
        for i in range(self.n_flows):
            key, subkey = jr.split(jr.PRNGKey(i))
            model = masked_autoregressive_flow(
            subkey,
            base_dist=Normal(jnp.zeros_like(self.chain[0,:])),
            transformer=RationalQuadraticSpline(knots=8, interval=4))
            self.flow_list += [eqx.tree_deserialise_leaves(self.harvest_path + "_flow_%s.eqx" %i, model)]
        return 0.




class Combine():

    def __init__(self, harvest_1, harvest_2):
        self.harvest_1 = harvest_1
        self.harvest_2 = harvest_2

    def __getattr__(self, name):
        if hasattr(self.harvest_1, name) and callable(getattr(self.harvest_1, name)):
            return getattr(self.harvest_1, name)
        elif hasattr(self.harvest_2, name) and callable(getattr(self.harvest_2, name)):
            return getattr(self.harvest_2, name)
        else:
            raise AttributeError(f"'Combine' object has no attribute '{name}'")


    def combine(self):

        #normalize the chains
        norm_chain_1 = (self.harvest_1.chain - self.harvest_2.mean) / self.harvest_2.std
        norm_chain_2 = (self.harvest_2.chain - self.harvest_1.mean) / self.harvest_1.std

        # get the weights
        flow_weight_list_2, flow_weight_list_1 = [], []
        for i in range(self.harvest_1.n_flows):
            flow_weight_list_2 += [np.asarray(self.harvest_1.flow_list[i].log_prob(norm_chain_2))]
        for i in range(self.harvest_2.n_flows):
            flow_weight_list_1 += [np.asarray(self.harvest_2.flow_list[i].log_prob(norm_chain_1))]

        ln_weights_2 = np.sum(np.vstack(flow_weight_list_2), axis = 0) / self.harvest_1.n_flows
        ln_weights_1 = np.sum(np.vstack(flow_weight_list_1), axis = 0) / self.harvest_2.n_flows

        #convert from log-likelihood to likelihood and update weights. Normalize mx(ln(weights)) to 0 to avoid overflow.
        ln_weights_1 -= np.max(ln_weights_1)
        ln_weights_2 -= np.max(ln_weights_2)

        chain_1_weights = self.harvest_1.weights * np.exp(ln_weights_1)
        chain_2_weights = self.harvest_2.weights * np.exp(ln_weights_2)

        return chain_1_weights, chain_2_weights



    def combine_subset(self, n_flows_1, n_flows_2):
        old_n_flows_1, old_n_flows_2 = self.harvest_1.n_flows, self.harvest_2.n_flows
        self.harvest_1.n_flows = n_flows_1 
        self.harvest_2.n_flows = n_flows_2 
        chain_1_weights, chain_2_weights = self.combine()
        #reset
        self.harvest_1.n_flows = old_n_flows_1
        self.harvest_2.n_flows = old_n_flows_2
        return chain_1_weights, chain_2_weights
    









 
