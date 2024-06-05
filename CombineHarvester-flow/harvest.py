import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from flowjax.bijections import RationalQuadraticSpline
from flowjax.distributions import Normal
from flowjax.flows import masked_autoregressive_flow

from CombineHarvesterFlow.utils import (WeightedMaximumLikelihoodLoss,
                                    fit_to_data_weight)


class Harvest():
    """Class to harvest the chains and train the flows."""
    def __init__(self, harvest_path, chain, n_flows, weights=None, random_seed=42):
        """Initializes the Harvest object.

        Parameters
        ----------
        harvest_path : string
            Path to save the models.
        chain : array
            The chain to train the flows on.
        n_flows : int
            Number of flows to train.
        weights : array, optional
            Weights for the chain, by default None
        random_seed : int, optional
            Random seed for the training of the flows, by default 42
        """
        self.harvest_path = harvest_path
        self.chain = chain
        self.n_flows = n_flows
        self.random_seed = random_seed

        self.weights = weights
        if self.weights is None:
            self.weights = np.ones_like(self.chain[:, 0])

    def _normalize_data(self):
        """Normalize the chain"""
        self.mean = np.average(self.chain, weights=self.weights, axis=0)
        self.std = (np.average((self.chain - self.mean)**2, weights=self.weights, axis=0)) ** 0.5
        self.norm_chain = (self.chain - self.mean) / self.std

    def _train_models(self):
        """Train the flows"""
        self.flow_list = []
        x = self.norm_chain
        for i in range(self.n_flows):
            key = jax.random.PRNGKey(self.random_seed + i)
            key, subkey = jax.random.split(key)
            flow = masked_autoregressive_flow(
                subkey,
                base_dist=Normal(jnp.zeros(x.shape[1])),
                transformer=RationalQuadraticSpline(knots=8, interval=4),
            )

            key, subkey = jax.random.split(key)
            flow, losses = fit_to_data_weight(
                weights=self.weights, key=subkey, dist=flow, x=x,
                learning_rate=1e-3, loss_fn=WeightedMaximumLikelihoodLoss()
            )
            self.flow_list += [flow]

    def harvest(self):
        """Harvest the chains and train the flows."""
        self._normalize_data()
        print('Training the flows')
        self._train_models()

    def save_models(self):
        """Save the models"""
        np.save(self.harvest_path + '_mean.npy', self.mean)
        np.save(self.harvest_path + '_std.npy', self.std)
        np.save(self.harvest_path + '_weights.npy', self.weights)
        np.save(self.harvest_path + '_norm_chain.npy', self.norm_chain)
        np.save(self.harvest_path + '_chain.npy', self.chain)
        for _ in range(len(self.flow_list)):
            eqx.tree_serialise_leaves(self.harvest_path + f'_flow_{_}.eqx', self.flow_list[_])

    def load_models(self):
        """Load the models"""
        self.mean = np.load(self.harvest_path + '_mean.npy')
        self.std = np.load(self.harvest_path + '_std.npy')
        self.weights = np.load(self.harvest_path + '_weights.npy')
        self.norm_chain = np.load(self.harvest_path + '_norm_chain.npy')
        self.chain = np.load(self.harvest_path + '_chain.npy')
        self.flow_list = []
        for i in range(self.n_flows):
            key, subkey = jr.split(jr.PRNGKey(i))
            model = masked_autoregressive_flow(
                subkey, base_dist=Normal(jnp.zeros_like(self.chain[0,:])),
                transformer=RationalQuadraticSpline(knots=8, interval=4)
            )
            self.flow_list += [
                eqx.tree_deserialise_leaves(self.harvest_path + "_flow_%s.eqx" % i, model)]
