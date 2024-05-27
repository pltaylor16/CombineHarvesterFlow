import numpy as np


class Combine():
    """Combine two Harvest objects.
    """
    def __init__(self, harvest_1, harvest_2):
        """Initializes the Combine object

        Parameters
        ----------
        harvest_1 : Harvest
            First Harvest object.
        harvest_2 : Harvest
            Second Harvest object.
        """
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
        """Combine the two chains

        Returns
        -------
        (array, array)
            The combined weights for the two chains.
        """
        # normalize the chains
        norm_chain_1 = (self.harvest_1.chain - self.harvest_2.mean) / self.harvest_2.std
        norm_chain_2 = (self.harvest_2.chain - self.harvest_1.mean) / self.harvest_1.std

        # get the weights
        flow_weight_list_2, flow_weight_list_1 = [], []
        for i in range(self.harvest_1.n_flows):
            flow_weight_list_2 += [np.asarray(self.harvest_1.flow_list[i].log_prob(norm_chain_2))]
        for i in range(self.harvest_2.n_flows):
            flow_weight_list_1 += [np.asarray(self.harvest_2.flow_list[i].log_prob(norm_chain_1))]

        ln_weights_2 = np.sum(np.vstack(flow_weight_list_2), axis=0) / self.harvest_1.n_flows
        ln_weights_1 = np.sum(np.vstack(flow_weight_list_1), axis=0) / self.harvest_2.n_flows

        # convert from log-likelihood to likelihood and update weights. Normalize mx(ln(weights)) to 0 to avoid overflow.
        ln_weights_1 -= np.max(ln_weights_1)
        ln_weights_2 -= np.max(ln_weights_2)

        chain_1_weights = self.harvest_1.weights * np.exp(ln_weights_1)
        chain_2_weights = self.harvest_2.weights * np.exp(ln_weights_2)

        return chain_1_weights, chain_2_weights

    def combine_subset(self, n_flows_1, n_flows_2):
        """Combine the two chains with a subset of the flows

        Parameters
        ----------
        n_flows_1 : int
            Number of flows to use for the first chain.
        n_flows_2 : int
            Number of flows to use for the second chain.

        Returns
        -------
        (array, array)
            The combined weights for the two chains.
        """
        old_n_flows_1, old_n_flows_2 = self.harvest_1.n_flows, self.harvest_2.n_flows
        self.harvest_1.n_flows = n_flows_1
        self.harvest_2.n_flows = n_flows_2
        chain_1_weights, chain_2_weights = self.combine()

        # reset
        self.harvest_1.n_flows = old_n_flows_1
        self.harvest_2.n_flows = old_n_flows_2
        return chain_1_weights, chain_2_weights
