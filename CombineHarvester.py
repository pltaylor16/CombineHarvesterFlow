
import numpy as np 
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import device_put, pmap
from flowjax.bijections import RationalQuadraticSpline
from flowjax.distributions import Normal
from flowjax.flows import masked_autoregressive_flow
from flowjax.train import fit_to_data
import threading



class Harvest():

    def __init__(self, harvest_name, chain=None, n_flows=None, weights=None, random_seed=42):
        self.harvest_name= harvest_name
        self.chain = chain
        self.n_flows = n_flows
        self.weights = weights
        self.random_seed = random_seed

    def _normalize_data(self):
        if self.weights == None:
            self.weights = np.ones_like(chain.shape[0])
        self.mean = np.average(data, weights=weights, axis = 0)
        self.std = (np.average((data - mean)**2, weights=weights, axis = 0)) ** 0.5
        self.norm_chain = (self.chain - self.mean) / self.std
        return 0.

    def _process_on_device(self, start, end, device):
        def _device_specific_computation(i):
            x = device_put(self.norm_chain, device)
            key = jax.random.PRNGKey(i)
            key, subkey = jax.random.split(key)
            flow = masked_autoregressive_flow(
                subkey,
                base_dist=Normal(jnp.zeros(x.shape[1])),
                transformer=RationalQuadraticSpline(knots=8, interval=4),
            )

            key, subkey = jax.random.split(key)
            flow, losses = fit_to_data(subkey, flow, x, learning_rate=1e-3)

            #add the model to the list
            self.flow_list[i] = flow

        # Run the loop on the specified device
        for i in range(start, end):
            random.seed(self.random_seed)
            _device_specific_computation(i+device+random.randint(-2**20, 2**20))

    def _train_models(self):
        self.flow_list = [None] * self.n_flows

        # Get available GPUs and CPUs
        gpus = [device for device in jax.devices() if device.device_kind == 'Gpu']
        cpus = [device for device in jax.devices() if device.device_kind == 'Cpu']
        
        # Prefer GPUs, fall back to CPUs if no GPUs are available
        if gpus:
            devices = gpus
        else:
            devices = cpus

        # Total number of iterations
        total_iterations = self.n_flows

        # Number of devices
        num_devices = len(devices)

        # Calculate the number of iterations per device
        iterations_per_device = total_iterations // num_devices
        remainder = total_iterations % num_devices


        start = 0
        for i, device in enumerate(devices):
            end = start + iterations_per_device + (1 if i < remainder else 0)
            thread = threading.Thread(target=_process_on_device, args=(self, start, end, device))
            threads.append(thread)
            thread.start()
            start = end

        # Join threads
        for thread in threads:
            thread.join()

        return 0.

    def harvest(self):
        self._normalize_data()
        self._train_models()

    def save_models(self):
        np.save(self.harvest_name + '_mean.npy', self.mean)
        np.save(self.harvest_name + '_std.npy', self.std)
        np.save(self.harvest_name + '_weights.npy', self.weights)
        np.save(self.harvest_name + '_norm_chain.npy', self.norm_chain)
        for _ in len(self.flow_list):
            eqx.tree_serialise_leaves(self.harvest_name + f'_flow_{_}.eqx', flow_list[_])
        return 0. 

    def load_models(self):
        self.mean = np.load(self.harvest_name + '_mean.npy')
        self.std = np.load(self.harvest_name + '_std.npy')
        self.weights = np.load(self.harvest_name + '_weights.npy')
        self.norm_chain = np.load(self.harvest_name + '_norm_chain.npy')
        #still need to load the flows
        return 0.



class Combine():

    def __init__(self, harvest_1, harvest_2):
        self.harvest_1 = harvest_1
        self.harvest_2 = harvest_2

    def combine(self):

 


'''
import numpy as np 
from sklearn.mixture import GaussianMixture 
from math import pi as pi
from math import inf as inf
from sklearn.preprocessing import StandardScaler


class Harvest():

    def __init__(self, chain, n_models=10, n_components=None, weights=None, n_reweight_samples=10000, n_samples =10000, scaler=None):

        self.chain = chain
        self.n_models = n_models
        self.n_components = n_components
        self.weights = weights
        self.n_reweight_samples = n_reweight_samples
        self.n_samples = n_samples
        self.scaler = scaler

    def _resample_chain(self):
        self.resampled_chain = self.chain
        if self.weights is not None:
            print ('Redrawing {} Points Since Chain is Weighted'.format(self.n_reweight_samples))
            normalized_weights = self.weights / np.sum(self.weights) 
            sampled_indices = np.random.choice(a=len(self.chain), size=self.n_reweight_samples, replace=True, p=normalized_weights)
            self.resampled_chain = self.chain[sampled_indices]
    
    def _normalize_chain(self):
        if self.scaler == None:
            print('CAUTION: You Will Want to Pass a Shared Data Scaler When Combining Chains!!!')
            self.scaler = StandardScaler()
            self.resampled_chain = self.scaler.fit_transform(self.resampled_chain)

    def _denormalize_chain(self):
        self.resampled_chain = self.scaler.inverse_transform(self.resampled_chain)

    def _fit_gmm(self):
        gmm = GaussianMixture(self.n_components)
        gmm.fit(self.resampled_chain)
        return gmm
        
    def _get_n_components_convergence(self, test_pts = np.logspace(0, 3, 20, dtype = int)):
        print ('Determining Optimal Number of Mixture Components')
        best_n_components = 0 
        best_score = -inf
        self.scores = []
        for test in (test_pts):
            self.n_components = test
            gmm = self._fit_gmm()
            score = gmm.score(self.resampled_chain)
            print('Testing {} Gaussian Components. Score: {}'.format(test, score))
            if  score > best_score:
                best_n_components = test 
            self.scores.append(gmm.score(self.resampled_chain))
        self.n_components = best_n_components

    def train(self):
        self._resample_chain()
        self._normalize_chain()
        if self.n_components == None:
            self._get_n_components_convergence()
        print ('Fitting {} GMM Models with {} Components to Data'.format(self.n_models, self.n_components))
        self.models = [self._fit_gmm() for i in range(self.n_models)]
        self._denormalize_chain()


    def resample(self):
        n_draws = np.random.multinomial(self.n_samples, np.ones(self.n_models) / self.n_models)
        out_array_list = []
        for i in range(len(self.models)):
            out_i = self.scaler.inverse_transform(self.models[i].sample(n_samples = n_draws[i])[0])
            out_array_list.append(out_i)
        return np.vstack(out_array_list)




class Combine():

    def __init__(self, model_1, model_2, prior_1=None, prior_2=None, prior_target=None, Jacobian_1=None, Jacobian_2=None, n_samples = 10000):

        self.model_1 = model_1
        self.model_2 = model_2
        self.prior_1 = prior_1
        self.prior_2 = prior_2
        self.prior_target = prior_target
        self.Jacobian_1 = Jacobian_1
        self.Jacobian_2 = Jacobian_2
        self.n_samples = n_samples


    def load_model(self):
        pass


    def _reweight_chain(self):
        pass

    def _multiply_gaussians(self, mean1, cov1, mean2, cov2):
        inv_cov1 = np.linalg.inv(cov1)
        inv_cov2 = np.linalg.inv(cov2)
        cov_product = np.linalg.inv(inv_cov1 + inv_cov2)
        mean_product = cov_product @ (inv_cov1 @ mean1 + inv_cov2 @ mean2)
        k = np.shape(mean1)[0]
        denominator = ((2 * pi) ** k * np.linalg.det(cov1 + cov2)) ** 0.5
        exponent = np.exp(-0.5 * (mean1- mean2) @ np.linalg.inv((cov1 + cov2)) @ (mean1-mean2))
        prefactor = 1. / denominator * exponent
        return mean_product, cov_product, prefactor

    def _create_product_gmm(self, gmm1, gmm2):
        n_components1 = gmm1.means_.shape[0]
        n_components2 = gmm2.means_.shape[0]
        
        new_means = []
        new_covs = []
        new_weights = []
        new_precisions = []
        
        for i in range(n_components1):
            for j in range(n_components2):
                mean1, cov1, weight1 = gmm1.means_[i], gmm1.covariances_[i], gmm1.weights_[i]
                mean2, cov2, weight2 = gmm2.means_[j], gmm2.covariances_[j], gmm2.weights_[j]
                
                new_mean, new_cov, prefactor = _multiply_gaussians(self, mean1, cov1, mean2, cov2)
                new_precesions = np.linalg.inv(new_cov)
                new_means.append(new_mean)
                new_covs.append(new_cov)
                new_weights.append((prefactor * weight1 * weight2))
                new_precisions.append(new_precesions)
        
        new_weights = np.array(new_weights)
        new_weights /= new_weights.sum()  
        
        # Create a new GMM
        new_gmm = GaussianMixture(n_components=len(new_weights))
        new_gmm.weights_ = new_weights
        new_gmm.means_ = np.array(new_means)
        new_gmm.covariances_ = np.array(new_covs)
        return new_gmm


    def draw_joint_samples(self):
        pass


    def save_joint_samples(self):
        pass


'''