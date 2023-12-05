""" Interpolation by Gaussian Processes """

import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import torch
import gpytorch
from gpytorch.kernels import ScaleKernel, RBFKernel, PeriodicKernel
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.constraints import Interval
from gpytorch.priors import NormalPrior
from utils import normalize_data, plot_GP, multiple_plot_GP


def interpolation_GP_arn(X_observation, y_observation, type_kernel, type_package, name_specie):

    npts = 100
    val_min = X_observation_tensor.min()
    val_max = X_observation_tensor.max()
    t = torch.linspace(val_min.item(), val_max.item(), npts)
    
    if type_package == "Pytorch" :
        
        if type_kernel == "Normal" :
            k_per = ScaleKernel(PeriodicKernel(lengthscale_constraint=10)) 
            model_per = SingleTaskGP(X_observation, y_observation,
                                 covar_module=k_per)
            
        elif type_kernel == "Range" :
            k_per = ScaleKernel(PeriodicKernel(period_length_constraint=Interval(30, 45), lengthscale=10)) # Scale kernel adds the amplitude hyperparameter to the kernel
            model_per = SingleTaskGP(X_observation, y_observation,
                                 covar_module=k_per)
            
        elif type_kernel == "Fixed" :
            k_per = ScaleKernel(PeriodicKernel()) # Scale kernel adds the amplitude hyperparameter to the kernel
            model_per = SingleTaskGP(X_observation, y_observation, covar_module=k_per)
            # fixing the period hyperparameter
            model_per.covar_module.base_kernel.period_length = 40
            # disabling training for it once it has been fixed. 
            model_per.covar_module.base_kernel.raw_period_length.requires_grad_(False)
            
        elif type_kernel == "Prior" :
            period_prior= NormalPrior(40, 20)
            k_per = ScaleKernel(PeriodicKernel(period_length_prior=period_prior, lengthscale_prior=gpytorch.priors.GammaPrior(concentration=.9, rate=0.5)))
            model_per = SingleTaskGP(X_observation, y_observation, covar_module=k_per)
        
        mll_per = ExactMarginalLogLikelihood(model_per.likelihood, model_per)
        fit_gpytorch_model(mll_per, max_retries=30);
    
        
        
        preds_per = model_per(t)
        mean_per = preds_per.mean
        var_per = preds_per.variance
        
    
    
        print(f'Period: {model_per.covar_module.base_kernel.period_length.item()}')
        print(f'Amplitude: {model_per.covar_module.outputscale.item()}')
        print(f'Lengthscale: {model_per.covar_module.base_kernel.lengthscale.item()}')
        print(f'Noise: {model_per.likelihood.noise.item()}')
    
    elif type_package == "Sklearn" :
        
        k = ScaleKernel(RBFKernel(period_length=35)) # Scale kernel adds the amplitude hyperparameter to the kernel
        model = SingleTaskGP(X_observation, y_observation,
                             covar_module=k)
        
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll);
        
        preds = model(t)
        mean_per = preds.mean
        var_per = preds.variance
        
        with torch.no_grad():
            fig, axes = plt.subplots(1,1)
            axes.scatter(X_observation, y_observation, marker='x', color='black') # data
    
            axes.plot(t, mean_per, color='darkblue', linewidth=2)
            axes.fill_between(t, mean_per - 2*torch.sqrt(var_per), mean_per + 2*torch.sqrt(var_per),
                               alpha=.2, color='darkblue')
            axes.set_xlabel('Time $t$ (hours)')
            axes.set_title(str(name_specie))
            fig.tight_layout()
            fig.savefig("results/"+str(name_specie)+".png", dpi=300)
            plt.show()
            
    return mean_per, var_per, t


############# RNA-SEQ #########################################################
with open('data/Jeu_9/data/data_dict_concentration_cc.dat', 'rb') as fichier:
    rna_seq_data = pkl.load(fichier)
    
list_species = ["ARNTL", "CLOCK", "CRY1", "CRY2", "NR1D1", "PER1", "PER2", "PER3", "RORA"]
X_observation = rna_seq_data["CTs"]*3
kernel_method = "Range"
package_method = "Pytorch"

res_rna_seq = {name : {"X_observation_tensor" :{}, "y_observation_tensor_normalized" :{}, "time" :{}, "mean_specie":{}, "var_specie":{}} for name in list_species}

for i, name in enumerate(list_species) : 
    y_observation = np.concatenate((rna_seq_data["ctrl"][name]["pts"][:,0], rna_seq_data["ctrl"][name]["pts"][:,1], rna_seq_data["ctrl"][name]["pts"][:,2]), axis=0)

    # Missing data handling
    masque_nan = ~np.isnan(y_observation)
    X_observation_filtre = np.array(X_observation)[masque_nan]
    y_observation_filtre = y_observation[masque_nan]
    
    # Processing
    X_observation_filtre = np.array([[val] for val in X_observation_filtre])
    y_observation_filtre = np.array([[val] for val in y_observation_filtre])
    
    # Convert lists to PyTorch tensors
    X_observation_tensor = torch.tensor(X_observation_filtre, dtype=torch.double).view(-1, 1)  # Adjust the view size if necessary
    y_observation_tensor = torch.tensor(y_observation_filtre, dtype=torch.double).view(-1, 1)  # Adjust the view size if necessary

    # Normalize data
    y_observation_tensor_normalized, y_mean, y_std = normalize_data(y_observation_tensor)
    
    mean_specie, var_specie, time = interpolation_GP_arn(X_observation_tensor, y_observation_tensor_normalized, type_kernel = kernel_method, type_package = package_method, name_specie = name)
    plot_GP(X_observation_tensor, y_observation_tensor_normalized, mean_specie, time, var_specie, name, "RNA_Seq")

    res_rna_seq[name]["X_observation_tensor"] = X_observation_tensor
    res_rna_seq[name]["y_observation_tensor_normalized"] = y_observation_tensor_normalized
    res_rna_seq[name]["time"] = time
    res_rna_seq[name]["mean_specie"] = mean_specie
    res_rna_seq[name]["var_specie"] = var_specie
    
multiple_plot_GP(list_species, res_rna_seq, "RNA_Seq")

    

############# PROTEINS ########################################################
with open('data/dict_comparison', 'rb') as fichier:
    prot_data = pkl.load(fichier)
    
list_species = list(prot_data.keys())
kernel_method = "Range"
package_method = "Pytorch"

res_prot = {name : {"X_observation_tensor" :{}, "y_observation_tensor_normalized" :{}, "time" :{}, "mean_specie":{}, "var_specie":{}} for name in list_species}

for i, name in enumerate(list_species) : 
    dataset = [key for key in prot_data[name].keys() if key.startswith('Jeu')]
    
    X_observation = np.concatenate([prot_data[name][data]["cts"] for data in dataset])
    y_observation = np.concatenate([prot_data[name][data]["pts"] for data in dataset])

    # Missing data handling
    masque_nan = ~np.isnan(y_observation)
    X_observation_filtre = np.array(X_observation)[masque_nan]
    y_observation_filtre = y_observation[masque_nan]
    
    # Processing
    X_observation_filtre = np.array([[val] for val in X_observation_filtre])
    y_observation_filtre = np.array([[val] for val in y_observation_filtre])
    
    # Convert lists to PyTorch tensors
    X_observation_tensor = torch.tensor(X_observation_filtre, dtype=torch.double).view(-1, 1)  # Adjust the view size if necessary
    y_observation_tensor = torch.tensor(y_observation_filtre, dtype=torch.double).view(-1, 1)  # Adjust the view size if necessary

    # Normalize data
    y_observation_tensor_normalized, y_mean, y_std = normalize_data(y_observation_tensor)
    
    mean_specie, var_specie, time = interpolation_GP_arn(X_observation_tensor, y_observation_tensor_normalized, type_kernel = kernel_method, type_package = package_method, name_specie = name)
    plot_GP(X_observation_tensor, y_observation_tensor_normalized, mean_specie, time, var_specie, name, "Prot")

    res_prot[name]["X_observation_tensor"] = X_observation_tensor
    res_prot[name]["y_observation_tensor_normalized"] = y_observation_tensor_normalized
    res_prot[name]["time"] = time
    res_prot[name]["mean_specie"] = mean_specie
    res_prot[name]["var_specie"] = var_specie
    
multiple_plot_GP(list_species, res_prot, "Prot")
    

    
    