""" Useful functions """
import torch
import matplotlib.pyplot as plt

def normalize_data(y):
    y_mean, y_std = y.mean(), y.std()
    y_normalized = (y - y_mean) / y_std

    return y_normalized, y_mean, y_std


def denormalize_data(y_normalized, mean, std):
    y = (y_normalized * std) + mean

    return y


def plot_GP(X_observation, y_observation, mean_per, t, var_per, name_specie, type_data):
    with torch.no_grad():
        plt.scatter(X_observation, y_observation, marker='x', color='black')  # data
        plt.plot(t, mean_per.detach().numpy(), color='darkblue', linewidth=2)  # Detach the tensor
        plt.fill_between(t, (mean_per - 2 * torch.sqrt(var_per)).detach().numpy(),
                         (mean_per + 2 * torch.sqrt(var_per)).detach().numpy(),
                         alpha=.2, color='darkblue')
        plt.xlabel('Time $t$ (hours)')
        plt.ylabel('Concentration (nmol/L)')
        plt.title(str(name_specie))
        plt.tight_layout()
        plt.savefig("results/" + str(type_data) + "_" + str(name_specie) + ".png", dpi=300)
        plt.show()
        
        
def multiple_plot_GP(list_species, dict_data, type_data):
    num_rows = 2  
    num_cols = -(-len(list_species) // num_rows) 

    # Ajout du subplot à partir du dictionnaire dict_data
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5*num_rows))

    for i, name in enumerate(list_species):
        row, col = i // num_cols, i % num_cols  # Calcul de la position du subplot dans la grille
        X_observation_tensor = dict_data[name]["X_observation_tensor"]
        y_observation_tensor_normalized = dict_data[name]["y_observation_tensor_normalized"]
        mean_specie = dict_data[name]["mean_specie"]
        time = dict_data[name]["time"]
        var_specie = dict_data[name]["var_specie"]

        axes[row, col].scatter(X_observation_tensor, y_observation_tensor_normalized, marker='x', color='black')  # data

        axes[row, col].plot(time, mean_specie.detach().numpy(), color='darkblue', linewidth=2)  # Detach the tensor
        axes[row, col].fill_between(time, (mean_specie - 2 * torch.sqrt(var_specie)).detach().numpy(),
                                    (mean_specie + 2 * torch.sqrt(var_specie)).detach().numpy(),
                                    alpha=.2, color='darkblue')
        axes[row, col].set_xlabel('Time $t$ (hours)')
        axes[row, col].set_ylabel('Concentration (nmol/L)')
        axes[row, col].set_title(str(name))

    # Supprimer les sous-graphiques inutilisés
    for i in range(len(list_species), num_rows * num_cols):
        fig.delaxes(axes.flatten()[i])

    fig.tight_layout()
    plt.savefig("results/GP_"+str(type_data)+".png", dpi=300)
    plt.show()