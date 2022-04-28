""" Auxiliary plotting functions"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def plot_color(samples, y_labels, X_mb, scale, titl):
    fig = plt.figure(figsize=(5, 5))
    gs = gridspec.GridSpec(1, 1)
    gs.update(wspace=0.5, hspace=0.5)
    S = (np.ones_like(X_mb[:, 0])) * 5 #25.  # circle size


    # plot test-data
    plt.scatter(X_mb[:, 0], X_mb[:, 1], s=S, color='k', alpha=0.4, marker='o')
    # plot generated data
    S = (np.ones_like(y_labels)) * 25.  # circle size
    plt.scatter(samples[:, 0], samples[:,1], s=S, c= y_labels, alpha=0.99, marker='o')

    plt.title(titl)
    plt.xlabel(r'$x_1$', fontsize=20)
    plt.ylabel(r'$x_2$', fontsize=20)
    plt.axis([-scale, scale, -scale, scale])
    plt.grid(True)
    return fig


def plot_losses_over_steps():
    if not os.path.exists('./output_files'):
        os.makedirs('./output_files')

    fig = plt.figure()

    plt.plot(D_loss_plots)
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.ylim(-5, 15)  # (-15, 15)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(['Adversarial (D_loss-G_loss)'], loc='upper right', fontsize=14)
    plt.savefig(fname + '/LIP_plots_S_pen_INTERPOLATE/out_Loss_plots/ccgan_D_Loss_beta' + str(beta) + 'gamma_' + str(
        gamma) + '.png', bbox_inches='tight')
    plt.close(fig)

    fig = plt.figure()

    # plt.plot(x_idx, D_loss_plots)
    plt.plot(G_loss_plots)
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    # plt.ylim(-5, 15) # (-15, 15)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(['Generator (G_loss)'], loc='upper right', fontsize=14)
    plt.savefig(fname + '/LIP_plots_S_pen_INTERPOLATE/out_Loss_plots/ccgan_G_Loss_beta' + str(beta) + 'gamma_' + str(
        gamma) + '.png', bbox_inches='tight')
    plt.close(fig)