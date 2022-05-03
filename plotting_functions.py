""" Auxiliary plotting functions"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# 2-D plot (used in SwissRoll dataset)
def plot_color(samples, y_labels, X_mb, scale, titl):
    fig = plt.figure(figsize=(5, 5))
    gs = gridspec.GridSpec(1, 1)
    gs.update(wspace=0.5, hspace=0.5)
    S = (np.ones_like(X_mb[:, 0])) * 5  # 25.  # circle size

    # plot test-data
    plt.scatter(X_mb[:, 0], X_mb[:, 1], s=S, color='k', alpha=0.4, marker='o')
    # plot generated data
    S = (np.ones_like(y_labels)) * 25.  # circle size
    plt.scatter(samples[:, 0], samples[:, 1], s=S, c=y_labels, alpha=0.99, marker='o')

    plt.title(titl)
    plt.xlabel(r'$x_1$', fontsize=20)
    plt.ylabel(r'$x_2$', fontsize=20)
    plt.axis([-scale, scale, -scale, scale])
    plt.grid(True)
    return fig


def plot_losses_over_steps(D_loss_plots, G_loss_plots, save_fname, alpha):
    if not os.path.exists('./output_files/' + save_fname + '/loss_plots'):
        os.makedirs('./output_files/' + save_fname + '/loss_plots')

    fig = plt.figure()

    plt.plot(D_loss_plots)
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.ylim(-5, 15)  # (-15, 15)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(['Adversarial (D_loss-G_loss)'], loc='upper right', fontsize=14)
    plt.savefig('./output_files/' + save_fname + '/loss_plots/ccgan_D_Loss_alpha' + str(alpha) + '.png',
                bbox_inches='tight')
    plt.close(fig)

    fig = plt.figure()

    plt.plot(G_loss_plots)
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    # plt.ylim(-5, 15) # (-15, 15)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(['Generator (G_loss)'], loc='upper right', fontsize=14)
    plt.savefig('./output_files/' + save_fname + '/loss_plots/ccgan_G_Loss_alpha' + str(alpha) + '.png',
                bbox_inches='tight')
    plt.close(fig)


# 50-D marginal histograms (used in Synthetic dataset)
def plot_50genes(samples, real_samples,  scale=16.5):
    fig = plt.figure(figsize=(90, 40))
    gs = gridspec.GridSpec(1, 1)
    gs.update(wspace=0.5, hspace=0.5)
    edges = np.concatenate(([-10], np.arange(-0.5, 16, 0.5), [20]), axis=0) # [-1, 0:0.5:16, 20]

    # samples_LOC = samples.numpy()

    for i in range(50):
        plt.subplot(5, 10, i+1)
        plt.hist(real_samples[:, i], edges, weights=np.ones(len(real_samples[:, i])) / len(real_samples[:, i]), density=False, alpha=0.5, facecolor='b') # normed! as in matlab
        plt.title('gene = %i' % i) # plt.title('iter = %i' % it)
        # the histogram of the data
        plt.hist(samples[:, i], edges, weights=np.ones(len(samples[:, i])) / len(samples[:, i]), density=False, alpha=0.5, facecolor='g') # normed! as in matlab

        plt.xlim(-1.0, scale)
        plt.ylim(0, 1.0)

    return fig
