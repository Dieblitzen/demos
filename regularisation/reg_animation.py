## AUTHOR: Samar Khanna, 2021

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from scipy.special import legendre


def linear_h(w, x):
    return (w @ x.T).T

def grad_linear_h(w, x):
    return x

def mse_loss_f(h, w, x, y):
    # loss = (1/y.shape[0]) * np.sum((h(w, x) - y) ** 2)
    loss = np.sum((h(w, x) - y) ** 2)
    return loss

def grad_mse_loss_f(h, grad_h, w, x, y):
    # grad_loss = (1/y.shape[0]) * np.sum(2*(h(w, x) - y) * grad_h(w, x), axis=0).reshape(1, -1)
    grad_loss = np.sum(2*(h(w, x) - y) * grad_h(w, x), axis=0).reshape(1, -1)
    return grad_loss

def l2_reg(w):
    return np.linalg.norm(w[..., :-1]) ** 2

def grad_l2_reg(w):
    grad = 2 * w
    grad[..., -1] = 0
    return grad

def l1_reg(w):
    return np.sum(np.abs(w[..., :-1]))

def grad_l1_reg(w):
    grad = np.sign(w)
    grad[..., -1] = 0
    return grad

def linf_reg(w):
    return np.max(np.abs(w[..., :-1]))

def grad_linf_reg(w):
    grad = np.zeros_like(w)
    grad[..., np.argmax(w[..., :-1])] = 1
    return grad

def grad_descent(steps, loss_f, grad_loss_f, h, grad_h, x, y, 
                 lr=0.005, reg_c=0.01, reg_f=None, grad_reg_f=None):
    xy = np.hstack([x, y])
    
    w = np.zeros((1, x.shape[1]))  # size 1 by d
    thetas = [w]
    losses = [loss_f(h, w, x, y)]
    for e in range(steps):
        # Shuffle data
        np.random.shuffle(xy)
        _x, _y = xy[:, :-1], xy[:, -1:]

        reg, grad_reg = 0, 0
        if reg_f is not None and grad_reg_f is not None:
            reg = reg_f(w)
            grad_reg = grad_reg_f(w)

        grad = grad_loss_f(linear_h, grad_linear_h, w, _x, _y) + reg_c * grad_reg

        # Stairway to hell
        w = w - lr * grad

        loss = loss_f(h, w, _x, _y) + reg_c * reg

        thetas.append(w)
        losses.append(loss)
    
    return thetas, losses


def featurize(x):
    x = x.reshape(-1, 1)
    return np.hstack([
        x,
        sum([p*(x**(2-i)) for i, p in enumerate(legendre(2))]),
        # sum([p*(x**(3-i)) for i, p in enumerate(legendre(3))]),
        sum([p*(x**(4-i)) for i, p in enumerate(legendre(4))]),
        sum([p*(x**(5-i)) for i, p in enumerate(legendre(5))]),
        sum([p*(x**(6-i)) for i, p in enumerate(legendre(6))]),
        sum([p*(x**(7-i)) for i, p in enumerate(legendre(7))]),
        sum([p*(x**(8-i)) for i, p in enumerate(legendre(8))]),
        # -(x+0.05)**3, 
        np.cos(x), 
        np.ones((x.shape[0], 1))
    ])


def plot_animation(x_data, y_data, feature_f, h, thetas_and_losses,
                   num_frames=500, x_support=(-0.6, 0.6), y_support=(-0.2, 0.25)):

    xmin, xmax = x_support
    ymin, ymax = y_support

    num_plots = len(thetas_and_losses)
    d = thetas_and_losses[0][0][0].shape[1]
    fig, axes = plt.subplots(2, num_plots, figsize=(14, 6), gridspec_kw={'height_ratios': [3, 1]})

    axes[0, 0].set_title("No Regularization")
    axes[0, 1].set_title("L1 Regularization")
    axes[0, 2].set_title("L∞ Regularization")

    for i in range(num_plots):
        axes[0, i].set_xlim(xmin, xmax)
        axes[0, i].set_ylim(ymin, ymax)
    
    for i in range(num_plots):
        axes[1, i].set_yticks([])
        axes[1, i].set_ylabel(f"w{i+1}")
        axes[1, i].set_xticks(list(range(thetas1[0].shape[1])))
        axes[1, i].set_xticklabels(
            ['x', 'x^2', 'x^4', 'x^5', 'x^6', 'x^7', 'x^8', 'cos(x)', 'bias'],
            fontsize='x-small'
        )

    scats = []
    lines = []
    loss_texts = []
    weight_vizs = []
    weight_texts = []  # List of lists
    for i in range(num_plots):
        # Scatter plots for the data points
        scat = axes[0, i].scatter([], [], c='r', marker='x')
        scats.append(scat)

        # Line plots for the predicted curves
        line, = axes[0, i].plot([], [], 'b', lw=1)
        lines.append(line)

        # Text for displaying loss
        loss_text = axes[0, i].text((xmin+xmax)/2, 0.85*ymax, "", ha='center')
        loss_texts.append(loss_text)

        # The image plot of the (1, d) weight vector
        weight_viz = axes[1, i].imshow(np.zeros((1, d)), vmin=-1, vmax=1, cmap='RdYlGn')
        weight_vizs.append(weight_viz)

        # The text labels for each weight vector value
        w_texts = [axes[1, i].text(j, 0, "", ha='center', fontsize='small') for j in range(d)]
        weight_texts.append(w_texts)


    # These are the regression data points that the function should fit
    scatter_data = np.hstack([x_data.reshape(-1, 1), y_data.reshape(-1, 1)])

    # These are the input points fed to the function as the weights change
    pred_line_x = np.linspace(-0.5, 0.5, num=50)
    pred_line_feats = feature_f(pred_line_x)

    # These are the indices of the thetas to use for the animation (favouring those that show change)
    def get_significant_ani_inds(losses, total_num, pct_diff=0.05):
        ani_indices = [0]
        for curr_ind in range(1, len(losses)):
            prev_ind = ani_indices[-1]
            if np.abs(1 - losses[curr_ind]/losses[prev_ind]) >= pct_diff:
                ani_indices.append(curr_ind)

        ani_indices = np.array(ani_indices)
        if len(ani_indices) < total_num:
            remaining = [i for i in range(len(losses)) if i not in set(ani_indices)]
            prob_dist = np.array([-(x - len(remaining)) for x in range(len(remaining))])
            chosen = np.random.choice(
                remaining, total_num - len(ani_indices), replace=False, #p=prob_dist/prob_dist.sum()
            )

            ani_indices = np.concatenate((ani_indices, chosen))
            ani_indices.sort()

        return ani_indices
    
    combined_losses = sum([np.array(losses) for theta, losses in thetas_and_losses])
    ani_indices = get_significant_ani_inds(combined_losses, num_frames, pct_diff=0.01)
    
    def init():
        # line1.set_data([], [])
        # line2.set_data([], [])
        # scat1.set_offsets(scatter_data)
        # scat2.set_offsets(scatter_data)
        # return [scat1, scat2, line1, line2,]
        return []  # Don't really think I need this function

    def animate(i):
        # global ani_indices
        if i >= num_frames:
            return []

        ind = ani_indices[i]

        artists = []
        for j in range(num_plots):
            thetas, losses = thetas_and_losses[j]

            w = thetas[ind]
            loss = losses[ind]

            # Regression outputs using current weights
            pred = np.array(h(w, pred_line_feats).reshape(-1))

            scat = scats[j]
            scat.set_offsets(scatter_data)

            line = lines[j]
            line.set_data(pred_line_x, pred)

            loss_text = loss_texts[j]
            loss_text.set_text("Loss value: {:.4f}".format(loss))

            w_norm = w/np.linalg.norm(w[:, :-1])
            w_norm[:, -1] = w[:, -1]  # Don't want to normalize bias

            w_viz = weight_vizs[j]
            w_viz.set_data(w_norm)

            w_texts = weight_texts[j]
            for w_text, w_val in zip(w_texts, w_norm.reshape(-1)):
                w_text.set_text("{:.2f}".format(w_val))
            
            artists.extend([scat, line, loss_text, w_viz])
            artists.extend(w_texts)
        
        return artists

    ani = animation.FuncAnimation(fig, animate, num_frames+20, init_func=init, interval=50, blit=True)
    plt.show()
    # ani.save("animation6.gif", writer='imagemagick', fps=30)
    ani.save("animation6.mp4", writer='ffmpeg', fps=30)
    return None



if __name__ == "__main__":
    # truth_f = lambda x: (x + 0.2) * (x - 0.1) * (x - 0.25)
    # truth_f = lambda x: 0.5 * (x + 0.25) * (x - 0.25)
    truth_f = lambda x: 2 * x

    # X and Y data for regression
    x_data = np.linspace(-0.5, 0.5, num=10) 
    y_data = np.array([truth_f(x) for x in x_data])

    # Add some noise to regression labels
    num_tweak = len(y_data)
    y_data[np.random.choice(y_data.shape[0], num_tweak, replace=False)] += np.random.normal(scale=0.5, size=num_tweak)
    # y_data[4] -= 0.5
    # y_data[-10] += 0.5

    X = featurize(x_data)  # shape (n, d) (includes dummy 1 for bias)
    Y = y_data.reshape(-1, 1)  # shape (n, 1)
    XY = np.hstack([X, Y])  # shape (n, d + 1)

    thetas1, losses1 = grad_descent(
        10000, mse_loss_f, grad_mse_loss_f, linear_h, grad_linear_h,
        X, Y, lr=0.002, #reg_c=0.5, reg_f=linf_reg, grad_reg_f=grad_linf_reg
    )

    thetas2, losses2 = grad_descent(
        10000, mse_loss_f, grad_mse_loss_f, linear_h, grad_linear_h,
        X, Y, lr=0.002, reg_c=0.75, reg_f=l1_reg, grad_reg_f=grad_l1_reg
    )

    thetas3, losses3 = grad_descent(
        10000, mse_loss_f, grad_mse_loss_f, linear_h, grad_linear_h,
        X, Y, lr=0.002, reg_c=2.0, reg_f=linf_reg, grad_reg_f=grad_linf_reg
    )


    # Initial plot of data
    plt.scatter(x_data, y_data, c='r', marker='x')
    plt.title('Plot of regression points')
    plt.ylabel('y')
    plt.xlabel('x')

    # Animation plot
    plot_animation(
        x_data, y_data,
        featurize, linear_h, 
        [(thetas1, losses1), (thetas2, losses2), (thetas3, losses3)],
        num_frames=300,
        x_support=(-0.6, 0.6),
        # y_support=(-0.2, 0.25),
        y_support=(-2, 2)
    )
    plt.show()
