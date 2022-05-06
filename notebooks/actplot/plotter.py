import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

def plot_activation(x, y, text_x, text_y, formula_string):
    fig = plt.figure(figsize=(7, 3), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])

    # 5pts 2 spaces 1 pt 2 spaces
    ax.grid(True, color='0.6', dashes=(5, 2, 1, 2))

    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_visible(False)
    ax.plot((1), (0), ls="", marker=">", ms=8,
              color="k", transform=ax.get_yaxis_transform(), clip_on=False)
    ax.plot((0), (1), ls="", marker="^", ms=8,
              color="k", transform=ax.get_xaxis_transform(), clip_on=False)

    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))

    ax.plot(x, y, linewidth=2)

    ax.text(text_x, text_y,
              formula_string, size='xx-large',
              bbox=dict(boxstyle='round', facecolor='wheat'))
    
    plt.show()

def plot_tanh():
    x = np.linspace(-2.5, 2.5, 1000)
    tanh = ( 2 / (1 + np.exp(-2*x) ) ) -1

    plot_activation(
        x, tanh,
        -2.5, 0.8,
        r'$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$'
    )

def plot_sigmoid():
    x = np.linspace(-6, 6, 1000)
    sigmoid = 1 / (1 + np.exp(-x) )

    plot_activation(
        x, sigmoid,
        -6, 0.8,
        r'$\sigma(x) = \frac{1}{1 + e^{-x}}$'
    )

def plot_relu():
    x = np.linspace(-2, 2, 1000)
    relu = np.maximum(0, x)

    plot_activation(
        x, relu,
        -2, 1.7,
        r'$f(x) = \max(0, x)$'
    )

def plot_leakyrelu():
    x = np.linspace(-2, 2, 1000)
    leaky_relu = np.maximum(0.1 * x, x)

    plot_activation(
        x, leaky_relu,
        -2, 1.7,
        r'$f(x) = \max(0.1x, x)$'
    )
