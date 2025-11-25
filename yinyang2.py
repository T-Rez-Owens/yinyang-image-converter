import numpy as np
import matplotlib.pyplot as plt

def yin_yang(R=1.0):
    t = np.linspace(0, 2*np.pi, 1000)

    # Outer circle
    x_outer = R*np.cos(t)
    y_outer = R*np.sin(t)

    # Inner swirl circles centers
    C1 = (0, +R/2)
    C2 = (0, -R/2)

    # Eyes radius
    re = R/6

    # The S-curve is made from TWO semicircles:
    # Lower half of the top inner circle
    t_top = np.linspace(np.pi, 2*np.pi, 500)
    x_top = (R/2)*np.cos(t_top)
    y_top = (R/2) + (R/2)*np.sin(t_top)

    # Upper half of the bottom inner circle
    t_bot = np.linspace(0, np.pi, 500)
    x_bot = (R/2)*np.cos(t_bot)
    y_bot = -(R/2) + (R/2)*np.sin(t_bot)

    # Eyes
    x_eye1 = re*np.cos(t)
    y_eye1 = -R/2 + re*np.sin(t)

    x_eye2 = re*np.cos(t)
    y_eye2 = +R/2 + re*np.sin(t)

    fig, ax = plt.subplots(figsize=(6,6))

    ax.plot(x_outer, y_outer, 'k', linewidth=2)
    ax.plot(x_top, y_top, 'k', linewidth=2)
    ax.plot(x_bot, y_bot, 'k', linewidth=2)
    ax.plot(x_eye1, y_eye1, 'k', linewidth=2)
    ax.plot(x_eye2, y_eye2, 'k', linewidth=2)

    ax.set_aspect('equal')
    ax.axis('off')
    plt.show()
# Run the yin-yang visualization
if __name__ == "__main__":  
    yin_yang(R=1.0)