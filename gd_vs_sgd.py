import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import random

style.use('fivethirtyeight')

fig = plt.figure(figsize = (12,8))
fig.tight_layout()
plt.subplots_adjust(hspace = 0.4)
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)

ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

data = pd.read_csv('./lr.csv')
x = np.array(data.X)
y = np.array(data.Y)

a0 = 6
b0 = 100

a1 = 6
b1 = 100

J_gd = []
J_sgd = []

#Gradient Descent
def gd(i):
    global a0
    global b0

    learning_rate = 0.000001

    J = ((a0 * x + b0) - y)**2
    J_gd.append(J.sum())

    if(len(J_gd) > 5 and i < 6):
        J_gd.pop(0)

    if(len(J_gd) > 50):
        J_gd.pop(0)

    ax3.clear()
    ax3.set_title("Loss of GD")
    ax3.plot(np.arange(i-len(J_gd)+1 ,i+1), np.array(J_gd),label = "{:.2f}".format(J.sum()))
    ax3.legend()
    
    ax1.clear()
    ax1.set_xlim(-20,120)
    ax1.set_ylim(-20,200)
    ax1.set_title("Gradient Descent")
    ax1.plot(x, y, 'ro')
    ax1.plot(np.arange(-10, 120), a0 * np.arange(-10, 120) + b0, 'b',linewidth = 3)

    y_ = a0 * x + b0
        
    J_a = 2 * (y_ - y) * x
    J_b = 2 * (y_ - y)
        
    a0 = a0 - J_a.sum() * learning_rate
    b0 = b0 - J_b.sum() * learning_rate * 500

#Stochastic Gradient Descent
def sgd(i):
    
    global a1
    global b1
    
    learning_rate = 0.0001
    
    J = ((a1 * x + b1) - y)**2
    J_sgd.append(J.sum())
    
    if(len(J_sgd) > 5 and i < 6):
        J_sgd.pop(0)
    
    if(len(J_sgd) > 50):
        J_sgd.pop(0)
    
    ax4.clear()
    ax4.set_title("Loss of SGD")
    ax4.plot(np.arange(i-len(J_sgd)+1 ,i+1), np.array(J_sgd),label = "{:.2f}".format(J.sum()))
    ax4.legend()
    
    ax2.clear()
    ax2.set_xlim(-20,120)
    ax2.set_ylim(-20,200)
    ax2.set_title("Stochastic Gradient Descent")
    ax2.plot(x, y, 'ro')
    ax2.plot(np.arange(-10, 120), a1 * np.arange(-10, 120) + b1, 'b',linewidth = 3)
        
    x0 = random.choice(x)
    index = np.where(x == x0)
        
    y_ = a1 * x0 + b1
        
    J_a = 2 * (y_ - y[index]) * x0
    J_b = 2 * (y_ - y[index])
        
    a1 = a1 - J_a * learning_rate
    b1 = b1 - J_b * learning_rate * 500
    
anim1 = animation.FuncAnimation(fig, gd, interval=0)
anim2 = animation.FuncAnimation(fig, sgd, interval=0)
plt.show()
