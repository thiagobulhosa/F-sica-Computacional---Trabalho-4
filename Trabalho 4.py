#Thiago dos Santos
#Thiago Bulhosa
#Pablo Montel

from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
from runner_data import get_runner_data as grd
from scipy.optimize import minimize
from sympy import limit

def v_of_t(a0, b, t):
    tan = np.tanh(np.sqrt(a0) * np.sqrt(b) * t)
    return (tan / np.sqrt(b / a0))

def x_of_t(a0, b, t):
    return np.log(np.cosh(t * np.sqrt(a0 * b))) / b

def t_of_x(a0, b, x):
    tx = np.linspace(0, 100, 1000)
    xt = x_of_t(a0, b, tx)
    spl = interp1d(xt, tx, kind='cubic', assume_sorted=True)
    return spl(x)

def chi2(p, tobs, xobs):
    a0 = p[0]
    b = p[1]
    t_model = t_of_x(a0, b, xobs)
    dy2 = (tobs - t_model) ** 2
    return dy2.sum()

a0 = 8.5
p = 1.2
cd = 0.5
a=1
massa = 80
B = (1 / (2 * massa)) * p * cd * a
t = np.linspace(0, 10, 1000)
v = v_of_t(a0, B, t)
x = x_of_t(a0, B, t)

plt.figure(1)
plt.title("Velocidade v(t)")
plt.plot(t, v, '-b')
plt.figure(2)
plt.title("Posição x(t)")
plt.plot(t, x, '-r')
plt.show()

for i in range(3):
    corredor = grd()
    m = minimize(chi2, x0=[a0, B], args=(corredor[1], corredor[0]), method='Nelder-Mead', tol=1.e-8)
    print(f'Corredor {i + 1}')
    print(f'a0  = {m.x[0]:.2f} m/s2.')
    print(f'B = {m.x[1]:.2f}')

    t = np.linspace(0, 10, 1000)
    xmin = x_of_t(m.x[0], m.x[1], t)
    vmin = v_of_t(m.x[0],m.x[1],t)
    limite = limit(v,t,np.inf)
    print("Velocidade máxima do corredor:",vmin.max())
    print("Limite da velocidade:",max(limite))

    A = (m.x[1] * 2 * 80 / 1.2) * 0.5
    print("Seção Reta do corpo do atleta=",A)

    plt.figure(3)
    plt.plot(t, vmin)
    plt.figure(4)
    plt.plot(t, xmin,"-r")
    plt.show()