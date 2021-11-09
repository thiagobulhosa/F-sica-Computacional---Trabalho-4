import numpy as np

def get_runner_data ():
    
    runner = input ('Enter runner name: bolt, gay, powell: ')

    if runner == 'bolt':
        t = np.loadtxt ('bolt.txt', unpack=True)
        t_reaction = 0.146
    elif runner == 'gay':
        t = np.loadtxt ('gay.txt', unpack=True)
        t_reaction = 0.144
    elif runner == 'powell':
        t = np.loadtxt ('powell.txt', unpack=True)
        t_reaction = 0.134
    else:
        exit ('You must choose a runner. Try again.')

    x = np.linspace (10.0, 100.0, 10)

    return x, t, t_reaction
