""" @author: gnleo """

import numpy as np


""" Fitness Schaffer f6(x,y) 
ref: https://www.cs.unm.edu/~neal.holts/dga/benchmarkFunction/schafferf6.html
"""
def fitness_schaffer_6(x, y):
    num = np.sin( np.sqrt( ( (x*x) + (y*y) ) ) ) ** 2 - 0.5
    den = 1 + ( 0.001 * ( (x*x) + (y*y) )) ** 2

    return (0.5 - (num / den))


""" Fitness Schaffer f6(x,y) deslocado """
def fitness_displacement_schaffer_6(x, y):
    num = np.sin( np.sqrt( ( (x*x) + (y*y) ) ) ) ** 2 - 0.5
    den = 1 + ( 0.001 * ( (x*x) + (y*y) )) ** 2

    return (999.5 - (num / den))


""" Fitness Schaffer f4(x,y) 
ref: https://www.sfu.ca/~ssurjano/schaffer4.html
"""
def fitness_schaffer_4(x, y):
    num = np.cos((np.sin(abs(x**2 - y**2)))) ** 2 - 0.5
    den = (1 + (0.001 * (x**2 + y**2) )) ** 2

    return (0.5 + (num / den))


""" Fitness OneMax f([x,x,x,x])
ref: https://subscription.packtpub.com/book/data/9781838557744/5/ch05lvl1sec28/the-onemax-problem
"""
def fitness_one_max(cromosso):
    soma = 0
    for i in range(len(cromosso)):
        soma = soma + cromosso[i]

    return soma


""" Fitness Cross-in-Tray Function f(x,y)
ref: https://www.sfu.ca/~ssurjano/crossit.html
"""
def fitness_cross_in_tray(x, y):
    fact1 = np.sin(x)*np.sin(y)
    fact2 = np.exp((abs(100 - np.sqrt(x**2+y**2)/np.pi)))

    return -0.0001 * ((abs(fact1*fact2) + 1) ** 0.1)


""" Fitness Ackley Function f(x,y)
ref: https://en.wikipedia.org/wiki/Ackley_function
"""
def fitness_ackley(x,y):
    CONST_A = 20
    CONST_B = 0.2
    CONST_C = 2 * np.pi

    square_root = np.sqrt(0.5 * (x**2 + y**2))
    exponential = np.exp(0.5 * (np.cos(CONST_C * x) + np.cos(CONST_C * y)) )

    return - CONST_A * np.exp( (-(CONST_B) * square_root)) - exponential + np.e + CONST_A

