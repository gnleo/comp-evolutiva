import os

PATH = os.getcwd() + '/02/1/12/'

for i in range(10):
    os.makedirs(PATH + 'loop_{}'.format(i))