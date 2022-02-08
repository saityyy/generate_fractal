import numpy as np


def linear(x, y):
    return x, y


def sinusoidal(x, y):
    return np.sin(x), np.sin(y)


def spherical(x, y):
    r = (x**2+y**2)
    return x/r, y/r


def swirl(x, y):
    r = (x**2+y**2)
    return x*np.sin(r)-y*np.cos(r), x*np.cos(r)+y*np.sin(r)


def horseshoe(x, y):
    r = (x**2+y**2)
    return (x-y)*(x+y)/r, (2*x*y)/r  # overflow error


def polar(x, y):
    r = (x**2+y**2)**(1/2)
    theta = np.arctan(x/y)
    return theta/np.pi, r-1


def hand_kerchief(x, y):
    r = (x**2+y**2)**(1/2)
    theta = np.arctan(x/y)
    return r*np.sin(theta+r), r*np.cos(theta-r)


def heart(x, y):
    r = (x**2+y**2)**(1/2)
    theta = np.arctan(x/y)
    return r*np.sin(theta*r), -r*np.cos(theta*r)


def disc(x, y):
    r = (x**2+y**2)**(1/2)
    theta = np.arctan(x/y)
    return theta*np.sin(np.pi*r)/np.pi, theta*np.cos(np.pi*r)/np.pi


def spiral(x, y):
    r = (x**2+y**2)**(1/2)
    theta = np.arctan(x/y)
    return (np.cos(theta)+np.sin(r))/r, (np.sin(theta)-np.cos(r))/r


def hyperbolic(x, y):
    r = (x**2+y**2)**(1/2)
    theta = np.arctan(x/y)
    return np.sin(theta)/r, r*np.cos(theta)


func_collection = [linear,
                   sinusoidal,
                   spherical,
                   swirl,
                   polar,
                   hand_kerchief,
                   heart,
                   disc,
                   spiral,
                   hyperbolic]
