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


def diamond(x, y):
    r = (x**2+y**2)**(1/2)
    theta = np.arctan(x/y)
    return np.sin(theta)*np.cos(r), np.cos(theta)*np.sin(r)


def ex(x, y):
    r = (x**2+y**2)**(1/2)
    theta = np.arctan(x/y)
    p0, p1 = np.sin(theta+r), np.cos(theta-r)
    return r*(p0**3+p1**3), r*(p0**3-p1**3)


def julia(x, y):
    r = (x**2+y**2)**(1/2)
    theta = np.arctan(x/y)
    omega = np.random.rand()*np.pi
    return r**(1/2)*np.cos(theta/2+omega), r**(1/2)*np.sin(theta/2+omega)


def bent(x, y):
    if x >= 0 and y >= 0:
        return x, y
    elif x < 0 and y >= 0:
        return 2*x, y
    elif x >= 0 and y < 0:
        return x, y/2
    else:
        return 2*x, y/2


def fisheye(x, y):
    r = (x**2+y**2)**(1/2)
    return 2/(r+1)*y, 2/(r+1)*x


def power(x, y):
    r = (x**2+y**2)**(1/2)
    theta = np.arctan(x/y)
    return r**np.sin(theta)*np.cos(theta), r**np.sin(theta)*np.sin(theta)


def eyefish(x, y):
    r = (x**2+y**2)**(1/2)
    return 2/(r+1)*x, 2/(r+1)*y


def bubble(x, y):
    r = (x**2+y**2)**(1/2)
    return 4/(r**2+4)*x, 4/(r**2+4)*y


def cylinder(x, y):
    return np.sin(x), y


func_collection = [linear,
                   sinusoidal,
                   spherical,
                   # swirl,
                   polar,
                   hand_kerchief,
                   heart,
                   disc,
                   spiral,
                   hyperbolic,
                   diamond,
                   ex,
                   # julia,
                   bent,
                   fisheye,
                   power,
                   eyefish,
                   bubble,
                   cylinder]
