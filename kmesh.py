# @Copyright 2018 Kristjan Haule 
from tanmesh import *

def Give_k_mesh(Nk,kF,cutoff,k0=1e-10):
    x0 = 2*kF/(6.*Nk/2)
    #x0 = 2*kF/(4.*Nk/2)
    om = GiveTanMesh(x0, kF, Nk/3) #[1:]
    om = hstack( (-om[::-1],om[1:]) ) + kF
    dh = om[-2]-om[-3]
    L = cutoff-om[-2]
    Nr = int(L/dh+1)
    dx = (L-1e-7)/(Nr-1.)
    kx = zeros(len(om)+Nr-2)
    kx[:len(om)] = om[:]
    kx[0]=k0
    kx[len(om)-1:] = dx*arange(1,Nr) + om[-2]
    return kx
