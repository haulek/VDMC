#!/usr/bin/env python
# @Copyright 2018 Kristjan Haule 
import os
from scipy import *
from pylab import *

def Run0(beta,Debug=False):

    kxb = load('kxb.npy')
    kx = array([0.5*(kxb[i+1]+kxb[i]) for i in range(len(kxb)-1)]) # from binning mesh to vertex mesh
    dh_k = array([kxb[i+1]-kxb[i] for i in range(len(kxb)-1)])     # integrating dk; now kx and dh_k is exactly compatible with QMC binning
    
    Rc00 = load('Rc.00.npy')
    Nkbin3, Nlth = shape(Rc00)
    Nlth-=1
    gam = Rc00[:,0].real+1.0  # only l=0 contributes at q=0

    f1 = open('order_BKA_1.dat', 'a')
    f2 = open('order_BKA_2.dat', 'a')
    
    
    p0 = load('p0.00.npy')
    print >> f1, '#', 1, 0, p0[0]
    print >> f2, '#', 1, 0, p0[0]
    NCases = [(norder, Corder) for norder in range(start_order,MaxOrder+1) for Corder in range(norder)]
    
    PQw1 = p0[0] # first order value
    PQw2 = p0[0]
    res1 = {1 : p0[0]}
    res2 = {1 : p0[0]}
    for norder in range(start_order,MaxOrder+1):
        for Corder in range(norder):
            fname2 = 'Pcof2_lmbda_'+str(lmbda)+'_'+str(norder)+'_corder_'+str(Corder)+'.npy'
            if not os.path.isfile(fname2):
                print >> sys.stderr, 'File', fname2, 'does not exists. sciping....'
                continue
            QMC2 = load(fname2)
            fname1 = 'Pcof1_lmbda_'+str(lmbda)+'_'+str(norder)+'_corder_'+str(Corder)+'.npy'
            if not os.path.isfile(fname1):
                print >> sys.stderr, 'File', fname1, 'does not exists. sciping....'
                continue
            QMC1 = load(fname1)

            qmc1 = QMC1[0,:,0,0]
            qmc2 = QMC2[0,:,0,:,0,0]

            V1 = qmc1 + sum(qmc2,axis=1)  # all diagrams which need vertex from the right
            V2 = qmc2                     # diagrams need vertex from the left and the right
            
            dPQ1 = dot(V1,gam) * beta
            dPQ2 = dot(gam, dot(V2,gam) ) * beta
            
            PQw1 += dPQ1
            PQw2 += dPQ2
            
            print >> f1, '#', norder, Corder, dPQ1
            print >> f2, '#', norder, Corder, dPQ2
        #result += str(norder) + ' ' + str(PQw) + '\n'
        res1[norder]=PQw1
        res2[norder]=PQw2
    #print result,
    print >> f1, '# ', res1
    print >> f1, lmbda,
    for n in res1.keys():
        print >> f1, -res1[n],
    print >> f1
    print >> f2, '# ', res2
    print >> f2, lmbda,
    for n in res2.keys():
        print >> f2, -res2[n],
    print >> f2
    f1.close()
    f2.close()
    
if __name__ == '__main__':

    if len(sys.argv)<2:
        print 'Give input filename of Pcof_?_corder_?.npy'
        sys.exit(1)
    
    lmbda = sys.argv[1]
    execfile('params2.py')
    beta = p['beta']
    #lmbda = p['lmbda']
    start_order, MaxOrder = 3, p['MaxOrder']
    Run0(beta,Debug=True)
