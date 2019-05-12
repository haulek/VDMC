#!/usr/bin/env python
# @Copyright 2018 Kristjan Haule 
import os
from scipy import *
from pylab import *

def Run0(QMC_filename,beta,Debug=False):

    kxb = load('kxb.npy')
    kx = array([0.5*(kxb[i+1]+kxb[i]) for i in range(len(kxb)-1)]) # from binning mesh to vertex mesh
    dh_k = array([kxb[i+1]-kxb[i] for i in range(len(kxb)-1)])     # integrating dk; now kx and dh_k is exactly compatible with QMC binning
    
    Rc00 = load('Rc.00.npy')
    Nkbin3, Nlth = shape(Rc00)
    Nlth-=1

    p0 = load('p0.00.npy')
    print '#', 1, 0, p0[0]
    #result =  '1 '+ str( p0[0] ) + '\n'
    start_order, MaxOrder = 3,5
    #NCases = [(norder, Corder) for norder in range(start_order,MaxOrder+1) for Corder in [0]+range(2,norder)]
    NCases = [(norder, Corder) for norder in range(start_order,MaxOrder+1) for Corder in range(norder)]
    
    PQw = p0[0] # first order value
    res = {1 : p0[0]}
    for norder in range(start_order,MaxOrder+1):
        for Corder in range(norder): #[0]+range(2,norder):
            filename = QMC_filename[:-14]+str(norder)+'_corder_'+str(Corder)+'.npy'
            if not os.path.isfile(filename):
                print >> sys.stderr, 'File', filename, 'does not exists. sciping....'
                continue
            
            QMC = load(filename)
            Nthbin, Nkbin, Nthbin2, Nkbin2 = shape(QMC) # theta-binnibg, k-binning
            if Nthbin!=Nthbin2:
                print 'WARNING : It seems Nthbin != Nthbin2'
            if Nkbin!=Nkbin2:
                print 'WARNING : It seems Nkbin != Nkbin2'
            
            dth = 2.0/Nthbin                                               # dcos(theta)
            dth2 = 2.0/Nthbin2
            
            # QMC data can be summed over k and cos(theta) to obtain usual polarization (function of Q and t only).
            #QMC_PQ = dot(sum(QMC,axis=0), dh_k)*dth
            
            if (len(kx) != Nkbin or Nkbin!=Nkbin3):
                print 'ERROR : Number of k-bins from QMC and tpvertex.py does not match len(kx)=', len(kx), 'Nkbin=', Nkbin, 'Nkbin2=', Nkbin2
                
            QMCs = sum(QMC,axis=(0,2))
            gm = (Rc00[:,0].real + 1.0)*dh_k[:]*sqrt(dth*dth2)
            
            dPQ = dot(gm, dot(QMCs[:,:],gm)) * beta
            PQw += dPQ
            print '#', norder, Corder, dPQ
        #result += str(norder) + ' ' + str(PQw) + '\n'
        res[norder] = PQw
    print '# ', res
    print lmbda,
    for n in res.keys():
        print -res[n],
    print

if __name__ == '__main__':

    if len(sys.argv)<2:
        print 'Give input filename of Pcof_?_corder_?.npy'
        sys.exit(1)
    
    QMC_filename = sys.argv[1]
    execfile('params2.py')
    beta = p['beta']
    lmbda = p['lmbda']
    Run0(QMC_filename,beta,Debug=True)
