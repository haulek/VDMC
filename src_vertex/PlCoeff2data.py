#!/usr/bin/env python
import os
from scipy import *
from pylab import *
from numpy.polynomial import legendre
from Frequency2Legendre import *
# @Copyright 2018 Kristjan Haule and Kun Chen    

if __name__ == '__main__':
    col=['b','g','r','c','m','y','k','w']
    if len(sys.argv)<2:
        print 'Give input filename of Pcof_?_corder_?.txt'
        sys.exit(1)
    
    QMC_filename = sys.argv[1]
    execfile('params.py')
    for k in p.keys():
        #print k, p[k]
        exec( k+' = '+str(p[k]) )
    kF = (9*pi/4.)**(1./3.) /rs
    print '----'
    print 'rs=', rs
    print 'kF=', kF
    print 'T=', (1./beta)/(kF**2), 'EF'
    print 'cutoffk=', cutoffk
    print 'cutoffq=', cutoffq
    print 'lmbda=', lmbda
    print 'dmu=', dmu
    
    Nt=50
    tx = linspace(0,beta,Nt)
    print 'Reading ', QMC_filename
    PQ_tot = loadtxt(QMC_filename)
    
    if os.path.exists('qxb.npy'): # this is discrete run, hence no q-interpolation needed
        qx = load('qxb.npy')
        Pq_tot = zeros( (len(tx), len(qx)) )
        for iq in range(len(qx)):
            Pq_tot[:,iq] = legendre.legval( 2*tx/beta-1., PQ_tot[iq,:])
        
        Pqlt = transpose(PQ_tot)
    else:
        #Nq=100
        qx = linspace(1e-3,cutoffq,p['Nq'])
        Pq_tot = legendre.leggrid2d( 2*tx/beta-1., 2*qx/cutoffq-1., PQ_tot)
    
        lmax_t = shape(PQ_tot)[0]-1
        Pqlt = zeros( (lmax_t+1,len(qx)) )
        for lt in range(lmax_t+1):
            Pqlt[lt,:] = legendre.legval( 2*qx/cutoffq-1., PQ_tot[lt,:])
            
    # saving data
    data = zeros( (shape(Pq_tot)[0]+1, len(qx)) )
    data[0,:] = qx[:]/kF
    for it in range(len(tx)):
        data[it+1,:] = Pq_tot[it,:]

    root, ext = os.path.splitext(QMC_filename)
    savetxt(root+'.tau', data.transpose() )

    iOm = zeros(p['nom']+p['ntail'],dtype=intc)
    iOm[:nom] = range(p['nom'])
    iOm[nom:]= logspace(log10(p['nom']+1),log10(20*p['nom']),p['ntail'])
    Ker = Get_P2Om_Transform(lmax_t, iOm, beta) # Transforms from Legendre Basis to Matsubara Frequency basis.
        
    pq = dot(Ker, Pqlt)  # pq[iOm,iq] = \sum_lt Ker_even[iOm,lt]*Pqlt[lt,iq]
    # saving data
    data = zeros( (len(iOm)+1, len(qx)) )
    data[0,:] = qx[:]/kF
    for im in range(len(iOm)):
        data[im+1,:] = pq[im,:].real
    savetxt(root+'.omg', data.transpose() )
