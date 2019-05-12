#!/usr/bin/env python
# @Copyright 2018 Kristjan Haule 
from scipy import *
from pylab import *
from numpy.polynomial import legendre
from Frequency2Legendre import *
import samplewj as sw

def GetOmg():
    norder = 1
    Corder = 0
    fname = 'Pcof_'+str(norder)+'_corder_'+str(Corder)+'.omg'
    dat = loadtxt(fname).transpose()
    qx = dat[0]
    Pw = dat[1:]
    plot(qx, Pw[0], label='N='+str(norder) )
    
    for norder in range(3,6):
        for Corder in [0]+range(2,norder):
            fname = 'Pcof_'+str(norder)+'_corder_'+str(Corder)+'.omg'
            dat = loadtxt(fname).transpose()
            Pw += dat[1:]
        plot(qx, Pw[0], label='N='+str(norder) )
    legend(loc='best')
    show()
        
def get_cutoff_in_data(dr):
    cnf = dict()
    execfile(dr+'/params.py',cnf)
    pt = cnf['p']
    return pt['cutoffq']

def ChangeGrids(p0,qx0,qx,iOm):
    # from p0[qx0,iOm0]   to  _p0_[qx,iOm]
    #
    _p0_ = zeros((len(qx),len(iOm)))
    for iw in range(len(iOm)):
        fp0 = interpolate.UnivariateSpline(qx0, p0[:,iw], s=0)
        _p0_[:,iw] = fp0(qx)
    return _p0_

class  Fourier:
    def __init__(self, iOm, nom, tx, beta):
        self.nom = nom
        self.iOm = iOm
        # Inverse Fourier Transform
        self.iOm_rest = array(range(iOm[nom-1]+1, iOm[-1]+1))
        self.tx = tx
        self.beta = beta
        
        X1 = 2.*pi/beta* tensordot(iOm[1:nom], tx, axes=0)  # iOm * tau
        self.cs1 = cos(X1)
        X2 = 2.*pi/beta* tensordot(self.iOm_rest, tx, axes=0)    # iOm * tau
        self.cs2 = cos(X2)
        
    def InverseF(self, DVi, iq):
       sm0 = 1./self.beta * DVi[0].real  # iOm=0 (i.e., first point) is special
       sm1 = 2./self.beta * dot(DVi[1:self.nom].real, self.cs1)
       sm2 = zeros(len(self.tx))
       if iq>0: # at small q, there is only Om=0 contribution.
           fw = interpolate.UnivariateSpline( self.iOm[self.nom-1:], DVi[self.nom-1:].real, s=0 )
           Pqw_rest = fw(self.iOm_rest)
           sm2 = 2./self.beta * dot(Pqw_rest, self.cs2)
       sm = sm0+sm1+sm2
       return sm

if __name__ == '__main__':
    # Here we have calculation with the larer cutoff
    col=['b','g','r','c','m','y','k','w']

    
    execfile('params.py')
    for k in p.keys():
        print k, p[k]
        exec( k+' = '+str(p[k]) )
    kF = (9*pi/4.)**(1./3.) /rs
    n0 = 3/(4*pi*rs**3)
    Ex0 = -3./(2*pi)*(9*pi/4)**(1./3.) * 1/rs
    
    dr = 'data'  # BK data has to be in another directory
    MaxOrder=4
    print 'MaxOrder=', MaxOrder
    
    print '----'
    print 'rs=', rs
    print 'kF=', kF
    print 'T=', (1./beta)/(kF**2), 'EF'
    print 'cutoffk=', cutoffk
    print 'cutoffq=', cutoffq
    print 'lmbda=', lmbda
    print 'dmu=', dmu

    hf = sw.HartreeFock(kF, cutoffk, beta, lmbda, dmu)
    kx_, epsx_ = hf.get()
    kx_ *= kF
    epsx_ *= kF**2
    #Ex = hf.Exchange()
    #print 'Ex=', Ex
    Ex = hf.Exchange2()  # precise exchange at finite temperature
    print 'Ex(T)=', Ex, 'bu Ex0=', Ex0

    # Matsubara mesh needs to be compatible with tpvertex.py. Make sure this is the case!
    iOm = zeros(nom+ntail,dtype=intc)
    iOm[:nom] = range(nom)
    iOm[nom:]= logspace(log10(nom+1),log10(20*nom),ntail)

    # Loading data from tpvertex.py
    p2 = load('p2.dat.npy')   # Baym Kadanoff Polarization
    qx0 = load('qx.dat.npy')  # q-mesh
    p0 = load('wp0.dat.npy')  # RPA polarization
    print 'shape(p0)=', shape(p0)
    print 'shape(qx)=', shape(qx0)
    print 'shape(p2)=', shape(p2)
    
    print 'cutoffq is =', cutoffq/kF    
    cutoffq = get_cutoff_in_data(dr)
    print 'cutoffq is now=', cutoffq/kF
    
    Nq=200
    qx = linspace(1e-3,cutoffq,Nq)
    # We could have just a single time point, but it does not hurt to have a few....
    Nt=10
    tx = linspace(0,beta,Nt)
    
    CheckP0=False
    if CheckP0:
        Omg = 2*iOm*pi/beta
        hf = sw.HartreeFock(kF, cutoffk, beta, lmbda, dmu)
        kx_, epsx_ = hf.get()
        kx_ *= kF
        epsx_ *= kF**2
        epsf = interpolate.CubicSpline(kx_,epsx_)
        e_q = epsf(qx0)
        
        p0t = sw.InverseFourierBoson_new(beta, tx, nom, iOm, p0, e_q)

        for it in range(0,len(tx),2):
            plot(qx0/kF, p0t[:,it])
        show()
        
        p2t = sw.InverseFourierBoson_new(beta, tx, nom, iOm, p2, e_q)
        
        for it in range(0,len(tx),2):
            plot(qx0/kF, p2t[:,it])
        show()

    # First we will do firts order BK, which is obtained analytically in this directory
    Vq = 8*pi/qx0**2
    VP = zeros( shape(p2) )
    VP0 = zeros( shape(p0) )
    for im in range(shape(p2)[1]):
        VP[:,im] = Vq[:] * p2[:,im].real
        VP0[:,im] = Vq[:] * p0[:,im].real
    DV = VP/(1.0-VP) - VP0
    
    for im in range(shape(p2)[1]):
        DV[:,im]  *= qx0**2
        VP0[:,im] *= qx0**2
    


    if False:
        for im in range(0,nom,2):
            plot(qx0/kF, DV[:,im])
    
    
    four = Fourier(iOm, nom, tx, beta)

    DVt = zeros((len(qx0),len(tx)))
    for iq in range(len(qx0)):
        DVt[iq,:] = four.InverseF(DV[iq,:], iq)
    
    fDVt = interpolate.UnivariateSpline(qx0, DVt[:,0], s=0)
    Intg = fDVt.integral(0,qx0[-1])
    Intg_rest = fDVt.integral(qx[-1],qx0[-1])  # difference between 3*kF and 20*kF integral
    #Intc = fDVt.integral(10*kF, qx0[-1])
    #print 'Intc/Intg=', Intc/Intg
    Ec = -1/(2*pi)**2 * fDVt.integral(0,qx0[-1])
    
    print 'Ec[N=1]=', Ec/n0, 'E_potential[N=1]=', Ex+Ec/n0, 'Intg=', Intg, 'n0=', n0

    plot(qx0/kF, DVt[:,0], 'o-', label='N=1')
    
    norder = 1
    Corder = 0
    fname = dr+'/Pcof_'+str(norder)+'_corder_'+str(Corder)+'.txt'
    PQ = loadtxt(fname)
    Nlt, Nlq = shape(PQ)
    
    print 'shape0=', shape(PQ)
    for norder in range(3,MaxOrder+1):
        for Corder in [0]+range(2,norder):
            fname = 'Pcof_'+str(norder)+'_corder_'+str(Corder)+'.txt'
            print 'fn=', fname
            PQt = loadtxt(fname)
            PQ += PQt[:,:Nlq]

    if CheckP0:
        Pqt = legendre.leggrid2d( 2*tx/beta-1., 2*qx/qx[-1]-1., PQ)
        for it in range(0,len(tx),2):
            plot(qx/kF, Pqt[it,:])
        show()
    
    Pqlt = zeros( (Nlt,len(qx)) )
    for lt in range(Nlt):
        Pqlt[lt,:] = legendre.legval( 2*qx/qx[-1]-1., PQ[lt,:])
    
    lmax_t = Nlt-1
    
    Ker = Get_P2Om_Transform(lmax_t, iOm, beta) # Transforms from Legendre Basis to Matsubara Frequency basis.
    Pqw = dot(Ker, Pqlt)  # Pqw[iOm,iq] = \sum_lt Ker_even[iOm,lt]*Pqlt[lt,iq]
    ####
    # Interpolate bubble on the new qx mesh with smaller cutoff
    _p0_ = ChangeGrids(p0,qx0,qx,iOm)

    if False:
        for im in range(0,nom,2):
            plot(qx/kF, Pqw[im,:].real, '.')
            plot(qx0/kF, p2[:,im].real, '-')
        show()
        sys.exit(0)
    
    Vq = 8*pi/qx**2
    VP = zeros( (len(qx), len(iOm)) )
    VP0 = zeros( (len(qx), len(iOm)) )
    for im in range(len(iOm)):
        VP[:,im] = Vq[:] * Pqw[im,:].real
        VP0[:,im] = Vq[:] * _p0_[:,im].real
        
    DV = VP/(1.0-VP) - VP0
    
    for im in range(shape(p2)[1]):
        DV[:,im]  *= qx**2
        VP0[:,im] *= qx**2

    if False:
        for iq in range(0,len(qx),3):
            plot(qx/kF, DV[:,iq], 'o-')
        show()
        sys.exit(0)
    if False:
        for im in range(0,nom,2):
            plot(qx/kF, DV[:,im], '.')
    
    
    DVt = zeros((len(qx),len(tx)))
    for iq in range(len(qx)):
        DVt[iq,:] = four.InverseF(DV[iq,:], iq)

    fDVt = interpolate.UnivariateSpline(qx, DVt[:,0], s=0)
    #Intg = fDVt.integral(0,qx[-1])
    Intg = fDVt.integral(0,2.5*kF)
    #Ec = -1/(2*pi)**2 * (Intg+Intg_rest)
    Ec = -1/(2*pi)**2 * (Intg)
    print 'Ecfinal[N='+str(MaxOrder)+']= ', Ec/n0, 'E_potential[N='+str(MaxOrder)+']=', Ec/n0+Ex, 'Intg=', Intg

    plot(qx/kF, DVt[:,0],'s-', label='N='+str(MaxOrder))
    legend(loc='best')
    grid()
    show()
