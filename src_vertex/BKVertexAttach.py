#!/usr/bin/env python
# @Copyright 2018 Kristjan Haule 
import os
from scipy import *
from pylab import *
from numpy.polynomial import legendre
import tpvert as tpv
from Frequency2Legendre import *
import samplewj as sw
import glob


def Run(QMC_filename,beta):
    print 'Running on', QMC_filename
    
    kxb = load('kxb.npy')
    kx = array([0.5*(kxb[i+1]+kxb[i]) for i in range(len(kxb)-1)]) # from binning mesh to vertex mesh
    dh_k = array([kxb[i+1]-kxb[i] for i in range(len(kxb)-1)])     # integrating dk; now kx and dh_k is exactly compatible with QMC binning
    
    QMC = load(QMC_filename)
    Nthbin, Nkbin = shape(QMC)[:2]
    
    X = -1 + 2*(0.5+array(range(Nthbin)))/Nthbin                   # QMC cos(theta) points
    dth = 2.0/Nthbin                                               # dcos(theta)

    # QMC data can be summed over k and cos(theta) to obtain usual polarization (function of Q and t only).
    QMC_PQ = tensordot(sum(QMC,axis=0), dh_k, axes=([0],[0]))*dth # equivalent to : \sum_ik (\sum_ith QMC[ith,ik, lt1,lq1])*dk[ik] * dth
    
    DISCRETE=False
    if os.path.isfile('qxb.npy'):
        qx = load('qxb.npy')
        DISCRETE=True
        
        root, ext = os.path.splitext(QMC_filename)
        
        p2 = load('p2.npy') # p2[iq,iw] -- first order
        
        Nq, Nlt = shape(QMC)[2:] # theta-binnibg, k-binning, maxl_time, maxl_q
        Nlt-=1 # array size is lmax+1
        
        iOm = load('iOm.dat.npy')
        iOm = array(iOm,dtype=intc)
        
        Ker = Get_P2Om_Transform(Nlt, iOm, beta) # Transforms from Legendre Basis to Matsubara Frequency basis.
        Ker = transpose(Ker) # now we have Ker[lt,iOm]

        # Evaluates Legendre Polynomials
        Rc = load('Rc.0.npy') # Rc[iw,ik,ltheta]

        print 'shape(QMC)=', shape(QMC), 'shape(Rc)=', shape(Rc), 'shape(p2)=', shape(p2)
        
        Nlth2 = shape(Rc)[2]
        Plx = zeros((len(X), Nlth2))
        for l in range(Nlth2):
            Plx[:,l] = special.eval_legendre(l,X)
        
        PQw_tot = zeros((len(qx),len(iOm)))
        for iq in range(len(qx)):
            CQMC = QMC[:,:,iq,:]   # CQMC[ith,ik,lt]
            QMCW = dot(CQMC, Ker)  # OMCW[ith,ik,iOm] = \sum_lt CQMC[ith,ik,lt]  * Ker[lt,iOm]

            Rc = load('Rc.'+str(iq)+'.npy') # Rc[iw,ik,ltheta]
            nOm, Nkbin2, Nlth2 = shape(Rc)
            Nlth2 -= 1
            if (len(kx) != Nkbin or Nkbin!=Nkbin2):
                print 'ERROR : Number of k-bins from QMC and tpvertex.py does not match len(kx)=', len(kx), 'Nkbin=', Nkbin, 'Nkbin2=', Nkbin2
            if nOm!=len(iOm) or nOm != shape(p2)[1]:
                print 'ERROR : Length om iOm not correct nOm=', nOm, 'len(iOm)=', len(iOm), 'len(p2)=', len(p2)
            
            for iw in range(len(iOm)):
                Rcc = transpose(Rc[iw,:,:]) # Rcc[l,ik] = Rc[iw,ik,l]
                for l in range(shape(Rcc)[0]): Rcc[l,:] *= dh_k[:] * dth  # Rcc[l,ik] *= dh_k[ik]*dth
                # Vertex[ith,ik] = \sum_l Plx[ith,l]*Rc[iw,ik,l]*dh_k[ik]*dth
                Vertex = dot(Plx,Rcc)
                # PQw_tot[iq,iw] = sum_{ik,ith} QMCW[ith,ik,iw]*Vertex[ith,ik]
                PQw_tot[iq,iw] = tensordot(QMCW[:,:,iw], Vertex, axes=([0,1],[0,1]) ).real
        dat = vstack( (qx/kF,PQw_tot.transpose()) ).transpose() 
        savetxt(root+'.omg', dat)
        Nt=50
        tau = linspace(0,beta,Nt)
        p2t = sw.InverseFourierBoson_new(beta, tau, p['nom'], iOm, PQw_tot, zeros(len(qx)), 0)
        
        dat = vstack( (qx/kF, p2t.transpose()) ).transpose()
        savetxt(root+'.tau', dat)
        
    else:
        RClf = load('RClf.npy')
    
        Nlt, Nlq = shape(QMC)[2:] # theta-binnibg, k-binning, maxl_time, maxl_q
        Nlt-=1 # array size is lmax+1
        Nlq-=1 # array size is lmax+1
        (Nkbin2, Nlt2, Nlth2, Nlq2) = shape(RClf) # vertex has similar but different sizes
        Nlt2-=1
        Nlth2-=1
        Nlq2-=1
    
        if (len(kx) != Nkbin or Nkbin!=Nkbin2):
            print 'ERROR : Number of k-bins from QMC and tpvertex.py does not match len(kx)=', len(kx), 'Nkbin=', Nkbin, 'Nkbin2=', Nkbin2
    
        Vertex = zeros((Nthbin, Nkbin2, Nlt2+1, Nlq2+1))
        for ik in range(Nkbin):
            dk_dth = dh_k[ik] * dth
            for lt in range(Nlt2+1):
                for lq in range(Nlq2+1):  # changing from l_theta to cos(theta) mesh, compatible with binning
                    Vertex[:,ik,lt,lq] = legendre.legval(X, RClf[ik,lt,:,lq]) * dk_dth
    

        # Now closing QMC data with BK-vertex, i.e., summing over k and theta
        PQ = tensordot(QMC, Vertex, axes=([0,1],[0,1]) )  #  PQ(lt1,lq1,lt2,lq2) = \sum_{ith,ik} QMC[ith,ik, lt1,lq1] * Vertex[ith,ik, lt2,lq2]
        # The result is now expanded in terms of P_lt(2t'/beta-1)*P_lt2(2(t-t')/beta-1) * P_lq(2q/c_q-1) P_lq2(2q/c_q-1)
        # Need coefficients of the convolution between two legendres, i.e., convolution(P_lt1,P_lt2) = beta * \sum_{lt} P_lt
        conv_time = tpv.Plconv2(max(Nlt2,Nlt))

        # now using these coefficients, to perform convolution in time
        wPQ = tensordot(PQ, conv_time[:(Nlt2+1),:(Nlt+1),:], axes=([0,2],[1,0])) # wPQ[lq1,lq2,lt] = PQ[lt1,lq1,lt2,lq2]*conv_time[lt2,lt1,lt]
        wPQ *= beta
        # Convolution in time is done. Now we only have PQ written in terms of two Legendres, i.e., P_lq1(2q-1) P_lq2(2q-1). We want single P_lq(2q-1).
        # Need coefficients to transform.
        conv_q = tpv.Plconv3(max(Nlq2,Nlq))
        # Now the coefficients for product of two legendres is used to get single legendre expansion.
        # Now we have expansion in terms of P_{lt}(2t/beta-1) P_{lq}(2q/q_c-1) * BKCorrection_PQ[lt,lq]
        BKCorrection_PQ = tensordot(wPQ, conv_q[:(Nlq+1),:(Nlq2+1),:], axes=([0,1],[0,1]))  # BKCorrection_PQ[lt,lq] = wPQ[lq1,lq2,lt]*conv_q[lq1,lq2,lq]
        
        # This sets all odd terms in time to zero, which makes function symmetric in time. This should be satisfied by the polarization.
        # But careful... if this is not satisifed, you should not use that.
        BKCorrection_PQ[1::2,:]=0.0
        QMC_PQ[1::2,:]=0.0
        
        # Now summing up. It turns out that the number of coefficients can be different, hence we will carefully sum up.
        # PQ_tot = QMC_PQ + BK_correction_PQ
        PQ_tot = zeros( ( max(shape(QMC_PQ)[0],shape(BKCorrection_PQ)[0]), max(shape(QMC_PQ)[1],shape(BKCorrection_PQ)[1]) ) )
        nt,nq = shape(QMC_PQ)
        PQ_tot[:nt,:nq] += QMC_PQ[:,:]
        nt,nq = shape(BKCorrection_PQ)
        PQ_tot[:nt,:nq] += BKCorrection_PQ[:,:]
        
        root, ext = os.path.splitext(QMC_filename)
        savetxt(root+'.txt',  PQ_tot)

if __name__ == '__main__':
    execfile('params.py')
    beta = p['beta']
    
    if len(sys.argv)<2:
        print 'Give input filename of Pcof_?_corder_?.npy'
        sys.exit(1)
    
    #QMC_filename = 'Pcof_1_corder_0.npy'
    QMC_filename = sys.argv[1]

    filename = QMC_filename[:-14]+'?_corder_?.npy'
    filenames = glob.glob(filename)
    for filename in filenames:
        Run(filename,beta)
    
