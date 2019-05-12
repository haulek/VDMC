#!/usr/bin/env python
# @Copyright 2018 Kristjan Haule 
Parallel = True
from copy import deepcopy
from pylab import *
from scipy import *
from scipy import integrate
from scipy import special
from scipy import interpolate
if Parallel : from mpi4py import MPI
import time
import datetime
import bisect
import samplewj as sw
import sys
import io
import os
from kmesh import *

def dToccurence(Norder, Counter_order):
    Toccurence = 1.0
    if (Norder,Counter_order) == (4,0):
        Toccurence = 10.
    if (Norder,Counter_order) == (5,4):
        Toccurence = 10.
    if (Norder,Counter_order) == (5,0):
        Toccurence = 80.
    if (Norder,Counter_order) == (6,5):
        Toccurence = 100.
    if (Norder,Counter_order) == (6,0):
        Toccurence = 200.
    return Toccurence

def mpiSplitArray(rank,size,leng):
    def SplitArray(irank,size,leng):
        if leng % size==0:
            pr_proc = int(leng/size)
        else:
            pr_proc = int(leng/size+1)
        if (size<=leng):
            iqs,iqe = min(irank*pr_proc,leng) , min((irank+1)*pr_proc,leng)
        else:
            rstep=(size+1)/leng
            if irank%rstep==0 and irank/rstep<leng:
                iqs = irank/rstep
                iqe = iqs+1
            else:
                if irank/rstep<leng:
                    iqs = irank/rstep
                    iqe = irank/rstep
                else:
                    iqs = leng-1
                    iqe = leng-1
        return iqs,iqe
    sendcounts=[]
    displacements=[]
    for irank in range(size):
        iqs,iqe = SplitArray(irank,size,leng)
        sendcounts.append((iqe-iqs))
        displacements.append(iqs)
    iqs,iqe = SplitArray(rank,size,leng)
    return iqs,iqe, array(sendcounts,dtype=int), array(displacements,dtype=int)

def GetCounterTermPolarization(hf, fln, Vtype, my_mpi, p, dmu, lmbda, kx, epsx):
    kF, beta = p.kF, p.beta
    Recompute = 1
    if my_mpi.rank == my_mpi.master:
        if os.path.isfile(fln+'0.tau'): # checking if the file has identical parameters : kF, beta, dmu, lmbda
            fo = open(fln+'0.tau', 'r')
            data = fo.next().split()
            _kF_, _beta_, _dmu_, _lmbda_ = map(float,data[2:6])
            if ( (abs(_kF_-kF) < 1e-6) and (abs(_beta_-beta) < 1e-6) and (abs(_dmu_-dmu) < 1e-6) and (abs(_lmbda_-lmbda) < 1e-6) ):
                Recompute = 0
            else:
                print 'kF=', kF, ',', _kF_, ' beta=', beta, ',', _beta_, ' dmu=', dmu, ',', _dmu_, ' lmbda=', lmbda, ',', _lmbda_
        if Recompute:
            print 'In GetCounterTermPolarization... it seems we will need to Recompute bubbles'
        else:
            print 'In GetCounterTermPolarization... it seems we can keep the bubbles'
    
    if Parallel : 
        Recompute = comm.bcast( Recompute , root=my_mpi.master)
    print 'Recompute=', Recompute, 'rank=', my_mpi.rank

    if Recompute: # We have counter terms, hence need Polarization 
        Nt=max( 500, 2*(int(beta)/2) )
        tau = linspace(0,beta,Nt)
        qx = linspace(1e-3,cutoffk,Nq)
        q_kF = qx/kF
        iqs,iqe,sendcounts,displacements = mpiSplitArray(my_mpi.rank, my_mpi.size, len(q_kF) )
        
        #print my_mpi.rank, ' iqs=', iqs, 'iqe=', iqe
        # Calculating the exact integral
        Px = zeros( (iqe-iqs,Nt), dtype=float )
        Nt2 = Nt/2
        #print iqs,iqe, q_kF[iqs:iqe]

        iOm = zeros(nom+ntail,dtype=intc)
        iOm[:nom] = range(nom)
        iOm[nom:]= logspace(log10(nom+1),log10(20*nom),ntail)
        
        ypy = sw.Compute_Y_P_Y_(beta, lmbda, dmu, kF, cutoffk, 1)
        Pqw = zeros( (iqe-iqs,len(iOm)), dtype=float)

        for iq,Qa in enumerate(kF*q_kF[iqs:iqe]):
            Py = hf.P0(Qa,tau[:Nt2])
            Px[iq,:Nt2] = Py[:]
            Px[iq,Nt2:] = Py[::-1]

            Pqw[iq,:] = ypy.Pw0(Qa, iOm)
            
            #print 'iq=', iq, 'Qa=', Qa, 't=', tau[:Nt2], 'Px=', Px[iq,:]
            #print 'shape(Py)=', shape(Py), 'shape Px=', shape(Px[iq,:Nt2]), 'shape Px=', shape(Px[iq,Nt2:])
            #print 'Now Px=', Px[0,:], Px[1,:]
        #print 'Finally Px=', Px
        
        if Parallel and my_mpi.size>1:
            if my_mpi.rank == my_mpi.master:
                Px_gathered = zeros( (len(q_kF),len(tau)) )
                Pqw_gathered = zeros( (len(q_kF),len(iOm)) )
            else:
                Px_gathered = None
                Pqw_gathered = None
            dsize = len(tau)
            comm.Gatherv(Px,[Px_gathered,sendcounts*dsize,displacements*dsize,MPI.DOUBLE])
            dsize = len(iOm)
            comm.Gatherv(Pqw,[Pqw_gathered,sendcounts*dsize,displacements*dsize,MPI.DOUBLE])
            if my_mpi.rank == my_mpi.master:
                Px = Px_gathered
                Pqw = Pqw_gathered
            else:
                Px = zeros( (len(q_kF),len(tau)) )
                Pqw = zeros( (len(q_kF),len(iOm)) )
            #comm.Bcast(Px, root=master)
        
        if my_mpi.rank==my_mpi.master:
            fo = open(fln+'0.tau', 'w')
            print >> fo, len(tau), len(q_kF), kF, beta, dmu, lmbda
            for i in range(len(tau)):
                print >> fo, tau[i]
            fo.close()
            

            # DEFINITELY REMOVE
            #Px=zeros(shape(Px))
            savetxt(fln+'1.tau', vstack( (q_kF*kF,transpose(Px)*beta) ).transpose())
            #savetxt(fln+'1.tau', vstack( (q_kF,transpose(Px)*beta) ).transpose())
            savetxt(fln+'1_.tau', vstack( (tau,Px*beta) ).transpose() )
        
            Pqt = Px.copy()
            #print 'Starting fourier', beta, shape(iOm), shape(tau), shape(Pqt)
            #print 'Pqt=', Pqt
            #print 'iOm=', iOm
            #print 'tau=', tau
            
            if False:
                Pqw = sw.FourierBoson(beta, iOm, tau, Pqt)  # Pqw[iq,iw]
                pqw = Pqw.real
            else:
                pqw = Pqw
            
            savetxt(fln+'1.omg', vstack( (q_kF*kF, (pqw).transpose()*beta) ).transpose())
            savetxt(fln+'1_.omg', vstack( (iOm*2*pi/beta, pqw*beta) ).transpose())
            
            feps = interpolate.UnivariateSpline(kx,epsx, s=0)
            e_q = array([feps(Qa) for Qa in (kF*q_kF)])
            pw = copy(pqw)
            for ii in range(1,5):
                p0t = sw.InverseFourierBoson_new(beta, tau, nom, iOm, pw, e_q, ii)
                p0t = transpose(p0t)
                if ii==1:
                    c='_'
                else:
                    c=''
                savetxt(c+fln+str(ii)+'.tau', vstack( (q_kF*kF,p0t*beta) ).transpose())
                #savetxt(c+fln+str(ii)+'.tau', vstack( (q_kF,p0t*beta) ).transpose())
                savetxt(c+fln+str(ii)+'_.tau', vstack( (tau,transpose(p0t)*beta) ).transpose() )
                pw *= pqw
    if Parallel:
        comm.Barrier()

#def sample_static(lmbda, lmbda_spct, dmu, Norder, Counter_order, BKA=False, Hugenholtz=True, DynamicCT=False, Debug=False):
#    """ Takes care of unscreened static interaction only.
#        Samples polarization for any t and Q by Metropolis.
#        The normalization is done by adding a constant diagram, which should not suffers from ergodicity.
#    """
#    HF=True
#    norder = Norder  # real order of diagram
#    if Counter_order >= 2 : # when we deal with counter-terms (Counter_order!=0 or !=1), their order is actually smaller
#        norder = Counter_order
#    
#    my_mpi = sw.my_mpi()
#    if Parallel:
#        comm = MPI.COMM_WORLD
#        my_mpi.size = comm.Get_size()
#        my_mpi.rank = comm.Get_rank()
#    
#    p = sw.params()
#    if BKA:
#        p.Nthbin  = Nthbin #10
#        #p.Nkbin   = 50
#    p.Nlt     = 24
#    p.Nlq     = 18
#    
#    p.kF      = kF
#    p.beta    = beta
#    p.cutoffq = cutoffq
#    p.cutoffk = cutoffk
#    
#    p.Nitt    = int(Nitt) #50000000 #100000000
#    
#    p.V0norm  = 40**norder * 1e-3 * (10./beta)**2 * (2.1*kF)**norder
#    p.V0exp   = 4.
#    p.Pr      = [0.65,0.25,0.1]
#    p.Qring   = 0.5
#    p.dRk     = 1.3
#    p.dkF     = 0.5*kF
#    p.iseed   = random.randint(2**10)+my_mpi.rank*10
#    p.tmeassure = 2
#    p.Ncout    = Ncout # 500000
#    p.Nwarm    = 10000
#    p.lmbdat   = 0.0
#    p.Nq       = Nq
#    p.Nt       = 10
#    
#    if Hugenholtz:
#        filename = sinput+'/loops_Pdiags.'+str(Norder)+'_cworder_'+str(Counter_order)+'_'
#    else:
#        if Counter_order==0:
#            filename = sinput+'/loops_Pdiags.'+str(Norder)+'_nobubble_'
#        elif Counter_order==1:
#            filename = sinput+'/loops_Pdiags.'+str(Norder)+'_bubbles_'
#        else:
#            filename = sinput+'/loops_Pdiags.'+str(Norder)+'_corder_'+str(Counter_order)+'_'
#        
#
#    if not BKA:
#        (diagsG, diagSign, Loop_index, Loop_type, Loop_vertex, diagVertex, indx, Vtype) = GetPDiags(filename, BKremove='')
#    elif BKA==1:
#        (diagsG, diagSign, Loop_index, Loop_type, Loop_vertex, diagVertex, indx, Vtype) = GetPDiags(filename, BKremove='right')
#    elif BKA==2:
#        (diagsG, diagSign, Loop_index, Loop_type, Loop_vertex, diagVertex, indx, Vtype) = GetPDiags(filename, BKremove='both')
#    else:
#        print 'ERROR Unrecognized BKA in sample_static_Q0'
#        sys.exit(1)
#
#    if (len(diagsG)==0):
#        print 'Nothing to do'
#        return
#
#    if len(Vtype)==0:
#        Vtype = zeros((len(diagsG),norder))
#    if (rank==master):
#        fout = io.open('snohup_'+str(Norder)+'_corder_'+str(Counter_order), 'ab')
#        print >> fout, '# time: ', datetime.datetime.now(), 'iseed=', p.iseed, " running sample_static"
#        print >> fout, 'rs=', rs, 'kF=', kF, 'beta=', beta, 'MaxOrder=', MaxOrder, 'cutoffq=', cutoffq, 'cutoffk=', cutoffk, 'dmu_1=', dmu_1
#    else:
#        #fout = io.open('snohup.'+str(rank)+'_'+str(Norder)+'_corder_'+str(Counter_order), 'ab')
#        fout = sys.stdout
#
#    print 'Nk=', Nk
#    kxb = Give_k_mesh(Nk, kF, cutoffk, 0.0)  #extra
#    dcutoff = p.cutoffk-p.cutoffq
#    hf = sw.HartreeFock(p.kF, p.cutoffk+dcutoff, p.beta, lmbda, dmu)
#    kx, epsx = hf.get()
#
#    if DynamicCT:
#        GetCounterTermPolarization(hf, 'Pqt', Vtype, my_mpi, p, dmu, lmbda, kx, epsx)
#
#    if BKA==1:
#        if DynamicCT:
#            C_Pln, Pbin = sw.sample_static_fast_VHFD(fout, lmbda, lmbda_spct, p, kx, epsx, diagsG, diagSign, Loop_index, Loop_type, Vtype, indx, kxb, my_mpi)
#        else:
#            C_Pln, Pbin = sw.sample_static_fast_VHFC(fout, lmbda, lmbda_spct, p, kx, epsx, diagsG, diagSign, Loop_index, Loop_type, Vtype, indx, kxb, my_mpi)
#    elif BKA==0:
#        if DynamicCT:
#            C_Pln, Pbin = sw.sample_static_fast_HFD(fout, lmbda, lmbda_spct, p, kx, epsx, diagsG, diagSign, Loop_index, Loop_type, Vtype, indx, my_mpi)
#        else:
#            C_Pln, Pbin = sw.sample_static_fast_HFC(fout, lmbda, lmbda_spct, p, kx, epsx, diagsG, diagSign, Loop_index, Loop_type, Vtype, indx, my_mpi)
#    else:
#        print 'ERROR Unrecognized BKA in sample_static'
#        sys.exit(1)
#        
#    if my_mpi.rank == my_mpi.master:
#        qx = p.cutoffq*(0.5+arange(p.Nq))/p.Nq
#        savetxt('Pbin_'+str(Norder)+'_corder_'+str(Counter_order), vstack( (qx/kF,Pbin) ).transpose())
#        if BKA:
#            save('kxb', kxb)
#            save('Pcof_'+str(Norder)+'_corder_'+str(Counter_order), C_Pln )
#        else:
#            savetxt('Pcof_'+str(Norder)+'_corder_'+str(Counter_order), C_Pln )
#        
#        print >> fout, '# time: ', datetime.datetime.now()
#        
#        if Debug:
#            # The mesh in k where points are centered in the middle
#            tx = p.beta*(0.5+arange(p.Nt))/p.Nt
#            tx = hstack( ([1e-6],tx) )
#            
#            import matplotlib.pyplot as plt
#            
#            # Evaluates Legendre Polynomials
#            Pls = zeros((p.Nlq+1,len(qx)))
#            for l in range(p.Nlq+1):
#                Pls[l,:] = special.eval_legendre(l,2*qx/p.cutoffq-1.)
#            tub = zeros((len(tx),p.Nlt+1))
#            for l in range(0,p.Nlt+1,2):
#                tub[:,l] = special.eval_legendre(l,2*tx/p.beta-1.)
#
#            if BKA:
#                C_Pln_large = copy.deepcopy(C_Pln)
#                
#                dth = 2.0/p.Nthbin;
#                dk = array([kxb[ik+1]-kxb[ik] for ik in range(len(kxb)-1)])
#                C_Pln = zeros((p.Nlt+1,p.Nlq+1))
#                for lt in range(0,p.Nlt+1,2):
#                    for lq in range(p.Nlq+1):
#                        for ik in range(len(kxb)-1):
#                            C_Pln[lt,lq] += sum(C_Pln_large[:,ik,lt,lq])*dth*dk[ik]
#            
#            # Multiplies Legendre Polynomials with the coefficients
#            Ptq = dot(dot(tub, C_Pln),Pls)
#            
#            smallk = 1e-11
#            p21c = sw.PO2(kF, p.beta, p.cutoffk, p.cutoffq, smallk)
#            # Calculating the exact integral
#            Px = zeros((len(tx),p.Nq))
#            Nt_need = p.Nt/2+2
#            
#            for iq,Qa in enumerate(qx):
#                tau = tx[:Nt_need]
#                Px[:Nt_need,iq] = hf.P0(Qa,tau)
#                print Qa, max(Px[:,iq])
#                    
#            col=['b','g','r','c','m','y','k','w']
#            for it in range(p.Nt/2+2):
#                plt.plot(qx/kF, Px[it,:], ':'+col[it%8], label='t='+str(tx[it])+' exact '+str(my_mpi.rank))
#                plt.plot(qx/kF, Ptq[it,:], '-'+col[it%8], label='t='+str(tx[it])+' leg.b.'+str(my_mpi.rank))
#                if it>0:
#                    plt.plot(qx/kF, Pbin[it-1,:], '.'+col[it%8], label='t='+str(tx[it])+' bin')
#                
#            plt.legend(loc='best')
#            plt.show()


def sample_static_discrete(lmbda, lmbda_spct, dmu, Norder, Counter_order, BKA=False, Hugenholtz=True, DynamicCT=False, Debug=False):
    """ Takes care of unscreened static interaction only.
        Samples polarization for any t and Q by Metropolis.
        The normalization is done by adding a constant diagram, which should not suffers from ergodicity.
    """
    HF=True
    norder = Norder  # real order of diagram
    if Counter_order >= 2 : # when we deal with counter-terms (Counter_order!=0 or !=1), their order is actually smaller
        norder = Counter_order
    
    my_mpi = sw.my_mpi()
    if Parallel:
        comm = MPI.COMM_WORLD
        my_mpi.size = comm.Get_size()
        my_mpi.rank = comm.Get_rank()
    
    p = sw.params()
    p.Toccurence = dToccurence(Norder, Counter_order)
    if BKA:
        p.Nthbin  = Nthbin #10
        #p.Nkbin   = 50
    p.Nlt     = 24
    p.Nlq     = 18
    
    p.kF      = kF
    p.beta    = beta
    p.cutoffq = cutoffq
    p.cutoffk = cutoffk
    
    p.Nitt    = int(Nitt) #50000000 #100000000
    
    p.V0norm  = 40**norder * 1e-3 * (10./beta)**2 * (2.1*kF)**norder
    p.V0exp   = 4.
    p.Pr      = [0.65,0.25,0.1]
    p.Qring   = Qring # 0.5
    p.dRk     = 1.3
    p.dkF     = dkF0*kF
    p.iseed   = random.randint(2**10)+my_mpi.rank*10
    p.tmeassure = 2
    p.Ncout    = Ncout # 500000
    p.Nwarm    = 10000
    p.lmbdat   = 0.0
    p.Nq       = Nq
    p.Nt       = 10

    qx = linspace(1e-5,p.cutoffq,p.Nq)
    
    if my_mpi.rank == my_mpi.master:
        save('qxb', qx)

    if Hugenholtz:
        filename = sinput+'/loops_Pdiags.'+str(Norder)+'_cworder_'+str(Counter_order)+'_'
    else:
        if Counter_order==0:
            filename = sinput+'/loops_Pdiags.'+str(Norder)+'_nobubble_'
        elif Counter_order==1:
            filename = sinput+'/loops_Pdiags.'+str(Norder)+'_bubbles_'
        else:
            filename = sinput+'/loops_Pdiags.'+str(Norder)+'_corder_'+str(Counter_order)+'_'
        

    if not BKA:
        (diagsG, diagSign, Loop_index, Loop_type, Loop_vertex, diagVertex, indx, Vtype) = GetPDiags(filename, BKremove='')
    elif BKA==1:
        (diagsG, diagSign, Loop_index, Loop_type, Loop_vertex, diagVertex, indx, Vtype) = GetPDiags(filename, BKremove='right')
    elif BKA==2:
        (diagsG, diagSign, Loop_index, Loop_type, Loop_vertex, diagVertex, indx, Vtype) = GetPDiags(filename, BKremove='both')
    else:
        print 'ERROR Unrecognized BKA in sample_static_Q0'
        sys.exit(1)

    if (len(diagsG)==0):
        print 'Nothing to do'
        return

    if len(Vtype)==0:
        Vtype = zeros((len(diagsG),norder))
    if (rank==master):
        fout = io.open('snohup_'+str(Norder)+'_corder_'+str(Counter_order), 'ab')
        print >> fout, '# time: ', datetime.datetime.now(), 'iseed=', p.iseed, " running sample_static_discrete"
        print >> fout, 'rs=', rs, 'kF=', kF, 'beta=', beta, 'MaxOrder=', MaxOrder, 'cutoffq=', cutoffq, 'cutoffk=', cutoffk, 'dmu_1=', dmu_1
    else:
        #fout = io.open('snohup.'+str(rank)+'_'+str(Norder)+'_corder_'+str(Counter_order), 'ab')
        fout = sys.stdout

    print 'Nk=', Nk
    kxb = Give_k_mesh(Nk, kF, cutoffk, 0.0)  #extra
    dcutoff = p.cutoffk-p.cutoffq
    hf = sw.HartreeFock(p.kF, p.cutoffk+dcutoff, p.beta, lmbda, dmu)
    kx, epsx = hf.get()

    if DynamicCT:
        GetCounterTermPolarization(hf, 'Pqt', Vtype, my_mpi, p, dmu, lmbda, kx, epsx)

    if BKA==1:
        C_Pln, Pbin = sw.sample_static_Discrete_VHFC(fout, qx, lmbda, lmbda_spct, p, kx, epsx, diagsG, diagSign, Loop_index, Loop_type, Vtype, indx, kxb, my_mpi)
    elif BKA==0:
        C_Pln, Pbin = sw.sample_static_Discrete_HFC(fout, qx, lmbda, lmbda_spct, p, kx, epsx, diagsG, diagSign, Loop_index, Loop_type, Vtype, indx, my_mpi)
    else:
        print 'ERROR Unrecognized BKA in sample_static'
        sys.exit(1)
        
    if my_mpi.rank == my_mpi.master:
        savetxt('Pbin_'+str(Norder)+'_corder_'+str(Counter_order), vstack( (qx/kF,Pbin) ).transpose())
        if BKA:
            save('kxb', kxb)
            save('Pcof_'+str(Norder)+'_corder_'+str(Counter_order), C_Pln )
        else:
            savetxt('Pcof_'+str(Norder)+'_corder_'+str(Counter_order), C_Pln )
        
        print >> fout, '# time: ', datetime.datetime.now()



def sample_combined_discrete_heavy(lmbda, lmbda_spct, dmu, Norder, Counter_order, qx, kxb, my_mpi):
    """ Takes care of unscreened static interaction only.
        Samples polarization for any t and Q by Metropolis.
        The normalization is done by adding a constant diagram, which should not suffers from ergodicity.
    """
    norder = Norder  # real order of diagram
    if Counter_order >= 2 : # when we deal with counter-terms (Counter_order!=0 or !=1), their order is actually smaller
        norder = Counter_order
    
    p = sw.params()
    p.Toccurence = dToccurence(Norder, Counter_order)
    p.Nthbin  = Nthbin
    p.Nlt     = Nlt # 24
    #p.Nlq     = 18
    p.kF      = kF
    p.beta    = beta
    p.cutoffq = cutoffq
    p.cutoffk = cutoffk
    p.Nitt    = int(Nitt)
    p.V0norm  = 40**norder * 1e-3 * (10./beta)**2 * (2.1*kF)**norder
    p.V0exp   = 4.
    p.Pr      = [0.65,0.25,0.1]
    p.Qring   = Qring # 0.5
    p.dRk     = 1.3
    p.dkF     = dkF0*kF  # before it was 0.5*kF
    p.iseed   = random.randint(2**10)+my_mpi.rank*10
    p.tmeassure = 2
    p.Ncout    = Ncout
    p.Nwarm    = 10000
    p.lmbdat   = 0.0
    p.Nq       = Nq
    p.Nt       = 10

    filename = sinput+'/loops_Pdiags.'+str(Norder)+'_cworder_'+str(Counter_order)+'_'

    (diagsG, diagSign, Loop_index, Loop_type, Loop_vertex, diagVertex, indx, Vtype) = GetPDiags(filename, BKremove='')
    
    if len(Vtype)==0:
        Vtype = zeros((len(diagsG),norder))
    if (rank==master):
        fout = io.open('snohup_'+str(Norder)+'_corder_'+str(Counter_order), 'ab')
        print >> fout, '# time: ', datetime.datetime.now(), 'iseed=', p.iseed, " running sample_static_discrete"
        print >> fout, 'rs=', rs, 'kF=', kF, 'beta=', beta, 'MaxOrder=', MaxOrder, 'cutoffq=', cutoffq, 'cutoffk=', cutoffk, 'dmu_1=', dmu_1
    else:
        #fout = io.open('snohup.'+str(rank)+'_'+str(Norder)+'_corder_'+str(Counter_order), 'ab')
        fout = sys.stdout

    dcutoff = p.cutoffk-p.cutoffq
    hf = sw.HartreeFock(p.kF, p.cutoffk+dcutoff, p.beta, lmbda, dmu)
    kx, epsx = hf.get()

    (C_Pln2, C_Pln1, C_Pln0, Pbin, Pq0) = sw.sample_combined_discrete(fout, qx, lmbda, lmbda_spct, p, kx, epsx,
                                                                      diagsG, diagSign, Loop_index, Loop_type, Vtype, indx, kxb, my_mpi)
    
    if my_mpi.rank == my_mpi.master:
        savetxt('Pbin_'+str(Norder)+'_corder_'+str(Counter_order), vstack( (qx/kF,Pbin) ).transpose())
        save('kxb', kxb)
        save('Pcof2_'+str(Norder)+'_corder_'+str(Counter_order), C_Pln2 )
        save('Pcof1_'+str(Norder)+'_corder_'+str(Counter_order), C_Pln1 )
        save('Pcof0_'+str(Norder)+'_corder_'+str(Counter_order), C_Pln0 )
        
        savetxt('Pq0_'+str(Norder)+'_corder_'+str(Counter_order), vstack( (qx/kF,Pq0[:,0],Pq0[:,1]) ).transpose())
        # (Nthbin, Nk, Nq, Nlt) = (10, 160, 120, 24)
        # maximum that can be afforder:
        # (Nthbin, Nk, Nq, Nlt) = (10, 40, 40, 24) ~ 1.5G
        paver1 = sum(Pbin[:,0])
        
        print >> fout, 'Result : ', paver1, Pq0[0,0], '+-', Pq0[0,1]
        print >> fout, '# time: ', datetime.datetime.now()


def sample_combined_discrete(lmbda, lmbda_spct, dmu, Norder, Counter_order, qx, kxb, Vertex, Ker_iOm_lt, my_mpi):
    """ Takes care of unscreened static interaction only.
        Samples polarization for any t and Q by Metropolis.
        The normalization is done by adding a constant diagram, which should not suffers from ergodicity.
    """
    norder = Norder  # real order of diagram
    if Counter_order >= 2 : # when we deal with counter-terms (Counter_order!=0 or !=1), their order is actually smaller
        norder = Counter_order
    
    p = sw.params()
    p.Toccurence = dToccurence(Norder, Counter_order)
    p.Nthbin  = Nthbin
    p.Nlt     = Nlt # 24
    #p.Nlq     = 18
    p.kF      = kF
    p.beta    = beta
    p.cutoffq = cutoffq
    p.cutoffk = cutoffk
    p.Nitt    = int(Nitt)
    p.V0norm  = 40**norder * 1e-3 * (10./beta)**2 * (2.1*kF)**norder
    p.V0exp   = 4.
    p.Pr      = [0.65,0.25,0.1]
    p.Qring   = Qring # 0.2     # before it was 0.5
    p.dRk     = 1.3
    p.dkF     = dkF0*kF  # before it was 0.5*kF
    p.iseed   = random.randint(2**10)+my_mpi.rank*10
    p.tmeassure = 2
    p.Ncout    = Ncout
    p.Nwarm    = 10000
    p.lmbdat   = 0.0
    p.Nq       = Nq
    p.Nt       = 10
    
    filename = sinput+'/loops_Pdiags.'+str(Norder)+'_cworder_'+str(Counter_order)+'_'
    
    (diagsG, diagSign, Loop_index, Loop_type, Loop_vertex, diagVertex, indx, Vtype) = GetPDiags(filename, BKremove='')
    
    if len(Vtype)==0:
        Vtype = zeros((len(diagsG),norder))
    if (rank==master):
        fout = io.open('snohup_'+str(Norder)+'_corder_'+str(Counter_order), 'ab')
        print >> fout, '# time: ', datetime.datetime.now(), 'iseed=', p.iseed, " running sample_static_discrete"
        print >> fout, 'rs=', rs, 'kF=', kF, 'beta=', beta, 'MaxOrder=', MaxOrder, 'cutoffq=', cutoffq, 'cutoffk=', cutoffk, 'dmu_1=', dmu_1
    else:
        #fout = io.open('snohup.'+str(rank)+'_'+str(Norder)+'_corder_'+str(Counter_order), 'ab')
        fout = sys.stdout

    dcutoff = p.cutoffk-p.cutoffq
    hf = sw.HartreeFock(p.kF, p.cutoffk+dcutoff, p.beta, lmbda, dmu)
    kx, epsx = hf.get()

    (C_Pln2, C_Pln1, C_Pln0, Pbin, Pq0) = sw.sample_combined_discrete2(fout, qx, lmbda, lmbda_spct, p, kx, epsx, 
                                                                      diagsG, diagSign, Loop_index, Loop_type, Vtype, indx, kxb,
                                                                      Vertex, Ker_iOm_lt, my_mpi)
    
    if my_mpi.rank == my_mpi.master:
        savetxt('Pbin_'+str(Norder)+'_corder_'+str(Counter_order), vstack( (qx/kF,Pbin) ).transpose())
        save('kxb', kxb)
        save('Pcof2_'+str(Norder)+'_corder_'+str(Counter_order), C_Pln2 )
        save('Pcof1_'+str(Norder)+'_corder_'+str(Counter_order), C_Pln1 )
        save('Pcof0_'+str(Norder)+'_corder_'+str(Counter_order), C_Pln0 )
        savetxt('Pcof2_'+str(Norder)+'_corder_'+str(Counter_order), vstack( (qx/kF, C_Pln2[:,0], C_Pln2[:,-1]) ).T )
        savetxt('Pcof1_'+str(Norder)+'_corder_'+str(Counter_order), vstack( (qx/kF, C_Pln1[:,0], C_Pln1[:,-1]) ).T )
        savetxt('Pcof0_'+str(Norder)+'_corder_'+str(Counter_order), vstack( (qx/kF, C_Pln0[:,0], C_Pln0[:,-1]) ).T )
        
        savetxt('Pq0_'+str(Norder)+'_corder_'+str(Counter_order), vstack( (qx/kF,Pq0[:,0],Pq0[:,1]) ).transpose())
        # (Nthbin, Nk, Nq, Nlt) = (10, 160, 120, 24)
        # maximum that can be afforder:
        # (Nthbin, Nk, Nq, Nlt) = (10, 40, 40, 24) ~ 1.5G
        paver1 = sum(Pbin[:,0])
        
        print >> fout, 'Result : ', paver1, Pq0[0,0], '+-', Pq0[0,1]
        print >> fout, '# time: ', datetime.datetime.now()

        
def sample_Q0(Q_external, lmbda, lmbda_spct, dmu, Norder, Counter_order, BKA=0, Debug=False):
    """ Takes care of unscreened static interaction only.
        Samples polarization for any t and Q by Metropolis.
        The normalization is done by adding a constant diagram, which should not suffers from ergodicity.
    """
    HF=True
    norder = Norder  # real order of diagram
    if Counter_order >= 2 : # when we deal with counter-terms (Counter_order!=0 or !=1), their order is actually smaller
        norder = Counter_order
    
    my_mpi = sw.my_mpi()
    if Parallel:
        comm = MPI.COMM_WORLD
        my_mpi.size = comm.Get_size()
        my_mpi.rank = comm.Get_rank()
    
    p = sw.params()
    p.Toccurence = dToccurence(Norder, Counter_order)
    p.Nthbin  = 1
    p.Nlt     = 1
    p.Nlq     = 1
    
    p.kF      = kF
    p.beta    = beta
    p.cutoffq = cutoffq
    p.cutoffk = cutoffk
    
    p.Nitt    = int(Nitt)
    
    p.V0norm  = 40**norder * 1e-3 * (10./beta)**2 * (2.1*kF)**norder
    p.V0exp   = 4.
    p.Pr      = [0.65,0.25,0.1]
    p.Qring   = Qring # 0.5
    p.dRk     = 1.3
    p.dkF     = dkF0*kF
    p.iseed   = random.randint(2**10)+my_mpi.rank*10
    p.tmeassure = 2
    p.Ncout    = Ncout
    p.Nwarm    = Nwarm # 10000
    p.lmbdat   = 0.0
    p.Nq       = 100
    p.Nt       = 10
    
    filename = sinput+'/loops_Pdiags.'+str(Norder)+'_cworder_'+str(Counter_order)+'_'
    (diagsG, diagSign, Loop_index, Loop_type, Loop_vertex, diagVertex, indx, Vtype) = GetPDiags(filename, BKremove='')
     
    if (len(diagsG)==0):
        print 'Nothing to do'
        return
    if len(Vtype)==0:
        Vtype = zeros((len(diagsG),norder))
    
    if (rank==master):
        fout = io.open('pnohup_lmbda_'+str(lmbda)+'_'+str(Norder)+'_corder_'+str(Counter_order), 'ab')
        print >> fout, '# time: ', datetime.datetime.now(), 'iseed=', p.iseed
        print >> fout, 'rs=', rs, 'kF=', kF, 'beta=', beta, 'MaxOrder=', MaxOrder, 'cutoffq=', cutoffq, 'cutoffk=', cutoffk, 'dmu_1=', dmu_1
    else:
        fout = sys.stdout

    print 'Nk=', Nk
    kxb = Give_k_mesh(Nk, kF, cutoffk, 0.0)  #extra
    dcutoff = p.cutoffk-p.cutoffq
    hf = sw.HartreeFock(p.kF, p.cutoffk+dcutoff, p.beta, lmbda, dmu)
    kx, epsx = hf.get()
    
    if my_mpi.rank==0:
        savetxt('hf_epsk', vstack( (kx,epsx) ).transpose())

    C_Pln2, C_Pln1, C_Pln0, Pbin, Pq0 = sw.sample_Q0W0(Q_external, BKA, fout, lmbda, lmbda_spct, p, kx, epsx, diagsG, diagSign, Loop_index, Loop_type, Vtype, indx, kxb, my_mpi)
    
    Paver=0.
    _sigma_old_=0.
    _sigma_ = 0.
    if my_mpi.rank == my_mpi.master:
        Paver = sum(Pbin)
        N2 = len(Pbin)
        _Paver2_ = N2 * sqrt(sum(Pbin*Pbin)/N2)
        _sigma_old_ = sqrt( (_Paver2_ - Paver)*(Paver + _Paver2_)/N2 )
        _sigma_ = Pq0[0,1]
        
        tx = p.beta*(0.5+arange(p.Nt))/p.Nt
        #qx = p.cutoffq*(0.5+arange(p.Nq))/p.Nq
        savetxt('Pbin_lmbda_'+str(lmbda)+'_'+str(Norder)+'_corder_'+str(Counter_order), vstack( (tx,Pbin) ).transpose())
        save('kxb', kxb)
        save('Pcof2_lmbda_'+str(lmbda)+'_'+str(Norder)+'_corder_'+str(Counter_order), C_Pln2 )
        save('Pcof1_lmbda_'+str(lmbda)+'_'+str(Norder)+'_corder_'+str(Counter_order), C_Pln1 )
        save('Pcof0_lmbda_'+str(lmbda)+'_'+str(Norder)+'_corder_'+str(Counter_order), C_Pln0 )
        
        Paver2 = (C_Pln0[0,0] + sum(C_Pln1) + sum(C_Pln2))*beta
        
        print >> fout, 'Result : ', Paver, Paver2, Pq0[0,0], '+-', _sigma_, _sigma_old_
        print >> fout, '# time: ', datetime.datetime.now()
    Paver,_sigma_ = comm.bcast( (Paver, _sigma_), root=my_mpi.master)
    return (Paver, _sigma_)

def sample_static_Q0(Q_external, lmbda, lmbda_spct, dmu, Norder, Counter_order, BKA=0, Hugenholtz=True, DynamicCT=False, Debug=False):
    """ Takes care of unscreened static interaction only.
        Samples polarization for any t and Q by Metropolis.
        The normalization is done by adding a constant diagram, which should not suffers from ergodicity.
    """
    HF=True
    norder = Norder  # real order of diagram
    if Counter_order >= 2 : # when we deal with counter-terms (Counter_order!=0 or !=1), their order is actually smaller
        norder = Counter_order
    
    my_mpi = sw.my_mpi()
    if Parallel:
        comm = MPI.COMM_WORLD
        my_mpi.size = comm.Get_size()
        my_mpi.rank = comm.Get_rank()
    
    p = sw.params()
    p.Toccurence = dToccurence(Norder, Counter_order)
    if BKA:
        p.Nthbin  = Nthbin #10
        #p.Nkbin   = 50
    p.Nlt     = 1
    p.Nlq     = 1
    
    p.kF      = kF
    p.beta    = beta
    p.cutoffq = cutoffq
    p.cutoffk = cutoffk
    
    p.Nitt    = int(Nitt) #50000000 #100000000
    
    p.V0norm  = 40**norder * 1e-3 * (10./beta)**2 * (2.1*kF)**norder
    p.V0exp   = 4.
    p.Pr      = [0.65,0.25,0.1]
    p.Qring   = Qring # 0.5
    p.dRk     = 1.3
    p.dkF     = dkF0*kF
    p.iseed   = random.randint(2**10)+my_mpi.rank*10
    p.tmeassure = 2
    p.Ncout    = Ncout #500000
    p.Nwarm    = 10000
    p.lmbdat   = 0.0
    p.Nq       = 100
    p.Nt       = 10
    
    if Hugenholtz:
        filename = sinput+'/loops_Pdiags.'+str(Norder)+'_cworder_'+str(Counter_order)+'_'
    else:
        if Counter_order==0:
            filename = sinput+'/loops_Pdiags.'+str(Norder)+'_nobubble_'
        elif Counter_order==1:
            filename = sinput+'/loops_Pdiags.'+str(Norder)+'_bubbles_'
        else:
            filename = sinput+'/loops_Pdiags.'+str(Norder)+'_corder_'+str(Counter_order)+'_'
        
    if not BKA:
        (diagsG, diagSign, Loop_index, Loop_type, Loop_vertex, diagVertex, indx, Vtype) = GetPDiags(filename, BKremove='')
    elif BKA==1:
        (diagsG, diagSign, Loop_index, Loop_type, Loop_vertex, diagVertex, indx, Vtype) = GetPDiags(filename, BKremove='right')
    elif BKA==2:
        (diagsG, diagSign, Loop_index, Loop_type, Loop_vertex, diagVertex, indx, Vtype) = GetPDiags(filename, BKremove='both')
    else:
        print 'ERROR Unrecognized BKA in sample_static_Q0'
        sys.exit(1)
        
    if (len(diagsG)==0):
        print 'Nothing to do'
        return
    
    if len(Vtype)==0:
        Vtype = zeros((len(diagsG),norder))
    if (rank==master):
        fout = io.open('pnohup_lmbda_'+str(lmbda)+'_'+str(Norder)+'_corder_'+str(Counter_order), 'ab')
        print >> fout, '# time: ', datetime.datetime.now(), 'iseed=', p.iseed
        print >> fout, 'rs=', rs, 'kF=', kF, 'beta=', beta, 'MaxOrder=', MaxOrder, 'cutoffq=', cutoffq, 'cutoffk=', cutoffk, 'dmu_1=', dmu_1
    else:
        #fout = io.open('snohup.'+str(rank)+'_'+str(Norder)+'_corder_'+str(Counter_order), 'ab')
        fout = sys.stdout

    print 'Nk=', Nk
    kxb = Give_k_mesh(Nk, kF, cutoffk, 0.0)  #extra
    dcutoff = p.cutoffk-p.cutoffq
    hf = sw.HartreeFock(p.kF, p.cutoffk+dcutoff, p.beta, lmbda, dmu)
    kx, epsx = hf.get()
    
    if my_mpi.rank==0:
        savetxt('hf_epsk', vstack( (kx,epsx) ).transpose())

    if DynamicCT:
        GetCounterTermPolarization(hf, 'Pqt', Vtype, my_mpi, p, dmu, lmbda, kx, epsx)
        
    if not BKA:
        if DynamicCT:
            C_Pln, Pbin = sw.sample_static_Q0W0_HFD(fout, lmbda, lmbda_spct, p, kx, epsx, diagsG, diagSign, Loop_index, Loop_type, Vtype, indx, my_mpi)
        else:
            C_Pln, Pbin, Pbin2 = sw.sample_static_Q0W0_HFC(Q_external, fout, lmbda, lmbda_spct, p, kx, epsx, diagsG, diagSign, Loop_index, Loop_type, Vtype, indx, my_mpi)
    elif BKA==1:
        if DynamicCT:
            C_Pln, Pbin = sw.sample_static_Q0W0_VHFD(fout, lmbda, lmbda_spct, p, kx, epsx, diagsG, diagSign, Loop_index, Loop_type, Vtype, indx, kxb, my_mpi)
        else:
            C_Pln, Pbin, Pbin2 = sw.sample_static_Q0W0_VHFC(Q_external, fout, lmbda, lmbda_spct, p, kx, epsx, diagsG, diagSign, Loop_index, Loop_type, Vtype, indx, kxb, my_mpi)
    elif BKA==2:
        if DynamicCT:
            C_Pln, Pbin = sw.sample_static_Q0W0_VSHFD(fout, lmbda, lmbda_spct, p, kx, epsx, diagsG, diagSign, Loop_index, Loop_type, Vtype, indx, kxb, my_mpi)
        else:
            C_Pln, Pbin, Pbin2 = sw.sample_static_Q0W0_VSHFC(Q_external, fout, lmbda, lmbda_spct, p, kx, epsx, diagsG, diagSign, Loop_index, Loop_type, Vtype, indx, kxb, my_mpi)
    else:
        print 'ERROR Unrecognized BKA in sample_static_Q0'
        sys.exit(1)
        
    Paver=0.
    _sigma_=0.
    if my_mpi.rank == my_mpi.master:
        Paver = sum(Pbin)
        N2 = len(Pbin2)
        _Paver2_ = N2 * sqrt(sum(Pbin2*Pbin2)/N2)
        _sigma_ = sqrt( (Paver - _Paver2_)*(Paver + _Paver2_)/(N2*my_mpi.size) )
         
        tx = p.beta*(0.5+arange(p.Nt))/p.Nt
        #qx = p.cutoffq*(0.5+arange(p.Nq))/p.Nq
        savetxt('Pbin_lmbda_'+str(lmbda)+'_'+str(Norder)+'_corder_'+str(Counter_order), vstack( (tx,Pbin) ).transpose())
        if BKA:
            save('kxb', kxb)
            save('Pcof_lmbda_'+str(lmbda)+'_'+str(Norder)+'_corder_'+str(Counter_order), C_Pln )
            Paver2 = Paver
        else:
            Paver2 = C_Pln[0,0]*beta
        print >> fout, 'Result : ', Paver, Paver2, '+-', _sigma_
        print >> fout, '# time: ', datetime.datetime.now()
    Paver,_sigma_ = comm.bcast( (Paver, _sigma_), root=my_mpi.master)
    return (Paver, _sigma_)


def _ferm_(x):
    if (x>200):
        return 0.
    else:
        return 1./(exp(x)+1.)
ferm = vectorize(_ferm_)

def sample_Density(lmbda, lmbda_spct, dmu, Norder, Counter_order, Switch_Off_Interaction_Counterterm=False, Hugenholtz=True, DynamicCT=False):
    """ 
    """
    norder = Norder  # real order of diagram
    if Counter_order >= 2 : # when we deal with counter-terms (Counter_order!=0 or !=1), their order is actually smaller
        norder = Counter_order
    
    my_mpi = sw.my_mpi()
    if Parallel:
        comm = MPI.COMM_WORLD
        my_mpi.size = comm.Get_size()
        my_mpi.rank = comm.Get_rank()
        
    p = sw.params()
    p.lmbda_counter_scale = 1.
    if (Switch_Off_Interaction_Counterterm):
        p.lmbda_counter_scale =0.0
    p.Toccurence = dToccurence(Norder, Counter_order)
    p.kF      = kF
    p.beta    = beta
    p.cutoffq = cutoffq
    p.cutoffk = cutoffk
    
    p.Nitt    = int(Nittd)
    
    p.V0norm  = 40**norder/5 * 1e-3 * (10./beta)**2 * (2.1*kF)**norder
    p.V0exp   = 4.
    p.Pr      = [0.65,0.25,0.1]
    p.Qring   = Qring # 0.5
    p.dRk     = 1.3
    p.dkF     = dkF0*kF 
    p.iseed   = random.randint(2**10)+my_mpi.rank*10
    p.tmeassure = 2
    p.Ncout    = Ncout #500000
    p.Nwarm    = 10000
    p.lmbdat   = 0.0
    p.Nq       = 100
    p.Nt       = 10

    #print 'p.lmbda_counter_scale=', p.lmbda_counter_scale
    # same: kF, beta, dmu, lmbda
    dcutoff = p.cutoffk-p.cutoffq
    hf = sw.HartreeFock(p.kF, p.cutoffk+dcutoff, p.beta, lmbda, dmu)
    kx, epsx = hf.get()
    
    if (Norder==1):
        k = kx*kF
        to_integrate = k**2 * ferm( epsx*(kF**2*p.beta) )
        N0 = 2*4*pi/(2*pi)**3 * integrate.simps(to_integrate, x=k)
        return (N0, 0.0)
    
    if Hugenholtz:
        filename = sinput+'/loops_Pdiags.'+str(Norder)+'_cworder_'+str(Counter_order)+'_'
    else:
        if Counter_order==0:
            filename = sinput+'/loops_Pdiags.'+str(Norder)+'_nobubble_'
        elif Counter_order==1:
            filename = sinput+'/loops_Pdiags.'+str(Norder)+'_bubbles_'
        else:
            filename = sinput+'/loops_Pdiags.'+str(Norder)+'_corder_'+str(Counter_order)+'_'
    
    if (rank==master):
        fout = io.open('dnohup_'+str(Norder)+'_corder_'+str(Counter_order), 'ab')
        print >> fout, 'rs=', rs, 'kF=', kF, 'beta=', beta, 'MaxOrder=', MaxOrder, 'cutoffq=', cutoffq, 'cutoffk=', cutoffk, 'dmu_1=', dmu_1
    else:
        #fout = io.open('dnohup.'+str(rank)+'_'+str(Norder)+'_corder_'+str(Counter_order), 'ab')
        fout = sys.stdout
        
    (diagsG, diagSign, Loop_index, Loop_type, Loop_vertex, diagVertex, indx, Vtype) = GetPDiags(filename)
    (diagsG, diagSign, Loop_index, Loop_type, Loop_vertex, diagVertex, indx, Vtype) = KeepDensityDiagrams2(diagsG, diagSign, Loop_index, Loop_type, Loop_vertex, diagVertex, indx, Vtype)
    #print 'diagsG=', diagsG, 'diagSign=', diagSign, 'Vtype=', Vtype

    #### TEMPORARY
    #diagSign = array(diagSign[:,0])

    
    #if len(Vtype)==0:
    #    Vtype = zeros((len(diagsG),norder))
    
    if DynamicCT:
        GetCounterTermPolarization(hf, 'Pqt', Vtype, my_mpi, p, dmu, lmbda, kx, epsx) 
        (N0, sigmaN0, Ekin, sigmaEkin) = sw.sample_Density_HFD(fout, lmbda, lmbda_spct, p, kx, epsx, diagsG, diagSign, Loop_index, Loop_type, Vtype, indx, my_mpi)
    else:
        (N0, sigmaN0, Ekin, sigmaEkin) = sw.sample_Density_HFC(fout, lmbda, lmbda_spct, p, kx, epsx, diagsG, diagSign, Loop_index, Loop_type, Vtype, indx, my_mpi)
    
    if my_mpi.rank == my_mpi.master:
        # Now normalizing the curve
        #norm = (1/(2*pi)**3)**norder *  p.beta**(norder-1) # one order less, because it is set to constant
        #N0 *= norm
        #sigmaN0 *= norm
        print >> fout, 'The value of density is ', N0, 'with error', sigmaN0, 'Ekin=', Ekin, 'sigmaEkin=', sigmaEkin 
    data = array([N0,sigmaN0,Ekin,sigmaEkin])
    data = comm.bcast( data, root=my_mpi.master)
    return tuple(data)
    

def sample_dynamic(lmbda=1e-5):
    """ Using Hartree-Fock G, and the screened interaction.
        Samples polarization for any t and Q by Metropolis.
        The normalization is done by adding a self-consistently determined diagram, which should not suffers from ergodicity.
    """
    qw, tauw, Wq0, Wom0 = Get_HF_W(rs)
    
    Norder=2
    
    p = sw.params()
    p.Toccurence = dToccurence(Norder, Counter_order)
    p.kF      = kF
    p.beta    = beta
    p.cutoffq = cutoffq
    p.cutoffk = cutoffk

    p.Nitt    = int(Nitt) 

    p.V0norm  = 4**norder * 1e-2 * (10./beta)**2
    p.V0exp   = 4.
    p.Pr      = [0.65,0.25,0.1]
    p.Qring   = Qring # 0.5
    p.dRk     = 1.3
    p.dkF     = dkF0*kF
    p.iseed   = 1 #random.randint(2**31)
    p.tmeassure = 2
    p.Ncout    = 500000
    p.Nwarm    = 100000
    p.lmbdat   = 0.0
    p.Nq       = 100 # 100
    p.Nt       = 10  #10

    Nq = p.Nq #100
    Nt = p.Nt #10
    # The mesh in k where points are centered in the middle
    qx = p.cutoffq*(0.5+arange(Nq))/Nq
    tx = p.beta*(0.5+arange(Nt))/Nt
    tx = hstack( ([1e-6],tx) )

    # Calculating the Hartree-Fock dispersion
    hf = sw.HartreeFock(kF, p.cutoffk, beta,lmbda, 0.0)
    kx, epsx = hf.get()
    #kx, epsx = Get_HF_eps(rs)

    (diagsG, diagSign, Loop_index, Loop_type, Loop_vertex, diagVertex, indx, Vtype) = GetPDiags( 'input/loops_Pdiags.'+str(Norder)+'_nobubble_' )
    #print 'diagVertex=', diagVertex
    
    C_Pln, Pbin = sw.sample_dynamic(p, diagsG, diagSign, diagVertex, Loop_index, Loop_type, qw, tauw, Wq0, Wom0, kx, epsx)
    
    # Now normalizing the curve
    dq_binning = p.cutoffq/p.Nq
    dt_binning = p.beta/p.Nt
    norm = ( 1/(2*pi)**3 )**Norder * p.beta**(2*Norder-1) * (4*pi*p.cutoffq**3/3)
    
    for ls in range(shape(C_Pln)[0]):
        for l in range(shape(C_Pln)[1]):
            C_Pln[ls,l] *= norm * ((2.*l+1.)/p.cutoffq) * ((2*ls+1.)/p.beta)

    Pbin *= (norm/(dq_binning * dt_binning))

    # Evaluates Legendre Polynomials
    Pls = zeros((shape(C_Pln)[1],len(qx)))
    for l in range(shape(C_Pln)[1]):
        Pls[l,:] = special.eval_legendre(l,2*qx/p.cutoffq-1.)
    tub = zeros((len(tx),shape(C_Pln)[0]))
    for l in range(0,shape(C_Pln)[0],2):
        tub[:,l] = special.eval_legendre(l,2*tx/p.beta-1.)
    # Multiplies Legendre Polynomials with the coefficients
    Ptq = dot(dot(tub, C_Pln),Pls)


    Px = zeros((len(tx),Nq))
    Nt_need = Nt/2+2
    for iq,Qa in enumerate(qx):
        #Px[:,iq] = sw.P0_fast(Qa,tx,p.beta,kF,p.cutoffk)
        tau = tx[:Nt_need]
        Px[:Nt_need,iq] = hf.P0(Qa,tau)
        if (iq%10==0): print Qa, min(Px[:,iq])

    
    col=['b','g','r','c','m','y','k','w']
    for it in range(Nt/2+2):
        plot(qx/kF, Px[it,:], ':'+col[it%8], label='t='+str(tx[it])+' exact')
        plot(qx/kF, Ptq[it,:], '-'+col[it%8], label='t='+str(tx[it])+' leg.b.')
        if it>0:
            plot(qx/kF, Pbin[it-1,:], '.'+col[it%8], label='t='+str(tx[it])+' bin')
    grid()
    legend(loc='best')
    show()
    
    
def GetW0(rs):
    qx = loadtxt('input/q0.rs_'+str(rs))
    data = loadtxt('input/Wq0.rs_'+str(rs)).transpose()
    tau = copy(data[0,:])
    Wqm = array(data[1:,:], order='C')
    return (qx, tau, Wqm)

def Get_HF_W(rs):
    #qx = loadtxt('input/qHF.rs_'+str(rs))
    data1 = loadtxt('input/Wom0HF.rs_'+str(rs)).transpose()
    qx = copy(data1[0,:])
    Wom0 = copy(data1[1,:])
    data = loadtxt('input/WqHF.rs_'+str(rs)).transpose()
    tau = copy(data[0,:])
    Wqm = array(data[1:,:], order='C')
    return (qx, tau, Wqm, Wom0)

def Get_HF_eps(rs):
    data = loadtxt('input/ekHF.rs_'+str(rs)).transpose()
    kx = copy(data[0,:])
    epsx = copy(data[1,:])
    return (kx, epsx)



def ReadPDiagrams(filename):
    fi = open(filename, 'r')
    diagsG=[]
    Vtype=[]
    indx=[]
    s=';'
    while (s):
        try:
            s = fi.next()
            ii, s2 = s.split(' ', 1)
            indx.append( int(ii) )
            if ';' in s2:
                diag, vtyp = s2.split(';',1)
                Vtype.append( eval(vtyp) )
                diagsG.append( eval(diag) )
            else:
                diag = s2
                diagsG.append( eval(diag) )
                
        except StopIteration:
            break
    return (diagsG, indx, Vtype)

def GetPDiags(fname, BKremove=''):
    #fname = 'input/loops_Pdiags.'+str(Norder)+'_'
    fi = open(fname, 'r')
    s = fi.next()

    diagsG = []
    Loop_vertex = []
    Loop_type = []
    Loop_index = []
    diagVertex = []
    diagSign = []
    indx = []
    Vtype = []
    i=0
    while (s):
        try:
            s = fi.next()
            ii, s2 = s.split(' ', 1)
            indx.append( int(ii) )
            if ';' in s2:
                diag, vtyp = s2.split(';',1)
                Vtype.append( eval(vtyp) )
                diagsG.append( eval(diag) )
            else:
                diagsG.append( eval(s2) )
            s = fi.next()
            ii, n, lvertex = s.split(' ', 2)
            Loop_vertex.append( eval(lvertex.partition('#')[0]) )
            s = fi.next()
            ii, n, vtype = s.split(' ',2)
            Loop_type.append( eval(vtype.partition('#')[0]) )
            s = fi.next()
            ii, n, vind = s.split(' ',2)
            Loop_index.append( eval(vind.partition('#')[0]) )
            s = fi.next()
            ii, n, vind = s.split(' ',2)
            diagVertex.append( eval(vind.partition('#')[0]) )
            s = fi.next()
            ii, n, vind = s.split(' ',2)
            sgn = eval(vind.partition('#')[0])
            if type(sgn)==tuple:
                diagSign.append( sgn )
            else:
                diagSign.append( (sgn,) )
            i+=1
        except StopIteration:
            break
    if BKremove: # remove diagrams, which are not allowed in Baym-Kadanoff approach
        invalid = Find_BK_Ladders(diagsG, Vtype, BKremove)
        print 'BK-invalid=', len(invalid)
        for i in invalid[::-1]:
            del diagsG[i]
            del diagSign[i]
            del diagVertex[i]
            del indx[i]
            del Loop_index[i]
            del Loop_type[i]
            del Loop_vertex[i]
            if (len(Vtype)>0):
                del Vtype[i]
    #else:
    diagsG   = array(diagsG , dtype=int )
    diagVertex = array(diagVertex, dtype=int)
    indx = array(indx, dtype=int)
    Vtype = array(Vtype, dtype=int)
    return (diagsG, diagSign, Loop_index, Loop_type, Loop_vertex, diagVertex, indx, Vtype)

def Find_BK_Ladders(diagsG, Vtype, BKremove='right'):
    def partner(i):
        return (i/2)*2 + (1-i%2)

    if not (BKremove=='right' or BKremove=='left' or BKremove=='both'):
        print 'Unrecognized Type for BKremove=', BKremove, ' it should be one of right|left|both'
        sys.exit(1)
        
    Norder = len(diagsG[0])/2
    # First we check which diagrams have ladder on the right side.
    # Those are double-counted in BK approach, and must be removed.
    invalid_diags = []
    for id in range(len(diagsG)):
        i_diagsG = zeros(2*Norder,dtype=int)
        for i in range(2*Norder):
            i_diagsG[diagsG[id][i]]=i

        if BKremove=='right' or BKremove=='both':
            post_0 = diagsG[id][0]
            pred_0 = i_diagsG[0]
            if partner(pred_0)==post_0 :  # it has ladder as the first interaction line on the right
                if len(Vtype)>0:          # but should not be counter-term interaction
                    vtype = Vtype[id][pred_0/2]
                else:
                    vtype = 0
                if vtype==0:
                    invalid_diags.append( id ) # it is normal interaction line, hence diagram should be removed
                    continue
        
        if BKremove=='left' or BKremove=='both':
            post_1 = diagsG[id][1]
            pred_1 = i_diagsG[1]
            if partner(pred_1)==post_1 :  # it has ladder as the first interaction
                if len(Vtype)>0:          # but should not be counter-term interaction
                    vtype = Vtype[id][pred_1/2]
                else:
                    vtype = 0
                if vtype==0: invalid_diags.append( id ) # normal interaction line
                
    return invalid_diags


def KeepDensityDiagrams2(diagsG, diagSign, Loop_index, Loop_type, Loop_vertex, diagVertex, indx, Vtype):
    diagsG_=[]
    diagSign_=[]
    Loop_index_=[]
    Loop_type_=[]
    Loop_vertex_=[]
    diagVertex_=[]
    indx_=[]
    Vtype_=[]
    for i in range(len(diagsG)):
        if diagsG[i][0]==1:
            ifound=-1
            for ip in range(len(Loop_vertex[i])):
                if Loop_vertex[i][ip]==[0,1]:
                    ifound=ip
                    break
            if ifound<0:
                print 'ERROR: We could not find [0,1] in Loop_index of diagram ', diagsG[i], '! It should not happen!'
            diagsG_.append( diagsG[i] )
            diagSign_.append( diagSign[i] )
            diagVertex_.append( diagVertex[i] )
            indx_.append( indx[i] )
            if (len(Vtype)>0): Vtype_.append( Vtype[i] )
            lindex = Loop_index[i][:]
            del lindex[ifound]
            Loop_index_.append( lindex )
            ltype = Loop_type[i][:]
            del ltype[ifound]
            Loop_type_.append( ltype )
            lvertex = Loop_vertex[i][:]
            del lvertex[ifound]
            Loop_vertex_.append( lvertex)

    diagsG_   = array(diagsG_ , dtype=int )
    #diagSign_ = array(diagSign_, dtype=float)
    diagVertex_ = array(diagVertex_, dtype=int)
    indx_ = array(indx_, dtype=int)
    Vtype_ = array(Vtype_, dtype=int)
    return (diagsG_ , diagSign_, Loop_index_, Loop_type_, Loop_vertex_, diagVertex_, indx_, Vtype_)

def static_Q0_t0_first_order(lmbda, dmu, debug=False):
    cutoffq = 3*kF
    cutoffk = cutoffq + 1.2*kF
    Nt = 4
    Q = cutoffq
    
    hf = sw.HartreeFock(kF, cutoffk, beta, lmbda, dmu)
    kx, epsx = hf.get()
    tau = linspace(0+1e-5,beta-1e-5,Nt);
    
    Pq = hf.P0(Q, tau);
    if debug:
        plot(tau, Pq)
        show()
    return Pq[0]

def static_Q0_first_order(lmbda, dmu, debug=False):
    cutoffk = 3*kF
    Nt = 4
    Q = 0.001
    
    hf = sw.HartreeFock(kF, cutoffk, beta,lmbda, dmu)
    kx, epsx = hf.get()
    tau = linspace(0+1e-5,beta-1e-5,Nt);
    
    Pq = hf.P0(Q, tau);
    if debug:
        plot(tau, Pq)
        show()
    return sum(Pq)/len(Pq)

def ComputeSPCounter(lmbda, MaxOrder, n0, dmu_1=20, small=1e-6, small_error=0.05):
    
    def Density_Order_One(dmu, n0, kF, cutoffk, beta, lmbda):
        hf = sw.HartreeFock(kF, cutoffk*2, beta, lmbda, dmu)
        kx, epsx = hf.get()
        k = kx*kF
        to_integrate = k**2 * ferm( epsx*(kF**2*beta) )
        N0 = 2*4*pi/(2*pi)**3 * integrate.simps(to_integrate, x=k)
        return N0-n0

    def Ekin_Order_One(dmu, kF, cutoffk, beta, lmbda):
        hf = sw.HartreeFock(kF, cutoffk*2, beta, lmbda, dmu)
        kx, epsx = hf.get()
        k = kx*kF
        to_integrate = k**4 * ferm( epsx*(kF**2*beta) )
        Ekn = 2*4*pi/(2*pi)**3 * integrate.simps(to_integrate, x=k)
        return Ekn

    # First we compute dmu, which corresponds to the Hartree-Fock chemical potential. This is done analytically
    dmu=0.0
    Ekn0=0.0
    if rank==master: 
        fo = open('Density_order_'+str(MaxOrder)+'_lmbda_'+str(lmbda)+'.dat', 'a')
        print >> fo, '# time: ', datetime.datetime.now()
        dmu0 = dmu
        dNc0 = Density_Order_One(dmu0, n0, kF, cutoffk, beta, lmbda)
        #print >> fo, dmu0, dNc0
        for i in range(100): # Looking for change of sign
            dmu -= sign(dNc0)/dmu_1  # We need a reasonable next point to bracket zero
            dNc = Density_Order_One(dmu, n0, kF, cutoffk, beta, lmbda)
            #print >> fo, dmu, dNc
            fo.flush()
            if (dNc*dNc0 < 0) :
                ab = (dmu0,dmu)
                break
            dNc0 = dNc
            dmu0 = dmu
        dmu = optimize.brentq(Density_Order_One, ab[0],ab[1], args=(n0,kF, cutoffk, beta, lmbda) )
        dNc = Density_Order_One(dmu, n0, kF, cutoffk, beta, lmbda)
        Ekn0 = Ekin_Order_One(dmu, kF, cutoffk, beta, lmbda)
        print >> fo, '---- at order 1 dmu=', dmu, 'lmbda=', lmbda, 'n(1)-n0=', dNc, 'n0=', n0, 'Ekin_0=', Ekn0
        #print >> fo, dmu, dNc
        fo.flush()

    if Parallel:
        dmu = comm.bcast(dmu, root=0)
        Ekn0 = comm.bcast(Ekn0, root=0)
        
    # First we compute only the single-particle counter term of the order (N,N-1), which corresponds to a bubble with one dot on it.
    case = (3,2) # here lmbda_Vq is set to zero, and lmbda_spc = 1.0
    lmbda_spct=[1.]
    (Nc_0,sigmaNc_0,Ekin_0,sigmaEkin_0) = sample_Density(lmbda, lmbda_spct, dmu, case[0], case[1], True, Hugenholtz=Hugenholtz, DynamicCT=DynamicCT)
    if rank==master:
        print >> fo, "%2d %2d %s %12.8f %s %8.3g %s %12.8f %s %8.3g %s %12.8f %s %8.3g " % (case[0], case[1], 'dn=', Nc_0, '+-', sigmaNc_0, 'n-n(1)=', Nc_0, '+-', sigmaNc_0, 'Ekin=', Ekin_0, '+-', sigmaEkin_0), 'lmbda_spct=', lmbda_spct
        print >> fo, '-------------- '
        fo.flush()

    lmbda_spct=[0.]
    for norder in range(3,MaxOrder+1):
        ii=norder-3
        
        if DynamicCT or Hugenholtz:
            NCases0 = [(norder, Corder) for Corder in [0]+range(2,norder)]
        else:
            NCases0 = [(norder, Corder) for Corder in range(0,norder)]
        
        dNtot0 = 0.
        Nc_error=0.
        Ekn = Ekn0
        Ekn_error = 0.
        for case in NCases0:
            (Nc,sigmaNc,Ekin,sigmaEkin) = sample_Density(lmbda, lmbda_spct, dmu, case[0], case[1], False, Hugenholtz=Hugenholtz, DynamicCT=DynamicCT)
            dNtot0 += Nc
            Nc_error += sigmaNc
            Ekn += Ekin
            Ekn_error += sigmaEkin
            if rank==master:
                print >> fo, "%2d %2d %s %12.8f %s %8.3g %s %12.8f %s %8.3g %s %12.8f %s %8.3g %s %10.6f" % (case[0], case[1], 'dn=', Nc, '+-', sigmaNc, 'n-n(1)=', dNtot0, '+-', Nc_error, 'Ekin=', Ekn, '+-', Ekn_error, 'dEkn=', Ekin), 'lmbda_spct=', lmbda_spct
                fo.flush()
        
        lmbda_spct[ii] = -dNtot0/Nc_0
        # add the last part to satisfy charge neutrality
        (Nc, Ekin)  = (lmbda_spct[ii] * Nc_0, lmbda_spct[ii] * Ekin_0 )
        (sigmaNc, sigmaEkin) = ( abs(lmbda_spct[ii]) * sigmaNc_0, abs(lmbda_spct[ii]) * sigmaEkin_0 )
        #
        dNtot0 += Nc
        Nc_error += sigmaNc
        Ekn += Ekin
        Ekn_error += sigmaEkin
        if rank==master:
            print >> fo, "%2d %2d %s %12.8f %s %8.3g %s %12.8f %s %8.3g %s %12.8f %s %8.3g %s %10.6f" % (norder, 2, 'dn=', Nc, '+-', sigmaNc, 'n-n(1)=', dNtot0, '+-', Nc_error, 'Ekin=', Ekn, '+-', Ekn_error, 'dEkn=', Ekin), 'lmbda_spct=', lmbda_spct
            fo.flush()
            Ekn0 = Ekn # now update Ekn0 with the kinetic energy up to this order

        if norder<MaxOrder:
            lmbda_spct.append(0.0)
        
    if rank==master: 
        print >> fo, '# time: ', datetime.datetime.now()
        fo.close()
    
    return (dmu, lmbda_spct)

def ComputeChemicalPotential(dmu, lmbda, MaxOrder, n0, dmu_1=20, FIT=True, error=1e-5):
    NCases = [(norder, Corder) for norder in range(1,MaxOrder+1) for Corder in range(norder)]
    if MaxOrder>1:
        NCases.remove((2,1))
        NCases.remove((2,0))  # this one does not change the chemical potential
    xs=[]
    ys=[]
    if rank==master: 
        fo = open('Density_order_'+str(MaxOrder)+'.dat', 'a')
        print >> fo, '# time: ', datetime.datetime.now()
    for imu in range(-1,100):
        # Computing the total density with MC
        Ntot = 0.
        Nc_error=0.
        if rank==master: print >> fo, '---- start with dmu=', dmu, 'lmbda=', lmbda, 'MaxOrder=', MaxOrder, 'n0=', n0
        for case in NCases:
            (Nc,sigmaNc) = sample_Density(lmbda, dmu, case[0], case[1])
            Ntot += Nc
            Nc_error += sigmaNc
            if rank==master:
                print >> fo, case[0], case[1], Nc
                fo.flush()
        Nc_error *= 1/sqrt(len(NCases))
        if Nc_error<1e-11: Nc_error=1e-4 # on single CPU we do not have an estimate.
        if rank==master:
            print >> fo, 'at_mu= ', dmu, 'total_n=   ', Ntot, 'error_n=', Nc_error
            fo.flush()
        fi = Ntot-n0   # This is the value of the function n-n0
        xs.append(dmu) # Remember this data point 
        ys.append(fi)  #
        if imu<0:
            dmu -= sign(Ntot-n0)/dmu_1  # At the first step we need a reasonable second point
        else:
            if FIT: # Using fitting to a straight line
                # fitting the line through the points
                _xs_ = array(xs)
                _ys_ = array(ys)
                Sx = sum(_xs_)
                Sy = sum(_ys_)
                Sxx = sum(_xs_**2)
                Sxy = sum(_xs_*_ys_)
                S = len(_xs_)
                Dlt = S*Sxx-Sx**2
                # point where the line vanishes is the next approximation
                dmu = (Sx * Sxy - Sxx * Sy)/(S * Sxy - Sx*Sy)
                # error estimate of density
                error_y = Nc_error * (sqrt(abs(Sxx/Dlt)) + sqrt(abs(S/Dlt))*abs(dmu))
            else: # Probabilistic bisection
                # at i==0, we linearly interpolate bwteen the two points. At i>0, we use probabilistic bisection.
                dmu = dmu - 1./(imu+1) * fi * (xs[-2]-xs[-1])/(ys[-2]-ys[-1])
                
            diff = fabs(xs[-2]-xs[-1])
            #dmu_dn = sum([abs((xs[i]-xs[i-1])/(ys[i]-ys[i-1])) for i in range(1,len(xs))])
            #error_mu = error*dmu_1/sqrt(len(xs)) #* 30

            if rank==master: print >> fo, 'mu_new=', dmu, '(n-n0)/n0=', fi/n0, 'error_y=', error_y, 'diff_x=', diff
            
            #if ( diff < error_mu):
            if ( diff < 0.5*dmu_1*error_y and abs(fi/n0)<0.05 ):
                if rank==master:
                    print >> fo, 'dmu_best=', dmu, 'lmbda=', lmbda, 'MaxOrder=', MaxOrder, 'diff=', diff
                    print >> fo, '# time: ', datetime.datetime.now()
                    print >> fo, '#---------------------------------------------------------------------'
                    fo.close()
                return dmu
    if rank==master:
        print >> fo, 'dmu_best=', dmu, 'lmbda=', lmbda, 'MaxOrder=', MaxOrder
        fo.close()
    return dmu

        
if __name__ == '__main__':
    DynamicCT=False
    BKA=False
    Nthbin = 10
    Nlt    = 24
    Nw     = 1
    Qring  = 0.2 # before was 0.5
    dkF0   = 0.2 # before was 0.5
    execfile('params.py')
    for k in p.keys():
        #print k, p[k]
        exec( k+' = '+str(p[k]) )
    kF = (9*pi/4.)**(1./3.) /rs
    n0 = 3/(4*pi*rs**3)
    #print '----'
    #print 'rs=', rs
    #print 'kF=', kF
    #print 'n0=', n0
    #print 'T=', (1./beta)/(kF**2), 'EF'
    #print 'cutoffk=', cutoffk
    #print 'cutoffq=', cutoffq
    #print 'lmbda=', lmbda
    #print 'dmu=', dmu
    
    my_mpi = sw.my_mpi()
    if Parallel:
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()
        master=0
        my_mpi.rank = rank
        my_mpi.size = size
    #if Parallel:
    #    comm = MPI.COMM_WORLD
    #    size = comm.Get_size()
    #    rank = comm.Get_rank()
    #    master=0

    start_order = 2
    #if BKA: start_order=3
    if DynamicCT or Hugenholtz:
        NCases = [(norder, Corder) for norder in range(start_order,MaxOrder+1) for Corder in [0]+range(2,norder)]
    else:
        NCases = [(norder, Corder) for norder in range(start_order,MaxOrder+1) for Corder in range(norder)]
    if (2,1) in NCases: NCases.remove((2,1))
    
    if icase=='density' or icase=='scan':
        NCases = [(1,0)] + NCases
        for lmbda in _lmbdas_:
            if Existing_dmu:
                dmu = dmus[lmbda]
                lmbda_spct = lmbda_spcts[lmbda]
            else:
                (dmu, lmbda_spct) = ComputeSPCounter(lmbda, MaxOrder, n0, dmu_1, 5e-6, 0.13)
                continue
                
            if (rank==master):
                fo = open('order_'+str(MaxOrder)+'.dat', 'a')
                print >> fo, '# time: ', datetime.datetime.now()
                print >> fo, '# lmbda=', lmbda
            fq_tot = zeros((MaxOrder+1))
            for case in NCases:
                if case[0]==1:
                    fq = -beta * static_Q0_first_order(lmbda, dmu)
                    _sigma_ = 0.0
                else:
                    #mfq,_sigma_ = sample_static_Q0(Q_external, lmbda, lmbda_spct, dmu, case[0], case[1], BKA=BKA, Hugenholtz=Hugenholtz, DynamicCT=DynamicCT)
                    mfq,_sigma_ = sample_Q0(Q_external, lmbda, lmbda_spct, dmu, case[0], case[1], BKA)
                    fq = -mfq
                    
                fq_tot[case[0]] += fq
                if (rank==master):
                    print >> fo, '# ', case[0], case[1], fq, _sigma_
                    fo.flush()
            if (rank==master):
                print >> fo, '# total=', fq_tot.tolist()
                print >> fo, lmbda, " %17.13f "*(len(fq_tot)-1) % tuple([sum(fq_tot[:i]) for i in range(2,MaxOrder+2)])
                print >> fo, '# time: ', datetime.datetime.now()
                fo.close()
            
    elif icase=='computeP' or icase=='q_dep':
        if Existing_dmu:
            dmu = dmus[lmbda]
            lmbda_spct = lmbda_spcts[lmbda]
        else:
            (dmu, lmbda_spct) = ComputeSPCounter(lmbda, MaxOrder, n0, dmu_1, 5e-7, 0.05)

        NEW=True
        if (NEW):
            qx = linspace(1e-5,p['cutoffq'],p['Nq'])
            kxb = Give_k_mesh(p['Nk'], kF, p['cutoffk'], 0.0)
            if my_mpi.rank == 0:
                print 'using dmu=', dmu, 'lmbda=', lmbda_spct
                save('qxb', qx)

            p_vrt= {'rs':p['rs'], 'beta':p['beta'], 'lmbda':p['lmbda'], 'dmu':dmu, 'Nlt':Nlt, 'lmax':p['lmax'], 'Nq':p['Nq'], 'Nk':p['Nk'],
                    'Nt':p['Nt'], 'nom':p['nom'], 'ntail':p['ntail'], 'cutoffq':p['cutoffq'], 'cutoffk':p['cutoffk'],
                    'SaveLadder':False, 'SaveAll':False, 'Nthbin':Nthbin, 'SaveRc':False}
            import ntpvertex
            ntpvertex.ConstructVertex(p_vrt)
            
            iOm = array( load('iOm.dat.npy') ,dtype=intc)
            Nw = min(Nw, len(iOm))
            Vertex = zeros( ( Nthbin, len(kxb)-1, len(qx), Nw ), dtype=complex)
            for iq in range(len(qx)):
                V = load('Vertex.'+str(iq)+'.npy')  # Vertex[X,ik,iq,iw]
                Vertex[:,:,iq,:] = V[:,:,:Nw]
            Ker_iOm_lt = load('Ker_iOm_lt.npy')[:Nw,:]
            
        #NCases=[(3,0)]
        for case in NCases:
            if NEW:
                sample_combined_discrete(lmbda, lmbda_spct, dmu, case[0], case[1], qx, kxb, Vertex, Ker_iOm_lt, my_mpi)
                #sample_combined_discrete_heavy(lmbda, lmbda_spct, dmu, case[0], case[1], qx, kxb, my_mpi)
            else:
                sample_static_discrete(lmbda, lmbda_spct, dmu, case[0], case[1], BKA=BKA, Hugenholtz=Hugenholtz)
            #sample_static(lmbda, lmbda_spct, dmu, case[0], case[1], BKA=BKA, Hugenholtz=Hugenholtz, DynamicCT=DynamicCT, Debug=False)
    else:
        print 'Not yet implemented'
