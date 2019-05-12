#!/usr/bin/env python
# @Copyright 2018 Kristjan Haule 
import sys
import os
import time
from scipy import *
from scipy import special
from scipy import integrate
from scipy import interpolate
from scipy import optimize
from scipy import linalg
#from scipy.linalg import lapack
from pylab import *
from tanmesh import *
from Frequency2Legendre import *
from kmesh import *
import tpvert as tpv
import samplewj as sw
Parallel = True
if Parallel : from mpi4py import MPI

def jVj(kx,ky,lmax,lmbda):
    x_max = 10.
    c1,c2 = 70.,30.

    vv = tpv.jVj(kx,ky,lmax,lmbda,x_max)
    
    k1 = min(kx,ky)/sqrt(lmbda)
    k2 = max(kx,ky)/sqrt(lmbda)
    
    if k1>50. or (k1>10. and k1>0.5*k2):
        #print 'currently v=', vv.tolist()
        l_correct = int((x_max*k1-c2)/c1)
        l_correct = min(lmax,l_correct)
        #print 'l_correct=', l_correct
        for l in range(l_correct+1):  # need to correct
            x_m = (c1*l + c2)/k1
            v1 = tpv.jVj_single(kx,ky,l,lmbda,x_m)
            k1,k2 = kx/sqrt(lmbda), ky/sqrt(lmbda)
            v2 = (-special.expi(-x_m*(1-(k1-k2)*1j)) + (-1)**l * special.expi(-x_m*(1-(k1+k2)*1j)) )/(k1*k2*lmbda)
            #print 'at l=', l, 'x_m=', x_m, 'v1=', v1, 'v2=', v2.real
            vv[l] = v1 + v2.real
    return vv

def _ferm_(x):
    if (x>200):
        return 0.
    else:
        return 1./(exp(x)+1.)
ferm = vectorize(_ferm_)
def _dferm_(x):
    if abs(x)>200:
        return 0.
    ex = exp(x)
    return -1./((ex+1.)*(1./ex+1.))
dferm = vectorize(_dferm_)


def Check_Vkk():
    lmax=10
    cutoffq = 3.0*kF
    cutoffk = cutoffq + 1.0*kF + 3.0/beta
    kx = Give_k_mesh(120, kF, cutoffk)
    dh_k = array([0.5*(kx[0]+kx[1])]+[0.5*(kx[i+1]-kx[i-1]) for i in range(1,len(kx)-1)] + [0.5*(kx[-1]-kx[-3])])

    which_kx = range(0,len(kx),len(kx)/4)
    
    dt1,dt2=0,0
    Vkk = zeros((len(kx),len(kx),lmax+1))
    for ik in which_kx:
        k1 = kx[ik]
        for jk,k2 in enumerate(kx):
            tt1 = time.clock()
            jvj = jVj(k1,k2,lmax,lmbda)
            tt2 = time.clock()
            Vkk[ik,jk,:] = jvj[:]
            Vkk[jk,ik,:] = jvj[:]
            tt3 = time.clock()
            dt1 += tt2-tt1
            dt2 += tt3-tt2
            print 't1=', dt1, 't2=', dt2
        print ik

    
    Vsm = zeros( (len(kx), len(kx)) )
    for ik in which_kx:
        k1 = kx[ik]
        for jk,k2 in enumerate(kx):
            Vsm[ik,jk] += sum([Vkk[ik,jk,l]*(2*l+1) for l in range(lmax+1)])
        print ik

    col=['b','g','r','c','m','y','k','w']
    for i,ik in enumerate(which_kx):
        plot(kx, Vsm[ik,:], col[i]+'-', label='V_kk at k/kF='+str(kx[ik]/kF))
        plot(kx, 2./( (kx[ik]-kx)**2 + lmbda), col[i]+'.', label='exact Vkk at k/kF='+str(kx[ik]/kF) )
        
    legend(loc='best')
    show()

def Check_Bubble():
    lmax=10
    Nq = 40
    Nt = max(100,int(2*beta))
    nom = int(2*beta)
    ntail = 50
    cutoffq = 3.0*kF
    cutoffk = cutoffq + 1.0*kF + 3.0/beta

    # k-mesh
    kx = Give_k_mesh(120, kF, cutoffk)
    dh_k = array([0.5*(kx[0]+kx[1])]+[0.5*(kx[i+1]-kx[i-1]) for i in range(1,len(kx)-1)] + [0.5*(kx[-1]-kx[-3])])
    # q-mesh
    cc=4.
    qx = GiveTanMesh(cutoffq/(Nq*cc), cutoffq, Nq)+1e-3
    # time mesh
    cc=8.
    tx = GiveTanMesh(beta/(cc*Nt), beta/2., Nt)
    tau = array(hstack( (tx[:],(-tx[::-1]+beta)[1:]) ))
    # imaginary frequency mesh
    iOm = zeros(nom+ntail,dtype=int)
    iOm[:nom] = range(nom)
    iOm[nom:]= logspace(log10(nom+1),log10(20*nom),ntail)

    ### Now computing Bubble using old HartreeFock class
    hf = sw.HartreeFock(kF, cutoffk, beta, lmbda, dmu)
    kx_, epsx_ = hf.get()
    kx_ *= kF
    epsx_ *= kF**2
    epsf = interpolate.CubicSpline(kx_,epsx_)
    fm = ferm(beta*epsx_)
    df = fm*(1-fm) * kx_**2 * (1/pi**2)
    print 'trapz=', integrate.trapz(df, x=kx_)
    print 'simps=', integrate.simps(df, x=kx_)
    Px = zeros((len(tx),Nq))
    for iq,Qa in enumerate(qx):
        Px[:,iq] = hf.P0(Qa,tx)
        print Qa, max(Px[:,iq])

    # Now computing from the new class <Y|P|Y>
    ypy = tpv.Compute_Y_P_Y(beta, lmbda, dmu, kF, cutoffk, lmax)

    p0 = zeros( (len(qx), len(iOm)), dtype=float)
    for iq,q in enumerate(qx):
        P_p_P = zeros( ( len(kx), lmax+1, lmax+1, len(iOm) ), dtype=complex )
        for ik,k in enumerate(kx):
            pp = ypy.Run(k, q, iOm)  # pp[l1,l2,iOm]
            norm_dk = dh_k[ik]*k**2/(pi**2)
            p0[iq,:] += pp[0,0,:].real * norm_dk
            
    print 'Starting inverse fourier'
    p0t = sw.InverseFourierBoson_new(beta, tau, nom, iOm, p0, 7.0)
    
    col=['b','g','r','c','m','y','k','w']
    for i,it in enumerate(range(0,len(tau)/2+1,5)):
        plot(qx, p0t[:,it], col[i%8]+'-', label='<Y|P|Y> it='+str(it))
        plot(qx, Px[it,:], col[i%8]+'.', label='P_q(t)  it='+str(it))
    legend(loc='best')
    show()         
    

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

def ConstructVertex(p, Short=False):
    if Parallel:
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()
        master=0
    else:
        size = 1
        rank = 0
        master = 0
        
    lmax    = p['lmax']      # expansion for the angle
    lmax_t  = p['lmax_t']    # expansion order for time variable
    lmax_k  = p['lmax_k']    # expansion order for momentum variable k
    lmax_q  = p['lmax_q']    # expansion order for external momentum variable q
    Nq      = p['Nq']
    Nk      = p['Nk']
    Nt      = p['Nt']
    nom     = p['nom']
    ntail   = p['ntail']
    cutoffq = p['cutoffq']
    cutoffk = p['cutoffk']
    beta    = p['beta']
    lmbda   = p['lmbda']
    dmu     = p['dmu']     
    kF = (9*pi/4.)**(1./3.) /p['rs']
        
    small = 1e-9     # how small should be the matrix element to be ignored
    Nlast = ntail/10 # high frequency moment is computed from a fraction of last points
    
    # k-mesh
    if p.has_key('QMC_kmesh') and not p['QMC_kmesh']:
        # old-kmesh
        kx = Give_k_mesh(Nk, kF, cutoffk,1e-7)
        dh_k = array([0.5*(kx[0]+kx[1])]+[0.5*(kx[i+1]-kx[i-1]) for i in range(1,len(kx)-1)] + [0.5*(kx[-1]-kx[-2])])
        #dh_k = array([0.5*(kx[0]+kx[1])]+[0.5*(kx[i+1]-kx[i-1]) for i in range(1,len(kx)-1)] + [0.5*(kx[-1]-kx[-3])])
    else: # QMC compatible mesh of k-points. Essential for higher orders.
        # new k-mesh
        kxb = Give_k_mesh(Nk, kF, cutoffk, 0.0)
        kx = array([0.5*(kxb[i+1]+kxb[i]) for i in range(len(kxb)-1)])
        dh_k = array([kxb[i+1]-kxb[i] for i in range(len(kxb)-1)])  # now kx and dh_k is exactly compatible with QMC binning
        
    DISCRETE=False
    if os.path.isfile('qxb.npy'):
        qx = load('qxb.npy')
        DISCRETE=True
    else:
        # q-mesh
        #cc=4.
        #qx = GiveTanMesh(cutoffq/(Nq*cc), cutoffq, Nq)+1e-3
        qx = linspace(1e-3,cutoffq,p['Nq'])

    # time mesh
    cc=8.
    tx = GiveTanMesh(beta/(cc*Nt), beta/2., Nt)
    tau = array(hstack( (tx[:],(-tx[::-1]+beta)[1:]) ))
    # imaginary frequency mesh
    iOm = zeros(nom+ntail,dtype=intc)
    iOm[:nom] = range(nom)
    iOm[nom:]= logspace(log10(nom+1),log10(20*nom),ntail)

    if rank==0 and DISCRETE:
        save('iOm.dat', iOm)
        
    compareQ=False
    if compareQ:
        ### Computing Bubble using old HartreeFock class
        hf = sw.HartreeFock(kF, cutoffk, beta, lmbda, dmu)
        kx_, epsx_ = hf.get()
        kx_ *= kF
        epsx_ *= kF**2
        epsf = interpolate.CubicSpline(kx_,epsx_)
        fm = ferm(beta*epsx_)
        df = fm*(1-fm) * kx_**2 * (1/pi**2)
        print 'trapz=', integrate.trapz(df, x=kx_)
        print 'simps=', integrate.simps(df, x=kx_)
        Px = zeros((len(tx),Nq))
        for iq,Qa in enumerate(qx):
            Px[:,iq] = hf.P0(Qa,tx)
            print Qa, max(Px[:,iq])
    
    # Now computing from the new class <Y|P|Y>
    ypy = tpv.Compute_Y_P_Y(beta, lmbda, dmu, kF, cutoffk, lmax)
    
    iks,ike,sendcounts,displacements = mpiSplitArray(rank, size, len(kx) )
    dsize = len(kx)*(lmax+1)
    #print 'rank=', rank, 'iks=', iks, 'ike=', ike, 'ike-iks=', ike-iks
    tt0=time.time()
    # Now computing Vkk matrix, which now also included dk for efficiency
    Vkk_dh = zeros( (ike-iks,len(kx),lmax+1) )
    for ik,k1 in enumerate(kx[iks:ike]):
        for jk,k2 in enumerate(kx):
            jvj = jVj(k1,k2,lmax,lmbda)
            Vkk_dh[ik,jk,:] = jvj[:] * ((2./pi) * k2**2 * dh_k[jk]) # careful : here we add dh for efficiency in later use
        #print 'done ik=', ik
    print 't_{Vkk}=', time.time()-tt0
    
    if Parallel and size>1:
        if rank == 0:
            Vkk_gathered = zeros( (len(kx),len(kx),lmax+1) )
        else:
            Vkk_gathered = None
        comm.Gatherv(Vkk_dh,[Vkk_gathered,sendcounts*dsize,displacements*dsize,MPI.DOUBLE])
        if rank == 0:
            Vkk_dh = Vkk_gathered
        else:
            Vkk_dh = zeros( (len(kx),len(kx),lmax+1) )
        comm.Bcast(Vkk_dh, root=0)
    #if rank==0:
    #    for ik in range(len(kx)):
    #        savetxt('Vkk.'+str(ik), vstack( (kx/kF, Vkk_dh[ik,:,0], Vkk_dh[ik,:,1], Vkk_dh[ik,:,2], Vkk_dh[ik,:,3])).transpose() )
    
    
    gamma = zeros( ((lmax+1)*len(kx), (lmax+1)*len(kx)), dtype=complex )
    Id = identity( (lmax+1)*len(kx) ,dtype=complex)
    iOm_last = iOm[-Nlast:]                               # the last few points to find high frequency
    wOm = 2*iOm*pi/beta * 1j                              # Matsubara frequency in correct units
    (LF_Ker_even, LF_Ker_odd) = Get_LF_Ker(lmax_t, iOm[-1]+1, beta)  # constructs the matrix to transform from frequency to Legendre Coefficients

    X_qx = (2*qx[:]/cutoffq-1.)                           # for computing Legendres in qx variable

    iqs,iqe,sendcounts,displacements = mpiSplitArray( rank, size, len(qx) )
    print 'iqs:iqe=', str(iqs)+':'+str(iqe)
    
    p0 = zeros( (iqe-iqs, len(iOm)), dtype=float)  # P0 bubble
    p2 = zeros( (iqe-iqs, len(iOm)), dtype=float)  # second order diagram

    K_to_Legendre = False
    if K_to_Legendre:
        X_kx = (2*kx[:]/cutoffk-1.)                           # for computing Legendres in kx variable
        Clll = zeros( (iqe-iqs,lmax+1,lmax_t+1,lmax_k+1) )    # The array of Legendre coefficients for the vertex. This is final result
    
    RCll = zeros( (iqe-iqs,lmax+1,lmax_t+1,len(kx)) )     # The array of coefficients, but with k-variable kept on the mesh
    dt1, dt2, dt3, dt4, dt5, dt6, dt7, dt8, dt9, dt10, dt11 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

    ll = array(range(lmax+1))
    for iq,q in enumerate(qx[iqs:iqe]):
        p02 = zeros( (len(iOm), len(kx), lmax+1), dtype=complex )
        P_p_P = zeros( ( len(kx), lmax+1, lmax+1, len(iOm) ), dtype=complex )
        tt1 = time.time()
        for ik,k in enumerate(kx):
            pp = ypy.Run(k, q, iOm)  # pp[l1,l2,iOm]
            # just saving it
            P_p_P[ik,:,:,:] = pp[:,:,:]  # P_p_P[ik,l1,l2,iOm]
            # bubble
            norm_dk = dh_k[ik]*k**2/(pi**2)
            p0[iq,:] += pp[0,0,:].real * norm_dk  # bubble: p0[iq,iOm]=\sum_k P_p_P[l1=0,l2=0,iOm]*k^2*dk/(pi**2)
            # preparing for second order diagram
            for l in range(lmax+1):
                #p0_2[:, ik*(lmax+1)+l] = pp[0,l,:] * norm_dk
                p02[:,ik,l] = pp[0,l,:] * norm_dk / sqrt(2*l+1)  # p02[iOm,ik,l] = P_p_P[ik,l1=0,l2=l,iOm] /sqrt(2l+1) * k^2*dk/pi^2
        
        tt2 = time.time()
        dt1 += tt2-tt1
        
            
        Rc = zeros((len(iOm),len(kx),lmax+1),dtype=complex)
        for iw in range(len(iOm)):
            tt4 = time.time()
            tpv.AssemblyGamma(gamma, Vkk_dh, P_p_P, iw)
            tt5 = time.time()
            ## Here is an equivalet to AssemblyGamma, but much slower code
            #for ik1 in range(len(kx)):
            #    for ik2 in range(len(kx)):
            #        for l1 in range(lmax+1):
            #            for l2 in range(lmax+1):
            #                gammar[ik1*(lmax+1)+l1, ik2*(lmax+1)+l2] = Vkk_dh[ik1,ik2,l1] * P_p_P[ik2,l1,l2,iw]

            gammac = linalg.inv(Id + gamma)  # Important inversion, which takes most of the time. 

            R1 = zeros((lmax+1)*len(kx), dtype=complex)
            for ik in range(len(kx)):
                R1[:] += gammac[:,ik*(lmax+1)+0]  # this sum: \sum_k' (1+gammac)_{k,l;k',l'=0) is equivalent to closing the four leg into three leg object
            
            for ik in range(len(kx)):
                Rc[iw,ik,:] = sqrt(2*ll+1) * R1[ik*(lmax+1):(ik+1)*(lmax+1)] # sqrt(2*l+1) is added for proper normalization, so that R is normalized to P_l rather than Y_{l0}
            # Rc is the ladder in the form
            # Rc[iOm,ik,l] = \sqrt(2l+1)\sum_{k''} (I+2/pi*gamma)^{-1}_{k,l;k'',0}*dk''
            tt6 = time.time()
            # Finally, this gives the first order ladder approximation, i.e.,
            # p2[iq,iOm] = \sum_{l,k} dk*k^2/(pi^2) p_{l1=0,l2=l,k} * \sum_{k''} (I+2/pi*gamma)^{-1}_{k,l;k'',0}*dk''
            p2[iq,:] = (p02 * Rc).sum(-1).sum(-1).real # p2[iq,iw] = \sum_{ik,l} p02[iw,ik,l] * Rc[iw,ik,l]
            tt7 = time.time()
            
            dt3 += tt5-tt4
            dt4 += tt6-tt5
            dt5 += tt7-tt6
        print 'iq=', iq+iqs, 't_{Y|P|Y}=', dt1, 't_{Assembly_Gamma}=', dt3, 't_{Gamma_Inverse}=', dt4, 't_{p0-contribution}=', dt5

        # Rc[iw=0,ik,l]
        if iq==0 and iqs==0:
            Rc_minus_constant = copy(Rc[0,:,:])
            Rc_minus_constant[:,0] -= 1.0        # subtracting the constant, which subtacts the original diagram. Now vertex Rc goes to zero in infinity.
            save('Rc.00', Rc_minus_constant) # Rc[ik,l]
            save('p0.00', p2[iq,:])  # p2[iw]
            save('b0.00', p02[0,:,:]) # bubble
            
        if DISCRETE:
            #save('p2.'+str(iq+iqs), p2[iq,:])
            save('Rc.'+str(iq+iqs), Rc)
        

        if (Short): return # does not need to do anything else, because we just need p02 and Rc for scan
            
        tt8 = time.time()
        # Below we write ladders in basis of Legendre polynomials, i.e., ladders computed in k,q,l,iOm space are transformed to k,lq,l,lt basis.
        Cll = zeros( (lmax+1,lmax_t+1,len(kx)) )          # intermediate results Cll[ltheta,lt,ik]
        for ik in range(len(kx)):
            Rc[:,ik,0] -= 1.0                             # subtracting the constant, which subtacts the original diagram. Now vertex Rc goes to zero in infinity.
            for l in range(0,lmax+1):                     # Now Rc has only the correction with ladders, but not the origonal diagram.
                if max(abs(Rc[:,ik,l])) > small:          # if Rc is too small (like for l==0) we skip it
                    a_high = Rc[-Nlast:,ik,l].imag*iOm_last       # determines the high frequency tail of the imaginary part, which is propto 1/Om
                    b_high = Rc[-Nlast:,ik,l].real*iOm_last**2    # determines the high frequency tail of the real part, which is propto 1/Om^2
                    xa_high = -sum(a_high)/len(a_high) * (2*pi/beta)   # The complex high frequency is  xa/(wOm-xb)
                    xb_high = -sum(b_high)/len(b_high) * (2*pi/beta)**2/xa_high
                    Cl = zeros(lmax_t+1)
                    if ( abs(xa_high) > 1e-13 ):          # If coefficient is too small, we do not need to subtract. No high frequency tail there.
                        R_approx = xa_high/(wOm-xb_high)  # This is the tail, which is computed exactly
                        Rc[:,ik,l] -= R_approx
                        Cl = xa_high * SimplePole(xb_high, lmax_t, beta) # This computes the exact Legendre coefficients (for time) for the pole in Matsubara frequency.
                    Cl +=  tpv.LegendreInverseFourierBoson(Rc[:,ik,l], iOm, lmax_t, beta, nom, LF_Ker_even, LF_Ker_odd) # Here we transform from Matsubara frequency to Legendre Coefficients
                    Cll[l,:,ik] = Cl[:]  # Now vertex in Cll[ltheta,lt,ik]
        RCll[iq,:,:,:] = Cll[:,:,:] # RCll[iq,ltheta,lt,ik]
        tt9 = time.time()
        dt6 += tt9-tt8
        if K_to_Legendre:
            ### We will abandom that, because k-dependence is very sharp
            for l in range(lmax+1):
                for lt in range(lmax_t+1):
                    if max(abs(Cll[l,lt,:])) > small: # Projecting momentum k to Legendre coefficients
                        Clll[iq,l,lt,:] = tpv.ProjectToLegendre(X_kx, Cll[l,lt,:], lmax_k )
        tt10 = time.time()
        dt7 += tt10-tt9
        print 't_{Vertex-frequency-transform}=', dt6, 't_{Vertex-k-transform}', dt7

        # SendReceive Clll[iq,l,lt,lk]
    if Parallel and size>1:
        if rank == 0:
            p0_gathered = zeros( (len(qx), len(iOm)) )  # P0 bubble
            p2_gathered = zeros( (len(qx), len(iOm)) )  # second order diagram
            RCll_gathered = zeros( (len(qx),lmax+1,lmax_t+1,len(kx)) ) 
            if K_to_Legendre:
                Clll_gathered = zeros( (len(qx),lmax+1,lmax_t+1,lmax_k+1) )
        else:
            p0_gathered = None
            p2_gathered = None
            RCll_gathered = None
            if K_to_Legendre:
                Clll_gathered = None
        
        dsize = (lmax+1)*(lmax_t+1)*len(kx)
        comm.Gatherv(RCll,[RCll_gathered,sendcounts*dsize,displacements*dsize,MPI.DOUBLE])        
        
        dsize = len(iOm)
        comm.Gatherv(p0,[p0_gathered,sendcounts*dsize,displacements*dsize,MPI.DOUBLE])
        comm.Gatherv(p2,[p2_gathered,sendcounts*dsize,displacements*dsize,MPI.DOUBLE])
        
        if K_to_Legendre:
            dsize = (lmax+1)*(lmax_t+1)*(lmax_k+1)
            comm.Gatherv(Clll,[Clll_gathered,sendcounts*dsize,displacements*dsize,MPI.DOUBLE])
        
        if rank == 0:
            p0 = p0_gathered
            p2 = p2_gathered
            RCll = RCll_gathered
            if K_to_Legendre:
                    Clll = Clll_gathered
        else:
            p0 = zeros( (len(qx), len(iOm)), dtype=float)  # P0 bubble
            p2 = zeros( (len(qx), len(iOm)), dtype=float)  # second order diagram
            RCll = zeros( (len(qx),lmax+1,lmax_t+1,len(kx)) ) 
            if K_to_Legendre:
                Clll = zeros( (len(qx),lmax+1,lmax_t+1,lmax_k+1) )    
        comm.Bcast(p0, root=0)
        comm.Bcast(p2, root=0)
        comm.Bcast(RCll, root=0)
        if K_to_Legendre:
            comm.Bcast(Clll, root=0)
        
    tt11 = time.time()
    
    WantToDoIt=False
    if WantToDoIt:
        iks,ike,sendcounts,displacements = mpiSplitArray( rank, size, len(kx) )
        RClf = zeros( (ike-iks,lmax_t+1,lmax+1,lmax_q+1) )
        for ik,k in enumerate(kx[iks:ike]):
            for l in range(lmax+1):
                for lt in range(lmax_t+1):
                    RClf[ik,lt,l,:] =  tpv.ProjectToLegendre(X_qx, RCll[:,l,lt,iks+ik], lmax_q)
                    
        if Parallel and size>1:
            if rank == 0:
                RClf_gathered = zeros( (len(kx),lmax_t+1,lmax+1,lmax_q+1) )
            else:
                RClf_gathered = None
            dsize = (lmax_t+1)*(lmax+1)*(lmax_q+1)
            comm.Gatherv(RClf,[RClf_gathered,sendcounts*dsize,displacements*dsize,MPI.DOUBLE])
    else:
        # Projecting momentum q to Legendre coefficients for the vertex.
        RClf = zeros( (len(kx),lmax_t+1,lmax+1,lmax_q+1) )
        for ik in range(len(kx)):  
            for l in range(lmax+1):
                for lt in range(lmax_t+1):
                    RClf[ik,lt,l,:] =  tpv.ProjectToLegendre(X_qx, RCll[:,l,lt,ik], lmax_q)
    
    tt12 = time.time()
    dt8 += tt12-tt11
    print 't_{Vertex-Q-transform}=', dt8

    if rank == 0: # This is the vertex we will need for Baym-Kadanoff approach
        save('RClf',RClf)

        if DISCRETE:
            save('p2', p2)
        
        dat = vstack( (qx/kF,p2.transpose()) ).transpose() 
        savetxt('Pcof_1_corder_0.omg', dat)
        
        hf = sw.HartreeFock(kF, qx[-1], beta, lmbda, dmu)
        qxx, epsx = hf.get()
        feps = interpolate.UnivariateSpline(qxx,epsx,s=0)
        e_q = feps(qx)
        Nt = 50
        tau = linspace(0,beta,Nt)
        p2t = sw.InverseFourierBoson_new(beta, tau, p['nom'], iOm, p2, e_q, 1)
        
        dat = vstack( (qx/kF,p2t.transpose()) ).transpose()
        savetxt('Pcof_1_corder_0.tau', dat)
        
        # Now converting ladder for the first order to Legendre Coefficients
        p2_lq = zeros( (len(iOm),lmax_q+1) )
        for im in range(len(iOm)):
            p2_lq[im,:] = tpv.ProjectToLegendre(X_qx, p2[:,im], lmax_q)
        p2_ll = zeros( (lmax_t+1,lmax_q+1) )
        for lq in range(lmax_q+1):
            # Here we transform from Matsubara frequency to Legendre Coefficients
            p2_ll[:,lq] =  tpv.LegendreInverseFourierBoson(p2_lq[:,lq], iOm, lmax_t, beta, nom, LF_Ker_even, LF_Ker_odd)
        ## Alternative way
        #p2_lt = zeros( (len(qx),lmax_t+1) )
        #for iq in range(len(qx)):
        #    # Here we transform from Matsubara frequency to Legendre Coefficients
        #    p2_lt[iq,:] =  tpv.LegendreInverseFourierBoson(p2[iq,:], iOm, lmax_t, beta, nom, LF_Ker_even, LF_Ker_odd)
        #p2_ll = zeros( (lmax_t+1,lmax_q+1) )
        #for lt in range(lmax_t+1):
        #    p2_ll[lt,:] = tpv.ProjectToLegendre(X_qx, p2_lt[:,lt], lmax_q)
        savetxt('Pcof_1_corder_0.txt', p2_ll)

        if p['SaveLadder'] or p['SaveAll']:
            for iw in range(len(iOm)):
                savetxt('wb2.'+str(iw), vstack( (qx/kF, p2[:,iw])).transpose() )
        if p['SaveAll']:
            for iw in range(len(iOm)):
                savetxt('wb0.'+str(iw), vstack( (qx/kF, p0[:,iw])).transpose() )

    # saving precious data
    if rank == 0:
        if p['SaveLadder'] or p['SaveAll']:
            print 'Starting inverse fourier'
            tt13 = time.time()
            p0t = sw.InverseFourierBoson_new(beta, tau, nom, iOm, p0, 7.0)
            p2t = sw.InverseFourierBoson_new(beta, tau, nom, iOm, p2, 7.0)
            tt14 = time.time()
            dt9 += tt14-tt13
            
            savetxt('times.dat', tau)
            for it in range(len(tau)):
                savetxt('b2.'+str(it), vstack( (qx/kF, p2t[:,it])).transpose() )
        if p['SaveAll']:
            for it in range(len(tau)):
                savetxt('b0.'+str(it), vstack( (qx/kF, p0t[:,it])).transpose() )
    
    if K_to_Legendre:
        # We have the following array Cllll[lt,lq,l,lk]
        Cllll = zeros( (lmax_t+1,lmax_q+1,lmax+1,lmax_k+1) )
        for l in range(lmax+1):
            for lt in range(lmax_t+1):
                for lk in range(lmax_k+1):  # Projecting momentum q to Legendre coefficients for the vertex
                    Cllll[lt,:,l,lk] =  tpv.ProjectToLegendre(X_qx, Clll[:,l,lt,lk], lmax_q)
        if rank == 0:
            save('Cllll', Cllll)
                    
        print 'START Ctl'
        cbg = tpv.Plconv(lmax_t)
        Ctl = zeros( (lmax_q+1,lmax_t/2+1, lmax_t+1, lmax+1, lmax_k+1) )
        # Here we will perform the convolution in time on the vertex. This is not needed anymore.
        # We have input vertex in the form Cllll[lt,lq,l,lk]
        # and output vertex in Ctl[lq,lt2,lt1,l,lk] 
        #                                  remember Cllll[lq,l,lk,lt]
        Ctl[:,0,0,:,:] = Cllll[0,:,:,:]  # Ctl[lq,lt2,lt,l,lk] = Cllll[lt,lq,l,lk] 
        for lt_outside in range(2,lmax_t+1,2):
            for lt_inside in range(1,lmax_t+1):
                dsum = zeros( (lmax_q+1,lmax+1,lmax_k+1) )
                for lt1 in range(lt_inside%2, lmax_t+1, 2):
                    dsum += cbg[lt_outside,lt1,lt_inside] * Cllll[lt1,:,:,:]
                Ctl[:,lt_outside/2,lt_inside,:,:] = dsum[:,:,:]   # Ctl[lq,lt_outside/2,lt_inside,l,lk] = dsum[lq,l,lk]
        if rank == 0:
            save('Ctl', Ctl)
            print 'Ctl saved'
            

        tt15 = time.time()
        # Below we check how good is Legendre representation, by computing the lowest order diagram
        # Currently we have the form Cllll[lt,lq,l,lk]
        Rc = zeros( (len(iOm), len(qx), len(kx), lmax+1), dtype=complex)
        Ker = Get_P2Om_Transform(lmax_t, iOm, beta)
        for l in range(lmax+1):
            R = zeros((lmax_t+1,len(qx),len(kx)), dtype=complex)
            for lt in range(lmax_t+1):
                R[lt,:,:] = legendre.leggrid2d(X_qx, X_kx, Cllll[lt,:,l,:]) #  R[lt,qx,kx] <= Cllll[lt,lq,l,lk]
            for ik in range(len(kx)):
                Rc[:,:,ik,l] = dot(Ker,R[:,:,ik])  # Rc[iOm,qx,kx,l] = Ker(iOm,lt)*R(lt,qx,kx)
        
        WB2 = zeros( (len(qx), len(iOm)), dtype=complex)
        for iq in range(0,len(qx)):
            p02 = load('p02.'+str(iq)+'.npy')
            for im in range(len(iOm)):
                csum = sum(Rc[im,iq,:,:]*p02[im,:,:])
                WB2[iq,im] = csum + sum(p02[im,:,0])
        
        tt16 = time.time()
        dt10 += tt16-tt15
        print 't_{Inverse-Fourier}=', dt9, 't_{check-ladders-from-legendre-representation}=', dt10+dt11

        if rank == 0:
            for iw in range(len(iOm)):
                savetxt('nwb2.'+str(iw), vstack( (qx/kF, WB2[:,iw].real)).transpose() )

    print 't_{Y|P|Y}=', dt1, 't_{Assembly_Gamma}=', dt3, 't_{Gamma_Inverse}=', dt4, 't_{p0-contribution}=', dt5
    print 't_{Vertex-frequency-transform}=', dt6, 't_{Vertex-k-transform}', dt7, 't_{Vertex-Q-transform}=', dt8
    print 't_{Inverse-Fourier}=', dt9, 't_{check-ladders-from-legendre-representation}=', dt10+dt11

    debug=False
    if debug and rank==0:
        col=['b','g','r','c','m','y','k']
        
        Ctlq = zeros( (lmax+1, len(qx), len(kx), lmax_t/2+1, lmax_t+1) )
        for lt_outside in range(0,lmax_t+1,2):
            for lt_inside in range(lmax_t+1):
                for l in range(lmax+1):                                                                                     # Ctl[lq,lt2,lt,l,lk] = Cllll[lt,lq,l,lk] 
                    Ctlq[l,:,:,lt_outside/2,lt_inside] = legendre.leggrid2d(X_qx, X_kx, Ctl[:,lt_outside/2,lt_inside,l,:]) #  R[l,kx,qx,lti,lto] <= Ctl[lq,lto,lti,l,lk] P_{lq}(xq)*_{lk}(xk)
        
        print 'Ctlq computed'
        
        b2w = zeros( (len(qx), len(iOm)), dtype=complex)
        for iq in range(len(qx)):
            p02 = load('p02.'+str(iq)+'.npy')
            p02int = zeros( (lmax_t+1) )
            ll = array(range(lmax_t+1))
            b21 = zeros(lmax_t/2+1)
            for ik in range(len(kx)):
                for l in range(lmax+1):
                    p02int = beta/(2*ll+1.) * tpv.LegendreInverseFourierBoson(p02[:,ik,l], iOm, lmax_t, beta, nom, LF_Ker_even, LF_Ker_odd)
                    b21[:] += dot(Ctlq[l,iq,ik,:,:],p02int[:])  # b21[lt_outside/2] = Ctlq[l,iq,ik,lt_outside/2,lt1]*p02int[lt1]
            
            b22 = zeros(lmax_t+1)
            for lt_outside in range(0,lmax_t+1,2):
                b22[lt_outside] = b21[lt_outside/2]
            b22 *= (2*ll+1.)
            b2w[iq,:] = dot(Ker,b22)
            # Adding the free term
            for im in range(len(iOm)):
                b2w[iq,im] += sum(p02[im,:,0])
        
        print 'Final b2 computed'
        
        for ii,iw in enumerate(range(0,len(iOm),10)):
            dd = loadtxt('wb2.'+str(iw)).transpose()
            plot(qx/kF, b2w[:,iw].real, '--'+col[ii%7])
            plot(dd[0], dd[1], '.'+col[ii%7])
        show()
        
        #t9 = time.time()
        #dt6 += t9-t8
        #print 'dt1=', dt1, 'dt2=', dt2, 'dt3=', dt3, 'dt4=', dt4, 'dt5=', dt5, 'dt6=', dt6
        
        #print B2[0,0], B2[1,0]
        #print loadtxt('wb2.0')[0,1]
        
        for ii,iw in enumerate(range(0,len(iOm),10)):
            dd = loadtxt('wb2.'+str(iw)).transpose()
            plot(qx/kF, WB2[:,iw].real, '--'+col[ii%7])
            plot(dd[0], dd[1], '.'+col[ii%7])
        show()

    
    
if __name__ == '__main__':
    Short=False
    if os.path.isfile('params2.py'):
        execfile('params2.py')
    else:
        execfile('params.py')
    
    for k in p.keys():
        print k, p[k]
        exec( k+' = '+str(p[k]) )
    kF = (9*pi/4.)**(1./3.) /rs
    print '----'
    print 'rs=', rs
    print 'kF=', kF
    print 'T=', (1./beta)/(kF**2), 'EF'
    print 'cutoffk=', cutoffk
    print 'cutoffq=', cutoffq
    print 'lmbda=', lmbda
    if dmus is not None and dmus.has_key(lmbda):
        dmu = dmus[lmbda]
        p['dmu']=dmu
    print 'dmu=', dmu

    #Check_Vkk()
    #Check_Bubble()
    
    ConstructVertex(p, Short=Short)
    
