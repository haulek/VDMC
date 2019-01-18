from scipy import *
from scipy import special
from scipy import interpolate
from pylab import *
from numpy.polynomial import legendre
import tpvert as tpv
# @Copyright 2018 Kristjan Haule and Kun Chen    

def GetAsymptoteSphBessels(nn, lmax):
    """
    Given input array of arbitrary integers, nn=[i,i+1,... i+k], corresponding to bosonic Matsubara points, 
    calculates the value of the spherical bessel functions on these points, i.e.
         j_l(nn*pi)     for l=[0,...lmax]
    It uses asymptotic expression for spherical bessel function.
    Note that this gives very precise j_l(n*pi) for n > 2*(lmax)^(8/5)
    """
    nn = array(nn)
    r1 = (-1)**nn/(nn*pi)
    r2 = (-1)**nn/(nn*pi)**2
    x2 = 1./(nn*pi)**2
    jla = zeros( (lmax+1,len(nn)) )
    for l in xrange(0,lmax+1,2): # even
        xm = l*(l+1.)/2.
        sm = xm
        for i in range(3,l+1,2):
            xm *= -(l-i+1)*(l-i+2)*(l+i-1.)*(l+i)/(4*(i-1.)*i) * x2
            sm += xm
        jla[l,:] = (-1)**(l/2) * r2 * sm
    for l in xrange(1,lmax+1,2): # odd
        xm = 1.
        sm = xm
        for i in range(2,l+1,2):
            xm *= -(l-i+1)*(l-i+2)*(l+i-1.)*(l+i)/(4*(i-1.)*i) * x2
            sm += xm
        jla[l,:] = (-1)**((l+1)/2) * r1 * sm
    return jla

def GetSphericalBessel_n_pi(iOm, lmax):
    """
    Given input integer array iOm of bosonic Matsubara points, it computes the spherical bessel functions in all 
    needed points, i.e.,
                    j_l(n*pi), where n=iOm[i]
    For small n, it calls scipy routine, while for large n, where the usual recursion is unstable, uses the 
    asymptotic expansion of bessel functions (see GetAsymptoteSphBessels )
    """
    jln = zeros( (lmax+1,len(iOm)) )
    nmax = int( 2.*(lmax)**(8./5.) )   # beyond this n we can use asimptotic expression for bessels.
    #print 'nmax=', nmax
    iOm_low  = filter(lambda n: n<nmax, iOm)
    iOm_high = array(iOm[len(iOm_low):],dtype=intc)
    # Here we use asymptoic expansion (up to the second order) for bessel functions
    #jln[:,len(iOm_low):] = GetAsymptoteSphBessels(iOm_high, lmax)
    jln[:,len(iOm_low):] = tpv.GetAsymptoteSphBessels(iOm_high, lmax)
    for ni,n in enumerate(iOm_low): # the rest are computed numerically.
        #jln[:,ni] = special.sph_jn(lmax, n*pi)[0]
        jln[:,ni] = special.spherical_jn(range(lmax+1), n*pi)
    return jln

    
def Get_LF_Ker(lmax, nOm_max, beta):
    """
    It computes the matrix of spherical bessel functions, so that we can transform from Matsubara frequency directly to
    Legendre polynomials coefficients.
    We will use the resulting Kernel as:
       Cl[2*l  ] = \sum_n LF_Ker_even[l,n] * Re(G(iOm[n]))
       Cl[2*l+1] = \sum_n LF_Ker_odd [l,n] * Im(G(iOm[n]))
       
    We derived that :
          LF_Ker_even[l,n] = [ delta(l,0)*delta(n,0) + (1-delta(n,0)) * 2*(-1)^n * j_l(n*pi) ] * (2*l+1)*(-1)^(l/2) / beta
          LF_Ker_odd [l,n] = (1-delta(n,0)) * 2 * (-1)^n j_l(n*pi) * (2*l+1) * (-1)^((l-1)/2) / beta 
    where the first is valid for l even, and the second for l odd.
    """
    nn = array(range(nOm_max))
    jln = GetSphericalBessel_n_pi(nn, lmax)
    LF_Ker_even = zeros(((lmax+2)/2,nOm_max))
    for l in range(0,lmax+1,2):
        l2 = l/2
        LF_Ker_even[l2,1:] = jln[l,1:] * 2*(-1)**nn[1:]
        if l==0: LF_Ker_even[l2,0] = 1.
        LF_Ker_even[l2,:] *= (-1)**l2 * (2*l+1.)/beta
    
    LF_Ker_odd  = zeros(((lmax+1)/2,nOm_max))
    for l in xrange(1,lmax+1,2):
        l2 = (l-1)/2
        LF_Ker_odd[l2,1:] = jln[l,1:] * 2*(-1)**nn[1:]
        LF_Ker_odd[l2,:] *= (-1)**l2 * (2*l+1.)/beta

    return (LF_Ker_even, LF_Ker_odd)

def LegendreInverseFourierBoson(Gom, iOm, lmax, beta, nom, LF_Ker_even, LF_Ker_odd):
    nOm_max = iOm[-1]+1
    #if LF_Ker_even==None or LF_Ker_odd==None:
    #    (LF_Ker_even, LF_Ker_odd) = Get_LF_Ker(lmax, nOm_max, beta)
    
    Omrest = array(range(iOm[nom-1]+1,nOm_max))
    fGom_real = interpolate.CubicSpline(iOm[nom-1:], Gom[nom-1:].real)
    Gm_real = zeros(nOm_max)
    Gm_real[:nom] = Gom[:nom].real
    Gm_real[nom:] = fGom_real( Omrest )
    fGom_imag = interpolate.CubicSpline(iOm[nom-1:], Gom[nom-1:].imag)
    Gm_imag = zeros(nOm_max)
    Gm_imag[:nom] = Gom[:nom].imag
    Gm_imag[nom:] = fGom_imag( Omrest )

    Cl_even = dot(LF_Ker_even,Gm_real)
    Cl_odd = dot(LF_Ker_odd, Gm_imag)
    Cl = zeros(lmax+1)
    for l in range(0,lmax+1,2):
        Cl[l] = Cl_even[l/2]
    for l in range(1,lmax+1,2):
        Cl[l] = Cl_odd[(l-1)/2]
    return Cl

def Get_P2Om_Transform(lmax_t, iOm, beta):
    """
    Transforms from Legendre Basis to Frequency basis. It is expressed as
    G(iOm_n) = \sum_l beta (i)^l (-1)^n j_l(n*pi) * c_l
    This function gives only the Kernel for this operation, i.e.,
    Ker(iOm_n, l) = beta * (i)^l (-1)^n j_l(n*pi)
    """
    jln = GetSphericalBessel_n_pi(iOm, lmax_t)
    Ker = zeros( (len(iOm), lmax_t+1), dtype=complex)
    for l_t in range(lmax_t+1):
        p = beta * (1j)**l_t
        for ni,n in enumerate(iOm):
            Ker[ni,l_t] = p * (-1)**n * jln[l_t,ni]
    return Ker

def GetBackFrequency(Cl, iOm):
    lmax = len(Cl)-1
    jln = GetSphericalBessel_n_pi(iOm, lmax)
    
    Gom = zeros(len(iOm), dtype=complex)
    for ni,n in enumerate(iOm):
        csum = 0j
        p = beta * (-1)**n
        i_l = 1
        for l in range(lmax+1):
            csum += Cl[l] * p * i_l * jln[l,ni]
            i_l *= 1j
        Gom[ni] = csum
    return Gom


def SimplePole(eps, lmax, beta):
    """ This transforms a simple pole-like expression for the Matsubara Green's function
      G(iOm) = 1/(iOm-eps)
      to Legendre polynomials basis.
      We derived the following relation
       C_l = -(2*l+1) i^l j_l(i*x)/(2*sh(x))
       where x = beta*eps/2
      The bessel function of imaginary argument is modified bessel-function i_l(x) = (-i)^l j_l(i*x), hence we can also write
       C_l = -(2*l+1) (-1)^l i_l(x)/(2*sh(x))

    This is useful to subtract the high frequency tails.
    """
    x = beta*eps/2.
    if abs(x)>25: # because special.sph_in(x) is an exponential function, x should not be large
        res1 = zeros(lmax+1)
        tpv.SimplePoleInside(res1,x,lmax)
        return res1
    
        #s = sign(x)  
        #sx = s*x
        #res1=zeros(lmax+1)
        #for l in xrange(lmax+1):
        #    z = 1.
        #    ak = 1.
        #    dsm = 1.
        #    for k in range(0,l):
        #        ak *= (l-k)*(l+k+1)/(2.*(k+1.))
        #        z *= -1/sx
        #        dsm += ak * z
        #    res1[l] = -0.5 * (2*l+1)*(-s)**l/x * dsm
        #return res1
    l = array(range(lmax+1))
    #iln = special.sph_in(lmax, x)[0]
    iln = special.spherical_in(range(lmax+1), x)
    res = (-1/(2*sinh(x)))*(2*l+1)*(-1)**l * iln
    return res 
    
if __name__ == '__main__':
    from tanmesh import *
    def _Gw_(n):
        wq = 1.
        iW = 2.*n*pi/beta * 1j
        return 1/(iW-wq) - 1/(iW+wq)
    Gw = vectorize(_Gw_)


    beta = 50
    nom = 2*beta
    ntail = 50
    lmax = 20

    #lmax=15
    #nn = array([154,164],dtype=intc)
    #print 'pyt', GetAsymptoteSphBessels(nn, lmax)[1,:]
    #print 'c++', tpv.GetAsymptoteSphBessels(nn, lmax)[1,:]
    #sys.exit(0)
    
    #c1 = GetAsymptoteSphBessels(range(20,30), lmax)
    #nn = array(range(20,30), dtype=intc)
    #c2 = tpv.GetAsymptoteSphBessels(nn, lmax)
    #for l in range(lmax+1):
    #    print l, c1[l,:]-c2[l,:]
    #sys.exit(0)

    print SimplePole(1.00254063672, lmax, beta)
    sys.exit(0)
    

    Nt = 100
    # time mesh
    cc=8.
    tx = GiveTanMesh(beta/(cc*Nt), beta/2., Nt)
    tau = array(hstack( (tx[:],(-tx[::-1]+beta)[1:]) ))
    
    iOm = zeros(nom+ntail,dtype=intc)
    iOm[:nom] = range(nom)
    iOm[nom:]= logspace(log10(nom+1),log10(20*nom),ntail)
    print 'nom=', nom, 'ntail=', ntail
    
    (LF_Ker_even, LF_Ker_odd) = Get_LF_Ker(lmax, iOm[-1]+1, beta) 
    Gom = Gw(iOm)
    Cl = LegendreInverseFourierBoson(Gom, iOm, lmax, beta, nom, LF_Ker_even, LF_Ker_odd)
    Cl2 = tpv.LegendreInverseFourierBoson(Gom, iOm, lmax, beta, nom, LF_Ker_even, LF_Ker_odd)
    print Cl
    print Cl2
    sys.exit(0)

    
    wOm = 2*iOm*pi/beta * 1j
    lmax=15
    gwtst = 1.0/(wOm-1.0) - 1.0/(wOm+1.)
    Cl0 = SimplePole(1.0, lmax,beta) - SimplePole(-1.0, lmax,beta) 

    Gom2 = GetBackFrequency(Cl0, iOm)
    Kr = Get_P2Om_Transform(lmax, iOm, beta)
    Gom3 = dot(Kr,Cl0)
    #print 'Gom=', GetBackFrequency(Cl0, range(154,165))
    #print 'Gom=', GetBackFrequencyDebug(Cl0, [164])
    #print 'Gom=', GetBackFrequencyDebug(Cl0, [154,164])
    
    #for ni,n in enumerate(iOm):
    #    print ni, n
    #print 'Cl0=', Cl0
    
    plot(iOm, gwtst.real, '.')
    #plot(iOm, gwtst.imag, '.')
    plot(iOm, Gom2.real, '-')
    #plot(iOm, Gom2.imag, '-')
    plot(iOm, Gom3.real, '--')
    #plot(iOm, Gom3.imag, '--')
    show()
    sys.exit(0)
    
    Gom = Gw(iOm)
    Cl = LegendreInverseFourierBoson(Gom, iOm, lmax, beta)
    print Cl

    Gt_expected = -cosh((tau-beta/2.))/sinh(beta/2.)
    Gt = legendre.legval(2*tau/beta-1., Cl)
    plot(tau, Gt)
    plot(tau, Gt_expected)
    show()
    
    Gom2 = GetBackFrequency(Cl, iOm)
    
    #plot(iOm, Gom.real, '-')
    #plot(iOm, Gom2.real, '-')
    plot(iOm, (Gom-Gom2).real, '-')
    show()
    
    
