// @Copyright 2018 Kristjan Haule 
#include <iostream>
#include <cmath>
#include <bitset>
#include <blitz/array.h>
#include "random.h"
#include "util.h"
#include "interpolate.h"
#include "polynomial.h"
#define _TIME
#include "timer.h"
#include "vectlist.h"

//typedef std::vector<int> Tsign;                                                                                                                                                               
typedef std::vector<double> Tsign;
typedef std::vector<Tsign> TdiagSign;


#ifndef FNAME
#ifdef NO_APPEND_FORTRAN
# define FNAME(x) x
#else
# define FNAME(x) x##_
#endif
extern "C" {
  void FNAME(dger)(const int* n1, const int* n2, const double* alpha, const double* x, const int* incx, const double* y, const int* incy, double* A, const int* lda);  
}
#endif

#ifndef _SAMPLE0
#define _SAMPLE0

//using namespace std;
//using namespace blitz;
namespace bl = blitz;

inline int et(int i){
  // for equal time we will use t[3]=t[2], t[5]=t[4], ... but we should not use t[1]!=t[0]
  return i==1 ? 1 : 2*(i/2);
}

inline double Vq(double q){ return 8*pi/(q*q+1e-6);}

class ScreenedCoulombV{
  double lmbda;
public:
  ScreenedCoulombV(double _lmbda_) : lmbda(_lmbda_){};
  double operator()(double q) const { return 8*pi/(q*q+lmbda);}
};

class CounterCoulombV{
  double lmbda;
public:
  CounterCoulombV(double _lmbda_) : lmbda(_lmbda_) {};
  double operator()(double q, int iexp) const {
    double Vq = 8*pi/(q*q+lmbda);
    if (iexp==0) return Vq;
    else return Vq*ipower( Vq * lmbda/(8*pi), iexp);
  }
};


class params{
public:
  double kF, beta;        // The fermi wave vector and inverse temperature
  double Toccurence;      // How often to check that V0norm is good. Default=1
  double lmbda_counter_scale; // Should be 1.0, unless we want to scale the lambda counter term, compared to lambda in Vq(lambda). Useful for computing the chemical potential 
  double cutoffk, cutoffq;//Cutoff for dependent momentum k and independent momentum q
  int Nitt;               // Total number of MC -steps
  double V0norm, V0exp;   // The value of measuring diagram, and its exponent for falling-off
  std::array<double,3> Pr;// List of probabilities to make each of the MC steps
  int Nq, Nt;             // Number of momentum k-points and time points
  double Qring;           // How often should we chose momentum in a ring around origin versus adding dk to k
  double dRk;             // For the above ring method, how large should be step
  double dkF;             // If not ring method, how much should we change momentum dk
  int iseed;              // Random number generator seed"
  int tmeassure;          // How often to take measurements
  int Ncout;              // How often to print logging information
  int Nwarm;              // How many warm-up steps
  double lmbdat;          // coefficient to reweight time
  int Nthbin;             // Number of bins for cos(theta) in computing vertex function
  int Nlt;                // Number of Legendre Polynomials for expansion of time
  int Nlq;                // Numbef of Legendre Polynomials for expansion of external momentum q
  params() : Toccurence(1.), lmbda_counter_scale(1.0), V0norm(1), V0exp(0), Pr{1/3.,1/3.,1/3.}, 
    Qring(0.0), dRk(1.4), iseed(0), tmeassure(2), Ncout(100000), 
	     Nwarm(10000), lmbdat(3.0), Nthbin(8), Nlt(20), Nlq(18) {}
};

class egass_Gk{
public:
  double beta;
  double kF, kF2;
  egass_Gk(double _beta_, double _kF_) : beta(_beta_), kF(_kF_), kF2(_kF_*_kF_)
  {};
  double operator()(double k, double t) const
  {
    //const double Emax = 100.;
    double epsk = k*k-kF2;
    if (fabs(t)<1e-16) return 1.0/(1+exp(beta*epsk)); // for t=0, we need to use G(0^-), due to definition of interaction!!!
    if (t>0){
      //if (beta*epsk > Emax) return -exp(-t*epsk);
      //if (beta*epsk < -Emax) return -exp((beta-t)*epsk);
      return -exp(-t*epsk)/(1+exp(-beta*epsk));
    } else{
      //if (beta*epsk > Emax) return exp(-(beta+t)*epsk);
      //if (beta*epsk < -Emax) return exp(-t*epsk);
      return exp(-(beta+t)*epsk)/(1+exp(-beta*epsk));
    }
  }
  double eps(double k) const { return k*k-kF2;  }
  void debug(const std::string& filename) const{}
};

#ifndef _FERM_
#define _FERM_
inline double ferm(double x){
  if (x>700) return 0.;
  return 1./(exp(x)+1.);
}
#endif

class Gk_HF{
public:
  Spline1D<double> epsx;
  bl::Array<double,1> kx;
  double beta, kF;
public:
  void SetUp(const bl::Array<double,1>& _kx_, const bl::Array<double,1>& _epsx_){
    kx.resize(_kx_.extent(0));
    epsx.resize(_epsx_.extent(0));
    for (int i=0; i<kx.extent(0); i++){
      kx(i) = _kx_(i)*kF; // The input function was meassured in terms of kF (just convention).
      epsx[i] = _epsx_(i)*kF*kF; // The energy was measured in units of kF^2 (just convention)
    }
    epsx.splineIt(kx);
  }
  Gk_HF(double _beta_, double _kF_, const bl::Array<double,1>& _kx_, const bl::Array<double,1>& _epsx_) :
    epsx(_epsx_.extent(0)), kx(_kx_.extent(0)), beta(_beta_), kF(_kF_)
  {
    /*
    for (int i=0; i<kx.extent(0); i++){
      kx(i) = _kx_(i)*kF; // The input function was meassured in terms of kF (just convention).
      epsx[i] = _epsx_(i)*kF*kF; // The energy was measured in units of kF^2 (just convention)
    }
    epsx.splineIt(kx);
    */
    SetUp(_kx_,_epsx_);
  };
  Gk_HF(double _beta_, double _kF_) : beta(_beta_), kF(_kF_){}
  double operator()(double k, double t) const
  {
    //const double Emax = 100.;
    double eps = epsx(Interp(k,kx));
    if (fabs(t)<1e-16)
      return 1.0/(1+exp(beta*eps)); // for t=0, we need to use G(0^-), due to definition of interaction!!!
    if (t>0){
      //if (beta*eps > Emax) return -exp(-t*eps);
      //if (beta*eps < -Emax) return -exp((beta-t)*eps);
      return -exp(-t*eps)/(1+exp(-beta*eps));
    } else{
      //if (beta*eps > Emax) return exp(-(beta+t)*eps);
      //if (beta*eps < -Emax) return exp(-t*eps);
      return exp(-(beta+t)*eps)/(1+exp(-beta*eps));
    }
  }
  double eps(double k) const { return  epsx(Interp(k,kx));}
  double epsk(const intpar& ik) const {return epsx(ik);}
  void debug(const std::string& filename) const{
    std::ofstream debg(filename.c_str());
    for (int ik=0; ik<kx.extent(0); ik++){
      debg<<kx(ik)<<" "<<epsx[ik]<<std::endl;
    }
  }
};


class meassureWeight{
  /*
   The operator() returns the value of a meassuring diagram, which is a function that we know is properly normalized to unity.
   We start with the flag self_consistent=0, in which case we use a simple function : 
                 f0(k) = theta(k<kF) + theta(k>kF) * (kF/k)^dexp
   Notice that f0(k) needs to be normalized so that \int f0(k) 4*pi*k^2 dk = 1.

   If given a histogram from previous MC data, and after call to Recompute(), it sets self_consistent=1, in which case we use
   separable approximation for the integrated function. Namely, if histogram(k) ~ h(k), then f1(k) ~ h(k)/k^2 for each momentum variable.
   We use linear interpolation for function g(k) and we normalize it so that \int g(k)*4*pi*k^2 dk = 1. 
   Notice that when performing the integral, g(k) should be linear function on the mesh i*dh, while the integral 4*pi*k^2*dk should be perform exactly.
   */
  int dexp;
  double kF, cutoff, integral0, dh;
  int self_consistent, Nbin;
  bl::Array<double,2> gx;
  int i_first_momentum;
  int Nloops;
  int Noff;  // Number of off-diagonal terms included, such as R(|r_5-r_4|)*R(|r_5-r_3|)*R(|r_5-r_2|)*...
  bl::Array<double,2> gx_off;
public:
  bl::Array<double,2> K_hist;
public:
  meassureWeight(double _dexp_, double _cutoff_, double _kF_, int _Nbin_, int _Nloops_, int _i_first_momentum_=1):
    dexp(_dexp_), kF(_kF_), cutoff(_cutoff_),
    Nbin(_Nbin_), i_first_momentum(_i_first_momentum_), Nloops(_Nloops_), K_hist(2*Nloops-1,Nbin)
  {
    integral0 =  (4*pi*ipower(kF,3)/3.) * ( 1 + 3./(dexp-3.) * ( 1 - ipower(kF/cutoff,dexp-3) ) );
    self_consistent=0;
    dh = cutoff/Nbin;
    K_hist=0;
    Noff = Nloops-1-i_first_momentum;
    //Noff = 0;
  }
  template<typename real>
  double f0(real x){
    if (x>kF) return ipower(kF/x,dexp)/integral0;
    else return 1./integral0;
  }
  template<typename real>
  double f1(real x, int ik){
    const double small=1e-16;
    // we will use discrete approximation for gx(ik, x)
    int ip = static_cast<int>(x/dh);
    if (ip>=Nbin) return 0;
    double res = gx(ik,ip);
    return res > small ? res : small;
  }
  double f1_off(double x, int ik){
    const double small=1e-16;
    int ip = static_cast<int>(x/dh);
    if (ip>=Nbin) return 0;
    double res = gx_off(ik,ip);
    return res > small ? res : small;
  }
  template<typename real>
  double operator()(const bl::Array<real,1>& amomentum, const bl::Array<bl::TinyVector<real,3>,1>& momentum)
  {
    double PQ_new = 1.0;
    if (! self_consistent){
      for (int ik=i_first_momentum; ik<amomentum.extent(0); ik++)
	PQ_new *= f0( amomentum(ik) );
    }else{
      for (int ik=i_first_momentum; ik<amomentum.extent(0); ik++)
	PQ_new *= f1( amomentum(ik), ik);

      for (int ik=Nloops-2; ik>Nloops-2-Noff; --ik){
	bl::TinyVector<real,3> dk = momentum(ik)-momentum(Nloops-1);
	PQ_new *= f1_off( norm(dk), ik );
      }
    }
    return PQ_new;
  }
  // We do not literly take histogram from MC, but we transform it slightly, so that large
  // peaks become less large, and small values are less small.
  //double trs(double x){ return pow(x,2./3.);}
  //double trs(double x){ return sqrt(fabs(x));}
  double trs(double x){ return fabs(x);}
  
  double intg3D(const bl::Array<double,1>& f, double dh){
    // integrates picewise constant function (which is assumed to be constant in each interval),
    // but with weight 4*pi*k^2 f(k) dk
    double dsum=0;
    for (int i=0; i<f.extent(0); i++) dsum += f(i)*((i+1)*(i+1)*(i+1)-i*i*i);
    return dsum * dh*dh*dh * 4*pi/3.;
  }
  
  Polynomial Normalization_X1_X2(int i5, const bl::Array<double,1>& g_off, const bl::Array<double,1>& g_diag){
    // Here we calculate the following integral analytically
    //
    //     F(r5) = Integrate[ r1 g_diag(r1) g_off(u) u , {r1,0,cutoff}, {u, |r1-r5|, r1+r5} ]
    //
    //     where g_diag(r1) and g_off(u) are picewise constant functions on the equidistant mesh i*Dlt (i=0,1,2,,,N-1)
    //     Introducing continuous and descreate variables:
    //            r1 = i1 + x, with i1 integer and x=[0,1]
    //            r5 = i5 + t, with i5 integer and t=[0,1]
    //     we can write
    //
    //     F(i5,t) = \sum_{i1} Integrate[ (i1+x) g_diag[i1]  Integrate[ g_off(u) u , {u, |i5-i1+t-x|, i5+i1+t+x}], {x,0,1}]
    //
    //     which is a polynomial in t, returned by this function, and depends on input variable i5
    //
    //  To calculate this function, we need to expand it, since g_off(u) is also discrete.
    //  We have to consider several special cases:
    //  case : i1 < i5:
    //         F(i5,t) =   \sum_{i1} g_diag[i1] * Integrate[(i1+x) u g_off[i5-i1-1], {x, t, 1}, {u, i5-i1 + t-x, i5-i1}]
    //                              +g_diag[i1] * Integrate[(i1+x) u g_off[i5-i1]  , {x, t, 1}, {u, i5-i1,  i5-i1+1}]
    //                              +g_diag[i1] * Integrate[(i1+x) u g_off[i5-i1]  , {x, 0, t}, {u, i5-i1 + t-x, i5-i1+1}]
    // +\sum_{j=|i1-i5|+1}^{i1+i5-1} g_diag[i1] * Integrate[(i1+x) u g_off[j]      , {x, 0, 1}, {u, j, j+1}]
    //                              +g_diag[i1] * Integrate[(i1+x) u g_off[i5+i1]  , {x, 0,1-t},{u, i1+i5, i1+i5 + t+x}]
    //                              +g_diag[i1] * Integrate[(i1+x) u g_off[i5+i1]  , {x, 1-t,1},{u, i1+i5, i1+i5+1}]
    //                              +g_diag[i1] * Integrate[(i1+x) u g_off[i5+i1+1], {x, 1-t,1},{u, i1+i5+1,i1+i5 + t+x}]
    //  case : i1 > i5
    //         F(i5,t) =   \sum_{i1} g_diag[i1] * Integrate[(i1+x) u g_off[i1-i5-1], {x, 0, t}, {u, i1-i5 + x-t, i1-i5}]
    //                              +g_diag[i1] * Integrate[(i1+x) u g_off[i1-i5]  , {x, 0, t}, {u, i1-i5, i1-i5+1}] + 
    //                              +g_diag[i1] * Integrate[(i1+x) u g_off[i1-i5]  , {x, t, 1}, {u, i1-i5 + x-t, i1-i5+1}]
    // +\sum_{j=|i1-i5|+1}^{i1+i5-1} g_diag[i1] * Integrate[(i1+x) u g_off[j]      , {x, 0, 1}, {u, j, j+1}]
    //                              +g_diag[i1] * Integrate[(i1+x) u g_off[i5+i1]  , {x, 0,1-t},{u, i1+i5, i1+i5 + t+x}] + 
    //                              +g_diag[i1] * Integrate[(i1+x) u g_off[i5+i1]  , {x, 1-t,1},{u, i1+i5, i1+i5+1}]
    //                              +g_diag[i1] * Integrate[(i1+x) u g_off[i5+i1+1], {x, 1-t,1},{u, i1+i5+1,i1+i5 + t+x}]
    //  case : i1 == i5 and i1 != 0
    //       F(i5,t) =   \sum_{i1}   g_diag[i1] * Integrate[(i1+x) u g_off[0]       , {x, 0, 1}, {u, |t-x|, 1}]
    //          +\sum_{j=1}^{2*i5-1} g_diag[i1] * Integrate[(i1+x) u g_off[j]       , {x, 0, 1}, {u, j, j+1}]
    //                              +g_diag[i1] * Integrate[(i1+x) u g_off[2*i5]    , {x, 0,1-t},{u, i1+i5, i1+i5 + t+x}] + 
    //                              +g_diag[i1] * Integrate[(i1+x) u g_off[2*i5]    , {x, 1-t,1},{u, i1+i5, i1+i5+1}]
    //                              +g_diag[i1] * Integrate[(i1+x) u g_off[2*i5+1]  , {x, 1-t,1},{u, i1+i5+1,i1+i5 + t+x}]
    //  case : i1 == i5 == 0
    //         F(i5,t) =   \sum_{i1} g_diag[i1] * Integrate[(i1+x) u g_off[0]       , {x, 0, 1-t},{u, |t-x|, t+x}]
    //                              +g_diag[i1] * Integrate[(i1+x) u g_off[0]       , {x, 1-t, 1},{u, |t-x|, 1}]
    //                              +g_diag[i1] * Integrate[(i1+x) u g_off[1]       , {x, 1-t,1},{u, 1, t+x}]
    Polynomial Ps(5); // Final polynomial in t for F(i5,t). It can be at most forth order polynomial.
    Ps = 0;
    for (int i1=0; i1<Nbin; ++i1){
      // Stands for F0_45 = \sum_{j=|i1-i5|}^{i1+i5} Integrate[(i1+x) u g_off[j], {x, 0, 1}, {u, j, j+1}]
      //   which is F0_45 = \sum_{j=|i1-i5|}^{i1+i5} (i1+1/2) * (j+1/2) * g_off[j]
      double F0_45=0; // 
      for (int j=abs(i5-i1); j<=i5+i1; ++j){
	if (j<Nbin)
	  F0_45 += g_off(j)*(j+0.5);
      }
      F0_45 *= (i1+0.5);
      // Now we start buliding the corrections at the beginning and the end of the [|i5-i1+t-x| , i5+i1+t+x] interval
      Polynomial P1(5); P1 = 0;
      if (i1==i5 && i5==0){ // the above integrals can be computed analytically, and are
	P1[0] =  -1./4. * g_off(0) + F0_45;
	P1[1] =   2./3. * g_off(0);
	P1[2] = -1./2.  * ( g_off(0) - g_off(1) );
	P1[4] =  1./24. * ( g_off(0) - g_off(1) );
      }else{
	// Correcting the beginning of the integral, where the exact integral starts at |i5-i1+t-x|
	// and the approximate F0_45 starts at |i5-i1|
	if (i5==i1){
	  P1[0] = -(1./8.  + 1./6. * i1) * g_off(0) + F0_45;
	  P1[1] =  (1./3.  + 1./2. * i1) * g_off(0);
	  P1[2] = -(1./4.  + 1./2. * i1) * g_off(0);
	}else if (i1<i5){
	  P1[0] = (-1./8. + 0.5*i1 * (i5-i1-1) + i5/3. ) * g_off(i5-i1-1) + F0_45;
	  P1[1] = ( 1./3. - i1 * (i5-i1-1) - i5/2. ) * g_off(i5-i1-1);
	  P1[2] = 0.5 * i1 * (i5-i1) * ( g_off(i5-i1-1) - g_off(i5-i1) ) - 0.5*(i1+0.5) * g_off(i5-i1-1);
	  P1[3] = ( g_off(i5-i1-1) - g_off(i5-i1) ) * i5/6.;
	  P1[4] = ( g_off(i5-i1-1) - g_off(i5-i1) ) / 24.;
	}else{ // i1>i5
	  P1[0] =-( i1*(i1-i5+1)/2. - i5/3. + 1./8.) * g_off(i1-i5) + F0_45;
	  P1[1] = ( i1*(i1-i5+1)    - i5/2. + 1./3.) * g_off(i1-i5);
	  P1[2] =  ( g_off(i1-i5-1) - g_off(i1-i5) ) * i1*(i1-i5)/2. - 0.5 * (i1+0.5) * g_off(i1-i5);
	  P1[3] = -( g_off(i1-i5-1) - g_off(i1-i5) ) * i5/6.;
	  P1[4] = -( g_off(i1-i5-1) - g_off(i1-i5) ) / 24.;
	}
	// now correcting the end of the integral, where the exact integral stops at i5+i1+t+x
	// and the approximate integrals stops at i5+i1+1
	if (i1+i5<Nbin){
	  P1[0] += -( i1*(i1+i5+1) + i5/3. + 1./4.) * 0.5 * g_off(i1+i5);
	  P1[1] +=  ( i1*(i1+i5+1) + i5/2. + 1./3.) *       g_off(i1+i5);
	  P1[2] += -( (i1+1)*(i1+i5+1) - (i1+0.5) ) * 0.5 * g_off(i1+i5);
	  P1[3] +=  g_off(i1+i5) * i5/6.;
	  P1[4] +=  g_off(i1+i5) / 24.;
	  if (i1+i5+1<Nbin){
	    P1[2] +=  (i1+1)*(i1+i5+1) * 0.5 * g_off(i1+i5+1); 
	    P1[3] += -g_off(i5+i1+1) * i5/6.;
	    P1[4] += -g_off(i5+i1+1) / 24.;
	  }
	}
      }
      P1 *= g_diag(i1);  // Need to multiply with the diagonal function g_diag[i1]
      Ps   += P1;        // Finally, sums polynomials over i1
    }
    return Ps;
  }
  /*
  template<typename real>
  void Recompute(const bl::Array<real,2>& K_hist, const bl::Array<real,2>& T_hist, bool SaveData=true)
  {
    int Ndim = K_hist.extent(0);
    int Ntime= T_hist.extent(0);
    if (K_hist.extent(1) != Nbin)
      std::cerr << "Wrong dimension of K_hist. Should be " << Nbin << " but is " << K_hist.extent(1) << std::endl;
    if (T_hist.extent(1) != Nbin)
      std::cerr << "Wrong dimension of T_hist. Should be " << Nbin << " but is " << T_hist.extent(1) << std::endl;
      
    gx.resize(Ndim,Nbin);
    self_consistent=1;
    // first smoothen the histogram, by averaging over three points. Then transform function so that large variations are removed.
    for (int ik=i_first_momentum; ik<Ndim; ik++){ 
      gx(ik,0) = trs( 0.5*(K_hist(ik,0)+K_hist(ik,1)) );
      for (int i=1; i<Nbin-1; i++)
	gx(ik,i) = trs( (K_hist(ik,i-1)+K_hist(ik,i)+K_hist(ik,i+1))/3. );
      gx(ik,Nbin-1) = trs( 0.5*(K_hist(ik,Nbin-2)+K_hist(ik,Nbin-1)) );

      double norm = 1/intg3D(gx(ik,bl::Range::all()), dh);
      for (int i=0; i<Nbin; i++) gx(ik,i) *= norm;
    }

    if (SaveData){
      for (int ik=i_first_momentum; ik<Ndim; ik++){
	std::ofstream out( (std::string("meassure_weight.")+std::to_string(ik)).c_str() );
	for (int i=0; i<Nbin; i++){
	  out<< dh*(i+0.5) << " "<< f1(dh*(i+0.5),ik) << " "<<f0(dh*(i+0.5)) <<std::endl;
	}
	out.close();
      }
    }
  }
  */
  void Recompute(bool SaveData=true)
  {
    //int Ndim = K_hist.extent(0);
    if (K_hist.extent(1) != Nbin)
      std::cerr << "Wrong dimension of K_hist. Should be " << Nbin << " but is " << K_hist.extent(1) << std::endl;
      
    gx.resize(Nloops,Nbin);
    gx_off.resize(Nloops-1,Nbin);
    self_consistent=1;
    // first smoothen the histogram, by averaging over three points. Then transform function so that large variations are removed.
    for (int ik=i_first_momentum; ik<Nloops; ik++){ 
      gx(ik,0) = trs( 0.5*(K_hist(ik,0)+K_hist(ik,1)) );
      for (int i=1; i<Nbin-1; i++)
	gx(ik,i) = trs( (K_hist(ik,i-1)+K_hist(ik,i)+K_hist(ik,i+1))/3. );
      gx(ik,Nbin-1) = trs( 0.5*(K_hist(ik,Nbin-2)+K_hist(ik,Nbin-1)) );

      double norm = 1/intg3D(gx(ik,bl::Range::all()), dh);
      for (int i=0; i<Nbin; i++) gx(ik,i) *= norm;
    }

    // Off-Diagonal first round of normalization
    for (int ik=0; ik<Nloops-1; ik++){
      int iik = ik+Nloops;
      gx_off(ik,0) = trs( 0.5*(K_hist(iik,0)+K_hist(iik,1)) );
      for (int i=1; i<Nbin-1; i++)
	gx_off(ik,i) = trs( (K_hist(iik,i-1)+K_hist(iik,i)+K_hist(iik,i+1))/3. );
      gx_off(ik,Nbin-1) = trs( 0.5*(K_hist(iik,Nbin-2)+K_hist(iik,Nbin-1)) );
    }
    for (int ik=0; ik<Nloops-1; ik++){
      double sm = sum(gx_off(ik,bl::Range::all()));
      double norm=1;
      if (sm!=0){
	norm = Nbin/sm;
      }else{
	cout<<"ERROR : The sum is zero, which should not happen!"<<endl;
      }
      gx_off(ik,bl::Range::all()) *= norm; // We make this off_diagonal function normalized so that each value is on average 1.0
      /*
      for (int i=0; i<Nbin; i++)
	if (std::isnan(gx_off(ik,i)))
	  cout<<"1 ERROR and entry in gx_off("<< ik <<","<<i<<") is nanx"<<endl;
      */
    }
    
    // Temporarily, just for debugging
    //gx_off(Range::all(),Range::all()) = 1.0;

    // We need to calculate the following integral
    //
    //  Norm = \int d^3r1...d^3r5  g5(r5) * g1(r1)*g15(|\vr1-\vr5|) * g2(r2)*g25(|\vr2-\vr5|) * ...* g4(r4)*g45(|\vr4-\vr5|)
    // 
    //  This can be turned into radial and angle integral. The important property is that angle between \vr_i-\vr_j| appears in a single term
    //  hence each term can be independetly integrated over phi, and over cos(theta_{ij}) = x_i
    //  We get the following result
    //
    //  Norm = 2*(2*pi)^5 Integrate[ r5^{2-4}*g5(r5) *
    //                               * Integrate[ r1*g1(r1)*u*g15(u) , {r1,0,cutoff}, {u, |r1-r5|, r1+r5} ]
    //                               * Integrate[ r2*g2(r2)*u*g25(u) , {r2,0,cutoff}, {u, |r2-r5|, r2+r5} ]
    //                               * Integrate[ r3*g3(r3)*u*g35(u) , {r3,0,cutoff}, {u, |r3-r5|, r3+r5} ]
    //                               * Integrate[ r4*g4(r4)*u*g45(u) , {r4,0,cutoff}, {u, |r4-r5|, r4+r5} ]
    //                               , {r5, 0, cutoff}]
    //
    //     In the above function "Normalization_X1_X2" we compute polynomial for
    //
    //     F( r5=(i5,t) ) = Integrate[ r1 g_diag(r1) g_off(u) u , {r1,0,cutoff}, {u, |r1-r5|, r1+r5} ]
    //
    //     where g_diag(r1) and g_off(u) are picewise constant functions on the equidistant mesh i*Dlt (i=0,1,2,,,N-1)
    //     Here
    //            r1 = i1 + x, with i1 integer and x=[0,1]
    //            r5 = i5 + t, with i5 integer and t=[0,1]
    //     and therefore
    //
    //     F(i5,t) = \sum_{i1} Integrate[ (i1+x) g_diag[i1]  Integrate[ g_off(u) u , {u, |i5-i1+t-x|, i5+i1+t+x}], {x,0,1}]
    //  The normalization is than
    //
    //  Norm = (4*pi)^5/2^4 Integrate[ r5^{2-4}*g5(r5) * F_{g1}(i5,t) * F_{g2}(i5,t) * F_{g3}(i5,t) * F_{g4}(i5,t), {r5, 0, cutoff}]
    //
    //  Norm = (4*pi*Dlt)^5/2^4 \sum_{i5} Integrate[ (i5+t)^{2-4} * g5[i5] * F_{g1}(i5,t) * F_{g2}(i5,t) * F_{g3}(i5,t) * F_{g4}(i5,t), {t,0,1}]
    //  
    {
      double dsum=0;
      for (int i5=0; i5<Nbin; ++i5){
	Polynomial Pt(1);  // Since there are Noff 4-th order polynomials, the order is Noff*4.
	Pt=0;     // Will hold the product F * F * F ...
	Pt[0]=1; // Starting with identity
	for (int ik=Nloops-2; ik>Nloops-2-Noff; --ik){
	  Polynomial Ps = Normalization_X1_X2( i5, gx_off(ik,bl::Range::all()), gx(ik,bl::Range::all()) );
	  Pt *= Ps;
	}
	Pt.CheckIsLegitimate();
	// Computes the integral Integrate[ (i5+t)^{2-Noff} * Pt(t), {t,0,1}]
	double Pt_Int = Pt.CmpIntegral(i5,2-Noff);
	if (std::isnan(Pt_Int))
	  cout<<"ERROR Pt_Int is nan at i5="<<i5<<" "<<endl;
	dsum += Pt_Int * gx(Nloops-1,i5);
      }

      dsum *= ipower( 4*pi*dh*dh*dh, Noff+1)/ipower(2.,Noff);
      double nrm = 1.0;
      if (dsum!=0 && Noff>0) nrm = 1./std::pow(fabs(dsum),1./Noff);

      gx_off(bl::Range::all(),bl::Range::all()) *= nrm;

      for (int ik=0; ik<Nloops-1; ik++)
	for (int i=0; i<Nbin; i++)
	  if (std::isnan(gx_off(ik,i)))
	    cout<<"ERROR and entry in gx_off("<< ik <<","<<i<<") is nanx"<<endl;
    }

    if (SaveData){
      for (int ik=i_first_momentum; ik<Nloops; ik++){
	std::ofstream out( (std::string("measure_weight.")+to_string(ik)).c_str() );
	for (int i=0; i<Nbin; i++){
	  out<< dh*(i+0.5) << " "<< f1(dh*(i+0.5),ik) << " "<<f0(dh*(i+0.5)) <<endl;
	}
	out.close();
      }
      for (int ik=0; ik<Nloops-1; ik++){
	std::ofstream out( (std::string("measure_weight.")+to_string(ik+Nloops)).c_str() );
	for (int i=0; i<Nbin; i++){
	  out<< dh*(i+0.5) << " "<< f1_off(dh*(i+0.5),ik) <<endl;
	}
	out.close();
      }
    }
  }
  template<typename real>
  void Add_to_K_histogram(double dk_hist, const bl::Array<bl::TinyVector<real,3>,1>& momentum, double cutoffq, double cutoffk)
  {
    if (i_first_momentum>0){
      double Q = norm(momentum(0));
      // external variable histogram
      if (Q<cutoffq){
	int iik = static_cast<int>(Q/cutoffq * Nbin);
	if (iik>=Nbin) iik=Nbin-1;
	K_hist(0,iik) += dk_hist;
      }
    }
    // histogram of other momenta, which we integrate over
    for (int ik=i_first_momentum; ik<Nloops; ik++){
      double k = norm(momentum(ik));
      if (k<cutoffk && k>1e-150){
	int iik = static_cast<int>(k/cutoffk * Nbin);
	if (iik>=Nbin) iik=Nbin-1;
	K_hist(ik,iik) += dk_hist/(k*k);
      }
    }
    // histogram for variable differences. We choose to use
    // the following combination of momenta
    //  |k_0-k_{N-1}|,  |k_1-k_{N-1}|,  |k_2-k_{N-1}|, ...., |k_{N-2}-k_{N-1}|
    //
    for (int ik=0; ik<Nloops-1; ik++){
      bl::TinyVector<real,3> dkv = momentum(ik)-momentum(Nloops-1);
      double dk = norm(dkv);
      if (dk < cutoffk && dk>1e-150 ){
	int iik = static_cast<int>(dk/cutoffk * Nbin);
	if (iik>=Nbin) iik=Nbin-1;
	double dd = dk_hist/(dk*dk);
	K_hist(ik+Nloops,iik) += dd;
      }
    }
  }
  double Normalize_K_histogram(){
    // We can get overflow during MPI for a long run. We should use a constant for normalization.
    // We could here normalize K_hist (with knorm), and than when we measure, we would
    // add instead of adding unity, we would add 1./knorm
    double dsum=0;
    for (int ik=0; ik<Nloops; ik++)
      for (int i=0; i<Nbin; i++)
	dsum += K_hist(ik,i);
    double dnrm = Nloops/dsum;
    if (std::isnan(dnrm)) cout<<"ERROR : Normalize_K_histogram encounter nan"<<endl;
    K_hist *= dnrm;
    return dnrm;
  }
};

template <class GK>
long double MoveVertex(double t2, double t1, double t1i, double t1o, double k1i, double k1o, const GK& Gk)
{
  /*
    Returns the following ratio:

    Gk(k1i,t2-t1i)*Gk(k1o,t1o-t2)
    -----------------------------
    Gk(k1i,t1-t1i)*Gk(k1o,t1o-t1)
  */
  double beta = Gk.beta;
  double ek1i = Gk.eps(k1i), ek1o = Gk.eps(k1o);
  double exp_ratio1a = -(t2-t1)*(ek1i-ek1o);
  double exp_ratio1b = 0;
  int sign1=1;
  if (t1i==t1) t1i = t1 + 3e-16*abs(t1);
  if (t1o==t1) t1o = t1 - 3e-16*abs(t1);
  if (t1>t1o && t1o>t2){ sign1*=-1; exp_ratio1b +=  ek1o;}
  if (t1<t1o && t1o<=t2){ sign1*=-1; exp_ratio1b += -ek1o;}
  if (t1>t1i && t1i>=t2){ sign1*=-1; exp_ratio1b += -ek1i;}
  if (t1<t1i && t1i<t2){ sign1*=-1; exp_ratio1b +=  ek1i;}
  double earg = exp_ratio1a + exp_ratio1b * beta;
  //return sign1 * exp(exp_ratio1a + exp_ratio1b * beta);
  /*
  if (earg < 709){
    return sign1 * exp(earg);
  }else{
    return 0; //sign1*numeric_limits<double>::max();
  }
  */
  if (earg < 11356){
    return sign1 * exp(static_cast<long double>(earg));
  }else{
    return 0; //sign1*numeric_limits<double>::max();
  }
}
void ComputeVertexCompanion(bl::Array<int,2>& vertex_companion, bl::Array<int,2>& vertex_incoming, bl::Array<int,3>& diagVertex, bl::Array<int,2>& is_companion, const bl::Array<int,2>& diagsG, const bl::Array<int,2>& i_diagsG)
{
  // First we go through all diagrams and check for type-two vertices. When we find a type-2 vertex, we find the companion vertex,
  // which is the vertex that shares a fermionic propagator with the type-two vertex. Then we check if the companion is preceeding
  // the type-2 vertex, or, it comes latter (vertex_incoming). We also change the type to companion from 1 to 3, because such vertex
  // can not be evaluated independently from the current type-two vertex. The two should always be computed together, because the
  // result is not a simple product of two independet terms.
  int Ndiags= diagsG.extent(0);
  int Norder= diagsG.extent(1)/2;
  vertex_companion=0;
  vertex_incoming=0;
  is_companion=0;
  for (int id=0; id<Ndiags; id++){
    for (int ii=1; ii<Norder; ii++){
      int itype = diagVertex(id,ii-1,1);
      if (itype==2){
	int i     = diagVertex(id,ii-1,0);
	int i_m = i_diagsG(id,i);
	int i_p = diagsG(id,i);
	for (int jj=1; jj<Norder; jj++){
	  if (diagVertex(id,jj-1,0)==i_m){
	    vertex_companion(id,ii-1)=jj;
	    diagVertex(id,jj-1,1) = 3; // changing type of companion to three
	    vertex_incoming(id,ii-1) = 1;
	    is_companion(id,i_m) = i;     // It says that i_m is companion of type-2 vertex, and we have now index for type-2 vertex, which is >0.
	    break;
	  }
	  if (diagVertex(id,jj-1,0)==i_p){
	    vertex_companion(id,ii-1)=jj;
	    diagVertex(id,jj-1,1) = 3; // changing type of companion to three
	    vertex_incoming(id,ii-1) = 0;
	    is_companion(id,i_p) = i;    // It says that i_p is not companion of type-2 vertex, and it gives the number of this type-2 vertex, which is >0.
	    break;
	  }
	  if (jj==Norder-1)
	    std::cerr<<"WARNING : It seems I could not find partner to type two Vertex, which should not happen. Is the vertex "<<i<<" in diag="<<id<<" really vertex type 2?"<<std::endl;
	}
      }
    }
  }
}

class Reweight_time{
  /* Reweighting the time variable with the function
        wt(t) ~ cosh((beta-2*t)*lambda/2) ~ exp(-lmbda*t) + exp(-(beta-t)*lmbda)
      Its normalized version is:
        wt(t) = cosh((beta-2*t)*lmbda/2) / sinh(beta*lmbda/2) * lmbda/2
      Its integral gives the function to generate random numers
       Int[w(t),{t,0,t}] == x  =>  
       x(t) =  beta/2 + asinh( (2*x-1)*sinh(beta*lmbda/2) )/lmbda
      The weight can also be expressed in terms of x, i.e.,
       w(t(x)) == wx(x) = sqrt[(1-2*x)^2 + 1/sinh(beta*lmbda/2)^2] * lmbda/2
   */
public:
  double beta, lmbda, r, lmbd2, q;
  Reweight_time(double _beta_, double _lmbda_) : beta(_beta_), lmbda(_lmbda_)
  {
    if (lmbda!=0){
      r = sinh(beta*lmbda/2);
      q = lmbda/(2*r);
    }else{
      r = 1.0;
      q = 1.0;
    }
    //lmbd2 = 2*lmbda;
  }
  double wt(double t)
  {
    if (lmbda==0) return 1.0;
    return q * cosh((beta-2*t)*lmbda/2);
  }
  double gtau(double x)
  {
    if (lmbda==0) return x*beta;
    return beta/2. + asinh(r*(2*x-1))/lmbda;
  }
  double wx(double x)
  {
    if (lmbda==0) return 1.0;
    return sqrt(fabs(1/(r*r) + (1-2*x)*(1-2*x)))*lmbda/2;
  }
};


template<typename real>
inline bool Find_new_k_point(bl::TinyVector<real,3>& k_new, real& ka_new, double& trial_ratio, const bl::TinyVector<real,3>& k, real ka, double cutoffk, RanGSL& drand, bool Qring, double dkF, double dRk)
{
  if (Qring){ // tryng new k-point in a ring with radius ka/dRk...ka*dRk
    ka_new = ka * (1.0/dRk + drand()*(dRk-1.0/dRk)); /// interval [ka/dRk,....,ka*dRk]
    if (ka_new > cutoffk){
      return false;  // outside the cutoff
    }else{              // new drawing the random angle in 3D
      double th = pi*drand(), phi = 2*pi*drand();
      double ka_sin_th = ka_new*sin(th);
      k_new = ka_sin_th*cos(phi), ka_sin_th*sin(phi), ka_new*cos(th);
      double ka_sin_old = (ka>0) ? sqrt(k[0]*k[0]+k[1]*k[1]) : 1e-100;
      trial_ratio = ka_sin_th/ka_sin_old; // k_new^2*sin(th_new)*(k_old*dRk-k_old/dRk)/(k_old^2*sin(th_old)*(k_new*dRk-k_new/dRk))
    }
  }else{          // trying new k-point by adding random dk to k=k+dk
    bl::TinyVector<real,3> dk( (2*drand()-1.)*dkF, (2*drand()-1.)*dkF, (2*drand()-1.)*dkF );
    k_new = k+dk;
    ka_new = norm(k_new);
    if (ka_new > cutoffk)
      return false;  // outside the cutoff
    else
      trial_ratio= 1.0;
  }
  return true;
}
template<typename real>
inline bool Find_new_Q_point(bl::TinyVector<real,3>& Q_new, real& Qa_new, double& trial_ratio, const bl::TinyVector<real,3>& Q, real Qa, double cutoffq, RanGSL& drand)
{
  Qa_new = cutoffq*drand();
  double th = pi*drand(), phi = 2*pi*drand();
  double sin_th = sin(th);
  double Qa_sin_th = Qa_new*sin_th;
  Q_new = Qa_sin_th*cos(phi), Qa_sin_th*sin(phi), Qa_new*cos(th);
  double sin_th_old = (Qa>0) ? sqrt(Q[0]*Q[0]+Q[1]*Q[1])/Qa : 1e-100;
  trial_ratio = sin_th/sin_th_old;
  return true;
}

template<typename real>
inline bool Find_new_discrete_Q_point(int& iiQ, bl::TinyVector<real,3>& Q_new, real& Qa_new, double& trial_ratio,
				     const bl::TinyVector<real,3>& Q, real Qa, const bl::Array<double,1>& qx, RanGSL& drand)
{
  //Qa_new = cutoffq*drand();
  iiQ = drand()*qx.extent(0);
  Qa_new = qx(iiQ);
  double th = pi*drand(), phi = 2*pi*drand();
  double sin_th = sin(th);
  double Qa_sin_th = Qa_new*sin_th;
  Q_new = Qa_sin_th*cos(phi), Qa_sin_th*sin(phi), Qa_new*cos(th);
  double sin_th_old = (Qa>0) ? sqrt(Q[0]*Q[0]+Q[1]*Q[1])/Qa : 1e-100;
  trial_ratio = sin_th/sin_th_old;
  return true;
}

template<typename real>
inline bool Find_new_zQ_point(bl::TinyVector<real,3>& Q_new, real& Qa_new, double& trial_ratio, const bl::TinyVector<real,3>& Q, real Qa, double cutoffq, RanGSL& drand)
{
  Qa_new = cutoffq*drand();
  Q_new = 0, 0, Qa_new;
  //double th = pi*drand(), phi = 2*pi*drand();
  //double sin_th = sin(th);
  //double Qa_sin_th = Qa_new*sin_th;
  //Q_new = Qa_sin_th*cos(phi), Qa_sin_th*sin(phi), Qa_new*cos(th);
  //double sin_th_old = (Qa>0) ? sqrt(Q[0]*Q[0]+Q[1]*Q[1])/Qa : 1e-100;
  //trial_ratio = sin_th/sin_th_old;
  trial_ratio = 1.0;
  return true;
}

class BitArray{
  std::bitset<8> *f;
  int N0, N;
public:
  BitArray(int _N_) : N(_N_)
  {
    N0 = (N%8==0) ? N/8 : N/8+1;
    f = new std::bitset<8>[N0];
  }
  int operator[](int i) const{
    return f[i/8][i%8];
  }
  void set(int i, int value){
    if (i>=0 && i<N)
      f[i/8].set(i%8,value);
  }
  void operator=(int value){
    if (value){ // all set to unity
      for (int j=0; j<N0; j++) f[j].set();
    }else{  // all set to zero
      for (int j=0; j<N0; j++) f[j].reset();
    }
  }
  ~BitArray()
  { delete []f; f=NULL;}
  int size() const {return N;}

  friend std::ostream& operator<< (std::ostream& stream, const BitArray& r);
};

std::ostream& operator<< (std::ostream& stream, const BitArray& r)
{
  for (int i=0; i< r.size(); i++) stream << r[i] << ",";
  stream << std::endl;
  return stream;
}

template<typename sint>
void Get_GVind(bl::Array<bl::Array<unsigned short,1>,1>& loop_Gkind, bl::Array<bl::Array<char,1>,1>& loop_Gksgn,
	       bl::Array<bl::Array<unsigned short,1>,1>& loop_Vqind, bl::Array<bl::Array<char,1>,1>& loop_Vqsgn,
	       int& Ngp, int& Nvp,
	       const std::vector<std::vector<std::vector<int> > >& Loop_index, const std::vector<std::vector<std::vector<int> > >& Loop_type,
	       const bl::Array<sint,2>& Gindx, const bl::Array<sint,2>& Vindx,
	       const bl::Array<int,1>& single_counter_index, const std::vector<double>& lmbda_spct,
	       bl::Array<char,2>& Vtype,
	       bool dynamic_counter=false, ostream& log=std::cout)
{/*  Given all the loops in all diagrams (Loop_indx and Loop_type) it calculates which propagtaors are
     changed when a single loop is being modified. Each loop carries a momentum k_i, and changing it requires
     one to modifiy all fermionic propagators stored in loop_Gkind(k_i,:). The momentum of each of these
     propagator contains either +k_i or -k_i, and such sign of momentum is contained in loop_Gksgn(k_i,:).
     Finally, Ngp is the maximum number of fermionic propagators that need to be changed when a single loop of 
     momentum is changed.
     For interaction, the corresponding quantities are loop_Vqind, loop_Vqsgn, and Nvp.
     Note that due to single-particle counter terms, Vq might need to be changed even when it does not contain particular loop. This is because 
     We define an effective V_q, which contains both the two-particle counter term, and the single-particle counter term, and the latter depends on
     density n_{k+q} of the nerby G-propagator.
  */
  int Ndiags = Loop_index.size();
  int Nloops = Loop_index[0].size();

  double sum_lmbda_spct=0.0;
  for (int i=0; i<lmbda_spct.size(); ++i) sum_lmbda_spct += lmbda_spct[i]*lmbda_spct[i];
  sum_lmbda_spct = sqrt(sum_lmbda_spct);
  const double small = 1e-10;
  //log<<"single_counter_index=" << single_counter_index << endl;
  // When single-particle counter terms is added, we first find inverse index for single_counter_index, which gives index to n_{k+q}.
  std::map<int,deque<int> > single_counter_index_inverse;
  if (sum_lmbda_spct>small){
    for (int ii_v=0; ii_v<single_counter_index.extent(0); ++ii_v){
      if (single_counter_index(ii_v)>=0){
	int ii_g = single_counter_index(ii_v);
	single_counter_index_inverse[ii_g].push_back(ii_v);
      }
    }
  }
  
  loop_Gkind.resize(Nloops);
  loop_Gksgn.resize(Nloops);
  loop_Vqind.resize(Nloops);
  loop_Vqsgn.resize(Nloops);
  Ngp=0; Nvp=0;
  for (int iloop=0; iloop<Nloops; iloop++){
    std::map<int,char> _loop_Gk_sign_, _loop_Vq_sgn_;
    for (int id=0; id<Ndiags; id++){ // updating momentum in all propagators
      const std::vector<int>& lindex = Loop_index[id][iloop];
      const std::vector<int>& ltype  = Loop_type[id][iloop];
      for (int i=0; i<lindex.size(); i++){
	int ltype_i  = ltype[i];
	int lindex_i = lindex[i];
	if ( abs(ltype_i)==1 ){
	  int ii = Gindx(id, lindex_i);
	  if (ii==std::numeric_limits<sint>().max()) continue; // When computing Density, we remove 0->1 propagator, and therefore ii=max, and should be removed
	  int isign = sign(ltype_i);
	  auto iter = _loop_Gk_sign_.find(ii);
	  if ( iter == _loop_Gk_sign_.end() ){// no key yet
	    _loop_Gk_sign_[ii] = isign;
	  }else{
	    if (isign != iter->second) log<<"ERROR : Should not happen 1!"<<std::endl;
	  }
	}else{
	  if (lindex_i>0){ // meassuring line does not get computed, hence we skip it.
	    int ii = Vindx(id, lindex_i);
	    int isign = sign(ltype_i);
	    auto iter = _loop_Vq_sgn_.find(ii);
	    //log<<" here iloop="<<iloop<<" Vindx(id,lindex_i)="<<ii<<" isign="<<isign << endl;
	    if ( iter == _loop_Vq_sgn_.end() ){
	      _loop_Vq_sgn_[ii] = isign;
	    }else{
	      if (isign != iter->second){
		log << "ERROR : Should not happend 2!"<<" isign="<<isign<<" while iter->second="<< (int)(iter->second) << std::endl;
		log << "  ii="<<ii<<" isign="<<isign<<" Vindx(id)="<<Vindx(id,bl::Range::all())<<endl;
	      }
	    }
	    //log <<"  _loop_Vq_sgn_=";
	    //for (auto ci = _loop_Vq_sgn_.begin(); ci!=_loop_Vq_sgn_.end(); ++ci) log << ci->first<<" "<< (int)(ci->second) <<" ; ";
	    //log << endl;
	  }
	}
      }
    }

    {
      // This part of the code exists because of the single-particle counter term (lmbda_spct).
      // Because we use the trick to combine the single-particle counter term with the two particle-counter term,
      // the value of the two-particle counter term is not simply Vq^2*lmbda, but has extra term:
      //   Vq*(Vq*lmbda)^i + lmbda_spct[i-1]/n_{k+q}, where n_{k+q} is density associated with the G-propagator inside the two-particle counter term.
      // This is because the two-particle counter term should be Vq*(Vq*lmbda)^i n_{k+q}, and the single-particle should be lmbda_spct, and we hide
      // the latter within the former.
      // The problem is that whenever n_{k+q} is changed (the G-propagator inside the two-particle counter term), we need to update the combined
      // counter term. We set sign=0 so that momentum is unchanged, but the value of the interaction is recomputed.
      for (int id=0; id<Ndiags; id++){ // updating momentum in all propagators
	const std::vector<int>& lindex = Loop_index[id][iloop];
	const std::vector<int>& ltype  = Loop_type[id][iloop];
	for (int i=0; i<lindex.size(); i++){ // we go over all vertices in this loop again
	  int ltype_i  = ltype[i];           // we need to do this again at the end, because this counter term might already be changed above
	  int lindex_i = lindex[i];          // and then we should not do anything.
	  if ( abs(ltype_i)==1 ){            // this is G-propagator, and maybe inside the two-particle counter tem?
	    if (dynamic_counter){
	      if ( (lindex_i%2==1) && Vtype(id,lindex_i/2)!=0){ // this is odd vertex, from which dynamic counter-term is going out/in.
		int ii_v = Vindx(id, lindex_i/2);   // When this propagator is changed, the Vertex is changed, and consequently the bubble is changed too.
		auto iter = _loop_Vq_sgn_.find(ii_v);           // Is this interaction changed already before, because the same loop goes through this interaction anyway.
		//log <<" DCT:  iloop="<<iloop<<" id="<<id<<" i="<<i<<" ii_v="<< ii_v << endl;
		if ( iter == _loop_Vq_sgn_.end() ){             // Not changed above (does not contain this loop), therefore add it to list to be updated newertheless
		  _loop_Vq_sgn_[ii_v] = 0;                      // the sign value should be zero, because momentum of the interaction does not change.
		}
	      }
	    }

	    if (sum_lmbda_spct>small){
	      int ii = Gindx(id, lindex_i);    // index for the G-propagator
	      //log <<"iloop="<<iloop<<" id="<<id<<" i="<<i<<" lindex_i="<<lindex_i<<" ii="<<ii<<" "<<std::endl;
	      if (ii==std::numeric_limits<sint>().max()) continue; // When computing Density, we remove 0->1 propagator, and therefore ii=max, and should be removed
	      // Trying to find interaction index ii_v, so that "single_counter_index(ii_v)==ii"
	      auto piv = single_counter_index_inverse.find(ii); // Does this G-propagaor appear in the "single_counter_index" list, than it is inside a two-particle counter term.
	      //log <<" SPCT:  iloop="<<iloop<<" id="<<id<<" i="<<i<<" ii="<< ii << endl;
	      if (piv != single_counter_index_inverse.end() ){  // This is standard search in a map: If it does not point at the end, we found a match
		for (int j=0; j<piv->second.size(); j++){
		  int ii_v = piv->second[j];                         // This interaction, i.e., two particle counter term, needs to be updated any time G propagator of index ii is changed.
		  //log << "   SPCT: Found in map ii_v = "<< ii_v << endl;
		  auto iter = _loop_Vq_sgn_.find(ii_v);           // Is this interaction changed already before, because the same loop goes through this interaction anyway.
		  //log << "   we found ii_v="<<ii_v<<" and iter="<< iter->first<< endl;
		  if ( iter == _loop_Vq_sgn_.end() ){             // Not changed above (does not contain this loop), therefore add it to list to be updated newertheless
		    _loop_Vq_sgn_[ii_v] = 0;                      // the sign value should be zero, because momentum of the interaction does not change.
		  }
		}
	      }
	    }
	  }
	}
      }
    }
    
    loop_Gkind(iloop).resize(_loop_Gk_sign_.size());
    loop_Gksgn(iloop).resize(_loop_Gk_sign_.size());
    auto iter = _loop_Gk_sign_.begin();
    for (int i=0; iter!=_loop_Gk_sign_.end(); ++iter,++i){
      loop_Gkind(iloop)(i) = iter->first;
      loop_Gksgn(iloop)(i) = iter->second;
    }
    Ngp = std::max(Ngp, static_cast<int>(_loop_Gk_sign_.size()));
    
    loop_Vqind(iloop).resize(_loop_Vq_sgn_.size());
    loop_Vqsgn(iloop).resize(_loop_Vq_sgn_.size());
    iter = _loop_Vq_sgn_.begin();
    for (int i=0; iter!=_loop_Vq_sgn_.end(); ++iter,++i){
      loop_Vqind(iloop)(i) = iter->first;
      loop_Vqsgn(iloop)(i) = iter->second;
    }
    Nvp = std::max(Nvp, static_cast<int>(_loop_Vq_sgn_.size()));
    
    //log << "loop_Gkind=" << loop_Gkind << std::endl;
    //log << "loop_Gksgn=" << loop_Gksgn << std::endl;
    //log << "loop_Vqind=" << loop_Vqind << std::endl;
    //log << "loop_Vqsgn=" << loop_Vqsgn << std::endl;
  }
}

template<typename sint>
void Get_GVind(bl::Array<bl::Array<unsigned short,1>,1>& loop_Gkind, bl::Array<bl::Array<char,1>,1>& loop_Gksgn,
	       bl::Array<bl::Array<unsigned short,1>,1>& loop_Vqind, bl::Array<bl::Array<char,1>,1>& loop_Vqsgn,
	       bl::Array<bl::Array<unsigned short,1>,1>& loop_Vqind2, bl::Array<bl::Array<char,1>,1>& loop_Vqsgn2,
	       int& Ngp, int& Nvp,  int& Nvp2,
	       const bl::Array<vecList,2>& Vqh2,
	       const std::vector<std::vector<std::vector<int> > >& Loop_index, const std::vector<std::vector<std::vector<int> > >& Loop_type,
	       const bl::Array<sint,2>& Gindx, const bl::Array<sint,2>& Vindx,
	       const bl::Array<int,1>& single_counter_index, const std::vector<double>& lmbda_spct,
	       const bl::Array<char,2>& Vtype,
	       bool dynamic_counter=false, ostream& log=std::cout)
{/*  Given all the loops in all diagrams (Loop_indx and Loop_type) it calculates which propagtaors are
     changed when a single loop is being modified. Each loop carries a momentum k_i, and changing it requires
     one to modifiy all fermionic propagators stored in loop_Gkind(k_i,:). The momentum of each of these
     propagator contains either +k_i or -k_i, and such sign of momentum is contained in loop_Gksgn(k_i,:).
     Finally, Ngp is the maximum number of fermionic propagators that need to be changed when a single loop of 
     momentum is changed.
     For interaction, the corresponding quantities are loop_Vqind, loop_Vqsgn, and Nvp.
     Note that due to single-particle counter terms, Vq might need to be changed even when it does not contain particular loop. This is because 
     We define an effective V_q, which contains both the two-particle counter term, and the single-particle counter term, and the latter depends on
     density n_{k+q} of the nerby G-propagator.
  */
  int Ndiags = Loop_index.size();
  int Nloops = Loop_index[0].size();

  double sum_lmbda_spct=0.0;
  for (int i=0; i<lmbda_spct.size(); ++i) sum_lmbda_spct += lmbda_spct[i]*lmbda_spct[i];
  sum_lmbda_spct = sqrt(sum_lmbda_spct);
  const double small = 1e-10;
  //log<<"single_counter_index=" << single_counter_index << endl;
  // When single-particle counter terms is added, we first find inverse index for single_counter_index, which gives index to n_{k+q}.
  std::map<int,deque<int> > single_counter_index_inverse;
  if (sum_lmbda_spct>small){
    for (int ii_v=0; ii_v<single_counter_index.extent(0); ++ii_v){
      if (single_counter_index(ii_v)>=0){
	int ii_g = single_counter_index(ii_v);
	single_counter_index_inverse[ii_g].push_back(ii_v);
      }
    }
  }
  
  loop_Gkind.resize(Nloops);
  loop_Gksgn.resize(Nloops);
  loop_Vqind.resize(Nloops);
  loop_Vqsgn.resize(Nloops);
  Ngp=0; Nvp=0;
  for (int iloop=0; iloop<Nloops; iloop++){
    std::map<int,char> _loop_Gk_sign_, _loop_Vq_sgn_;
    for (int id=0; id<Ndiags; id++){ // updating momentum in all propagators
      const std::vector<int>& lindex = Loop_index[id][iloop];
      const std::vector<int>& ltype  = Loop_type[id][iloop];
      for (int i=0; i<lindex.size(); i++){
	int ltype_i  = ltype[i];
	int lindex_i = lindex[i];
	if ( abs(ltype_i)==1 ){
	  int ii = Gindx(id, lindex_i);
	  if (ii==std::numeric_limits<sint>().max()) continue; // When computing Density, we remove 0->1 propagator, and therefore ii=max, and should be removed
	  int isign = sign(ltype_i);
	  auto iter = _loop_Gk_sign_.find(ii);
	  if ( iter == _loop_Gk_sign_.end() ){// no key yet
	    _loop_Gk_sign_[ii] = isign;
	  }else{
	    if (isign != iter->second) log<<"ERROR : Should not happen 1!"<<std::endl;
	  }
	}else{
	  if (lindex_i>0){ // meassuring line does not get computed, hence we skip it.
	    int ii = Vindx(id, lindex_i);
	    int isign = sign(ltype_i);
	    auto iter = _loop_Vq_sgn_.find(ii);
	    //log<<" here iloop="<<iloop<<" Vindx(id,lindex_i)="<<ii<<" isign="<<isign << endl;
	    if ( iter == _loop_Vq_sgn_.end() ){ // the first time we encounter this type of V propagator
	      _loop_Vq_sgn_[ii] = isign;
	    }else{ // has already appeared
	      if (isign != iter->second){ // if it had different sign previously, we have a problem... can not be equivalent
		log << "ERROR : Should not happend 2!"<<" isign="<<isign<<" while iter->second="<< (int)(iter->second) << std::endl;
		log << "  ii="<<ii<<" isign="<<isign<<" Vindx(id)="<<Vindx(id,bl::Range::all())<<endl;
	      }
	    }
	    //log <<"  _loop_Vq_sgn_=";
	    //for (auto ci = _loop_Vq_sgn_.begin(); ci!=_loop_Vq_sgn_.end(); ++ci) log << ci->first<<" "<< (int)(ci->second) <<" ; ";
	    //log << endl;
	  }
	}
      }
    }
    
    {
      // This part of the code exists because of the single-particle counter term (lmbda_spct).
      // Because we use the trick to combine the single-particle counter term with the two particle-counter term,
      // the value of the two-particle counter term is not simply Vq^2*lmbda, but has extra term:
      //   Vq*(Vq*lmbda)^i + lmbda_spct[i-1]/n_{k+q}, where n_{k+q} is density associated with the G-propagator inside the two-particle counter term.
      // This is because the two-particle counter term should be Vq*(Vq*lmbda)^i n_{k+q}, and the single-particle should be lmbda_spct, and we hide
      // the latter within the former.
      // The problem is that whenever n_{k+q} is changed (the G-propagator inside the two-particle counter term), we need to update the combined
      // counter term. We set sign=0 so that momentum is unchanged, but the value of the interaction is recomputed.
      for (int id=0; id<Ndiags; id++){ // updating momentum in all propagators
	const std::vector<int>& lindex = Loop_index[id][iloop];
	const std::vector<int>& ltype  = Loop_type[id][iloop];
	for (int i=0; i<lindex.size(); i++){ // we go over all vertices in this loop again
	  int ltype_i  = ltype[i];           // we need to do this again at the end, because this counter term might already be changed above
	  int lindex_i = lindex[i];          // and then we should not do anything.
	  if ( abs(ltype_i)==1 ){            // this is G-propagator, and maybe inside the two-particle counter tem?
	    if (dynamic_counter){
	      if ( (lindex_i%2==1) && Vtype(id,lindex_i/2)!=0){ // this is odd vertex, from which dynamic counter-term is going out/in.
		int ii_v = Vindx(id, lindex_i/2);   // When this propagator is changed, the Vertex is changed, and consequently the bubble is changed too.
		auto iter = _loop_Vq_sgn_.find(ii_v);           // Is this interaction changed already before, because the same loop goes through this interaction anyway.
		//log <<" DCT:  iloop="<<iloop<<" id="<<id<<" i="<<i<<" ii_v="<< ii_v << endl;
		if ( iter == _loop_Vq_sgn_.end() ){             // Not changed above (does not contain this loop), therefore add it to list to be updated newertheless
		  _loop_Vq_sgn_[ii_v] = 0;                      // the sign value should be zero, because momentum of the interaction does not change.
		}
	      }
	    }

	    if (sum_lmbda_spct>small){
	      int ii = Gindx(id, lindex_i);    // index for the G-propagator
	      //log <<"iloop="<<iloop<<" id="<<id<<" i="<<i<<" lindex_i="<<lindex_i<<" ii="<<ii<<" "<<std::endl;
	      if (ii==std::numeric_limits<sint>().max()) continue; // When computing Density, we remove 0->1 propagator, and therefore ii=max, and should be removed
	      // Trying to find interaction index ii_v, so that "single_counter_index(ii_v)==ii"
	      auto piv = single_counter_index_inverse.find(ii); // Does this G-propagaor appear in the "single_counter_index" list, than it is inside a two-particle counter term.
	      //log <<" SPCT:  iloop="<<iloop<<" id="<<id<<" i="<<i<<" ii="<< ii << endl;
	      if (piv != single_counter_index_inverse.end() ){  // This is standard search in a map: If it does not point at the end, we found a match
		for (int j=0; j<piv->second.size(); j++){
		  int ii_v = piv->second[j];                         // This interaction, i.e., two particle counter term, needs to be updated any time G propagator of index ii is changed.
		  //log << "   SPCT: Found in map ii_v = "<< ii_v << endl;
		  auto iter = _loop_Vq_sgn_.find(ii_v);           // Is this interaction changed already before, because the same loop goes through this interaction anyway.
		  //log << "   we found ii_v="<<ii_v<<" and iter="<< iter->first<< endl;
		  if ( iter == _loop_Vq_sgn_.end() ){             // Not changed above (does not contain this loop), therefore add it to list to be updated newertheless
		    _loop_Vq_sgn_[ii_v] = 0;                      // the sign value should be zero, because momentum of the interaction does not change.
		  }
		}
	      }
	    }
	  }
	}
      }
    }
    
    loop_Gkind(iloop).resize(_loop_Gk_sign_.size());
    loop_Gksgn(iloop).resize(_loop_Gk_sign_.size());
    auto iter = _loop_Gk_sign_.begin();
    for (int i=0; iter!=_loop_Gk_sign_.end(); ++iter,++i){
      loop_Gkind(iloop)(i) = iter->first;
      loop_Gksgn(iloop)(i) = iter->second;
    }
    Ngp = std::max(Ngp, static_cast<int>(_loop_Gk_sign_.size()));
    
    loop_Vqind(iloop).resize(_loop_Vq_sgn_.size());
    loop_Vqsgn(iloop).resize(_loop_Vq_sgn_.size());
    iter = _loop_Vq_sgn_.begin();
    for (int i=0; iter!=_loop_Vq_sgn_.end(); ++iter,++i){
      loop_Vqind(iloop)(i) = iter->first;
      loop_Vqsgn(iloop)(i) = iter->second;
    }
    Nvp = std::max(Nvp, static_cast<int>(_loop_Vq_sgn_.size()));
    
    //log << "loop_Gkind=" << loop_Gkind << std::endl;
    //log << "loop_Gksgn=" << loop_Gksgn << std::endl;
    //log << "loop_Vqind=" << loop_Vqind << std::endl;
    //log << "loop_Vqsgn=" << loop_Vqsgn << std::endl;
  }

  std::vector<std::map<int,int> > _loop_Vq2_sgn_(Nloops);
  int Norder = Vqh2.extent(1);
  for (int id=0; id<Ndiags; id++){ // updating momentum in all propagators
    for (int i=1; i<Norder; i++){  // all interaction propagators
      if (Vtype(id,i)>=10){        // Hugenholtz type
	int ii = Vindx(id, i);     // unique index for this propagator
	const vecList& v = Vqh2(id,i);   // what is Vq2 momentum for this propagator
	//log << " id="<< id<< " i="<< i << " ii="<< ii << " v=" << v << endl;
	for (auto k=v.m.begin(); k!=v.m.end(); ++k){
	  int iloop = k->first;
	  int isign = k->second;
	  auto iter2 = _loop_Vq2_sgn_[iloop].find(ii);
	  if (iter2 == _loop_Vq2_sgn_[iloop].end()){ // the first time we encounter this type of V propagator
	    _loop_Vq2_sgn_[iloop][ii] = isign;
	    //log << "     ... adding iloop="<< iloop<<" ii="<< ii << " isign="<< isign << endl;
	  }else{
	    //log << "     ... checking iloop="<< iloop<<" ii="<< ii << " isign="<< isign << " isign=" << iter2->second << endl;
	    if (isign != iter2->second){
	      log << "ERROR : Should not happend 5!"<<" isign="<<isign<<" while iter2->second="<< (int)(iter2->second) << std::endl;
	      log << "  ii="<<ii<<" isign="<<isign<<" Vindx(id)="<<Vindx(id,bl::Range::all())<<endl;
	    }
	  }
	}
      }
    }
  }
  /*
  log << "What we have so far " << endl;
  for (int iloop=0; iloop<Nloops; iloop++){
    log << " iloop=" << iloop <<endl;
    for (auto iter=_loop_Vq2_sgn_[iloop].begin(); iter != _loop_Vq2_sgn_[iloop].end(); ++iter){
      log << " ..... ii="<<iter -> first << " isign="<< iter->second << endl;
    }
  }
  */
  loop_Vqind2.resize(Nloops);
  loop_Vqsgn2.resize(Nloops);
  Nvp2=0;
  for (int iloop=0; iloop<Nloops; iloop++){
    //log << "iloop="<< iloop << " Vq.size=" << _loop_Vq2_sgn_[iloop].size() << endl;
    loop_Vqind2(iloop).resize(_loop_Vq2_sgn_[iloop].size());
    loop_Vqsgn2(iloop).resize(_loop_Vq2_sgn_[iloop].size());
    auto iter = _loop_Vq2_sgn_[iloop].begin();
    for (int i=0; iter!=_loop_Vq2_sgn_[iloop].end(); ++iter,++i){
      //log << "  iloop=" << iloop << " i=" << i << " ii=" << iter->first << " isign="<< iter->second << endl;
      loop_Vqind2(iloop)(i) = iter->first;
      loop_Vqsgn2(iloop)(i) = iter->second;
    }
    Nvp2 = std::max(Nvp2, static_cast<int>(_loop_Vq2_sgn_[iloop].size() ));
  }
}

template<typename sint>
void Get_GVind(bl::Array<bl::Array<unsigned short,1>,1>& loop_Gkind, bl::Array<bl::Array<char,1>,1>& loop_Gksgn,
	       bl::Array<bl::Array<unsigned short,1>,1>& loop_Vqind, bl::Array<bl::Array<char,1>,1>& loop_Vqsgn,
	       int& Ngp, int& Nvp,
	       const std::vector<std::vector<std::vector<int> > >& Loop_index, const std::vector<std::vector<std::vector<int> > >& Loop_type,
	       const bl::Array<sint,2>& Gindx, const bl::Array<sint,2>& Vindx,
	       const bl::Array<int,1>& single_counter_index, double lmbda_spct=0)
{/*  Given all the loops in all diagrams (Loop_indx and Loop_type) it calculates which propagtaors are
     changed when a single loop is being modified. Each loop carries a momentum k_i, and changing it requires
     one to modifiy all fermionic propagators stored in loop_Gkind(k_i,:). The momentum of each of these
     propagator contains either +k_i or -k_i, and such sign of momentum is contained in loop_Gksgn(k_i,:).
     Finally, Ngp is the maximum number of fermionic propagators that need to be changed when a single loop of 
     momentum is changed.
     For interaction, the corresponding quantities are loop_Vqind, loop_Vqsgn, and Nvp.
     Note that due to single-particle counter terms, Vq might need to be changed even when it does not contain particular loop. This is because 
     We define an effective V_q, which contains both the two-particle counter term, and the single-particle counter term, and the latter depends on
     density n_{k+q} of the nerby G-propagator.
  */
  int Ndiags = Loop_index.size();
  int Nloops = Loop_index[0].size();

  // When single-particle counter terms is added, we first find inverse index for single_counter_index, which gives index to n_{k+q}.
  std::map<int,int> single_counter_index_inverse;
  if (lmbda_spct!=0){
    for (int ii_v=0; ii_v<single_counter_index.extent(0); ++ii_v){
      if (single_counter_index(ii_v)>=0){
	int ii_g = single_counter_index(ii_v);
	single_counter_index_inverse[ii_g] = ii_v;
      }
    }
  }

  
  loop_Gkind.resize(Nloops);
  loop_Gksgn.resize(Nloops);
  loop_Vqind.resize(Nloops);
  loop_Vqsgn.resize(Nloops);
  Ngp=0; Nvp=0;
  for (int iloop=0; iloop<Nloops; iloop++){
    std::map<int,char> _loop_Gk_sign_, _loop_Vq_sgn_;
    for (int id=0; id<Ndiags; id++){ // updating momentum in all propagators
      const std::vector<int>& lindex = Loop_index[id][iloop];
      const std::vector<int>& ltype  = Loop_type[id][iloop];
      for (int i=0; i<lindex.size(); i++){
	int ltype_i  = ltype[i];
	int lindex_i = lindex[i];
	if ( abs(ltype_i)==1 ){
	  int ii = Gindx(id, lindex_i);
	  if (ii==std::numeric_limits<sint>().max()) continue; // When computing Density, we remove 0->1 propagator, and therefore ii=max, and should be removed
	  int isign = sign(ltype_i);
	  auto iter = _loop_Gk_sign_.find(ii);
	  if ( iter == _loop_Gk_sign_.end() ){// no key yet
	    _loop_Gk_sign_[ii] = isign;
	  }else{
	    if (isign != iter->second) std::cerr<<"ERROR : Should not happen 3!"<<std::endl;
	  }
	}else{
	  if (lindex_i>0){ // meassuring line does not get computed, hence we skip it.
	    int ii = Vindx(id, lindex_i);
	    int isign = sign(ltype_i);
	    auto iter = _loop_Vq_sgn_.find(ii);
	    if ( iter == _loop_Vq_sgn_.end() ){
	      _loop_Vq_sgn_[ii] = isign;
	    }else{
	      if (isign != iter->second) std::cerr<<"ERROR : Should not happend 4!"<<std::endl;
	    }
	  }
	}
      }
      
      // This part of the code exists because of the single-particle counter term (lmbda_spct).
      // Because we use the trick to combine the single-particle counter term with the two particle-counter term,
      // the value of the two-particle counter term is not simply Vq^2*lmbda, but has extra term:
      //   Vq*(Vq*lmbda)^i + lmbda_spct[i-1]/n_{k+q}, where n_{k+q} is density associated with the G-propagator inside the two-particle counter term.
      // This is because the two-particle counter term should be Vq*(Vq*lmbda)^i n_{k+q}, and the single-particle should be lmbda_spct, and we hide
      // the latter within the former.
      // The problem is that whenever n_{k+q} is changed (the G-propagator inside the two-particle counter term), we need to update the combined
      // counter term. We set sign=0 so that momentum is unchanged, but the value of the interaction is recomputed.
      if (lmbda_spct!=0){
	for (int i=0; i<lindex.size(); i++){ // we go over all vertices in this loop again
	  int ltype_i  = ltype[i];           // we need to do this again at the end, because this counter term might already be changed above
	  int lindex_i = lindex[i];          // and then we should not do anything.
	  if ( abs(ltype_i)==1 ){            // this is G-propagator, and maybe inside the two-particle counter tem?
	    int ii = Gindx(id, lindex_i);    // index for the G-propagator
	    //std::cout<<"iloop="<<iloop<<" id="<<id<<" i="<<i<<" lindex_i="<<lindex_i<<" ii="<<ii<<" "<<std::endl;
	    if (ii==std::numeric_limits<sint>().max()) continue; // When computing Density, we remove 0->1 propagator, and therefore ii=max, and should be removed
	    // Trying to find interaction index ii_v, so that "single_counter_index(ii_v)==ii"

	    cout<<" in single_counter_index_inverse looking for ii="<< ii << endl;
	    
	    auto piv = single_counter_index_inverse.find(ii); // Does this G-propagaor appear in the "single_counter_index" list, than it is inside a two-particle counter term.
	    if (piv != single_counter_index_inverse.end() ){  // If it does not point at the end, we found a match
	      int ii_v = piv->second;                         // This interaction, i.e., two particle counter term, needs to be updated any time G propagator of index ii is changed.
	      auto iter = _loop_Vq_sgn_.find(ii_v);           // Is this interaction changed already before, because the same loop goes through this interaction anyway.

	      cout<<" Found match ii_v = "<<ii_v << endl;
	       
	      if ( iter == _loop_Vq_sgn_.end() ){             // Not changed before, therefore add it to list to be updated
		_loop_Vq_sgn_[ii_v] = 0;                        // the sign value should be zero, because momentum of the interaction does not change.
	      }
	    }
	  }
	}
      }
    }
    loop_Gkind(iloop).resize(_loop_Gk_sign_.size());
    loop_Gksgn(iloop).resize(_loop_Gk_sign_.size());
    auto iter = _loop_Gk_sign_.begin();
    for (int i=0; iter!=_loop_Gk_sign_.end(); ++iter,++i){
      loop_Gkind(iloop)(i) = iter->first;
      loop_Gksgn(iloop)(i) = iter->second;
    }
    Ngp = std::max(Ngp, static_cast<int>(_loop_Gk_sign_.size()));
    
    loop_Vqind(iloop).resize(_loop_Vq_sgn_.size());
    loop_Vqsgn(iloop).resize(_loop_Vq_sgn_.size());
    iter = _loop_Vq_sgn_.begin();
    for (int i=0; iter!=_loop_Vq_sgn_.end(); ++iter,++i){
      loop_Vqind(iloop)(i) = iter->first;
      loop_Vqsgn(iloop)(i) = iter->second;
    }
    Nvp = std::max(Nvp, static_cast<int>(_loop_Vq_sgn_.size()));
    //std::cout<<"loop_Gkind="<<loop_Gkind<<std::endl;
    //std::cout<<"loop_Gksgn="<<loop_Gksgn<<std::endl;
    //std::cout<<"loop_Vqind="<<loop_Vqind<<std::endl;
    //std::cout<<"loop_Vqsgn="<<loop_Vqsgn<<std::endl;
  }
}

void Get_GVind(bl::Array<bl::Array<unsigned short,1>,1>& loop_Gkind, bl::Array<bl::Array<char,1>,1>& loop_Gksgn,
	       bl::Array<bl::Array<unsigned short,1>,1>& loop_Vqind, bl::Array<bl::Array<char,1>,1>& loop_Vqsgn,
	       int& Ngp, int& Nvp,
	       const std::vector<std::vector<std::vector<int> > >& Loop_index, const std::vector<std::vector<std::vector<int> > >& Loop_type,
	       const bl::Array<unsigned short,2>& Gindx, const bl::Array<unsigned short,2>& Vindx)
{/*  Given all the loops in all diagrams (Loop_indx and Loop_type) it calculates which propagtaors are
     changed when a single loop is being modified. Each loop carries a momentum k_i, and changing it requires
     one to modifiy all fermionic propagators stored in loop_Gkind(k_i,:). The momentum of each of these
     propagator contains either +k_i or -k_i, and such sign of momentum is contained in loop_Gksgn(k_i,:).
     Finally, Ngp is the maximum number of fermionic propagators that need to be changed when a single loop of 
     momentum is changed.
     For interaction, the corresponding quantities are loop_Vqind, loop_Vqsgn, and Nvp.
  */
  int Ndiags = Loop_index.size();
  int Nloops = Loop_index[0].size();

  loop_Gkind.resize(Nloops);
  loop_Gksgn.resize(Nloops);
  loop_Vqind.resize(Nloops);
  loop_Vqsgn.resize(Nloops);
  Ngp=0; Nvp=0;
  for (int iloop=0; iloop<Nloops; iloop++){
    std::map<int,char> _loop_Gk_sign_, _loop_Vq_sgn_;
    for (int id=0; id<Ndiags; id++){ // updating momentum in all propagators
      const std::vector<int>& lindex = Loop_index[id][iloop];
      const std::vector<int>& ltype  = Loop_type[id][iloop];
      for (int i=0; i<lindex.size(); i++){
	int ltype_i  = ltype[i];
	int lindex_i = lindex[i];
	if ( abs(ltype_i)==1 ){
	  int ii = Gindx(id, lindex_i);
	  int isign = sign(ltype_i);

	  auto iter = _loop_Gk_sign_.find(ii);
	  if ( iter == _loop_Gk_sign_.end() ){// no key yet
	    _loop_Gk_sign_[ii] = isign;
	  }else{
	    if (isign != iter->second) std::cerr<<"ERROR : Should not happen 5!"<<std::endl;
	  }
	  
	}else{
	  if (lindex_i>0){ // meassuring line does not get computed, hence we skip it.
	    int ii = Vindx(id, lindex_i);
	    int isign = sign(ltype_i);

	    auto iter = _loop_Vq_sgn_.find(ii);
	    if ( iter == _loop_Vq_sgn_.end() ){
	      _loop_Vq_sgn_[ii] = isign;
	    }else{
	      if (isign != iter->second) std::cerr<<"ERROR : Should not happend 6!"<<std::endl;
	    }
	  }
	}
      }
    }
    loop_Gkind(iloop).resize(_loop_Gk_sign_.size());
    loop_Gksgn(iloop).resize(_loop_Gk_sign_.size());
    auto iter = _loop_Gk_sign_.begin();
    for (int i=0; iter!=_loop_Gk_sign_.end(); ++iter,++i){
      loop_Gkind(iloop)(i) = iter->first;
      loop_Gksgn(iloop)(i) = iter->second;
    }
    Ngp = std::max(Ngp, static_cast<int>(_loop_Gk_sign_.size()));
    
    loop_Vqind(iloop).resize(_loop_Vq_sgn_.size());
    loop_Vqsgn(iloop).resize(_loop_Vq_sgn_.size());
    iter = _loop_Vq_sgn_.begin();
    for (int i=0; iter!=_loop_Vq_sgn_.end(); ++iter,++i){
      loop_Vqind(iloop)(i) = iter->first;
      loop_Vqsgn(iloop)(i) = iter->second;
    }
    Nvp = std::max(Nvp, static_cast<int>(_loop_Vq_sgn_.size()));
  }
}


int GetMax_NGp_trial(const bl::Array<unsigned short,2>& i_diagsG, const bl::Array<unsigned short,2>& Gindx, int Ng, int Ngp)
{/* This routine finds out how large should be array G_trial
  */
  int Ndiags = i_diagsG.extent(0);
  int Norder = i_diagsG.extent(1)/2;
  BitArray changed_G(Ng);
  int Ngp_tr = Ngp; // When changing momentum, we already know that G_trial needs to be Ngp
  // Now we check how large should be arrays for G_trial when changing time
  for (int itime=0; itime < Norder; ++itime){ // all possible time changes
    changed_G=0;            // which propagators are being changed?
    if (itime==0){         // this is the measuring time with vertex=0
      int ivertex=0;
      for (int id=0; id<Ndiags; id++){
	int i_pre_vertex = i_diagsG(id,ivertex);
	changed_G.set(Gindx(id,ivertex),1);         // these two propagators contain vertex=0.
	changed_G.set(Gindx(id,i_pre_vertex),1);
      }
    }else{
      for (int ivertex=2*itime; ivertex<=2*itime+1; ++ivertex){ // 2*itime and 2*itime+1 are changed at the same time, because they are equal
	for (int id=0; id<Ndiags; id++){
	  int i_pre_vertex = i_diagsG(id,ivertex);
	  changed_G.set(Gindx(id,ivertex),1);        // these two propagators are changed because of the new time.
	  changed_G.set(Gindx(id,i_pre_vertex),1);
	}
      }
    }
    int Ngp_trial=0;
    for (int ii=0; ii<Ng; ++ii)
      if (changed_G[ii]) Ngp_trial+=1;
    Ngp_tr = std::max(Ngp_tr,Ngp_trial);
  }
  return Ngp_tr;
}

void Where_to_Add_Single_Particle_Counter_Term(bl::Array<char,2>& Vtype,
					       int Ndiags, int Norder, const bl::Array<unsigned short,2>& diagsG)
{
  // At the interaction line with Vtype(id,j) we check if single-particle term should be added.
  // This occurs whenever the two-particle counter-term looks like a simple self-energy insertion of the Fock-type. 
  for (int id=0; id<Ndiags; id++)    // Checking where can we add single-particle counter terms. 
    for (int j=1; j<Norder; j++)     // skip meassuring line
      if (Vtype(id,j)==1){            // counter-term of order 1 only. It spans the vertices (2*j,2*j+1) in our convention.
	// the vertices are (2*j,2*j+1)
	if (diagsG(id,2*j)==2*j+1) // G-propagator must connect vertices 2*j and 2*j+1
	  // In this case G-propagator goes from 2*j to 2*j+1, and is stored in G(2*j)
	  Vtype(id,j) *= -1;        // Now we remember that this is different type of interaction, which has both two-particle and single-particle counter terms.
	else if (diagsG(id,2*j+1)==2*j)
	  // In this case G-propagator goes from 2*j+1 to 2*j, and is stored in G(2*j+1)
	  Vtype(id,j) *= -1;   // It is crucial to change the type of interaction, so that terms with both counter-terms and single counter terms are treated as different!
      }
}

void Where_to_Add_Single_Particle_Counter_Term(bl::Array<char,2>& Vtype, const std::vector<double>& lmbda_spct,
					       int Ndiags, int Norder, const bl::Array<unsigned short,2>& diagsG)
{
  double sum_lmbda_spct=0.0;
  for (int i=0; i<lmbda_spct.size(); i++) sum_lmbda_spct += lmbda_spct[i]*lmbda_spct[i];
  if (sum_lmbda_spct<1e-10) return; // no actual counter term
  
  // At the interaction line with Vtype(id,j) we check if single-particle term should be added.
  // This occurs whenever the two-particle counter-term looks like a simple self-energy insertion of the Fock-type. 
  for (int id=0; id<Ndiags; id++)    // Checking where can we add single-particle counter terms. 
    for (int j=1; j<Norder; j++)     // skip meassuring line
      for (int vtp=1; vtp<=lmbda_spct.size(); vtp++){
	// lmbda_spct[0] is combined with Vtype(id,j)==1, i.e., second order term in expansion
	// lmbda_spct[1] is combined with Vtype(id,j)==2, i.e., third order term in expansion
	//   and so on....                    // We do not need to take Vtype%10 because single-particle counter term are possible only in non-Hugenholtz interactions
	if ( Vtype(id,j) == vtp ){            // counter-term of order 1 only. It spans the vertices (2*j,2*j+1) in our convention.
	  // the vertices are (2*j,2*j+1)
	  if (diagsG(id,2*j)==2*j+1) // G-propagator must connect vertices 2*j and 2*j+1
	    // In this case G-propagator goes from 2*j to 2*j+1, and is stored in G(2*j)
	    Vtype(id,j) *= -1;        // Now we remember that this is different type of interaction, which has both two-particle and single-particle counter terms.
	  else if (diagsG(id,2*j+1)==2*j)
	    // In this case G-propagator goes from 2*j+1 to 2*j, and is stored in G(2*j+1)
	    Vtype(id,j) *= -1;   // It is crucial to change the type of interaction, so that terms with both counter-terms and single counter terms are treated as different!
	}
      }
}



// Nv = vindx.extent(0);
void Find_Single_Particle_Counter_G_index(bl::Array<int,1>& single_counter_index, 
					   int Ndiags, int Norder, int Nv, const bl::Array<unsigned short,2>& diagsG, const bl::Array<char,2>& Vtype,
					   const bl::Array<unsigned short,2>& Gindx, const bl::Array<unsigned short,2>& Vindx)
{
  // Once more we go through single-particle counter terms and set the index to the G-propagator n_{k+q}, which needs to be included in the combined
  // counter term. It is stored into array single_counter_index.
  single_counter_index.resize(Nv);
  single_counter_index = -1;
  for (int id=0; id<Ndiags; id++){
    for (int j=1; j<Norder; j++){
      int ii_v = Vindx(id,j); // If this interaction contains also the single-particle counter terms 
      if (Vtype(id,j) < 0){   // than we have set Vindx<0 before in "Where_to_Add_Single_Particle_Counter_Term" 
	// the vertices are (2*j,2*j+1)
	if (diagsG(id,2*j)==2*j+1){ // G_propagator goes from 2j->2j+1 
	  single_counter_index(ii_v) = Gindx(id,2*j);     // G_propagtaors is stored in 2j
	} else if (diagsG(id,2*j+1)==2*j){ // alternatively it goes from 2j+1->2j
	  single_counter_index(ii_v) = Gindx(id,2*j+1);   // and is hence stored in 2j+1
	}else{
	  std::cerr<<"ERROR : Should not happen as we should set Vtype to negative above"<<std::endl;
	}
      }
    }
  }
}


inline void PrintInfo(int itt, bool Qweight, double ka, double kmQ, double t, double PQ_new, double PQ, double occurence=0, std::ostream& log=std::cout)
{
  log.precision(5);
  if (! Qweight){
    log<<"+"<<std::right<<std::setw(7)<<itt/1.0e6<<std::left<<"M k="<<std::left<<std::setw(9)<<ka<<" k-Q="<<std::setw(9)<<kmQ<<" t="<<std::setw(9)<<t<<" PQ_new="<<std::setw(12)<<PQ_new<<" PQ_old="<<std::setw(12)<<PQ<<" ratio="<<std::setw(12)<<PQ_new/PQ<<" occurence="<<occurence<<std::endl;
  }else{
    log<<"-"<<std::right<<std::setw(7)<<itt/1.0e6<<std::left<<"M k="<<std::left<<std::setw(9)<<ka<<" k-Q="<<std::setw(9)<<kmQ<<" t="<<std::setw(9)<<t<<" PQ_new="<<std::setw(12)<<PQ_new<<" PQ_old="<<std::setw(12)<<PQ<<" ratio="<<std::setw(12)<<PQ_new/PQ<<" occurence="<<occurence<<std::endl;
  }
}
inline void PrintInfo_(int itt, bool Qweight, double ka, double kmQ, double t, double PQ_new, double PQ, double occurT=0, double occurence=0, std::ostream& log=std::cout)
{
  log.precision(5);
  if (! Qweight){
    log<<"+"<<std::right<<std::setw(7)<<itt/1.0e6<<std::left<<"M k="<<std::left<<std::setw(9)<<ka<<" k-Q="<<std::setw(9)<<kmQ<<" t="<<std::setw(9)<<t<<" PQ_new="<<std::setw(12)<<PQ_new<<" PQ_old="<<std::setw(12)<<PQ<<" ratio="<<std::setw(12)<<PQ_new/PQ<<" dhk="<<occurT<<" occurence="<<occurence<<std::endl;
  }else{
    log<<"-"<<std::right<<std::setw(7)<<itt/1.0e6<<std::left<<"M k="<<std::left<<std::setw(9)<<ka<<" k-Q="<<std::setw(9)<<kmQ<<" t="<<std::setw(9)<<t<<" PQ_new="<<std::setw(12)<<PQ_new<<" PQ_old="<<std::setw(12)<<PQ<<" ratio="<<std::setw(12)<<PQ_new/PQ<<" dhk="<<occurT<<" occurence="<<occurence<<std::endl;
  }
}
inline void _PrintInfo_(int itt, bool Qweight, int icase, double ka, double kmQ, double t, double PQ_new, double PQ, double occurT=0, double occurence=0, std::ostream& log=std::cout)
{
  log.precision(5);
  if (! Qweight){
    log<<"+"<<std::right<<std::setw(7)<<itt/1.0e6<<std::left<<"M icase="<<std::setw(1)<<icase<<" k="<<std::left<<std::setw(8)<<ka<<" k-Q="<<std::setw(8)<<kmQ<<" t="<<std::setw(8)<<t<<" PQ_new="<<std::setw(12)<<PQ_new<<" PQ_old="<<std::setw(12)<<PQ<<" ratio="<<std::setw(12)<<PQ_new/PQ<<" dhk="<<occurT<<" occurence="<<occurence<<std::endl;
  }else{
    log<<"-"<<std::right<<std::setw(7)<<itt/1.0e6<<std::left<<"M icase="<<std::setw(1)<<icase<<" k="<<std::left<<std::setw(8)<<ka<<" k-Q="<<std::setw(8)<<kmQ<<" t="<<std::setw(8)<<t<<" PQ_new="<<std::setw(12)<<PQ_new<<" PQ_old="<<std::setw(12)<<PQ<<" ratio="<<std::setw(12)<<PQ_new/PQ<<" dhk="<<occurT<<" occurence="<<occurence<<std::endl;
  }
}
#endif // _SAMPLE0
