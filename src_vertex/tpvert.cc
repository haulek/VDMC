// @Copyright 2018 Kristjan Haule 
#include <cstdint>
#include <ostream>
#include <deque>
#include <fstream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "util.h"
#include "interpolate.h"
#include "analytic.h"
#include <gsl/gsl_sf.h>
#include "slinalg.h"
#include "ypy.h"
//#include "sbessel.h"
//#include "fraction.h"

void GetAsymptoteSphBessels(Array<double,2>& jla, const Array<int,1>& nn, int lmax)
{
  //
  //    Given input array of arbitrary integers, nn=[i,i+1,... i+k], corresponding to bosonic Matsubara points, 
  //    calculates the value of the spherical bessel functions on these points, i.e.
  //         j_l(nn*pi)     for l=[0,...lmax]
  //    It uses asymptotic expression for spherical bessel function.
  //    Note that this gives very precise j_l(n*pi) for n > 2*(lmax)^(8/5)
  //
  Array<double,1> r1(nn.extent(0)), r2(nn.extent(0)), x2(nn.extent(0));
  for (int n=0; n<nn.extent(0); n++){
    double x = nn(n)*pi;
    int m1_nn = minus_1_2_n(nn(n));
    r1(n) = m1_nn/x;     //(-1)**nn/(nn*pi);
    r2(n) = m1_nn/(x*x); //(-1)**nn/(nn*pi)**2;
    x2(n) = 1./(x*x);    // 1/(nn*pi)**2;
  }
  Array<double,1> xm(nn.extent(0)), sm(nn.extent(0));
  //Array<double,2> jla(lmax+1,nn.extent(0));
  if (jla.extent(0) != lmax+1) cerr << "Wrong first dimension of jla in GetAsymptoteSphBessels" << endl;
  if (jla.extent(1) != nn.extent(0)) cerr << "Wrong second dimension of jla in GetAsymptoteSphBessels" << endl;
  
  for (int l=0; l<lmax+1; l+=2){ // even
    xm = l*(l+1.)/2.;
    sm(Range::all()) = xm(Range::all());
    for (int i=3; i<l+1; i+=2){
      xm *= -(l-i+1)*(l-i+2)*(l+i-1.)*(l+i)/(4*(i-1.)*i) * x2;
      sm += xm;
    }
    jla(l,Range::all()) = r2 * sm * minus_1_2_n(l/2);
  }
  for (int l=1; l<lmax+1; l+=2){ // odd
    xm = 1.;
    sm = xm;
    for (int i=2; i<l+1; i+=2){
      xm *= -(l-i+1)*(l-i+2)*(l+i-1.)*(l+i)/(4*(i-1.)*i) * x2;
      sm += xm;
    }
    jla(l,Range::all()) = r1 * sm * minus_1_2_n((l+1)/2);
  }
}

Array<double,1> _GetBesselInt_(double a, double b, const Array<double,1>& ja, const Array<double,1>& jb)
{// Given bessel functions in two points, like j_l(b) and j_l(a), it computes the integral
 //  In[l] == Integrate[ j_l(x), {x, a, b} ]
 // using the recursion satisfied by the bessel functions
 // It uses the well known recursion relation
 // (2*l+1) * d/dx j_l(x) = l * j_{l-1}(x) - (l+1) * j_{l+1}(x)
 //    which gives
 //  I_{l+1} = l/(l+1) * I_{l-1} - (2*l+1)/(l+1) j_l
 //
  int n_l = ja.extent(0);
  Array<double,1> In(n_l);
  In(0) = gsl_sf_Si(b) - gsl_sf_Si(a);    // Integrate[ sin(x)/x, {x,a,b}]
  if (n_l>1)
    In(1) = -jb(0) + ja(0);                 // In[1] = -sin(x)
  for (int l=1; l < n_l-1; l+=2){
    In(l+1) =  l/(l+1.) * In(l-1) - (2*l+1.)/(l+1.) * (jb(l)-ja(l));
    if ( l+2 < n_l )
      In(l+2) = (l+1.)/(l+2.) * In(l)   - (2*l+3.)/(l+2.) * (jb(l+1)-ja(l+1));
  }
  return In;
}
Array<double,2> _GetBesselMoment_(double a, double b, const Array<double,1>& In)
{// Given spherical bessel functions, their Integrals:
 //  In[l] == Integrate[ j_l(x), {x, a, b} ]
 // as computed GetBesselInt, it computes the first moments of bessel functions, i.e.,
 // Kn[l] == Integrate[ x * j_l(x), {x, a, b} ]
 // It uses well known recursion relation
 //  x j_{l+1} = (2l+1) j_l - x * j_{l-1}
 //  which gives
 //  K_{l+1} = (2l+1) I_l - K_{l-1}
 //
  int n_l = In.extent(0);
  Array<double,2> Kn(4,n_l);
  Kn(0,Range::all()) = In(Range::all());
  Kn(1,0) = -cos(b) + cos(a);                                            // Integrate[ x j0(x), {x, a, b}]
  Kn(2,0) = -b*cos(b) + sin(b) + a*cos(a) - sin(a);                      // Integrate[ x^2 j0(x), {x, a, b}]
  Kn(3,0) = (2.-b*b)*cos(b) + 2*b*sin(b) - (2.-a*a)*cos(a) - 2*a*sin(a); // Integrate[ x^3 j0(x), {x, a, b}]
  if (n_l>1){
    Kn(1,1) = In(0) - sin(b) + sin(a);                                     // Integrate[ x j1(x), {x, a, b}]
    Kn(2,1) = -2*cos(b) - b*sin(b) + 2*cos(a) + a*sin(a);                  // Integrate[ x^2 j1(x), {x, a, b}]
    Kn(3,1) = -3*b*cos(b) + sin(b) + (2.-b*b)*sin(b)  + 3*a*cos(a) - sin(a) - (2.-a*a)*sin(a); // Integrate[ x^3 j1(x), {x, a, b}]
  }
  //Kn(0) = -cos(b) + cos(a);        // Integrate[ x j0(x), {x, a, b}]
  //Kn(1) = In(0) - sin(b) + sin(a); // Integrate[ x j1(x), {x, a, b}]
  for (int ip=1; ip<4; ip++){
    for (int l=1; l < n_l-1; l+=2){
      Kn(ip,l+1) = -Kn(ip,l-1) + (2*l+1.) * Kn(ip-1,l);
      if ( l+2 < n_l )
	Kn(ip,l+2) = -Kn(ip,l)   + (2*l+3.) * Kn(ip-1,l+1);
    }
  }
  return Kn;
}

void GetAllBessels(Array<double,2>& jl1, Array<double,2>& jl2, Array<double,1>& rst, double k1, double k2, int lmax, double x_max, int N)
{ // computes bessel functions, as needed in function jVj, namely
  //    jl1[:,l] = j_l(k1*x)   and
  //    jl2[:,l] = j_l(k2*x), 
  // as well as
  //    res[:] = x*e^(-x)
  //
  jl1.resize(N,lmax+1);
  jl2.resize(N,lmax+1);
  rst.resize(N);
  double dh = x_max/(N-1.);
  Array<double,1> jtmp(lmax+1);
  for (int i=0; i<N; i++){
    double x = i*dh;
    //besselj( jtmp, lmax, k1*x);
    gsl_sf_bessel_jl_array (lmax, k1*x, jtmp.data());
    jl1(i, Range::all()) = jtmp;
    //besselj( jtmp, lmax, k2*x);
    gsl_sf_bessel_jl_array(lmax, k2*x, jtmp.data());
    jl2(i, Range::all()) = jtmp;
    rst(i) = x * exp(-x);
  }
}
void GetBesselMoments(Array<double,3>& Kn, double k2, const Array<double,2>& jl2, double x_max)
{// Computes bessel function moments as needed by jVj, namely
 //   Kn[i,n,l] = Integrate[ j_l(t) t^n, {t, k2*x_{i}, k2*x_{i+1}}]
 //   where input jl2[i,l] = j_l(x_i)
 //   
  int N = jl2.extent(0);
  int lmax = jl2.extent(1)-1;
  double dh = x_max/(N-1.);
  Array<double,1> In(lmax+1);
  Kn.resize(N,4,lmax+1);
  for (int i=1; i<N; i++){
    double x = i*dh;
    In = _GetBesselInt_((x-dh)*k2, x*k2, jl2(i-1,Range::all()), jl2(i,Range::all()) );
    Kn(i-1, Range::all(), Range::all()) = _GetBesselMoment_( (x-dh)*k2, x*k2, In );
  }
}
void EvaluateBesselIntegralWithMoments(Array<double,1>& Vl, double k2, const Array<double,3>& Kn, const Array<double,2>& jl1, Array<double,1>& rst, double x_max)
{//  Here we calculate the integral 
 //     Vl[l] == Integrate[ (2/x)*e^{-x} * x^2 * j_l(k1*x) * j_l(k2*x) , {x,0,x_max}]
 //     where  (2/x)*e^{-x} comes from Coulomb screened interaction of Yukawa type, and j_l are spherical bessel functions, needed to
 //  compute spherical expansion of the Coulomb repulsin V_{kk'} = \sum_{lm} Y_{lm}(k) <j_l| V_c |j_l> Y_{lm}^*(k').
 //  In this routine we assume that j_l(k2*x) is varying very fast, while j_l(k1*x) is varying slower.
 //  We assume that the part of the function fx == x*e^{-x}*j_l(k1*x) is well resolved on equidistant mesh of N point in the interval [0,x_max].
 //  For improving precision of the integral fx is splined by cubic spline on this equidistant mesh.
 //  Since j_l(k2*x) is oscilating very fast (because k2 is large) so that it can not be resolved on the existing mesh, we must calculate its integral analytically.
 //  We know that the cubic interpolation of fx function between pair of two points is
 //      fx(x) = q*f[i] + p*f[i+1] + (x[i+1]-x[i])^2*( (q^3-q) * f2[i] + (p^3-p) * f2[i+1])/6.
 //      where f[i] is the value of the funtion at x[i], p = (x-x[i])/(x[i+1]-x[i]) and q=1-p  and f2[i] is the second derivative of the function
 //  We then cast this polynomial interpolation into a more convenient form :
 //      fx(x) = a[0] + a[1] * x + a[2] * x^2 + a[3] * x^3
 //      Note that we name:   a[0] = c01+c02 ;  a[1] =  (dfdx+c12) ; a[2] = c2 ; a[3] = c3
 //   which can then be used to compute
 //    Vl[l] = \sum_i Integrate[ (a[0] + a[1] * x + a[2] * x^2 + a[3] * x^3 )  j_l(k2*x), {x, x_i, x_{i+1}}] =
 //    Vl[l] = \sum_i a[0]/k2   * Int[ j_l(t),     {t, x_i*k2, x_{i+1}*k2}] + a[1]/k^2 * Int[ j_l(t) * t,   {t, x_i*k2, x_{i+1}*k2}] +
 //                   a[2]/k2^3 * Int[ j_l(t) t^2, {t, x_i*k2, x_{i+1}*k2}] + a[3]/k^4 * Int[ j_l(t) * t^3, {t, x_i*k2, x_{i+1}*k2}]
 //
  int N = jl1.extent(0);
  int lmax = jl1.extent(1)-1;
  double dh = x_max/(N-1.);
  Array<double,1> xt(N);
  for (int i=0; i<N; i++) xt(i) = dh*i;
  Vl.resize(lmax+1);
  Spline1D<double> fx(N);
  for (int l=0; l<=lmax; l++){
    for (int i=0; i<N; i++) fx[i] = jl1(i,l) * rst(i);
    fx.splineIt(xt);
    double dsum=0.0;
    for (int i=0; i<N-1; i++){
      double xip = (i+1)*dh, xi = i*dh;
      double k2_2 = k2*k2;
      double k2_4 = k2_2*k2_2;
      double dK0 = Kn(i,0,l)/k2;
      double dK1 = Kn(i,1,l)/(k2_2);
      double dK2 = Kn(i,2,l)/(k2_2*k2);
      double dK3 = Kn(i,3,l)/k2_4;
      double df_dx = (fx[i+1]-fx[i])/dh;
      double c01 = fx[i+1] - xip * df_dx;
      double c02 = xi*xip*( fx.f2(i)*(2*xip-xi) + fx.f2(i+1)*(xip-2*xi))/(6.*dh);
      double c12 = -(fx.f2(i)*(2*xip*xip+2*xip*xi-xi*xi) + fx.f2(i+1)*(xip*xip-2*xip*xi-2*xi*xi))/(6.*dh);
      double df2dx = (fx.f2(i+1)-fx.f2(i))/dh;
      double c2 = (fx.f2(i+1) - xip * df2dx)/2.;
      double c3 = df2dx/6.;
      // The linear interpolation: f : c01  + df_dx * t
      // The cubic  interpolation: f : c01 + c02 + (dfdx + c12) * t + c2 * t^2 + c3 * t^3;
      //double fint = (c01 * dK0 + df_dx * dK1);
      double fint = (c01+c02) * dK0 + (df_dx + c12) * dK1 + c2 * dK2 + c3 * dK3;
      dsum += fint;
    }
    Vl(l) = dsum * 2. ;///lmbda;
  }
}

double EvaluateSingleBesselIntegralWithMoments(double k2, const Array<double,2>& Kn, const Array<double,1>& jl1, Array<double,1>& rst, double x_max)
{// The same function as "EvaluateBesselIntegralWithMoments", but computes only a single l, rather than all l's.
 // This is useful if x_max depends on l, hence j_l(k*x) need to be computed on different points for each l.
  int N = jl1.extent(0);
  double dh = x_max/(N-1.);
  Array<double,1> xt(N);
  for (int i=0; i<N; i++) xt(i) = dh*i;
  Spline1D<double> fx(N);
  
  for (int i=0; i<N; i++) fx[i] = jl1(i) * rst(i);
  fx.splineIt(xt);
  double dsum=0.0;
  for (int i=0; i<N-1; i++){
    double xip = (i+1)*dh, xi = i*dh;
    double k2_2 = k2*k2;
    double k2_4 = k2_2*k2_2;
    double dK0 = Kn(i,0)/k2;
    double dK1 = Kn(i,1)/(k2_2);
    double dK2 = Kn(i,2)/(k2_2*k2);
    double dK3 = Kn(i,3)/k2_4;
    double df_dx = (fx[i+1]-fx[i])/dh;
    double c01 = fx[i+1] - xip * df_dx;
    double c02 = xi*xip*( fx.f2(i)*(2*xip-xi) + fx.f2(i+1)*(xip-2*xi))/(6.*dh);
    double c12 = -(fx.f2(i)*(2*xip*xip+2*xip*xi-xi*xi) + fx.f2(i+1)*(xip*xip-2*xip*xi-2*xi*xi))/(6.*dh);
    double df2dx = (fx.f2(i+1)-fx.f2(i))/dh;
    double c2 = (fx.f2(i+1) - xip * df2dx)/2.;
    double c3 = df2dx/6.;
    // The linear interpolation: f : c01  + df_dx * t
    // The cubic  interpolation: f : c01 + c02 + (dfdx + c12) * t + c2 * t^2 + c3 * t^3;
    //double fint = (c01 * dK0 + df_dx * dK1);
    double fint = (c01+c02) * dK0 + (df_dx + c12) * dK1 + c2 * dK2 + c3 * dK3;
    dsum += fint;
  }
  return 2*dsum;///lmbda;
}

namespace py = pybind11;

py::array_t<double> jVj(double _k1_, double _k2_, int lmax, double lmbda, double x_max=10.)
{//  This function computes the matrix elements of the Coulomb interaction of Yukawa type in momentum space.
 //  We start with the Fourier transform of the Coulomb repulsion of the form:
 //    V_c(\vk1-\vk2) = Integrate[ e^{i(\vk1-\vk2)*\vr} * exp(-r*sqrt(lmbda))* 2/r  d^3r ]
 //  but we need it in the form
 //    V_c(\vk1-\vk2) = \sum_{lm} Y_{lm}(\vk1) V[l,k1,k2] Y_{lm}(\vk2)
 // This routine computed the matrix elements of V[l,k1,k2]/(4*pi)^2 .
 // They can be calculated by the following integral
 // V[l,k1,k2]/(4*pi)^2  = Integrate[ r^2 j_l(k1*r) * j_l(k2*r) * exp(-r*sqrt(lmbda))* 2/r , {r,0,Infinity}]
 // or
 // V[l,k1,k2]/(4*pi)^2  = Integrate[ j_l(k1/sqrt(lmbda)*x) * j_l(k2/sqrt(lmbda)*x) * x*exp(-x) , {x,0,Infinity}] *2/lmbda
 //
  static const int r=9;
  static const int N = (1<<r)+1; // 2^r+1

  py::array_t<double> _Vl_(lmax+1);
  py::buffer_info info = _Vl_.request();
  Array<double,1> Vl( (double*)info.ptr, info.shape[0], neverDeleteData);
  Vl=0;
  
  double k1, k2;
  if (_k1_<_k2_){ // The function is symmetric in k1,k2, however, we will choose k1<=k2 for convenience
    k1 = _k1_/sqrt(lmbda);
    k2 = _k2_/sqrt(lmbda);
  }else{ 
    k1 = _k2_/sqrt(lmbda);
    k2 = _k1_/sqrt(lmbda);
  }
  int icase = 0;
  if (k1 < 0.5*k2 && k2 > 5.0 ) // j(k2*x) is varying fast, while j(k1*x) can be interpolated
    icase = 1;      // Here j(k1*x) is interpolated by cubic spline, while integral of j(k2*x) is computed analytically
  // If both j(k1*x) and j(k2*x) vary very fast, we can use asymptotic expression in part of the interval
  //cout<<"icase="<<icase<<" k1="<<k1<<" k2="<<k2<<endl;
  Array<double,2> jl1, jl2;
  Array<double,1> rst;
  GetAllBessels(jl1, jl2, rst, k1, k2, lmax, x_max, N);
  
  if (icase==0){
    TinyVector<double,N> y;
    double dh = x_max/(N-1.);
    for (int l=0; l<=lmax; l++){
      for (int i=0; i<N; i++) y[i] = jl1(i,l) * jl2(i,l) * rst(i);
      Vl(l) = romberg2<N,r>(y, dh*(N-1.)) * 2./lmbda;
    }
  }else{
    Array<double,3> Kn;
    GetBesselMoments(Kn, k2, jl2, x_max);
    EvaluateBesselIntegralWithMoments(Vl, k2, Kn, jl1, rst, x_max);
    Vl *= 1/lmbda;
  }
  return _Vl_;
}
double jVj_single(double _k1_, double _k2_, int l, double lmbda, double x_max)
{// The same as "jVj", except here we calculate for a single l, rather that all l up to lmax.
  static const int r=9;
  static const int N = (1<<r)+1; // 2^r+1

  double Vl=0;
  
  double k1, k2;
  if (_k1_<_k2_){ // The function is symmetric in k1,k2, however, we will choose k1<=k2 for convenience
    k1 = _k1_/sqrt(lmbda);
    k2 = _k2_/sqrt(lmbda);
  }else{ 
    k1 = _k2_/sqrt(lmbda);
    k2 = _k1_/sqrt(lmbda);
  }
  int icase = 0;
  if (k1 < 0.5*k2 && k2 > 5.0 ) // j(k2*x) is varying fast, while j(k1*x) can be interpolated
    icase = 1;      // Here j(k1*x) is interpolated by cubic spline, while integral of j(k2*x) is computed analytically
  // If both j(k1*x) and j(k2*x) vary very fast, we can use asymptotic expression in part of the interval
  //cout<<"icase="<<icase<<" k1="<<k2<<" k2="<<k2<<endl;

  if (icase==0){
    TinyVector<double,N> y;
    double dh = x_max/(N-1.);
    for (int i=0; i<N; i++){
      double x = i*dh;
      double j1 = gsl_sf_bessel_jl(l, k1*x);
      double j2 = gsl_sf_bessel_jl(l, k2*x);
      y[i] = j1 * j2 * x * exp(-x);
    }
    Vl = romberg2<N,r>(y, x_max) * 2./lmbda;
  }else{
    Array<double,2> jl1, jl2;
    Array<double,1> rst;
    GetAllBessels(jl1, jl2, rst, k1, k2, l, x_max, N);
    Array<double,3> Kn;
    GetBesselMoments(Kn, k2, jl2, x_max);
    Vl = EvaluateSingleBesselIntegralWithMoments(k2, Kn(Range::all(),Range::all(),l), jl1(Range::all(),l), rst, x_max)/lmbda;
  }
  return Vl;
}

void AssemblyGamma(py::array_t<complex<double>,py::array::c_style>& _gamma_,
		   py::array_t<double,py::array::c_style>& _Vkk_,
		   py::array_t<complex<double>,py::array::c_style>& _P_p_P_, int iw)
{
  py::buffer_info info_g = _gamma_.request();
  if (info_g.ndim != 2) throw std::runtime_error("Number of dimensions for gamma must be 2");
  
  py::buffer_info info_v = _Vkk_.request();
  if (info_v.ndim != 3) throw std::runtime_error("Number of dimensions for Vkk must be 3");

  py::buffer_info info_pp = _P_p_P_.request();
  if (info_pp.ndim != 4) throw std::runtime_error("Number of dimensions for P_p_P must be 4");
  
  bl::Array<complex<double>,2> gamma( (complex<double>*)info_g.ptr, bl::shape(info_g.shape[0], info_g.shape[1]), bl::neverDeleteData);
  bl::Array<double,3> Vkk( (double*)info_v.ptr, bl::shape(info_v.shape[0], info_v.shape[1], info_v.shape[2]), bl::neverDeleteData);
  bl::Array<complex<double>,4> P_p_P( (complex<double>*)info_pp.ptr, bl::shape(info_pp.shape[0], info_pp.shape[1], info_pp.shape[2], info_pp.shape[3]), bl::neverDeleteData);

  int Nk = Vkk.extent(0);
  if (Vkk.extent(1) != Nk) cerr<<"Dimension 2 of Vkk is wrong"<<endl;
  int lmax = Vkk.extent(2)-1;
  if (P_p_P.extent(0)!=Nk) cerr<<"Dimension 1 of P_p_P is wrong"<<endl;
  if (P_p_P.extent(1)!=lmax+1) cerr<<"Dimension 2 of P_p_P is wrong"<<endl;
  if (P_p_P.extent(2)!=lmax+1) cerr<<"Dimension 3 of P_p_P is wrong"<<endl;
  int nOm = P_p_P.extent(3);
  if (gamma.extent(0) != (lmax+1)*Nk) cerr<<"Dimension 1 of gamma is wrong"<<endl;
  if (gamma.extent(1) != (lmax+1)*Nk) cerr<<"Dimension 2 of gamma is wrong"<<endl;
  
  //Vkk = zeros((len(kx),len(kx),lmax+1))
  //gammar = zeros( ((lmax+1)*len(kx), (lmax+1)*len(kx)), dtype=complex )
  //P_p_P = zeros( ( len(kx), lmax+1, lmax+1, len(iOm) ), dtype=complex )
  
  Array<complex<double>,3> p_p_p(Nk,lmax+1,lmax+1);
  for (int ik=0; ik<Nk; ik++)
    for (int l1=0; l1<=lmax; l1++)
      for (int l2=0; l2<=lmax; l2++)
	p_p_p(ik,l1,l2) = P_p_P(ik,l1,l2,iw);
    
  gamma=0;
  for (int ik1=0; ik1<Nk; ik1++){
    int iik1 = ik1*(lmax+1);
    for (int ik2=0; ik2<Nk; ik2++){
      int iik2 = ik2*(lmax+1);
      for (int l1=0; l1<=lmax; l1++){
	double v = Vkk(ik1,ik2,l1);
	for (int l2=0; l2<=lmax; l2++){
	  gamma(iik1+l1, iik2+l2) = v * p_p_p(ik2,l1,l2);
	  //if (abs(gamma(iik1+l1,iik2+l2))>1e10){
	  //  cout<<"at ik1="<<ik1<<" ik2="<<ik2<<" l1="<<l1<<" l2="<<l2<<" Vkk="<<v<<" p_p_p="<<p_p_p(ik2,l1,l2)<<endl;
	  //}
	}
      }
    }
  }
}

void ProjectToLegendre(Array<double,1>& cl, const Array<double,1>& xt, const Array<double,1>& ft, int lmax)
{//  Here we calculate the projection of a function ft(xt) to Legendre Polynomials up to l=lmax.
 //     cl[l] == (2*l+1)/2 * Integrate[ P_l(x) * fx(x) , {x,-1,1}]
 //     where  fx(x) is a smooth function, which is interpolated by spline.
 //  We know that the cubic interpolation of fx function between pair of two points is
 //      fx(x) = q*f[i] + p*f[i+1] + (x[i+1]-x[i])^2*( (q^3-q) * f2[i] + (p^3-p) * f2[i+1])/6.
 //      where f[i] is the value of the funtion at x[i], p = (x-x[i])/(x[i+1]-x[i]) and q=1-p  and f2[i] is the second derivative of the function
 //  We then cast this polynomial interpolation into a more convenient form :
 //      fx(x) = a[0] + a[1] * x + a[2] * x^2 + a[3] * x^3
 //      Note that we name:   a[0] = c01+c02 ;  a[1] =  (dfdx+c12) ; a[2] = c2 ; a[3] = c3
 //   which can then be used to compute
 //    cl[l] = \sum_i Integrate[ (a[0] + a[1] * x + a[2] * x^2 + a[3] * x^3 )  Pl(x), {x, x_i, x_{i+1}}] 
 //   In practice, we compute cPl(l,n) such that Pl(x) = \sum_n cPl(l,n) * x^n
 //   and then compute
 //   In[n] = Int[ x^n f(x), {x,-1,1}]
 //   In[n] = \sum_i (a0[i] (x_{i+1}^{n+1}-x_i^{n+1})/(n+1) + a1[i] * (x_{i+1}^{n+2}-x_i^{n+2})/(n+2) + a2[i] * (x_{i+1}^{n+3}-x_i^{n+3})/(n+3) + a3[i] * (x_{i+1}^{n+4}-x_i^{n+4})/(n+4) )
 //   cl[l] = \sum_n cPl(l,n) * In[n]
 //  
  int N = xt.extent(0);
  Spline1D<double> fx(N);
  for (int i=0; i<N; i++) fx[i] = ft(i);
  fx.splineIt(xt);
  
  Array<double,1> dK0(lmax+1), dK1(lmax+1), dK2(lmax+1), dK3(lmax+1);
  Array<double,1> Int(lmax+1);
  Int = 0;
  for (int i=0; i<N-1; i++){
    double xip = xt(i+1), xi = xt(i);
    double dh = xip-xi;
    double a = (i!=  0) ? xi  : -1.0;
    double b = (i!=N-2) ? xip :  1.0;
    double an = a, bn = b;
    for (int n=0; n<lmax+1; ++n){
      double am = an, bm = bn;// am = x_i^{n+1}; bm = x_{i+1}^{n+1}
      dK0(n) = (bm-am)/(n+1.);
      am *= a; bm *= b;       // am = x_i^{n+2}; bm = x_{i+1}^{n+2}
      dK1(n) = (bm-am)/(n+2.);
      am *= a; bm *= b;       // am = x_i^{n+3}; bm = x_{i+1}^{n+3}
      dK2(n) = (bm-am)/(n+3.);
      am *= a; bm *= b;       // am = x_i^{n+4}; bm = x_{i+1}^{n+4}
      dK3(n) = (bm-am)/(n+4.);
      an *= a; bn *= b;
    }
    double df_dx = (fx[i+1]-fx[i])/dh;
    double df2dx = (fx.f2(i+1)-fx.f2(i))/dh;
    double c01 = fx[i+1] - xip * df_dx;
    //double c02 = xi*xip*( fx.f2(i)*(2*xip-xi) + fx.f2(i+1)*(xip-2*xi))/(6.*dh);
    double c02 = 0.25*xi*xip*(fx.f2(i) + fx.f2(i+1) - 1./3.*(xip+xi)*df2dx);
    //double c12 = -(fx.f2(i)*(2*xip*xip+2*xip*xi-xi*xi) + fx.f2(i+1)*(xip*xip-2*xip*xi-2*xi*xi))/(6.*dh);
    double c12 = -0.5*(xip+xi)*fx.f2(i) + 0.5*(xi*xi-1./3.*dh*dh)*df2dx;
    double c2 = (fx.f2(i+1) - xip * df2dx)/2.;
    double c3 = df2dx/6.;
    // The linear interpolation: f : c01 + df_dx * t
    // The cubic  interpolation: f : c01 + c02 + (dfdx + c12) * t + c2 * t^2 + c3 * t^3;
    for (int n=0; n<lmax+1; ++n)
      Int(n) += (c01+c02) * dK0(n) + (df_dx + c12) * dK1(n) + c2 * dK2(n) + c3 * dK3(n);
  }
  Array<double,2> cPl;
  LegendreCoeff(cPl, lmax);
  cl=0;
  for (int l=0; l<=lmax; ++l){
    double dsum=0;
    for (int n=(l%2); n<=l; n+=2) dsum += cPl(l,n)*Int(n);
    cl(l) += dsum;
  }
  for (int l=0; l<=lmax; ++l) cl(l) *= (2*l+1.)/2.; // Normalization
}


void LegendreInverseFourierBoson(Array<double,1> Cl,
				 const Array<complex<double>,1>& Gom, const Array<int,1>& iOm, int lmax, double beta, int nom,
				 const Array<double,2>& LF_Ker_even,
				 const Array<double,2>& LF_Ker_odd)
{// Given input green's function in bosonic Matsubara points, it returns its expansion in terms of Legendre polynomials.
 // The inverse of this operation, which transforms from legendre coefficients c_l to Matsubara frequency, is easier to specify,
 //
 //   G(iOm_n) = \sum_l c_l beta i^l (-1)^n j_l(n*pi)
 // The operation from G(iOm_n) to c_l is computed by
 //   c_l = (-i)^l (2l+1) * 1/beta * \sum_{Om_n} (-1)^n j_l(n*pi) * G(iOm_n)
 //
  int nOm_max = iOm(iOm.extent(0)-1)+1;
  int iw0 = nom-1; int iw1 = iOm.extent(0);
  Array<double,1> iOmhigh(iw1-iw0);
  for (int i=iw0; i<iw1; ++i) iOmhigh(i-iw0) = iOm(i);
  Spline1D<double> fGom(iw1-iw0);
  for (int i=iw0; i<iw1; ++i) fGom[i-iw0] = Gom(i).real();
  fGom.splineIt(iOmhigh);

  Array<double,1> Gm(nOm_max);
  for (int i=0; i<nom; ++i) Gm(i) = Gom(i).real();
  int ia=0;
  for (int i=nom; i<nOm_max; ++i) Gm(i) = fGom( InterpLeft(i,ia,iOmhigh) );

  Array<double,1> Cl_even(LF_Ker_even.extent(0));
  firstIndex j;
  secondIndex k;
  Cl_even = sum( LF_Ker_even(j,k) * Gm(k), k);
  
  for (int i=iw0; i<iw1; ++i) fGom[i-iw0] = Gom(i).imag();
  fGom.splineIt(iOmhigh);
  
  for (int i=0; i<nom; ++i) Gm(i) = Gom(i).imag();
  ia=0;
  for (int i=nom; i<nOm_max; ++i) Gm(i) = fGom( InterpLeft(i,ia,iOmhigh) );

  Array<double,1> Cl_odd(LF_Ker_odd.extent(0));
  Cl_odd = sum( LF_Ker_odd(j,k) * Gm(k), k);
  
  Cl.resize(lmax+1);  
  for (int l=0; l<=lmax; l+=2)
    Cl(l) = Cl_even(l/2);
  for (int l=1; l<=lmax; l+=2)
    Cl(l) = Cl_odd((l-1)/2);
}

/*
void ProjectToLegendreDebug(Array<double,1>& cl, const Array<double,1>& xt, const Array<double,1>& ft, int lmax, const string& filename)
{//  Here we calculate the projection of a function ft(xt) to Legendre Polynomials up to l=lmax.
 //     cl[l] == (2*l+1)/2 * Integrate[ P_l(x) * fx(x) , {x,-1,1}]
 //     where  fx(x) is a smooth function, which is interpolated by spline.
 //  We know that the cubic interpolation of fx function between pair of two points is
 //      fx(x) = q*f[i] + p*f[i+1] + (x[i+1]-x[i])^2*( (q^3-q) * f2[i] + (p^3-p) * f2[i+1])/6.
 //      where f[i] is the value of the funtion at x[i], p = (x-x[i])/(x[i+1]-x[i]) and q=1-p  and f2[i] is the second derivative of the function
 //  We then cast this polynomial interpolation into a more convenient form :
 //      fx(x) = a[0] + a[1] * x + a[2] * x^2 + a[3] * x^3
 //      Note that we name:   a[0] = c01+c02 ;  a[1] =  (dfdx+c12) ; a[2] = c2 ; a[3] = c3
 //   which can then be used to compute
 //    cl[l] = \sum_i Integrate[ (a[0] + a[1] * x + a[2] * x^2 + a[3] * x^3 )  Pl(x), {x, x_i, x_{i+1}}] 
 //   In practice, we compute cPl(l,n) such that Pl(x) = \sum_n cPl(l,n) * x^n
 //   and then compute
 //   In[n] = Int[ x^n f(x), {x,-1,1}]
 //   In[n] = \sum_i (a0[i] (x_{i+1}^{n+1}-x_i^{n+1})/(n+1) + a1[i] * (x_{i+1}^{n+2}-x_i^{n+2})/(n+2) + a2[i] * (x_{i+1}^{n+3}-x_i^{n+3})/(n+3) + a3[i] * (x_{i+1}^{n+4}-x_i^{n+4})/(n+4) )
 //   cl[l] = \sum_n cPl(l,n) * In[n]
 //  
  int N = xt.extent(0);
  Spline1D<double> fx(N);
  for (int i=0; i<N; i++) fx[i] = ft(i);
  fx.splineIt(xt);

  
  ofstream out(filename.c_str());
  int M=100*N;
  for (int i=0; i<M; ++i){
    double x = -1 + 2.*i/(M-1.);
    out << x <<" "<< fx(Interp(x,xt)) << endl;
  }
  
  Array<double,1> dK0(lmax+1), dK1(lmax+1), dK2(lmax+1), dK3(lmax+1);
  Array<double,1> Int(lmax+1);
  Int = 0;
  for (int i=0; i<N-1; i++){
    double xip = xt(i+1), xi = xt(i);
    double dh = xip-xi;
    double a = (i!=  0) ? xi  : -1.0;
    double b = (i!=N-2) ? xip :  1.0;
    double an = a, bn = b;
    for (int n=0; n<lmax+1; ++n){
      double am = an, bm = bn;// am = x_i^{n+1}; bm = x_{i+1}^{n+1}
      dK0(n) = (bm-am)/(n+1.);
      am *= a; bm *= b;       // am = x_i^{n+2}; bm = x_{i+1}^{n+2}
      dK1(n) = (bm-am)/(n+2.);
      am *= a; bm *= b;       // am = x_i^{n+3}; bm = x_{i+1}^{n+3}
      dK2(n) = (bm-am)/(n+3.);
      am *= a; bm *= b;       // am = x_i^{n+4}; bm = x_{i+1}^{n+4}
      dK3(n) = (bm-am)/(n+4.);
      an *= a; bn *= b;
    }
    double df_dx = (fx[i+1]-fx[i])/dh;
    double df2dx = (fx.f2(i+1)-fx.f2(i))/dh;
    double c01 = fx[i+1] - xip * df_dx;
    //double c02 = xi*xip*( fx.f2(i)*(2*xip-xi) + fx.f2(i+1)*(xip-2*xi))/(6.*dh);
    double c02 = 0.25*xi*xip*(fx.f2(i) + fx.f2(i+1) - 1./3.*(xip+xi)*df2dx);
    //double c12 = -(fx.f2(i)*(2*xip*xip+2*xip*xi-xi*xi) + fx.f2(i+1)*(xip*xip-2*xip*xi-2*xi*xi))/(6.*dh);
    double c12 = -0.5*(xip+xi)*fx.f2(i) + 0.5*(xi*xi-1./3.*dh*dh)*df2dx;
    double c2 = (fx.f2(i+1) - xip * df2dx)/2.;
    double c3 = df2dx/6.;
    // The linear interpolation: f : c01 + df_dx * t
    // The cubic  interpolation: f : c01 + c02 + (dfdx + c12) * t + c2 * t^2 + c3 * t^3;
    for (int n=0; n<lmax+1; ++n)
      Int(n) += (c01+c02) * dK0(n) + (df_dx + c12) * dK1(n) + c2 * dK2(n) + c3 * dK3(n);
  }
  Array<double,2> cPl;
  LegendreCoeff(cPl, lmax);
  cl=0;
  for (int l=0; l<=lmax; ++l){
    double dsum=0;
    for (int n=(l%2); n<=l; n+=2) dsum += cPl(l,n)*Int(n);
    cl(l) += dsum;
  }
  for (int l=0; l<=lmax; ++l) cl(l) *= (2*l+1.)/2.; // Normalization



  /// debugging
  ofstream out2("_"+filename+"_");
  Array<double,1> cn(lmax+1);
  cn=0;
  for (int n=0; n<=lmax; ++n)
    for (int l=0; l<=lmax; ++l)
      cn(n) += cl(l)*cPl(l,n);
    
  for (int i=0; i<M; ++i){
    double x = -1 + 2.*i/(M-1.);

    double xn = 1.0;
    double val=0;
    for (int n=0; n<=lmax; ++n){
      val += cn(n)*xn;
      xn *= x;
    }
    out2 << x << " "<< val << endl;
  }
}
*/

void SimplePoleInside(Array<double,1>& res, double x, int lmax)
{
  // calculates modified spherical bessel function of the first kind for large argument x.
  double s = sign(x);
  double sx = s*x;
  //cout<<"s="<<s<<" sx="<<sx<<endl;
  res=0;
  for (int l=0; l<lmax+1; ++l){
    double z = 1.;
    double ak = 1.;
    double dsm = 1.;
    for (int k=0; k<l; ++k){
      ak *= (l-k)*(l+k+1)/(2.*(k+1.));
      z *= -1/sx;
      dsm += ak * z;
    }
    //cout<<l<<" dsm="<<dsm<<endl;
    int minus_s_2_l = s>0 ? minus_1_2_n(l) : 1; // (-s)^l
    //cout<<l<<" (-s)^l=" << minus_s_2_l << endl;
    //cout<<l<<" res=" << -0.5*(2*l+1)*minus_s_2_l/x * dsm << endl;
    res(l) = -0.5*(2*l+1)*minus_s_2_l/x * dsm;
  }
}


int binomialCoeff(int n, int k)
{
  int C[k+1];
  memset(C, 0, sizeof(C));
  C[0] = 1;  // nC0 is 1
  for (int i = 1; i <= n; i++){
    // Compute next row of pascal triangle using
    // the previous row
    for (int j = min(i, k); j > 0; j--)
      C[j] = C[j] + C[j-1];
  }
  return C[k];
}
/*
Fraction wconv(int n, int n1, int n2)
{
  // It gives the following convolution of polynomials, useful for convolution of Legendre Polynomials :
  // w(n,n1,n2) = Integrate[ z^n  Integrate[ x^{n2} (x-z-1)^{n1}, {x, z,1}] , {z,-1,1}]+
  //              Integrate[ z^n  Integrate[ x^{n2} (x-z+1)^{n1}, {x,-1,z}] , {z,-1,1}]
  //
  Fraction dsum = 0;
  if ( (n+n1+n2) % 2 == 0){
    for (int m=0; m<=n1; m++){
      //double p1 = binomialCoeff(n1, m)*4.*minus_1_2_n(m)/(n1+n2-m+1.);
      //double dsum1=0;
      Fraction dsum1 = 0;
      for (int k=(n%2); k<=m; k+=2)
	//dsum1 += binomialCoeff(m, k)/(n+k+1.);
	dsum1 += Fraction(binomialCoeff(m, k), (n+k+1));
      for (int k=(m+1)%2; k<=m; k+=2)
	//dsum1 -= binomialCoeff(m, k)/(n+k+n1+n2-m+2.);
	dsum1 -= Fraction(binomialCoeff(m, k), (n+k+n1+n2-m+2));
      long p1 = binomialCoeff(n1, m) * 4 * minus_1_2_n(m);
      long p2 = (n1+n2-m+1);
      dsum += dsum1 * Fraction(p1, p2);
    }
  }
  dsum.reduce();
  return dsum;
}

void Plconv_b(Array<double,3>& cbg, int lmax, int lmax1)
{
  // It gives the convolution of Legendre Polynomials for bosonic type functions, which is defined by
  // (P*P)(l1,l2,t) = Integrate[P_{l1}(2(t1-t)/beta-1) P_{l2}(2*t1/beta-1),{t1,t,beta}] +
  //                  Integrate[P_{l1}(2(t1-t)/beta+1) P_{l2}(2*t1/beta-1),{t1,0,t}]
  //                = beta \sum_l P_l(2t/beta-1) <l l1 | l2>
  // This is done in direct method, which is unstable for large l.
  int lmlmax = max(lmax,lmax1);
  Array<Fraction,2> cPl(lmlmax+1,lmlmax+1);
  LegendreCoeff(cPl,lmlmax);

  for (int l=0; l<lmlmax; l++){
    cout<<"l="<<l<<" ";
    for (int n=0; n<lmlmax; n++){
      cout << cPl(l,n) << " ";
    }
    cout<<endl;
  }

  Array<Fraction,3> w(lmax+1,lmax1+1,lmax1+1);
  for (int n=0; n<=lmax; n++)
    for (int n1=0; n1<=lmax1; n1++)
      for (int n2=0; n2<=lmax1; n2++){
  	w(n,n1,n2) = wconv(n,n1,n2);
      }
  
  //Array<double,3> cbg(lmax+1,lmax+1,lmax+1);
  cbg.resize(lmax1+1,lmax1+1,lmax+1);
  for (int l=0; l<=lmax; l++){
    for (int l1=0; l1<=lmax1; l1++){
      for (int l2=0; l2<=lmax1; l2++){
	Fraction dsum = 0;
	for (int n=0; n<=l; n++)
	  for (int n1=0; n1<=l1; n1++){
	    Fraction s=0;
	    for (int n2=0; n2<=l2; n2++){
	      s += cPl(l2,n2)*w(n,n1,n2);
	    }
	    Fraction t = cPl(l,n)*cPl(l1,n1);
	    t.reduce();
	    //cout<< " s=" << s << " t=" << t << endl;
	    dsum += t * s;
	  }
	cbg(l2,l1,l) = dsum * Fraction(2*l+1, 4);
      }
    }
  }
}
*/

void GetSphericalBessel_n_pi(Array<double,2>& jln, const Array<int,1>& iOm, int lmax)
{
  //    Given input integer array iOm of bosonic Matsubara points, it computes the spherical bessel functions in all 
  //    needed points, i.e.,
  //                    j_l(n*pi), where n=iOm[i]
  //    For small n, it calls scipy routine, while for large n, where the usual recursion is unstable, uses the 
  //    asymptotic expansion of bessel functions (see GetAsymptoteSphBessels )
  //Array<double,2> jln( (lmax+1,iOm.extent(0)) );
  jln.resize( lmax+1,iOm.extent(0) );
  int nmax = int( 2.*pow(lmax,8./5.) );   // beyond this n we can use asimptotic expression for bessels.
  Array<double,1> jtmp(lmax+1);
  // Here we compute them numerically in a usual way
  int ni=0;
  for (; ni<iOm.extent(0); ni++){
    int n = iOm(ni); // the rest are computed numerically.
    if (n>=nmax) break;
    gsl_sf_bessel_jl_array(lmax, n*pi, jtmp.data());
    jln(Range::all(), ni) = jtmp(Range::all());
  }
  Array<int,1> nn = iOm(Range(ni,toEnd));
  Array<double,2> jla(lmax+1,nn.extent(0));
  GetAsymptoteSphBessels(jla, nn, lmax);
  for (int ii=0; ni<iOm.extent(0); ++ii, ++ni){ // Here we use asymptotic expansion
    jln(Range::all(), ni) = jla(Range::all(),ii);
  }
}

void Plconv(Array<double,3>& cbg, int lmax){
  // Computes the convolution of two Legendre Polynomials in a stable way.
  // We want to compute:
  //  cbg(l2,l1,l) = (2l+1)/4 * Integrate[ P_l(z) Integrate[ P_{l1}(x-z-1) P_{l2}(x), {x,z,1}] , {z,-1,1} ] +
  //                 (2l+1)/4 * Integrate[ P_l(z) Integrate[ P_{l1}(x-z+1) P_{l2}(x), {x,-1,z}], {z,-1,1} ]
  // In Fourier space, this is equivalent to
  //
  //  cbg(l2,l1,l) = (2l+1) (-1)^{(l+l1-l2)/2} \sum_n (-1)^n j_l(n*pi) j_{l1}(n*pi) j_{l2}(n*pi}
  // 
  int lmax1 = lmax;
  cbg.resize(lmax1+1,lmax1+1,lmax+1);
  cbg = 0;
  cbg(0,0,0) = 1.0;
  int N=5000;
  Array<int,1> iOm(N);
  for (int i=0; i<iOm.size(); i++) iOm(i)=i+1;
  Array<double,2> jln;
  GetSphericalBessel_n_pi(jln, iOm, lmax);
  for (int l=1; l<=lmax; l++){
    for (int l1=1; l1<=lmax1; l1++){
      for (int l2=1; l2<=lmax1; l2++){
	if ( (l1+l2+l)%2 == 0){
	  double dsum = 0.0;
	  int minus_1_n = -1;
	  for (int n=0; n<iOm.extent(0); n++){
	    dsum += jln(l,n) * jln(l1,n) * jln(l2,n) * minus_1_n;
	    minus_1_n = -minus_1_n;
	  }
	  //cbg(l,l1,l2) = (2*l+1) * dsum * 2.0 * minus_1_2_n( (l1+l2+l)/2 + l1 +l);
	  cbg(l2,l1,l) = (2*l+1) * dsum * 2.0 * minus_1_2_n( (l1+l2+l)/2 + l2 );
	}
      }
    }
  }
  /*
  for (int l1=0; l1<=lmax1; l1++){
    for (int l2=0; l2<=lmax1; l2++){
      cout << l1 << " " << l2 << " :  ";
      for (int l=0; l<=lmax; l++){
	double t = cbg(l,l1,l2);
	if (fabs(t)<1e-10) t=0;
	cout << setw(9) << t << " ";
      }
      cout<<endl;
    }
  }
    for (int l1=0; l1<=lmax; l1++){
      double dsum = 0;
      for (int l=0; l<=lmax; l++) dsum += cbg(l,l1,l);
      cout << "l1=" << l1 << " dsum=" << dsum << endl;
    }
  */
}

void Plconv2(Array<double,3>& cbg, int lmax){
  // Computes the convolution of two Legendre Polynomials in a stable way.
  //  Notice that the arguments of the convolution are slightly different in this method than in Plconv.
  // This is closer to usually convolution, i.e., P(z-x)*P(x), as opposed to Plconv, which computes something similar to P(x-z)*P(x)
  // We want to compute:
  //  cbg(l2,l1,l) = (2l+1)/4 * Integrate[ P_l(z) Integrate[ P_{l1}(z-x-1) P_{l2}(x), {x,-1,z}], {z,-1,1} ] +
  //                 (2l+1)/4 * Integrate[ P_l(z) Integrate[ P_{l1}(z-x+1) P_{l2}(x), {x, z,1}], {z,-1,1} ]
  // In Fourier space, this is equivalent to
  //
  //  cbg(l2,l1,l) = (2l+1) (-1)^{(l-l1-l2)/2} \sum_n (-1)^n j_l(n*pi) j_{l1}(n*pi) j_{l2}(n*pi}
  // 
  int lmax1 = lmax;
  cbg.resize(lmax1+1,lmax1+1,lmax+1);
  cbg = 0;
  cbg(0,0,0) = 1.0;
  int N=5000;
  Array<int,1> iOm(N);
  for (int i=0; i<iOm.size(); i++) iOm(i)=i+1;
  Array<double,2> jln;
  GetSphericalBessel_n_pi(jln, iOm, lmax);
  for (int l=1; l<=lmax; l++){
    for (int l1=1; l1<=lmax1; l1++){
      for (int l2=1; l2<=lmax1; l2++){
	if ( (l1+l2+l)%2 == 0){
	  double dsum = 0.0;
	  int minus_1_n = -1;
	  for (int n=0; n<iOm.extent(0); n++){
	    dsum += jln(l,n) * jln(l1,n) * jln(l2,n) * minus_1_n;
	    minus_1_n = -minus_1_n;
	  }
	  //cbg(l,l1,l2) = (2*l+1) * dsum * 2.0 * minus_1_2_n( (l1+l2+l)/2 + l1 +l);
	  cbg(l2,l1,l) = (2*l+1) * dsum * 2.0 * minus_1_2_n( (l1+l2+l)/2 + l1+l2 );
	}
      }
    }
  }
}
void Plconv3(Array<double,3>& cbg, int lmax){
  // Computes the matrix elements between three Legendre Polynomials. 
  // We want to compute:
  //  cbg(l2,l1,l) = (2l+1)/2 * Integrate[ P_l2(x) P_l1(x) P_l(x), {x,-1,1}]
  // This is the most stable way to compute coefficient, although alternatives are possible,
  // and are coded in Plconv3b and Plconv3c.
  Array<double,2> Pln(lmax*2+1,lmax*2+1);
  LegendreCoeff(Pln,lmax*2);
  Array<double,2> Pnl(lmax*2+1,lmax*2+1);
  Pnl = Pln;
  InverseTriangularMatrix("L", Pnl);

  //cout<<"Pnl(0,:)="<<Pnl(0,Range::all())<<endl;
  //cout<<"Pnl(2,:)="<<Pnl(2,Range::all())<<endl;
  cbg.resize(lmax+1,lmax+1,2*lmax+1);
  cbg=0;
  for (int l2=0; l2<=lmax; l2++)
    for (int l1=0; l1<=lmax; l1++)
      for (int n1=l1%2; n1<=l1; n1+=2)
	for (int n2=l2%2; n2<=l2; n2+=2){
	  double pp = Pln(l2,n2)*Pln(l1,n1);
	  int n_n = n1+n2;
	  if (fabs(pp)>1e-6){
	    for (int l=0; l<=2*lmax; l++) cbg(l2,l1,l) += pp*Pnl(n_n,l);
	    //cout<<l2<<" "<<l1<<" "<<n1<<" "<<n2<<" "<< pp << " "<<cbg(l2,l1,Range::all())<<endl;
	    //cout<<"Pnl(0,:)="<<Pnl(0,Range::all())<<endl;
	  }
	}
  
  /*
  cout.precision(12);
  for (int l=0; l<lmax+1; l++)
    for (int n=0; n<lmax+1; n++)
      if (fabs(Pnl(l,n)>1e-4))
	cout<< setw(3)<<l<<" "<< setw(3)<<n<<" "<< Pnl(l,n) << endl;
  */
}
void Plconv3b(Array<double,3>& cbg, int lmax){
  // Computes the matrix elements between three Legendre Polynomials. 
  // We want to compute:
  //  cbg(l2,l1,l) = (2l+1)/2 * Integrate[ P_l2(x) P_l1(x) P_l(x), {x,-1,1}]
  // Uses direct method and computes integrals analytically
  Array<double,2> cPl(lmax+1,lmax+1);
  LegendreCoeff(cPl,lmax);

  cbg.resize(lmax+1,lmax+1,lmax+1);
  cbg=0;
  for (int l1=0; l1<=lmax; l1++){
    cbg(0,l1,l1)=1;
    cbg(l1,0,l1)=1;
    cbg(l1,l1,0)=1./(2*l1+1.);
  }
  for (int l2=1; l2<=lmax; l2++)
    for (int l1=1; l1<=lmax; l1++)
      for (int l=1; l<=lmax; l++){
	double dsum = 0;
	if ( (l+l1+l2)%2 == 0)
	  for (int n=l%2; n<=l; n+=2)
	    for (int n1=l1%2; n1<=l1; n1+=2){
	      double cc = cPl(l,n)*cPl(l1,n1);
	      for (int n2=l2%2; n2<=l2; n2+=2)
		if ( (n+n1+n2)%2 == 0) dsum += cc*cPl(l2,n2)/(n+n1+n2+1);
	    }
	cbg(l2,l1,l)=dsum*(2*l+1);
      }
  //cout<<" Strange "<<cbg(1,1,0)<<endl;
}

void Plconv3c(Array<double,3>& cbg, int lmax){
  // Computes the matrix elements between three Legendre Polynomials. 
  // We want to compute:
  //  cbg(l2,l1,l) = (2l+1)/2 * Integrate[ P_l2(x) P_l1(x) P_l(x), {x,-1,1}]
  //  Uses Fourier expansion, which is better for large l, but is very expensive
  // in this case and does not converge very fast.
  int N=5000;
  Array<int,1> iOm(N);
  for (int i=0; i<iOm.size(); i++) iOm(i)=i;
  Array<double,2> jln;
  GetSphericalBessel_n_pi(jln, iOm, lmax);
  Array<double,3> r(lmax+1,lmax+1,N);
  r=0;
  for (int l2=1; l2<=lmax; l2++){
    for (int l1=1; l1<=lmax; l1++){
      for (int n=0; n<N; n++){
	double dsum1 = 0.0;
	for (int n1=1; n1<N-n; n1++)  dsum1 += jln(l1,n1) * jln(l2,n+n1);
	double dsum2=0;
	for (int n1=1; n1<n; n1++)    dsum2 += jln(l1,n1) * jln(l2,n-n1);
	dsum2 *= minus_1_2_n( l1 );
	double dsum3=0;
	for (int n1=n+1; n1<N; n1++)  dsum3 += jln(l1,n1) * jln(l2,n1-n);
	dsum3 *= minus_1_2_n( l1+l2 );
	r(l1,l2,n)=dsum1+dsum2+dsum3;
	//cout<<setw(3)<<l2<<" "<<setw(3)<<l1<<" "<<n<<" "<< dsum <<endl;
      }
    }
  }
  cbg.resize(lmax+1,lmax+1,lmax+1);
  cbg=0;
  cbg=0;
  for (int l1=0; l1<=lmax; l1++){
    cbg(0,l1,l1)=1;
    cbg(l1,0,l1)=1;
    cbg(l1,l1,0)=1./(2*l1+1.);
  }
  for (int l2=1; l2<=lmax; l2++)
    for (int l1=1; l1<=lmax; l1++)
      for (int l=1; l<=lmax; l++){
	if ( (l+l1+l2)%2 == 0){
	  double dsum1 = 0;
	  for (int n=1; n<N; n++) dsum1 += jln(l,n) * r(l1,l2,n);
	  double dsum2 = 0;
	  for (int n=1; n<N; n++) dsum2 += jln(l,n) * r(l2,l1,n);
	  dsum2 *= minus_1_2_n(l);
	  cbg(l2,l1,l) = (dsum1+dsum2) * (2*l+1.) * minus_1_2_n( (l1+l2+l)/2 + l2);
	}
      }
}



PYBIND11_PLUGIN(tpvert) {
  py::module m("tpvert", "pybind11 wrap for spherical bessel integrals.");
  m.def("jVj", &jVj, "<j_l(k1*r)|2/r|j_l(k2*r)>");
  m.def("jVj_single", &jVj_single, "<j_l(k1*r)|2/r|j_l(k2*r)>");

  m.def("AssemblyGamma", &AssemblyGamma, "put together <j_l|V|j_l> and <Y|P|Y>");
  
  m.def("GiveTanMesh", [](double x0, double L, int Nw) -> py::tuple{
      Array<double,1> om, dom;
      GiveTanMesh(om, dom, x0, L, Nw);
      py::array_t<double, py::array::c_style> _om_(om.extent(0)), _dom_(om.extent(0));
      auto r = _om_.mutable_unchecked<1>();
      auto q = _dom_.mutable_unchecked<1>();
      for (int i = 0; i < r.shape(0); ++i){
        r(i) = om(i);
	q(i) = dom(i);
      }
      return py::make_tuple(_om_,_dom_);
    }, "Produces tangens mesh. Needs parameters: (container& om, container& dom, double& x0, double L, int Nw)"
    );
  py::class_<Compute_Y_P_Y>(m, "Compute_Y_P_Y", "Computes <Y_{l0}|P(omega,k,q)|Y_{l'0}> where P is bubble polarization.")
    .def(py::init<double,double,double,double,double,int>())
    .def("Run", [](class Compute_Y_P_Y& y_p_y, double k, double q, py::array_t<int>& iOm/*, int ik*/)->py::array_t<complex<double>>{
	py::buffer_info info = iOm.request();
	bl::Array<int,1> _iOm_((int*)info.ptr, info.shape[0], bl::neverDeleteData);
	py::array_t<complex<double>,py::array::c_style> _PpP_({y_p_y.lmax+1,y_p_y.lmax+1,_iOm_.extent(0)});
	py::buffer_info info2 = _PpP_.request();
	bl::Array<complex<double>,3> PpP((complex<double>*)info2.ptr, bl::shape(y_p_y.lmax+1,y_p_y.lmax+1,_iOm_.extent(0)), bl::neverDeleteData);
	y_p_y.Run(PpP, k, q, _iOm_);
	return _PpP_;
      }
      )
    .def("Pw0", [](class Compute_Y_P_Y& y_p_y, double q, py::array_t<int>& iOm)-> py::array_t<double>{
	py::buffer_info info = iOm.request();
	bl::Array<int,1> _iOm_((int*)info.ptr, info.shape[0], bl::neverDeleteData);
	
	py::array_t<double,py::array::c_style> _rm_(_iOm_.extent(0));
	py::buffer_info info2 = _rm_.request();
	bl::Array<double,1> rm((double*)info2.ptr, _iOm_.extent(0), bl::neverDeleteData);
	y_p_y.Pw0(q, _iOm_, rm);
	return _rm_;
      }
      )
    ;
  m.def("GetAsymptoteSphBessels", [](py::array_t<int,py::array::c_style>& _nn_, int lmax) -> py::array_t<double>{
      py::buffer_info info = _nn_.request();
      Array<int,1> nn((int*)info.ptr, info.shape[0], bl::neverDeleteData);
      py::array_t<double> _jla_({lmax+1,nn.extent(0)});
      py::buffer_info info_j = _jla_.request();
      Array<double,2> jla((double*)info_j.ptr, bl::shape(info_j.shape[0],info_j.shape[1]), bl::neverDeleteData);
      GetAsymptoteSphBessels(jla, nn, lmax);
      return _jla_;
    }," Computes bessel jl(x) for very large argument x using asymptotic expansion"
    );
  m.def("ProjectToLegendre", [](py::array_t<double,py::array::c_style>& _xt_, py::array_t<double,py::array::c_style>& _ft_, int lmax) -> py::array_t<double>{
      py::buffer_info info_x = _xt_.request();
      Array<double,1> xt((double*)info_x.ptr, info_x.shape[0], bl::neverDeleteData);
      py::buffer_info info_f = _ft_.request();
      Array<double,1> ft((double*)info_f.ptr, info_f.shape[0], bl::neverDeleteData);
      py::array_t<double> _cl_(lmax+1);
      py::buffer_info info_c = _cl_.request();
      Array<double,1> cl((double*)info_c.ptr, info_c.shape[0], bl::neverDeleteData);
      ProjectToLegendre(cl, xt, ft, lmax);
      return _cl_;
    }, " Given function f(x), we compute projection to Legendre Polynomials"
    );

  m.def("LegendreInverseFourierBoson", [](py::array_t<complex<double>,py::array::c_style>& _Gom_,
					  py::array_t<int,py::array::c_style>& _iOm_,
					  int lmax, double beta, int nom, 
					  py::array_t<double,py::array::c_style>& _LF_Ker_even_,
					  py::array_t<double,py::array::c_style>& _LF_Ker_odd_)->py::array_t<double>{
	  py::buffer_info info_G = _Gom_.request();
	  Array<complex<double>,1> Gom((complex<double>*)info_G.ptr, info_G.shape[0], bl::neverDeleteData);
	  py::buffer_info info_w = _iOm_.request();
	  Array<int,1> iOm((int*)info_w.ptr, info_w.shape[0], bl::neverDeleteData);
	  py::buffer_info info_even = _LF_Ker_even_.request();
	  Array<double,2> LF_Ker_even((double*)info_even.ptr, bl::shape(info_even.shape[0],info_even.shape[1]), bl::neverDeleteData);
	  py::buffer_info info_odd = _LF_Ker_odd_.request();
	  Array<double,2> LF_Ker_odd((double*)info_odd.ptr, bl::shape(info_odd.shape[0],info_odd.shape[1]), bl::neverDeleteData);

	  py::array_t<double> _Cl_(lmax+1);
	  py::buffer_info info_c = _Cl_.request();
	  Array<double,1> Cl((double*)info_c.ptr, info_c.shape[0], bl::neverDeleteData);
	  LegendreInverseFourierBoson(Cl, Gom, iOm, lmax, beta, nom, LF_Ker_even, LF_Ker_odd);
	  return _Cl_;
	}, "Given input green's function in bosonic Matsubara points, it returns its expansion in terms of Legendre polynomials."
	);
  
  m.def("SimplePoleInside", [](py::array_t<double,py::array::c_style>& _res_, double x, int lmax){
      py::buffer_info info = _res_.request();
      Array<double,1> res((double*)info.ptr, info.shape[0], bl::neverDeleteData);
      SimplePoleInside(res, x, lmax);
    }, "calculates modified spherical bessel function of the first kind for large argument x."
    );
  m.def("Plconv", [](int lmax) -> py::array_t<double>{
      py::array_t<double> _cbg_({lmax+1,lmax+1,lmax+1});
      py::buffer_info info = _cbg_.request();
      Array<double,3> cbg((double*)info.ptr, bl::shape(info.shape[0],info.shape[1],info.shape[2]), bl::neverDeleteData);
      Plconv(cbg, lmax);
      return _cbg_;
    }, "Computes the convolution of two Legendre Polynomials in a stable way."
    );
  m.def("Plconv2", [](int lmax) -> py::array_t<double>{
      py::array_t<double> _cbg_({lmax+1,lmax+1,lmax+1});
      py::buffer_info info = _cbg_.request();
      Array<double,3> cbg((double*)info.ptr, bl::shape(info.shape[0],info.shape[1],info.shape[2]), bl::neverDeleteData);
      Plconv2(cbg, lmax);
      return _cbg_;
    }, "Computes sligtly different convolution of two Legendre Polynomials in a stable way."
    );
  m.def("Plconv3", [](int lmax) -> py::array_t<double>{
      py::array_t<double> _cbg_({lmax+1,lmax+1,2*lmax+1});
      py::buffer_info info = _cbg_.request();
      Array<double,3> cbg((double*)info.ptr, bl::shape(info.shape[0],info.shape[1],info.shape[2]), bl::neverDeleteData);
      Plconv3(cbg, lmax);
      return _cbg_;
    }, "Computes the matrix elements between three Legendre Polynomials, i.e., P_l1(x)*P_l2(x) = sum_l c_l P_l(x) ."
    );
  m.def("Plconv3b", [](int lmax) -> py::array_t<double>{
      py::array_t<double> _cbg_({lmax+1,lmax+1,lmax+1});
      py::buffer_info info = _cbg_.request();
      Array<double,3> cbg((double*)info.ptr, bl::shape(info.shape[0],info.shape[1],info.shape[2]), bl::neverDeleteData);
      Plconv3b(cbg, lmax);
      return _cbg_;
    }, "Similar to Plconv3, but numerically less stable way"
    );
  m.def("Plconv3c", [](int lmax) -> py::array_t<double>{
      py::array_t<double> _cbg_({lmax+1,lmax+1,lmax+1});
      py::buffer_info info = _cbg_.request();
      Array<double,3> cbg((double*)info.ptr, bl::shape(info.shape[0],info.shape[1],info.shape[2]), bl::neverDeleteData);
      Plconv3c(cbg, lmax);
      return _cbg_;
    }, "Similar to Plconv3, but numerically less stable way"
    );
  return m.ptr();
}
