// @Copyright 2018 Kristjan Haule and Kun Chen
#include <iostream>
#include <blitz/array.h>
#include <fstream>
#include "util.h"
#include "interpolate.h"
#include "tanmesh.h"


void GiveDoubleExpMesh(bl::Array<double,1>& x, bl::Array<double,1>& dx, double a, double b, int Nx, double al=2.1){
  for (int i=0; i<Nx; i++){
    double t = i/(Nx-1.);
    double x0 = M_PI/2.*sinh(al*t);
    x(i) = a + (b-a) * tanh(x0);
    dx(i) = (b-a)*al*M_PI/2.*cosh(al*t)/ipower(cosh(x0),2);
  }
}
void GiveDoubleExpMesh2(bl::Array<double,1>& x, bl::Array<double,1>& dx, double a, double b, int Nx, double al=2.1){
  for (int i=0; i<Nx; i++){
    double t = -1 + 2*i/(Nx-1.);
    double x0 = M_PI/2.*sinh(al*t);
    x(i) = a + (b-a)/2. * (1. + tanh(x0));
    dx(i) = (b-a)*al*M_PI/2.*cosh(al*t)/ipower(cosh(x0),2);
  }
}
void ComposeTanMesh(bl::Array<double,1>& kx, int Nq, double L1, double L2){
  // L1 = kF L2=cutoff
  bl::Array<double,1> om, dom;
  double x0 = 2*L1/(10*Nq/2);
  GiveTanMesh(om, dom, x0, L1, Nq/2);
  om += L1;
  double dh = om(om.size()-1)-om(om.size()-2);
  int Nr = (L2-2*L1)/dh+1;
  double dx = (L2-2*L1-1e-7)/(Nr-1.);
  kx.resize(om.extent(0) + Nr-1);
  for (int i=0; i<om.size(); i++){
    kx(i) = om(i)+1e-10;
  }
  for (int i=1; i<Nr; i++){
    double x = 2*L1 + i*dx;
    kx(om.extent(0)+i-1) = x;
  }
}

inline double sqr(double x){return x*x;}

class HartreeFock{
public:
  static const int r=7;
  static const int N = (1<<r)+1; // 2^r+1
  static const int r_c=8;
  static const int N_c = (1<<r_c)+1; // 2^r_c+1
  double kF, cutoff, beta, lmbda, dmu; // dmu is a small chemical potential shift
  bl::TinyVector<double,N> fm, gm;
  bl::TinyVector<double,2*(N-1)+1> fm2;
  bl::Array<double,1> x, dx, x2, dx2;
private:
  bl::Array<bl::TinyVector<double,N_c>,1> fCu;
public:
  bl::Array<double,1> kx;
  Spline1D<double> epsx;
public:
  HartreeFock(double _kF_, double _cutoff_, double _beta_, double _lmbda_=0, double _dmu_=0, int Nq=256) :
    kF(_kF_), cutoff(_cutoff_), beta(_beta_), lmbda(_lmbda_), dmu(_dmu_),
    x(N), dx(N), x2(2*(N-1)+1), dx2(2*(N-1)+1),
    kx(Nq), epsx(Nq)
  {
    /*
    for (int i=0; i<Nq; i++){
      kx(i) = 1e-10 + (cutoff-1e-8) * i/(Nq-1.);
      epsx[i] = kx(i)*kx(i)-kF*kF;
    }
    */
    //cout<<"The size is "<< N << " x.size=" << x.size() << " dx.size=" << dx.size() <<" kx.size="<< kx.size() << " epsx.size="<< epsx.size() << endl;
    ComposeTanMesh(kx, Nq, kF, cutoff);
    epsx.resize(kx.extent(0));
    //cout<<"The size is now "<< N << " x.size=" << x.size() << " dx.size=" << dx.size() <<" kx.size="<< kx.size() << " epsx.size="<< epsx.size() << endl;
    for (int i=0; i<kx.extent(0); i++){
      epsx[i] = kx(i)*kx(i)-kF*kF;
    }
    epsx.splineIt(kx);
    //cout<< "epsx.f.size=" << epsx.f.size() << " epsx.f2.size="<< epsx.f2.size() << " epsx.dxi.size="<< epsx.dxi.size() << endl;
    GiveDoubleExpMesh(x, dx, 0, 1.0, x.extent(0));
    GiveDoubleExpMesh2(x2, dx2, 0, 1.0, x2.extent(0));
  }
  double dSx(double k, double q, const Spline1D<double>& epsx){
    return 0.5 * q * log((sqr(k+q)+lmbda)/(sqr(k-q)+lmbda)) * ferm(epsx(Interp(q,kx))*beta);
  }
  double Sx(double k, const Spline1D<double>& epsx)
  {
    if (k<kF){
      for (int i=0; i<x.size(); i++)  fm[i] = dSx(k, x(i)*k, epsx) * dx(i) * k;                  // [0....k] with dense mesh at k
      double rm1 = romberg2<N,r>(fm, 1.0)/(k*kF);
      for (int i=0; i<x2.size(); i++) fm2[i] = dSx(k, k + (kF-k)*x2(i), epsx) * dx2(i) * (kF-k); // [k,kF] with dense mesh at both k and kF
      double rm2 = romberg2<2*(N-1)+1,r+1>(fm2, 1.0)/(k*kF);
      for (int i=0; i<x.size(); i++) fm[i] = dSx(k, kF + (cutoff-kF)*(1 - x(i)), epsx) * dx(i) * (cutoff-kF); // [kF,cutoff] with dense mesh at kF
      double rm3 = romberg2<N,r>(fm, 1.0)/(k*kF);
      return rm1 + rm2 + rm3;
    }
    if (k>kF){
      for (int i=0; i<x.size(); i++) fm[i] = dSx(k, x(i)*kF, epsx) * dx(i) * kF;                  // [0...kF] with dense mesh at kF
      double rm1 = romberg2<N,r>(fm, 1.0)/(k*kF);
      for (int i=0; i<x2.size(); i++) fm2[i] = dSx(k, kF + (k-kF)*x2(i), epsx) * dx2(i) * (k-kF); // [kF...k] with dense mesh at both kF and k
      double rm2 = romberg2<2*(N-1)+1,r+1>(fm2, 1.0)/(k*kF);
      for (int i=0; i<x.size(); i++) fm[i] = dSx(k,  k + (cutoff-k)*(1 - x(i)), epsx) * dx(i) * (cutoff-k); // [k...cutoff] with dense mesh at k
      double rm3 = romberg2<N,r>(fm, 1.0)/(k*kF);
      return rm1 + rm2 + rm3;
    }
    // k==kF
    for (int i=0; i<x.size(); i++) fm[i] = dSx(k, x(i)*kF, epsx) * dx(i) * kF;
    double rm1 = romberg2<N,r>(fm, 1.0)/(k*kF);
    for (int i=0; i<x.size(); i++) fm[i] = dSx(k, kF + (cutoff-kF)*(1 - x(i)), epsx) * dx(i) * (cutoff-kF);
    double rm3 = romberg2<N,r>(fm, 1.0)/(k*kF);
    return rm1 + rm3;
  }
  void cmp()
  {
    for (int itt=0; itt<100; itt++){
      double diff=0;
      double sx_kF = Sx(kF,epsx);
      for (int i=0; i<epsx.size(); i++){
	double sx = Sx( kx(i), epsx);
	double epsn = kx(i)*kx(i) - kF*kF -2*kF/pi * (sx - sx_kF) - dmu;
	diff += fabs(epsn-epsx[i]);
	epsx[i] = epsn;
      }
      epsx.splineIt(kx);
      //clog<<"diff="<<diff<<endl;
      if (diff<1e-6) break;
    }

    /*
    ofstream bck2("bck2.dat");
    for (int i=0; i<epsx.size(); i++){
      double k = kx(i);
      double eps = epsx[i];
      double fm = -k*k*ferm(-eps*beta)*ferm(eps*beta)/(pi*pi);
      bck2 << k <<"  "<< fm << endl;
    }
    bck2.close();
    */
  }
  double ferm(double x) { return 1./(exp(x)+1.0); }
  double Sx0(double x) {
    if (x<1e-6) return 2.0;
    if (fabs(x-1.)<1e-6) return 1.0;
    return 1.0 + 0.5*(1/x-x)*log((x+1)/fabs(x-1));
  }

  double Cu(double k, double t)
  {
    double epsk = epsx(Interp(k,kx));
    if (epsk<0) return k * exp(t*epsk)/(1.+exp(beta*epsk));
    return k * exp(-(beta-t)*epsk)/(1.+exp(-beta*epsk));
  }
  double Cu(double k, double t, double epsk)
  {
    if (epsk<0) return k * exp(t*epsk)/(1.+exp(beta*epsk));
    return k * exp(-(beta-t)*epsk)/(1.+exp(-beta*epsk));
  }
  void Cins(bl::Array<double,1>& cins, double k, double Q, const bl::Array<double,1>& tau)
  {
    /*
    static const int r_c=8;
    static const int N_c = (1<<r_c)+1; // 2^r+1
    */
    double a = fabs(k-Q), b = k+Q;
    double dq = (b-a)/(N_c-1.);
    if ( fCu.extent(0) != tau.extent(0) ) fCu.resize( tau.extent(0) );
    ///*static*/ bl::Array<bl::TinyVector<double,N_c>,1> fCu(tau.extent(0));
    int ia=0;
    for (int iq=0; iq<N_c; iq++){
      double q = a + iq*dq;
      double epsk = epsx(InterpLeft(q,ia,kx));
      for (int it=0; it<tau.extent(0); it++) fCu(it)[iq] = Cu(q, tau(it), epsk);
    }
    for (int it=0; it<tau.extent(0); it++)
      cins(it) = romberg2<N_c,r_c>(fCu(it), b-a);
  }
  void P0qk(bl::Array<double,1>& p0t, double k, double Q, const bl::Array<double,1>& tau)
  {
    bl::Array<double,1> cins(tau.extent(0));
    Cins(cins, k, Q, tau);
    double epsk = epsx(Interp(k,kx));
    for (int it=0; it<tau.extent(0); it++){
      p0t(it) = Cu(k, beta-tau(it), epsk) * cins(it);
    }
  }
  bl::Array<double,1> P0(double Q, const bl::Array<double,1>& tau)
  {
    const int r=10;
    const int N = (1<<r)+1; // 2^r+1
    double a = 1e-11, b = cutoff;
    double dk = (b-a)/(N-1.);
    bl::Array<bl::TinyVector<double,N>,1> fP0(tau.extent(0));

    bl::Array<double,1> p0t(tau.extent(0));
    for (int i=0; i<N; i++){
      double k = a + i*dk; // k has size N
      P0qk(p0t, k, Q, tau);// p0t.size == tau.size
      for (int it=0; it<tau.extent(0); it++)
	fP0(it)[i] = p0t(it);
    }
    for (int it=0; it<tau.extent(0); it++)
      p0t(it) = -1./(2*pi*pi*Q) * romberg2<N,r>(fP0(it), b-a);
    return p0t;
  }
  
  double dEx(double k, double q, const Spline1D<double>& epsx){
    return q * ferm(epsx(Interp(q,kx))*beta) * log((k+q)/fabs(k-q));
  }
  double dExchange(double k, const Spline1D<double>& epsx, int ii)
  {//
   // Computes integral
   //  \int_0^{2*kF} dq  q * f(e_q) * log((k+q)/|k-q|)
   //
    //ofstream log(std::string("debuga.")+std::to_string(ii));
    //log.precision(15);
    double tcutoff=2*kF;
    if (k<kF){
      for (int i=0; i<x.size(); i++)  fm[i] = dEx(k, x(i)*k, epsx) * dx(i) * k;                  // [0....k] with dense mesh at k
      double rm1 = romberg2<N,r>(fm, 1.0);
      //for (int i=0; i<x.size(); i++) log << k*x(i)/kF << "  " << dEx(k, x(i)*k, epsx) <<endl;
      for (int i=0; i<x2.size(); i++) fm2[i] = dEx(k, k + (kF-k)*x2(i), epsx) * dx2(i) * (kF-k); // [k,kF] with dense mesh at both k and kF
      double rm2 = romberg2<2*(N-1)+1,r+1>(fm2, 1.0);
      //for (int i=0; i<x2.size(); i++) log << (k + (kF-k)*x2(i))/kF << "  " << dEx(k, k + (kF-k)*x2(i), epsx) << endl;
      for (int i=0; i<x.size(); i++) fm[i] = dEx(k, kF + (tcutoff-kF)*(1 - x(i)), epsx) * dx(i) * (tcutoff-kF); // [kF,cutoff] with dense mesh at kF
      double rm3 = romberg2<N,r>(fm, 1.0);
      //for (int i=x.size()-1; i>=0; i--) log << (kF + (tcutoff-kF)*(1 - x(i)))/kF << "  " << dEx(k, kF + (tcutoff-kF)*(1 - x(i)), epsx) << endl;
      //log<<"# rm= " << (rm1+rm2+rm3)/kF << endl;
      return rm1 + rm2 + rm3;
    }
    if (k>kF){
      for (int i=0; i<x.size(); i++) fm[i] = dEx(k, x(i)*kF, epsx) * dx(i) * kF;                  // [0...kF] with dense mesh at kF
      double rm1 = romberg2<N,r>(fm, 1.0);
      //for (int i=0; i<x.size(); i++) log << kF*x(i)/kF << "  " << dEx(k, x(i)*kF, epsx) <<endl;
      for (int i=0; i<x2.size(); i++) fm2[i] = dEx(k, kF + (k-kF)*x2(i), epsx) * dx2(i) * (k-kF); // [kF...k] with dense mesh at both kF and k
      double rm2 = romberg2<2*(N-1)+1,r+1>(fm2, 1.0);
      //for (int i=0; i<x2.size(); i++) log << (kF + (k-kF)*x2(i))/kF << "  " << dEx(k, kF + (k-kF)*x2(i), epsx) << endl;
      for (int i=0; i<x.size(); i++) fm[i] = dEx(k,  k + (tcutoff-k)*(1 - x(i)), epsx) * dx(i) * (tcutoff-k); // [k...cutoff] with dense mesh at k
      double rm3 = romberg2<N,r>(fm, 1.0);
      //for (int i=x.size()-1; i>=0; i--) log << (k + (tcutoff-k)*(1 - x(i)))/kF << "  " << dEx(k,  k + (tcutoff-k)*(1 - x(i)), epsx) << endl;
      //log<<"# rm= " << (rm1+rm2+rm3)/kF << endl;
      return rm1 + rm2 + rm3;
    }
    // k==kF
    for (int i=0; i<x.size(); i++) fm[i] = dEx(k, x(i)*kF, epsx) * dx(i) * kF;
    double rm1 = romberg2<N,r>(fm, 1.0);
    //for (int i=0; i<x.size(); i++) log << kF*x(i)/kF << "  " << dEx(k, x(i)*kF, epsx) <<endl;
    for (int i=0; i<x.size(); i++) fm[i] = dEx(k, kF + (tcutoff-kF)*(1 - x(i)), epsx) * dx(i) * (tcutoff-kF);
    double rm3 = romberg2<N,r>(fm, 1.0);
    //for (int i=x.size()-1; i>=0; i--) log << (kF + (tcutoff-kF)*(1 - x(i)))/kF << "  " << dEx(k, kF + (tcutoff-kF)*(1 - x(i)), epsx) << endl;
    //log<<"# rm= " << (rm1+rm3)/kF << endl;
    return rm1 + rm3;
  }
  double Exchange()
  { // Computes the integral
    //  -1/(pi^3 * n0) * \int_0^{2*kF} dk k f(e_k) \int_0^{2*kF} dq q f(e_q)  log((k+q)/|k-q|)
    //
    gm=0;
    for (int i=0; i<x.size(); i++){
      double k = x(i)*kF;
      if (i==0) k += 1e-10; /// exact zero gives nan for logarithm
      double fmk = ferm(epsx(Interp(k,kx))*beta);
      double VAL = dExchange(k, epsx, i)*k*fmk;
      gm[i] = VAL * dx(i)*kF;
    }
    double rm1 = romberg2<N,r>(gm, 1.0);
    double tcutoff=2*kF;
    gm=0;
    for (int i=x.size()-1; i>0; i--){
      double k = kF + (tcutoff-kF)*(1. - x(i));
      double fmk = ferm(epsx(Interp(k,kx))*beta);
      double VAL = dExchange(k, epsx, 2*x.size()-1-i )*k*fmk;
      gm[i] = VAL * dx(i) * (tcutoff-kF);
    }
    double rm3 = romberg2<N,r>(gm, 1.0);
    
    double n0 = (kF*kF*kF)/(3*pi*pi);
    double exc0 = -3./2.*kF/pi;

    double exc = -(rm1+rm3)/(pi*pi*pi) * 1/n0;
    //cout << "exc(T=0)="<< exc0 << endl;
    return exc;
  }
  double Exchange2()
  { // Computes the integral
    //  -1/(pi^3 * n0) * \int_0^{2*kF} dk k f(e_k) \int_0^{2*kF} dq q f(e_q)  log((k+q)/|k-q|)
    //
    //  by subtracting the exact value of the integral at T=0, and computing the correction to T=0 result.
    for (int i=0; i<x.size(); i++){
      double k = x(i)*kF;
      if (i==0) k += 1e-10;
      double fmk = ferm(epsx(Interp(k,kx))*beta);
      double theta = (k<kF) ? 1.0 : 0.0;
      double _fmk_ = fmk - theta;
      double res = k*kF + 0.5*(kF*kF-k*k)*log((kF+k)/fabs(kF-k));
      double VAL = (dExchange(k, epsx, 1000+i) + res )*k*_fmk_;
      gm[i] = VAL * dx(i) * kF;
      //cout << x(i) << "  " << gm[i] << endl;
    }
    double rm1 = romberg2<N,r>(gm, 1.0);
    double tcutoff=2*kF;
    gm[0]=0;
    for (int i=x.size()-1; i>0; i--){
      double k = kF + (tcutoff-kF)*(1. - x(i));
      double fmk = ferm(epsx(Interp(k,kx))*beta);
      double theta = (k<kF) ? 1.0 : 0.0;
      double _fmk_ = fmk - theta;
      double res = k*kF + 0.5*(kF*kF-k*k)*log((kF+k)/fabs(kF-k));
      double VAL = (dExchange(k, epsx, 1000+i+x.size()) + res)*k*_fmk_;
      gm[i] = VAL * dx(i) * (tcutoff-kF);
      //cout << x(i) << " " << gm[i] << endl;
    }
    double rm3 = romberg2<N,r>(gm, 1.0);
    
    double n0 = (kF*kF*kF)/(3*pi*pi);
    double exc0 = -3./2.*kF/pi;
    
    double exc = -(rm1+rm3)/(pi*pi*pi) * 1/n0;
    //cout << "exc(T=0)=" << exc0 << " correction=" << exc << endl;
    return exc + exc0;
  }
};


double Cu(double k, double t, double beta, double kF)
{
  double epsk = k*k-kF*kF;
  if (epsk<0){
    return k * exp(t*epsk)/(1.+exp(beta*epsk));
  }else{
    return k * exp(-(beta-t)*epsk)/(1.+exp(-beta*epsk));
  }
}

double Cins(double k, double Q, double t, double beta, double kF)
{
  const int r=8;
  const int N = (1<<r)+1; // 2^r+1
  double a = fabs(k-Q), b = k+Q;
  double dq = (b-a)/(N-1.);
  static bl::TinyVector<double,N> fCu;
  for (int i=0; i<N; i++){
    double q = a + i*dq;
    fCu[i] = Cu(q,t,beta,kF);
  }
  return romberg2<N,r>(fCu, b-a);
}

double Cu_manalytic(double k, double t, double beta, double kF)
{
  double epsk = k*k-kF*kF;
  if (epsk<0){
    return k * exp(t*epsk)* (1./(1.+exp(beta*epsk)) - 1.0);
  }else{
    return k * exp(-(beta-t)*epsk)/(1.+exp(-beta*epsk));
  }
}
double Cins_analytic(double k, double Q, double t, double beta, double kF)
{
  const int r=8;
  const int N = (1<<r)+1; // 2^r+1
  static bl::TinyVector<double,N> fCu;
  double a = fabs(k-Q), b = k+Q;
  //if (a>=b) return 0;
  if (a<kF && kF<=b){
    a = fabs(k-Q);
    b = kF;
    double dq = (b-a)/(N-1.);
    for (int i=0; i<N; i++)  fCu[i] = Cu_manalytic(a + i*dq, t,beta,kF);
    double part1 = romberg2<N,r>(fCu, b-a);
    
    a = kF;
    b = k+Q;
    dq = (b-a)/(N-1.);
    for (int i=0; i<N; i++) fCu[i] = Cu_manalytic(a + i*dq, t,beta,kF);
    double part2 = romberg2<N,r>(fCu, b-a);
    
    double kmq = fabs(k-Q);
    double te = t*(kmq*kmq-kF*kF);
    double analytic = fabs(te)>1e-6 ? (1.0-exp(te))/(2*t) : -0.5*(kmq*kmq-kF*kF)*(1. + te/2.*(1. + te/3.));
    return part1 + part2 + analytic;
  }else{
    double dq = (b-a)/(N-1.);
    for (int i=0; i<N; i++) fCu[i] = Cu_manalytic(a + i*dq,t,beta,kF);
    double analytic=0;
    if (a<kF && b<kF){
      analytic = fabs(t)>1e-7 ? (exp(t*(b*b-kF*kF))-exp(t*(a*a-kF*kF)))/(2*t) : 0.5*(b*b-a*a)*(1 + 0.5*(a*a + b*b - 2*kF*kF)*t);
    }
    return romberg2<N,r>(fCu, b-a) + analytic;
  }
}

double P0qk(double k, double Q, double t, double beta, double kF)
{
  //return Cu(k, beta-t,beta,kF) * Cins(k,Q,t,beta,kF);
  return Cu(k, beta-t,beta,kF) * Cins_analytic(k,Q,t,beta,kF);
}
double P0(double Q, double t, double beta, double kF, double cutoffk)
{
  const int r=10;
  const int N = (1<<r)+1; // 2^r+1
  double a = 1e-11, b = cutoffk;
  double dk = (b-a)/(N-1.);
  bl::TinyVector<double,N> fP0;
  for (int i=0; i<N; i++){
    double k = a + i*dk;
    fP0[i] = P0qk(k,Q,t,beta,kF);
  }
  return -1./(2*pi*pi*Q) * romberg2<N,r>(fP0, b-a);
}


bl::Array<double,1> Cins_analytic_fast(double k, double Q, const bl::Array<double,1>& tau, double beta, double kF)
{
  const int r=8;
  const int N = (1<<r)+1; // 2^r+1
  static bl::Array<bl::TinyVector<double,N>,1> fCu(tau.extent(0));
  double a = fabs(k-Q), b = k+Q;

  bl::Array<double,1> res(tau.extent(0));
  if (a<kF && kF<=b){
    a = fabs(k-Q);
    b = kF;
    double dq = (b-a)/(N-1.);
    for (int i=0; i<N; i++){
      double k = a + i*dq;
      double epsk = k*k-kF*kF;
      double wk = (epsk<0) ? k * (1./(1.+exp(beta*epsk)) - 1.0) : k /(1.+exp(-beta*epsk));
      for (int it=0; it<tau.extent(0); it++)
	fCu(it)[i] = (epsk<0) ? exp(tau(it)*epsk) * wk : exp(-(beta-tau(it))*epsk) * wk;
    }
    for (int it=0; it<tau.extent(0); it++)
      res(it) = romberg2<N,r>(fCu(it), b-a);
    
    a = kF;
    b = k+Q;
    dq = (b-a)/(N-1.);
    for (int i=0; i<N; i++){
      double k = a + i*dq;
      double epsk = k*k-kF*kF;
      double wk = (epsk<0) ? k * (1./(1.+exp(beta*epsk)) - 1.0) : k /(1.+exp(-beta*epsk));
      for (int it=0; it<tau.extent(0); it++)
	fCu(it)[i] = (epsk<0) ? exp(tau(it)*epsk) * wk : exp(-(beta-tau(it))*epsk) * wk;
    }
    for (int it=0; it<tau.extent(0); it++)
      res(it) += romberg2<N,r>(fCu(it), b-a);
    
    double kmq = fabs(k-Q);
    double eps_kmq = (kmq*kmq-kF*kF);
    for (int it=0; it<tau.extent(0); it++){
      double te = tau(it)*eps_kmq;
      res(it) += fabs(te)>1e-6 ? (1.0-exp(te))/(2*tau(it)) : -0.5*eps_kmq*(1. + te/2.*(1. + te/3.));
    }
    return res;
  }else{
    double dq = (b-a)/(N-1.);
    for (int i=0; i<N; i++){
      double k = a + i*dq;
      double epsk = k*k-kF*kF;
      double wk = (epsk<0) ? k * (1./(1.+exp(beta*epsk)) - 1.0) : k /(1.+exp(-beta*epsk));
      for (int it=0; it<tau.extent(0); it++)
	fCu(it)[i] = (epsk<0) ? exp(tau(it)*epsk) * wk : exp(-(beta-tau(it))*epsk) * wk;
    }
    for (int it=0; it<tau.extent(0); it++)
      res(it) = romberg2<N,r>(fCu(it), b-a);
    
    if (a<kF && b<kF){
      double eps_b = (b*b-kF*kF);
      double eps_a = (a*a-kF*kF);
      for (int it=0; it<tau.extent(0); it++){
	res(it) += fabs(tau(it))>1e-7 ? (exp(tau(it)*eps_b)-exp(tau(it)*eps_a))/(2*tau(it)) : 0.5*(eps_b-eps_a)*(1. + 0.5*(eps_a+eps_b)*tau(it));
      }
    }
    return res;
  }
}


bl::Array<double,1> P0qk_fast(double k, double Q, const bl::Array<double,1>& tau, double beta, double kF)
{
  bl::Array<double,1> cins = Cins_analytic_fast(k, Q, tau, beta, kF);
  double epsk = k*k-kF*kF;
  double wk =  (epsk<0) ? k /(1.+exp(beta*epsk)) : k /(1.+exp(-beta*epsk));
  for (int it=0; it<tau.extent(0); it++){
    double Cu =  (epsk<0) ? exp((beta-tau(it))*epsk) * wk : exp(-tau(it)*epsk) * wk;
    cins(it) *= Cu;
  }
  return cins;
}

bl::Array<double,1> P0_fast(double Q, const bl::Array<double,1>& tau, double beta, double kF, double cutoffk)
{
  const int r=10;
  const int N = (1<<r)+1; // 2^r+1
  double a = 1e-11, b = cutoffk;
  double dk = (b-a)/(N-1.);
  bl::Array<bl::TinyVector<double,N>,1> fP0(tau.extent(0));
  for (int i=0; i<N; i++){
    double k = a + i*dk;
    bl::Array<double,1> cins = P0qk_fast(k,Q,tau,beta,kF);
    for (int it=0; it<tau.extent(0); it++) fP0(it)[i] = cins(it);
  }
  bl::Array<double,1> p0(tau.extent(0));
  for (int it=0; it<tau.extent(0); it++) p0(it) = -1./(2*pi*pi*Q) * romberg2<N,r>(fP0(it), b-a);
  return p0;
}

/*
class P21{
public:
  double kF, kF2, beta, cutoffk, cutoffq, smallk;
  spline1D<double> S_x;
  mesh1D kx;
  
  double Sx(double q, double k){
    return q/(k*kF) * log((k+q)/fabs(k-q)) * 1.0/(1.0 + exp(beta*(q*q-kF2)));
  }
  double ferm(double k){
    return 1.0/(1.0 + exp(beta*(k*k-kF2)));
  }
  double Sx_minus(double q, double k, double fmk){
    return q/(k*kF) * log((k+q)/fabs(k-q)) * (ferm(q)-fmk);
  }
  P21(double _kF_, double _beta_, double _cutoffk_, double _cutoffq_, double _smallk_) :
    kF(_kF_), kF2(kF*kF), beta(_beta_), cutoffk(_cutoffk_), cutoffq(_cutoffq_), smallk(_smallk_){
    const int r=8;
    const int N = (1<<r)+1; // 2^r+1
    double a = smallk, b = cutoffk;
    double dk = (b-a)/(N-1.);
    double a_q = smallk+1e-10, b_q = cutoffq;
    double dq = (b_q-a_q)/(N-1.);
    bl::TinyVector<double,N> fP0;
    kx.resize(N);
    S_x.resize(N);
    for (int i=0; i<N; i++){
      double k = a + i*dk;
      kx[i] = k;
      //for (int j=0; j<N; j++) fP0[j] = Sx(a_q + j*dq, k);
      double fmk = ferm(k);
      for (int j=0; j<N; j++) fP0[j] = Sx_minus(a_q + j*dq, k, fmk);
      S_x[i] = romberg2<N,r>(fP0, b_q-a_q);
      S_x[i] += fmk/kF*( (b_q-a_q) + (b_q*b_q-k*k)*log((b_q+k)/fabs(b_q-k))/(2*k) - (a_q*a_q-k*k)*log((a_q+k)/fabs(a_q-k))/(2*k) );
    }
    kx.SetUp(0);
    double x1 = 0.5*(kx[1]+kx[0]);
    double df1 = (S_x[1]-S_x[0])/(kx[1]-kx[0]);
    double x2 = 0.5*(kx[2]+kx[1]);
    double df2 = (S_x[2]-S_x[1])/(kx[2]-kx[1]);
    double df0 = df1 + (df2-df1)*(kx[0]-x1)/(x2-x1);
    x1 = 0.5*(kx[N-1]+kx[N-2]);
    df1 = (S_x[N-1]-S_x[N-2])/(kx[N-1]-kx[N-2]);
    x2 = 0.5*(kx[N-2]+kx[N-3]);
    df2 = (S_x[N-2]-S_x[N-3])/(kx[N-2]-kx[N-3]);
    double dfn = df1 + (df2-df1)*(kx[N-1]-x1)/(x2-x1);
    S_x.splineIt(kx, df0, dfn);
  }
  double P1qk(double k, double Q, double t)
  {
    double epsk = k*k-kF2;
    double fk = 1/(exp(beta*epsk)+1.0);
    double fmk = 1.-fk;
    return S_x(kx.Interp(k)) * (t*fmk - (beta-t)*fk) * Cu(k,beta-t,beta,kF) * Cins(k,Q,t,beta,kF);
  }
  double P1(double Q, double t)
  {
    const int r=10;
    const int N = (1<<r)+1; // 2^r+1
    double a = smallk, b = cutoffk;
    double dk = (b-a)/(N-1.);
    bl::TinyVector<double,N> fP0;
    for (int i=0; i<N; i++){
      double k = a + i*dk;
      fP0[i] = P1qk(k,Q,t);
    }
    return -kF/(pi*pi*pi*Q) * romberg2<N,r>(fP0, b-a);
  }
};
*/

class PO2{
  static const int r=5;
  static const int N = (1<<r)+1; // 2^r+1
  bl::TinyVector<double,N> fm;
  bl::TinyVector<double,2*(N-1)+1> fm2;
  bl::Array<double,1> x, dx, x2, dx2;
  double kF, kF2, beta, cutoffk, cutoffq, smallk;
  Spline1D<double> S_x;
  bl::Array<double,1> kx;
public:
  PO2(double _kF_, double _beta_, double _cutoffk_, double _cutoffq_, double _smallk_) :
    x(N), dx(N), x2(2*(N-1)+1), dx2(2*(N-1)+1),
    kF(_kF_), kF2(kF*kF), beta(_beta_), cutoffk(_cutoffk_), cutoffq(_cutoffq_), smallk(_smallk_)
  {
    GiveDoubleExpMesh(x, dx, 0, 1.0, x.extent(0));
    GiveDoubleExpMesh2(x2, dx2, 0, 1.0, x2.extent(0));
    int Nq=128;
    ComposeTanMesh(kx, Nq, kF, cutoffk);
    S_x.resize(kx.extent(0));
    for (int i=0; i<kx.extent(0); i++) S_x[i] = Sx(kx(i));
    S_x.splineIt(kx);
  }
  double dSx(double k, double q){
    return q * log((k+q)/fabs(k-q)) * 1.0/(1.0 + exp(beta*(q*q-kF2))); // 1/(k*kF)
  }
  double Sx(double k)
  {
    if (k<kF){
      for (int i=0; i<x.size(); i++)  fm[i] = dSx(k, x(i)*k) * dx(i) * k;
      double rm1 = romberg2<N,r>(fm, 1.0)/(k*kF);
      for (int i=0; i<x2.size(); i++) fm2[i] = dSx(k, k + (kF-k)*x2(i)) * dx2(i) * (kF-k);
      double rm2 = romberg2<2*(N-1)+1,r+1>(fm2, 1.0)/(k*kF);
      for (int i=0; i<x.size(); i++) fm[i] = dSx(k, kF + (cutoffk-kF)*(1 - x(i))) * dx(i) * (cutoffk-kF);
      double rm3 = romberg2<N,r>(fm, 1.0)/(k*kF);
      return rm1 + rm2 + rm3;
    }
    if (k>kF){
      for (int i=0; i<x.size(); i++) fm[i] = dSx(k, x(i)*kF) * dx(i) * kF;
      double rm1 = romberg2<N,r>(fm, 1.0)/(k*kF);
      for (int i=0; i<x2.size(); i++) fm2[i] = dSx(k, kF + (k-kF)*x2(i)) * dx2(i) * (k-kF);
      double rm2 = romberg2<2*(N-1)+1,r+1>(fm2, 1.0)/(k*kF);
      for (int i=0; i<x.size(); i++) fm[i] = dSx(k,  k + (cutoffk-k)*(1 - x(i))) * dx(i) * (cutoffk-k);
      double rm3 = romberg2<N,r>(fm, 1.0)/(k*kF);
      return rm1 + rm2 + rm3;
    }
    // k==kF
    for (int i=0; i<x.size(); i++) fm[i] = dSx(k, x(i)*kF) * dx(i) * kF;
    double rm1 = romberg2<N,r>(fm, 1.0)/(k*kF);
    for (int i=0; i<x.size(); i++) fm[i] = dSx(k, kF + (cutoffk-kF)*(1 - x(i))) * dx(i) * (cutoffk-kF);
    double rm3 = romberg2<N,r>(fm, 1.0)/(k*kF);
    return rm1 + rm3;
  }
  void P1qk(bl::Array<double,1>& p1, double k, double Q, const bl::Array<double,1>& tau)
  {
    const int r=8;
    const int N = (1<<r)+1; // 2^r+1
    double a = fabs(k-Q), b = k+Q;
    double dq = (b-a)/(N-1.);
    static bl::Array<bl::TinyVector<double,N>,1> fCu(tau.extent(0));
    for (int i=0; i<N; i++){
      double q = a + i*dq;
      double epsq = q*q-kF*kF;
      double wCu = (epsq<0) ? q/(1.+exp(beta*epsq)) : q/(1.+exp(-beta*epsq));
      for (int it=0; it<tau.extent(0); it++)
	fCu(it)[i] = (epsq<0) ? wCu * exp(tau(it)*epsq) : wCu * exp(-(beta-tau(it))*epsq);
    }
    bl::Array<double,1> p1in(tau.extent(0));
    for (int it=0; it<tau.extent(0); it++)
      p1in(it) = romberg2<N,r>(fCu(it), b-a);
    
    double epsk = k*k-kF2;
    double fk = 1/(exp(beta*epsk)+1.0);
    double fmk = 1.-fk;
    double sx = S_x(Interp(k,kx));
    double wk = (epsk<0) ? sx * k /(1.+exp(beta*epsk)) : sx * k /(1.+exp(-beta*epsk));
    
    for (int it=0; it<tau.extent(0); it++){
      double cu = (epsk<0) ? exp((beta-tau(it))*epsk) : exp(-tau(it)*epsk);
      p1(it) = wk*cu * (tau(it)*fmk - (beta-tau(it))*fk) * p1in(it);
    }
  }
  bl::Array<double,1> P1(double Q, const bl::Array<double,1>& tau)
  {
    const int r=10;
    const int N = (1<<r)+1; // 2^r+1
    double a = smallk, b = cutoffk;
    double dk = (b-a)/(N-1.);
    
    bl::Array<bl::TinyVector<double,N>,1> fP0(tau.extent(0));
    bl::Array<double,1> p1(tau.extent(0));
    for (int i=0; i<N; i++){
      P1qk(p1, a + i*dk, Q, tau);
      for (int it=0; it<tau.size(); it++) fP0(it)[i] = p1(it);
    }
    for (int it=0; it<tau.extent(0); it++) p1(it) = -kF/(pi*pi*pi*Q) * romberg2<N,r>(fP0(it), b-a);
    return p1;
  }
};

class PO2_slow{
public:
  double kF, kF2, beta, cutoffk, cutoffq, smallk;
  Spline1D<double> S_x;
  bl::Array<double,1> kx;
  
  double Sx(double q, double k){
    return q/(k*kF) * log((k+q)/fabs(k-q)) * 1.0/(1.0 + exp(beta*(q*q-kF2)));
  }
  double ferm(double k){
    return 1.0/(1.0 + exp(beta*(k*k-kF2)));
  }
  double Sx_minus(double q, double k, double fmk){
    return q/(k*kF) * log((k+q)/fabs(k-q)) * (ferm(q)-fmk);
  }
  PO2_slow(double _kF_, double _beta_, double _cutoffk_, double _cutoffq_, double _smallk_) :
    kF(_kF_), kF2(kF*kF), beta(_beta_), cutoffk(_cutoffk_), cutoffq(_cutoffq_), smallk(_smallk_){
    const int r=8;
    const int N = (1<<r)+1; // 2^r+1
    double a = smallk, b = cutoffk;
    double dk = (b-a)/(N-1.);
    double a_q = smallk+1e-10, b_q = cutoffq;
    double dq = (b_q-a_q)/(N-1.);
    bl::TinyVector<double,N> fP0;
    kx.resize(N);
    S_x.resize(N);
    for (int i=0; i<N; i++){
      double k = a + i*dk;
      kx(i) = k;
      double fmk = ferm(k);
      for (int j=0; j<N; j++) fP0[j] = Sx_minus(a_q + j*dq, k, fmk);
      S_x[i] = romberg2<N,r>(fP0, b_q-a_q);
      S_x[i] += fmk/kF*( (b_q-a_q) + (b_q*b_q-k*k)*log((b_q+k)/fabs(b_q-k))/(2*k) - (a_q*a_q-k*k)*log((a_q+k)/fabs(a_q-k))/(2*k) );
    }
    S_x.splineIt(kx);
  }
  double P1qk(double k, double Q, double t)
  {
    double epsk = k*k-kF2;
    double fk = 1/(exp(beta*epsk)+1.0);
    double fmk = 1.-fk;
    return S_x(Interp(k,kx)) * (t*fmk - (beta-t)*fk) * Cu(k,beta-t,beta,kF) * Cins(k,Q,t,beta,kF);
  }
  double P1(double Q, double t)
  {
    const int r=10;
    const int N = (1<<r)+1; // 2^r+1
    double a = smallk, b = cutoffk;
    double dk = (b-a)/(N-1.);
    bl::TinyVector<double,N> fP0;
    for (int i=0; i<N; i++){
      double k = a + i*dk;
      fP0[i] = P1qk(k,Q,t);
    }
    return -kF/(pi*pi*pi*Q) * romberg2<N,r>(fP0, b-a);
  }
};
