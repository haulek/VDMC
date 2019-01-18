// @Copyright 2018 Kristjan Haule and Kun Chen    
#include "legendre.h"

class Build_theta_Mesh{
public:
  int Nt, Ntheta;
  bl::Array<double,1> xt, dxt;
  double beta, pc_beta;
public:  
  Build_theta_Mesh(double _beta_, int Nc=10, double pc=8., int r=6)
  {
    // pc --- how extended mesh around Fermi surface [-pc*kT,...pc*kT]
    // Nc --- number of extra points around the Fermi surface
    // r  --- Minimal number of points on theta mesh = 2^r+1
    beta = _beta_;
    //cout<<"r="<<r<<endl;
    Ntheta = (1<<r)+1;
    //cout<<"Ntheta="<<Ntheta<<" Nc="<<Nc<<endl;
    double x0=1./beta;
    pc_beta = pc/beta;
    GiveTanMesh(xt, dxt, x0, pc_beta/(1.+3.*(pc_beta)*(pc_beta)), Nc);
    Nt = xt.extent(0);
  }
  void Compute(bl::Array<double,1>& x, double k, double q, double kF){
    double x_critical = (kF*kF - k*k - q*q)/(2*k*q); // Fermi surface known analytically
    //cout<<"x_critical="<<x_critical<<endl;
    if (q>2*kF && x_critical > -1 && x_critical < 1){
      // special correction for large q, needs exponential mesh
      int Nx=300;
      x.resize(Nx);
      double gm=1.0;
      for (int i=0; i<Nx; i++){
	double t = 13./gm * (1. - i/(Nx-1.) );
	x(i) = -1 + 2.*exp(-t*gm);
      }
    }else if (x_critical > -1 && x_critical < 1 && 1/pc_beta > 10*(Ntheta/20.+1)){
      double L1 = xt(0) + x_critical + 1.;
      double L2 = 1. - (xt(Nt-1) + x_critical);
      if (L1 < 0) L1=0;
      if (L2 < 0) L2=0;
      int N1 = static_cast<int>(Ntheta * L1/(L1+L2));
      int N2 = Ntheta-N1;
      int Nat = 0, Nbt = Nt;
      if (xt(0)+x_critical<-1){
	//cout<<"N1="<<N1<<" N2="<<N2<<endl;
	while(Nat<Nt && xt(Nat)+x_critical<-1){
	  Nat+=1;
	  //cout<<"Increasing Nat to "<<Nat<<" so that xt0="<<xt(Nat)+x_critical<<endl;
	}
      }
      if (xt(Nt-1)+x_critical>1){
	//cout<<"N1="<<N1<<" N2="<<N2<<endl;
	while(Nbt>=0 && xt(Nbt-1)+x_critical>1) {
	  Nbt-=1;
	  //cout<<"Decreasing Nbt to "<<Nbt<<" so that xtn="<<xt(Nbt-1)+x_critical<<endl;
	}
      }
      x.resize(N1+N2+Nbt-Nat);
      for (int i=0; i<N1; ++i) x(i) = -1 + L1*(i+0.5)/(N1+0.0);
      for (int i=0; i<Nbt-Nat; ++i) x(N1+i) = x_critical + xt(i+Nat);
      for (int i=0; i<N2; ++i) x(N1+(Nbt-Nat)+i) = x_critical + xt(Nbt-1) + L2 * (i+0.5)/(N2+0.);
    } else if ( fabs(x_critical-1)<0.2 || fabs(x_critical+1)<0.2){
      bl::Array<double,1> dx(2*(Ntheta-1)+1);
      x.resize(2*(Ntheta-1)+1);
      GiveDoubleExpMesh(x, dx, 0, 1.0, x.extent(0), 1.5); // dense mesh at 1.0
      int N = x.extent(0);
      if (fabs(x_critical-1)<0.1){
	for (int i=0; i<N; i++) x(i) = -1 + 2*x(i); // 0->-1 ; 1-> 1
      } else{ 
	for (int i=0; i<=N/2; i++){
	  double x0 = 1-2*x(i); 
	  double xn = 1-2*x(N-i-1);
	  x(i) = xn;
	  x(N-i-1) = x0;
	}
      }
    }/*
    else if ( fabs(x_critical-1)<0.1 || fabs(x_critical+1)<0.1 ){
      int N = 2*(Ntheta-1)+1;
      x.resize(N);
      for (int i=0; i<N; ++i) x(i) = -1. + 2.*(i+0.5)/(N+0.);
    }*/else{
      x.resize(Ntheta);
      for (int i=0; i<Ntheta; ++i) x(i) = -1. + 2.*(i+0.5)/(Ntheta+0.);
    }
  }
};

namespace bl = blitz;
void GetPolyInt(bl::Array<std::complex<double>,1>& In, double a, double b, const std::complex<double>& z, double alpha, int N)
{
  //    Computes integrals: 
  //        Integrate[ x^n /(z + alpha * x ) , {x,a,b}]
  //    which can be shown to satisfy the following recursion relation:
  //        In[n+1] = ( (b^(n+1) - a^(n+1))/(n+1) - z*In[n] )/alpha
  //
  // In.resize(N+1);
  // In = 0;
  const double small = 1e-8;
  const double how_small = 0.6;
  std::complex<double> x = -alpha/z;
  //std::cout<<" a="<<a<<" b="<<b<<" z="<<z<<" alpha="<<alpha<<std::endl;
  if ( abs(x) < how_small){
    //  For small |alpha/z| the recursion is stable only when evaluating it downward.
    //  For this we need a good estimation of the value of In[N]. We derived the following exact 
    //  expression
    //         I[n] = 1/z * \sum_{i=0,infty} ( b^(n+1+i) - a^(n+1+i) )/(n+1+i) * (-alpha/z)^i
    //  which is evaluated for n=N. Then we use downward recursion
    //         In[n-1] = (b^n-a^n)/(z*n) - alpha/z * In[n]
    //std::cout<<"Downward"<<std::endl;
    double _bn_ = ipower(b,N); // b^N
    double _an_ = ipower(a,N); // a^N
    double  bn = _bn_ * b;     // b^(N+1)
    double  an = _an_ * a;     // a^(N+1)
    std::complex<double> x_n = 1.0;
    std::complex<double> In_N = 0;
    for (int n=0; n<1000; n++){
      std::complex<double> dI = (bn-an)/(N+1+n) * (x_n/z);
      //std::cout<<"n="<<n<<" dI="<<dI<<std::endl;
      In_N += dI;
      bn *= b;
      an *= a;
      x_n *= x;
      if ( abs(dI) < small) break;
    }
    In(N) = In_N;
    bn = _bn_; // b**N
    an = _an_; // a**N
    for (int n=N;  n>0; --n){
      In(n-1) = (bn-an)/n/z + x * In(n);
      bn /= b;
      an /= a;
    }
  } else{
    //  Upward recursion is very straighforard:
    //  In[n+1] = (b^(n+1) - a^(n+1))/(alpha*(n+1)) - z/alpha * In[n]
    //  and I[0] = log((z+alpha b)/(z+alpha a))/aloha
    //
    //std::cout<<"Upward"<<std::endl;
    In(0) = (log(z+alpha*b)-log(z+alpha*a))/alpha;
    double bn = b;  // b^(n+1)
    double an = a;  // a^(n+1)
    for (int n=0; n<N; ++n){
      In(n+1) = ((bn - an)/(n+1.) - z * In(n))/alpha;
      bn *= b;
      an *= a;
    }
  }
}

/*
void LegendreCoeff(bl::Array<double,2>& cPl, int lmax)
{
  // Construct coefficients cPl, such that the Legendre Polynomial  
  //         P[l](x) = \sum_i cPl[l,i] * x^i
  //
  cPl.resize(lmax+1,lmax+1);
  cPl = 0;
  cPl(0,0) = 1;
  cPl(1,1) = 1;
  for (int l=1; l<lmax; ++l){
    double c1 = (2*l+1.)/(l+1.);
    double c2 = l/(l+1.);
    for (int n=0; n<=l; ++n){
      cPl(l+1,n+1) = c1 * cPl(l,n);
      cPl(l+1,n)  -= c2 * cPl(l-1,n);
    }
  }
}
*/

#ifndef _FERM_
#define _FERM_
inline double ferm(double x){
  if (x>700) return 0.;
  else return 1./(exp(x)+1.);
}
#endif

inline double dferm(double x){
  if (fabs(x)>700) return 0.;
  double ex = exp(x);
  return -1./((ex+1.)*(1./ex+1.));
}

#ifdef _RENAME_COLLISION
class Compute_Y_P_Y_{
#else  
class Compute_Y_P_Y{
#endif  
  // Computes <Y_{l0}|P(omega,k,q)|Y_{l'0}>
  // We decided to have the form for P(t,k,q)
  //   P(t,k,q) = G_{k+q}(t) * G_k(-t)
  // which gives in Matsubara
  //
  //  P(iOm,k,q) = [f(e_k)-f(e_{k+q})]/(iOm + e_k - e_{k+q})
  // We use:
  //   eps(sqrt(k^2+q^2+2*k*q*x_i+2*k*q*(x-x_i))) = eps(k+q) + deps/dk * k*q*(x-x_i)/|k+q|
  //   f(eps(sqrt(k^2+q^2+2*k*q*x_i+2*k*q*(x-x_i)))) = f(eps(k+q)) + df/deps * deps/dk * k*q*(x-x_i)/|k+q|
  //  
  //  P(iOm,k,q,l1,l2) = \sqrt((2l_1+1)(2l_2+1))/2 * sum_{n1,n2} C_{l1,n1} C_{l2,n2} *
  //     Integrate[ x^{n1+n2}  ( f(e_k)-f(e_{k+q}) -df/deps*deps/dk*k*q/|k+q|*(x-x_i) )/(iOm + eps(k) - eps(k+q) - deps/dk * k*q/|k+q| *(x-x_i) ), {x,0.5*(x_i+x_{i-1}),0.5*(x_i+x_{i+1})}]
  //
public:
  double beta;
  int lmax;
  double kF;
  bl::Array<double,1> cn;
  Build_theta_Mesh btm;
  bl::Array<double,2> cPl;
  HartreeFock hf;
public:
#ifdef _RENAME_COLLISION
  Compute_Y_P_Y_
#else
  Compute_Y_P_Y
#endif  
  (double _beta_, double lmbda, double dmu, double _kF_, double cutoff, int _lmax_) :
    beta(_beta_), lmax(_lmax_), kF(_kF_), cn(lmax+1),
    btm(beta),  // creates mesh [-8*kt ... 8*kt] to resolve fermi surface
    hf(kF, cutoff, beta, lmbda, dmu)
  {
    hf.cmp();
    for (int l=0; l<lmax+1; ++l) cn(l) = sqrt((2*l+1.)/2.);
    LegendreCoeff(cPl, lmax);
  }
  void Run(bl::Array<complex<double>,3>& PpP, double k, double q, const bl::Array<int,1>& iOm, int ik_debug=-1)
  {
    if (PpP.extent(0)!=lmax+1 || PpP.extent(1)!=lmax+1 || PpP.extent(2)!=iOm.extent(0))
      PpP.resize(lmax+1,lmax+1,iOm.extent(0));

    bool debug=false;
    if (ik_debug>=0) debug = true;
    ofstream* log;
    if (debug){
      log = new ofstream( std::string("debug.")+std::to_string(ik_debug) );
      double x_critical = (kF*kF - k*k - q*q)/(2*k*q);
      (*log) << "# x_critical = "<< x_critical << endl;
    }
    
    bl::Array<double,1> x;
    btm.Compute(x, k, q, kF);
    int Nx = x.extent(0);
    
    double epsk = hf.epsx(Interp(k, hf.kx));  
    double fk = ferm(beta*epsk);

    vector<complex<double> > gz, zz;
    bl::Array<complex<double>,2> tz;
    if (debug){
      gz.resize(iOm.extent(0));
      zz.resize(iOm.extent(0));
      //tz.resize(iOm.extent(0));
      tz.resize(Nx,iOm.extent(0));
    }
    
    bl::Array<complex<double>,2> In(iOm.extent(0),2*lmax+1);
    In = 0;
    bl::Array<complex<double>,1> Kn(2*lmax+2);
    double k2q2 = k*k + q*q, kq2 = 2*k*q;
    int ia=0;
    cout.precision(12);
    for (int i=0; i<Nx; i++){
      double k_p_q = sqrt(k2q2 + kq2 * x(i));   // k+q
      intpar ip = InterpBoth(k_p_q, ia, hf.kx);
      double eps_kq  = hf.epsx(ip);           // eps(k+q)
      double deps_kq = hf.epsx.df(ip);        // d eps(k+q)/dk
      double deps_x = deps_kq * 0.5*kq2/k_p_q; // d eps(k+q) / dx
      double fkq = ferm(beta*eps_kq);           // f(eps(k+q))
      double df_eps = beta*dferm(beta*eps_kq);  // df(eps(k+q))/deps
      double df_dx = df_eps*deps_x;             // df(eps(k+q))/dx
      double a = i==0    ? -1 : 0.5*(x(i)+x(i-1));
      double b = i==Nx-1 ?  1 : 0.5*(x(i)+x(i+1));
      
      double alpha = -deps_x;
      double eta = -df_dx;
      double gamma0 = fk - fkq;
      double gamma = gamma0 - x(i)*eta;
      for (int iw=0; iw<iOm.extent(0); ++iw){
	complex<double> iOmw(0, 2*iOm(iw)*pi/beta+1e-15);
	complex<double> z0 = iOmw + epsk - eps_kq;
	complex<double> z = z0 - x(i) * alpha;
	GetPolyInt(Kn, a, b, z, alpha, 2*lmax+1);
	
	//double sm=0;
	//for (int n=0; n<2*lmax+1; ++n) sm += abs(Kn(n));
	    
	for (int n=0; n<2*lmax+1; ++n)
	  In(iw,n) += gamma * Kn(n) + eta * Kn(n+1);
	
	if (debug){
	  gz[iw] = gamma0/z0;
	  tz(i,iw) = In(iw,0);
	  zz[iw] = z0;
	  //if (norm(tz[iw])>1e10){
	    //if (ik==159 && iw==iOm.extent(0)-1 && fabs(x(i)+0.997418)<1e-5){
	    //cout <<  ik << " "<< iw << " "<< i << " In = " << In(iw,0) << endl;
	  //}				 //}
	}
      }
      if (debug){
	(*log) << x(i) << "  ";
	//for (int iw=0; iw<iOm.extent(0); ++iw) (*log) <<  gz[i].real() << " " << gz[i].imag() << "  ";
	for (int iw=0; iw<iOm.extent(0); ++iw) (*log) <<  tz(i,iw).real() << " " << tz(i,iw).imag() << "  ";
	/*
	int N = iOm.extent(0);
	(*log) << gz[N-1].real() << " " << tz[N-1].real() << " " << zz[N-1].real() << " " << zz[N-1].imag() << " ";
	(*log) << gz[N-2].real() << " " << tz[N-2].real() << " " << zz[N-2].real() << " " << zz[N-2].imag() << " ";
	(*log) << gz[N-3].real() << " " << tz[N-3].real() << " " << zz[N-3].real() << " " << zz[N-3].imag() << " ";
	(*log) << gz[N-4].real() << " " << tz[N-4].real() << " " << zz[N-4].real() << " " << zz[N-4].imag() << " ";
	*/
	(*log) << endl;
      }
    }
    /*
    if (debug){
      for (int iw=0; iw<iOm.extent(0); ++iw){
	(*log) << iOm(iw) << "  ";
	for (int i=0; i<Nx; i++) (*log) << tz(i,iw).real() << " " << tz(i,iw).imag() << " ";
	(*log) << endl;
      }
    }
    */
    /*
   //Here is the slower, but easier to read, equivalent of the below code
    for (int l1=0; l1<lmax+1; ++l1){
      for (int l2=0; l2<lmax+1; ++l2){
	for (int iw=0; iw<iOm.extent(0); ++iw){
	  complex<double> dsum = 0;
	  for (int n1=0; n1<l1+1; ++n1) // n1 needs to be only even or odd!
	    for (int n2=0; n2<l2+1; ++n2) // n2 need to be only even or odd!
	      dsum += cPl(l1,n1) * In(iw,n1+n2) * cPl(l2,n2);
	  PpP(l1,l2,iw) = dsum*cn(l1)*cn(l2);
	}
      }
    }
    */
    bl::Array<double,1> cc(2*lmax+1);
    for (int l1=0; l1<=lmax; ++l1){
      for (int l2=0; l2<=lmax; ++l2){
	cc=0;
	for (int n1=(l1%2); n1<=l1; n1+=2)
	  for (int n2=(l2%2); n2<=l2; n2+=2)
	    cc(n1+n2) += cPl(l1,n1)*cPl(l2,n2);
	cc *= cn(l1)*cn(l2);
	for (int iw=0; iw<iOm.extent(0); ++iw){
	  complex<double> dsum = 0;
	  // can be optimize
	  for (int n=(l1+l2)%2; n<=l1+l2; ++n) dsum += cc(n)*In(iw,n);
	  PpP(l1,l2,iw) = dsum;
	}
      }
    }
    /*
    cout<<"k="<<k<<" q="<<q<<endl;
    for (int l1=0; l1<lmax+1; ++l1){
      for (int l2=0; l2<lmax+1; ++l2){
	cout<<"c++ "<<l1<<" "<<l2<<" "<<PpP(l1,l2,0).real()<<" "<<PpP(l1,l2,0).imag()<<endl;
      }
    }
    */
    if (debug){
      delete log;
    }
  }


  void Pw0(double q, const bl::Array<int,1>& iOm, bl::Array<double,1>& rm)
  {
    bl::Array<complex<double>,3> PpP(1,1,iOm.extent(0));

    double cutoff = hf.cutoff;
    
    static int iiq=0;
    iiq++;
//#define __DEBUG__    
#ifdef __DEBUG__    
    ofstream log( std::string("debug.dat")+std::to_string(iiq));
    log.precision(15);
    log << "#  q/kF=" << q/kF << endl;
#endif
    
    rm.resize(iOm.extent(0));
    rm=0;

    {
      int Nw = 1<<(7-1);
      double tnw = tan(pi/(2*Nw));
      double x0 = 0.5*Nw/4.*tnw*tnw;
      bl::Array<double,1> x(2*Nw+1), dx(2*Nw+1);
      GiveTanMesh(x, dx, x0, 1.0, Nw);

      bl::Array<double,2> gm(iOm.extent(0), x.extent(0));
      gm=0;
      for (int i=0; i<x.size(); i++){
	double k = kF + x(i)*kF;
	if (k==0) k += 1e-10; /// exact zero gives nan for logarithm
	else if (k==kF) k += 1e-10;
	else if (k==2*kF) k-= 1e-10;

	Run(PpP, k, q, iOm/*, i+iiq*1000*/);

#ifdef __DEBUG__    
	log<< k/kF << "  ";
#endif	
	for (int iw=0; iw<iOm.extent(0); iw++){
	  gm(iw,i) = PpP(0,0,iw).real()*k*k * dx(i)*(2.0*Nw)*kF;
#ifdef __DEBUG__    
	  log << PpP(0,0,iw).real()*k*k << "  ";
#endif	  
	}
#ifdef __DEBUG__    
	log << endl;
#endif	
      }
      for (int iw=0; iw<iOm.extent(0); iw++)
	rm(iw) += romberg(gm(iw,bl::Range::all()), 1.0);
      

      const int N = ( 1<<7)+1; // 2^5+1
      x.resize(N);
      dx.resize(N);
      for (int i=0; i<N; i++){
	x(i) = i/(N-1.);
	dx(i) = 1.;
      }
      
      gm.resize(iOm.extent(0), x.extent(0));
      gm=0;
      for (int i=0; i<x.size(); i++){
	double k = 2*kF + x(i)*(cutoff-2*kF);
	if (i==0) k += 1e-10; /// exact zero gives nan for logarithm
	Run(PpP, k, q, iOm/*, iiq*1000 + i+x.size()*/);
	
#ifdef __DEBUG__    
	log<< k/kF << "  ";
#endif	
	for (int iw=0; iw<iOm.extent(0); iw++){
	  gm(iw,i) = PpP(0,0,iw).real()*k*k * dx(i)*(cutoff-2*kF);
#ifdef __DEBUG__    
	  log << PpP(0,0,iw).real()*k*k << "  ";
#endif	  
	}
#ifdef __DEBUG__    
	log << endl;
#endif	
      }

#ifdef __DEBUG__    
      log << "#  ";
#endif      
      for (int iw=0; iw<iOm.extent(0); iw++){
	rm(iw) += romberg(gm(iw,bl::Range::all()), 1.0);
#ifdef __DEBUG__    
	log << rm(iw) << "  ";
#endif	
      }
#ifdef __DEBUG__    
      log << std::endl;
#endif      
    }

    
    /*
    if (q<2.2*kF){
      const int N = ( 1<< 6)+1; // 2^5+1
      bl::Array<double,1> x(N), dx(N);
      GiveDoubleExpMesh(x, dx, 0, 1.0, x.extent(0), 1.5);
      bl::Array<double,2> gm(iOm.extent(0), x.extent(0));
      
      gm=0;
      for (int i=0; i<x.size(); i++){
	double k = x(i)*kF;
	if (i==0) k += 1e-10; /// exact zero gives nan for logarithm
	Run(PpP, k, q, iOm);

	log<< k/kF << "  ";
	for (int iw=0; iw<iOm.extent(0); iw++){
	  gm(iw,i) = PpP(0,0,iw).real()*k*k * dx(i)*kF;
	  log << PpP(0,0,iw).real()*k*k << "  ";
	}
	log << endl;
      }
      for (int iw=0; iw<iOm.extent(0); iw++) rm(iw) = romberg(gm(iw,bl::Range::all()), 1.0);
  
      double tcutoff=2*kF;
      gm=0;
      for (int i=x.size()-1; i>0; i--){
	double k = kF + (tcutoff-kF)*(1. - x(i));
	Run(PpP, k, q, iOm);

	log<< k/kF << "  ";
	for (int iw=0; iw<iOm.extent(0); iw++){
	  gm(iw,i) = PpP(0,0,iw).real()*k*k * dx(i)*(tcutoff-kF);
	  log << PpP(0,0,iw).real()*k*k << "  ";
	}
	log << endl;
      }
      for (int iw=0; iw<iOm.extent(0); iw++) rm(iw) += romberg(gm(iw,bl::Range::all()), 1.0);

      gm=0;
      for (int i=x.size()-1; i>=0; i--){
	double k = tcutoff + (cutoff-tcutoff)*(1. - x(i));
	Run(PpP, k, q, iOm);

	log<< k/kF << "  ";
	for (int iw=0; iw<iOm.extent(0); iw++){
	  gm(iw,i) = PpP(0,0,iw).real()*k*k * dx(i)*(cutoff-tcutoff);
	  log << PpP(0,0,iw).real()*k*k << "  ";
	}
	log << endl;
      }
      for (int iw=0; iw<iOm.extent(0); iw++) rm(iw) += romberg(gm(iw,bl::Range::all()), 1.0);
    }else{
      const int N = ( 1<< 5)+1; // 2^5+1
      bl::Array<double,1> x(N), dx(N);
      GiveDoubleExpMesh(x, dx, 0, 1.0, x.extent(0), 1.5);
      bl::Array<double,2> gm(iOm.extent(0), x.extent(0));
      
      //const int N = ( 1<< 6)+1; // 2^5+1
      //bl::Array<double,1> x2(N), dx(N);
      //GiveDoubleExpMesh2(x2, dx2, 0, 1.0, x2.extent(0));
      //bl::Array<double,2> gm2(iOm.extent(0), x2.extent(0));

      gm=0;
      for (int i=0; i<x.size(); i++){
	double k = x(i)*kF;
	if (i==0) k += 1e-10; /// exact zero gives nan for logarithm
	Run(PpP, k, q, iOm);

	log<< k/kF << "  ";
	for (int iw=0; iw<iOm.extent(0); iw++){
	  gm(iw,i) = PpP(0,0,iw).real()*k*k * dx(i)*kF;
	  log << PpP(0,0,iw).real()*k*k << "  ";
	}
	log << endl;
      }
      for (int iw=0; iw<iOm.extent(0); iw++) rm(iw) = romberg(gm(iw,bl::Range::all()), 1.0);
  
      double tcutoff=2*kF;
      gm=0;
      for (int i=x.size()-1; i>0; i--){
	double k = kF + (tcutoff-kF)*(1. - x(i));
	Run(PpP, k, q, iOm);

	log<< k/kF << "  ";
	for (int iw=0; iw<iOm.extent(0); iw++){
	  gm(iw,i) = PpP(0,0,iw).real()*k*k * dx(i)*(tcutoff-kF);
	  log << PpP(0,0,iw).real()*k*k << "  ";
	}
	log << endl;
      }
      for (int iw=0; iw<iOm.extent(0); iw++) rm(iw) += romberg(gm(iw,bl::Range::all()), 1.0);

      double rcutoff=q;
      gm=0;
      for (int i=0; i<x.size(); i++){
	double k = tcutoff + (rcutoff-tcutoff)*x(i);
	Run(PpP, k, q, iOm);

	log<< k/kF << "  ";
	for (int iw=0; iw<iOm.extent(0); iw++){
	  gm(iw,i) = PpP(0,0,iw).real()*k*k * dx(i)*(rcutoff-tcutoff);
	  log << PpP(0,0,iw).real()*k*k << "  ";
	}
	log << endl;
      }
      for (int iw=0; iw<iOm.extent(0); iw++) rm(iw) += romberg(gm(iw,bl::Range::all()), 1.0);

      gm=0;
      for (int i=x.size()-1; i>=0; i--){
	double k = rcutoff + (cutoff-rcutoff)*(1. - x(i));
	Run(PpP, k, q, iOm);

	log<< k/kF << "  ";
	for (int iw=0; iw<iOm.extent(0); iw++){
	  gm(iw,i) = PpP(0,0,iw).real()*k*k * dx(i)*(cutoff-rcutoff);
	  log << PpP(0,0,iw).real()*k*k << "  ";
	}
	log << endl;
      }
      for (int iw=0; iw<iOm.extent(0); iw++) rm(iw) += romberg(gm(iw,bl::Range::all()), 1.0);
    }
    */
    
    rm *= 1.0/(pi*pi);
  }
};
