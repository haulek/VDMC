// @Copyright 2018 Kristjan Haule and Kun Chen    
#ifndef _LEGENDRE_
#define _LEGENDRE_
#include <iostream>
#include <vector>
#include <deque>
#include <fstream>
#include <sstream>
#include <blitz/array.h>
//using namespace std;
//using namespace blitz;
namespace bl = blitz;

template <int lmax>
class LegendreQ{
public:
  int N;
  bl::TinyVector<double,lmax+1> XL;
  bl::Array<double,1> PPLX, PPLY;
  LegendreQ(int N_) : N(N_), PPLX(N), PPLY(N)//, XL(lmax+1)
  {
    for (int i=0; i<=lmax; i++)
      XL[i] = i/(i+1.);
  }
  void Compute(bl::Array<double,2>& Pl, const bl::Array<double,1>& X)
  {// Pl(N,lmax+1)
    for (int i=0; i<N; i++){
      Pl(i,0) = 1.0;
      Pl(i,1) = X(i);
    }
    if (lmax<=1) return;
    
    for (int i=0; i<N; i++){
      PPLX(i) = X(i)*X(i);
      PPLY(i) = PPLX(i)-1.0;
    }
    // Recursion for Legendre Polynomials
    for (int l=2; l<lmax; l++){
      for (int i=0; i<N; i++){
	Pl(i,l) = PPLX(i) + XL[l-1]*PPLY(i);
	PPLX(i) = X(i)*Pl(i,l);
	PPLY(i) = PPLX(i) - Pl(i,l-1);
      }
    }
    for (int i=0; i<N; i++)
      Pl(i,lmax) = PPLX(i) + XL[lmax-1]*PPLY(i);
  
    // correcting for x~1.0 where recursion does not work very well
    for (int i=0; i<N; i++){
      if (fabs(X(i))>0.9999999){
	for (int l=2; l<=lmax; l++){
	  Pl(i,l) = X(i)*Pl(i,l-1);
	}
      }
    }
  }
  
  void cmp_single(double x, bl::TinyVector<double,lmax+1>& pl)
  {
    pl[0] = 1.0;
    pl[1] = x;
    if (lmax<=1) return;
    double PPLX = x*x;
    double PPLY = PPLX-1.0;
    // Recursion for Legendre Polynomials
    for (int l=2; l<lmax; l++){
      pl[l] = PPLX + XL[l-1]*PPLY;
      PPLX = x * pl[l];
      PPLY = PPLX - pl[l-1];
    }
    pl[lmax] = PPLX + XL[lmax-1]*PPLY;
    // correcting for x~1.0 where recursion does not work very well
    if (fabs(x)>0.9999999){
      for (int l=2; l<=lmax; l++)
	pl[l] = x * pl[l-1];
    }
  }
};

class LegendrePl{
public:
  int lmax, N;
  bl::Array<double,1> XL;
  bl::Array<double,1> PPLX, PPLY;
  LegendrePl(){}
  LegendrePl(int lmax_, int N_=0) : lmax(lmax_), N(N_), XL(lmax+1), PPLX(N), PPLY(N)
  {
    for (int i=0; i<=lmax; i++) XL(i) = i/(i+1.);
  }
  void resize(int lmax_, int N_=0)
  {
    lmax = lmax_; N=N_;
    XL.resize(lmax+1);
    PPLX.resize(N);
    PPLY.resize(N);
    for (int i=0; i<=lmax; i++) XL(i) = i/(i+1.);
  }
  void Compute(bl::Array<double,2>& Pl, const bl::Array<double,1>& X)
  {// Pl(N,lmax+1)
    for (int i=0; i<N; i++){
      Pl(i,0) = 1.0;
      Pl(i,1) = X(i);
    }
    if (lmax<=1) return;
    
    for (int i=0; i<N; i++){
      PPLX(i) = X(i)*X(i);
      PPLY(i) = PPLX(i)-1.0;
    }
    // Recursion for Legendre Polynomials
    for (int l=2; l<lmax; l++){
      for (int i=0; i<N; i++){
	Pl(i,l) = PPLX(i) + XL(l-1)*PPLY(i);
	PPLX(i) = X(i)*Pl(i,l);
	PPLY(i) = PPLX(i) - Pl(i,l-1);
      }
    }
    for (int i=0; i<N; i++)
      Pl(i,lmax) = PPLX(i) + XL(lmax-1)*PPLY(i);
  
    // correcting for x~1.0 where recursion does not work very well
    for (int i=0; i<N; i++){
      if (fabs(X(i))>0.9999999){
	for (int l=2; l<=lmax; l++){
	  Pl(i,l) = X(i)*Pl(i,l-1);
	}
      }
    }
  }
  
  void cmp_single(double x, bl::Array<double,1>& pl, int lmaxt)
  {
    pl(0) = 1.0;
    pl(1) = x;
    if (lmaxt<=1) return;
    double PPLX = x*x;
    double PPLY = PPLX-1.0;
    // Recursion for Legendre Polynomials
    for (int l=2; l<lmaxt; l++){
      pl(l) = PPLX + XL(l-1)*PPLY;
      PPLX = x * pl(l);
      PPLY = PPLX - pl(l-1);
    }
    pl(lmaxt) = PPLX + XL(lmaxt-1)*PPLY;
    // correcting for x~1.0 where recursion does not work very well
    if (fabs(x)>0.9999999){
      for (int l=2; l<=lmaxt; l++)
	pl(l) = x * pl(l-1);
    }
  }
};


void LegendreP(bl::Array<double,2>& Pl, const bl::Array<double,1>& X, int lmax)
{
  int lmx = lmax+1;
  bl::Array<double,1> XL(lmx);
  for (int i=0; i<lmx; i++) XL(i) = i/(i+1.);

  int N = X.extent(0);
  Pl.resize(lmx,N);
  
  bl::Array<double,1> PPLX(N), PPLY(N);
  for (int i=0; i<N; i++){
    PPLX(i) = X(i)*X(i);
    PPLY(i) = PPLX(i)-1.0;
  }
  
  for (int i=0; i<N; i++){
    Pl(0,i) = 1.0;
    Pl(1,i) = X(i);
  }
  
  // Recursion for Legendre Polynomials
  for (int l=2; l<lmx-1; l++){
    for (int i=0; i<N; i++){
      Pl(l,i) = PPLX(i) + XL(l-1)*PPLY(i);
      PPLX(i) = X(i)*Pl(l,i);
      PPLY(i) = PPLX(i) - Pl(l-1,i);
    }
  }
  for (int i=0; i<N; i++)
    Pl(lmx-1,i) = PPLX(i) + XL(lmx-2)*PPLY(i);
  
  // correcting for x~1.0 where recursion does not work very well
  for (int j=0; j<N; j++){
    if (fabs(X(j))>0.9999999){
      for (int l=2; l<lmx; l++){
	Pl(l,j) = X(j)*Pl(l-1,j);
      }
    }
  }
}

void split(const std::string &s, char delim, std::vector<std::string> &elems){
  std::stringstream ss(s);
  std::string item;
  while (getline(ss, item, delim)){
    elems.push_back(item);
  }
}

double GetLegendreArgument(double x, double deltat, int Nbin)
{
  int i = static_cast<int>(x/deltat); // bin in equidistant mesh
  if (i>=Nbin) i=Nbin-1; // the point in or past the last bin
  return 2*(x-i*deltat)/deltat -1.;
}

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
/*
void LegendreCoeff(Array<Fraction,2>& cPl, int lmax)
{
  // Construct coefficients cPl, such that the Legendre Polynomial  
  //         P[l](x) = \sum_i cPl[l,i] * x^i
  //
  cPl.resize(lmax+1,lmax+1);
  cPl = 0;
  cPl(0,0) = 1;
  cPl(1,1) = 1;
  //cout << "cPl=" << cPl << endl;
  for (int l=1; l<lmax; ++l){
    Fraction c1( 2*l+1, l+1 );
    Fraction c2( l, l+1 );
    //cout << "l=" << l << " c1=" << c1;
    //cout << " c2=" << c2 <<endl;
    for (int n=0; n<=l; ++n){
      Fraction t1 = c1 * cPl(l,n);
      Fraction t2 = cPl(l+1,n) - c2 * cPl(l-1,n);
      
      //cout << "  n=" << n << " c(l+1,n+1)=" << t1 << endl;
      //cout << "  n=" << n << " c(l+1,n  )=" << t2 << endl;
      
      //cPl(l+1,n+1) = c1 * cPl(l,n);
      //cPl(l+1,n)  -= c2 * cPl(l-1,n);
      
      cPl(l+1,n+1) = t1;
      cPl(l+1,n)   = t2;
    }
  }
}
*/

/*
template<class container2D>
double InterpolateLegendreFunction(double xm, double x, double xp, const functionb<double>& Plm, const functionb<double>& Pl, const functionb<double>& Plp, const container2D& c, int Nbin, double beta, int lmax)
{
  if (c.extent(0)!=Nbin) cerr<<"First dimension of c is wrong : "<<c.extent(0)<<", "<<Nbin<<endl;
  if (c.extent(1)!=lmax+1) cerr<<"Second dimension of c is wrong : "<<c.extent(1)<<", "<<lmax+1<<endl;
  if (Pl.size()!=lmax+3) cerr<<"Dimension of Pl is wrong : "<<Pl.size()<<", "<<lmax+3<<endl;

  double deltat = beta/Nbin;
  // the left point
  double xbm = xm/deltat;
  int ibm = static_cast<int>(xbm);
  if (ibm>=Nbin) ibm=Nbin-1;
  // the central point
  double xb = x/deltat;
  int ib = static_cast<int>(xb);
  if (ib>=Nbin) ib=Nbin-1;
  // the right point
  double xbp = xp/deltat;
  int ibp = static_cast<int>(xbp);
  if (ibp>=Nbin) ibp=Nbin-1;
  // size of intervals
  double dlt_l = x-xm;
  double dlt_r = xp-x;
    
  //cout<<"ib[-1],ib[0],ib[1]="<<ibm<<" "<<ib<<" "<<ibp<<" dlt_l="<<dlt_l<<"dlt_r="<<dlt_r<<endl;
    
  double zsum = 0.;
  double weights = 0.;
  if (dlt_l>0){
    // Integration in the interval [xm,x]
    // First point j=ibm is special, because it has limits [xm,t_{j+1}]
    int j=ibm;                // first point, index for t_j
    double a = 2*(xbm-ibm) -1.;  // starting integration point
    double qsum0 = c(j,0)*(-a)/2.;
    for (int l=1; l<=lmax; l++)
      qsum0 -= c(j,l)*(Plm[l+1]-Plm[l-1])/(2*(2.*l+1.));
    double qsum1 = c(j,0)*(1.-a*a)/2. + c(j,1)*(-a*a*a)/3.;
    for (int l=2; l<=lmax; l++){
      double uPu = ( (l+1.)*(2*l-1.)*Plm[l+2] + (2*l+1.)*Plm[l] - l*(2*l+3.)*Plm[l-2] )/((2*l-1.)*(2*l+1.)*(2*l+3.));
      qsum1 -= c(j,l) * uPu;
    }
    double wgh0 = (deltat*(j+0.5)-xm)/dlt_l;
    double wgh1 = 0.25*deltat/dlt_l;
    if (ibm<ib){        // if more than single interval between [xm,x]
      qsum0 += c(j,0)*0.5;
      qsum1 += c(j,1)*1./3.;
      weights += wgh0*0.5;
    }
    zsum += wgh0*qsum0 + wgh1*qsum1;
    weights += wgh0*(-a)/2. + wgh1*(1.-a*a)/2.;
    // First point finished
    //cout<<j*beta/Nbin<<" "<<(j+1.)*beta/Nbin<<" "<<a<<" "<<1.<<" "<<weights * 2.*deltat/(dlt_l+dlt_r)<<endl;
    // Now process the inbetween intervals [t_j,t_{j+1}], which are contained in the entirety within [xm,x]
    for (int j=ibm+1; j<ib; j++){
      qsum0 = c(j,0);
      qsum1 = 2./3.*c(j,1);
      wgh0 = (deltat*(j+0.5)-xm)/dlt_l;
      wgh1 = 0.25*deltat/dlt_l;
      zsum += wgh0*qsum0 + wgh1*qsum1;
      weights += wgh0;
      //cout<<j*beta/Nbin<<" "<<(j+1.)*beta/Nbin<<" "<<-1.<<" "<<1.<<" "<<weights * 2.*deltat/(dlt_l+dlt_r)<<endl;
    }
    // Inbetween points finished
    // The last interval is again special, because integration is between [t_j,x]
    j=ib;
    double b = 2*(xb-ib)-1.;
    qsum0 = c(j,0)*(b)/2.;
    for (int l=1; l<=lmax; l++)
      qsum0 += c(j,l)*(Pl[l+1]-Pl[l-1])/(2.*(2.*l+1.));
    qsum1 = c(j,0)*(b*b-1.)/2. + c(j,1)*(b*b*b)/3.;
    for (int l=2; l<=lmax; l++){
      double uPu = ( (l+1.)*(2*l-1.)*Pl[l+2] + (2*l+1.)*Pl[l] - l*(2*l+3.)*Pl[l-2] )/((2*l-1.)*(2*l+1.)*(2*l+3.));
      qsum1 += c(j,l)*uPu;
    }
    wgh0 = (deltat*(j+0.5)-xm)/dlt_l;
    wgh1 = 0.25*deltat/dlt_l;
    if (ibm<ib){  // if more than single interval between [xm,x]
      qsum0 += c(j,0)*0.5;
      qsum1 += c(j,1)*1./3.;
      weights += wgh0*0.5;
    }
    zsum += wgh0*qsum0 + wgh1*qsum1;
    weights += wgh0 * (b)/2. + wgh1*(b*b-1.)/2.;
    //cout<<j*beta/Nbin<<" "<<(j+1)*beta/Nbin<<" "<<-1.<<" "<<b<<" "<<weights * 2.*deltat/(dlt_l+dlt_r)<<endl;
  }
  // Now we are finished with the interval [xm,x]
  if (dlt_r>0){
    // Now integrate the interval [x,xp]
    // The first interval[x,t_j] is special, because the integration starts above -1.
    int j=ib;
    double a = 2*(xb-ib)-1.;
    double qsum0 = c(j,0)*(-a)/2.;
    for (int l=1; l<=lmax; l++)
      qsum0 -= c(j,l)*(Pl[l+1]-Pl[l-1])/(2*(2.*l+1.));
    double qsum1 = c(j,0)*(1.-a*a)/2. + c(j,1)*(-a*a*a)/3.;
    for (int l=2; l<=lmax; l++){
      double uPu = ( (l+1.)*(2*l-1.)*Pl[l+2] + (2*l+1.)*Pl[l] - l*(2*l+3.)*Pl[l-2] )/((2*l-1.)*(2*l+1.)*(2*l+3.));
      qsum1 -= c(j,l) * uPu;
    }
    double wgh0 = (xp-deltat*(j+0.5))/dlt_r;
    double wgh1 = -0.25*deltat/dlt_r;
    if (ibp>ib){  // if more than single interval between [x,xp]
      qsum0 += c(j,0)*0.5;
      qsum1 += c(j,1)*1./3.;
      weights += wgh0 * 0.5;
    }
    zsum += wgh0*qsum0 + wgh1*qsum1;
    weights += wgh0 * (-a)/2. + wgh1 * (1.-a*a)/2.;
    // First point finished
    //cout<<j*beta/Nbin<<" "<<(j+1.)*beta/Nbin<<" "<<a<<" "<<1.<<" "<<weights * 2.*deltat/(dlt_l+dlt_r)<<endl;
    // Now process the inbetween intervals [t_j,t_{j+1}], which are contained in the entirety within [x,xp]
    for (int j=ib+1; j<ibp; j++){
      qsum0 = c(j,0);
      qsum1 = 2./3.*c(j,1);
      wgh0 = (xp-deltat*(j+0.5))/dlt_r;
      wgh1 = -0.25*deltat/dlt_r;
      zsum += wgh0*qsum0 + wgh1*qsum1;
      weights += wgh0;
      //cout<<j*beta/Nbin<<" "<<(j+1.)*beta/Nbin<<" "<<-1<<" "<<1.<<" "<<weights * 2.*deltat/(dlt_l+dlt_r)<<endl;
    }
    // Inbetween points finished
    // The last interval is again special, because integration is between [t_j,xp]
    j = ibp;
    double b = 2*(xbp-ibp)-1.;
    qsum0 = c(j,0)*(b)/2.;
    for (int l=1; l<=lmax; l++)
      qsum0 += c(j,l)*(Plp[l+1]-Plp[l-1])/(2.*(2.*l+1.));
    qsum1 = c(j,0)*(b*b-1.)/2. + c(j,1)*(b*b*b)/3.;
    for (int l=2; l<=lmax; l++){
      double uPu = ( (l+1.)*(2*l-1.)*Plp[l+2] + (2*l+1.)*Plp[l] - l*(2*l+3.)*Plp[l-2] )/((2*l-1.)*(2*l+1.)*(2*l+3.));
      qsum1 += c(j,l)*uPu;
    }
    wgh0 = (xp-deltat*(j+0.5))/dlt_r;
    wgh1 = -0.25*deltat/dlt_r;
    if (ibp>ib){ // if more than single interval between [x,xp]
      qsum0 += c(j,0)*0.5;
      qsum1 += c(j,1)*1./3.;
      weights += wgh0 * 0.5;
    }
    zsum += wgh0*qsum0 + wgh1*qsum1;
    weights += wgh0 * (b)/2. + wgh1 * (b*b-1.)/2.;
    //cout<<j*beta/Nbin<<" "<<(j+1.)*beta/Nbin<<" "<<-1<<" "<<b<<" "<<weights * 2.*deltat/(dlt_l+dlt_r)<<endl;
    // Now we are finished with the interval [x,xp]
  }
  weights *= 2./(dlt_l+dlt_r) * deltat;
  zsum *= 2./(dlt_l+dlt_r) * deltat;
  //cout<<"weights="<<weights<<endl;
  return zsum;
}
*/
/*
int main()
{
  // loading "dSml" file
  ifstream fdSml("dSml");
  deque<vector<double> > data;
  string line;
  while (fdSml.good()){
    getline(fdSml,line);
    if (line.length()==0) break;
    vector<string> items;
    split(line,' ',items);
    vector<double> dat(items.size()-1);
    for (int i=1; i<items.size(); i++) dat[i-1] = stod(items[i]);
    data.push_back(dat);
  }
  function2D<double> dSml(data.size(),data[0].size());
  for (int i=0; i<dSml.extent(0); i++)
    for (int j=0; j<dSml.extent(1); j++)
      dSml(i,j) = data[i][j];
  // what we have in dSml
  int Nbin = dSml.extent(0);
  int lmax = dSml.extent(1)-1;
  double beta=100.;
  // spacing
  double deltat = beta/Nbin;

  // the new grid will have so many points
  int Nx = 101;
  function1D<double> xx(Nx);
  for (int i=0; i<Nx; i++) xx[i] = i*beta/(Nx-1.);

  // Calculating Legendre Poly on this grid
  LegendreQ legendre(lmax+2,xx.size());
  function1D<double> Xs(xx.size());
  for (int i=0; i<xx.size(); i++)
    Xs[i] = GetLegendreArgument(xx[i],deltat,Nbin);
  function2D<double> Plc(xx.size(),lmax+3);
  legendre.Compute(Plc, Xs);
  
  for (int i=0; i<xx.size(); i++){
    double xm, x, xp;
    functionb<double> *Plm, *Pl, *Plp;
    function1D<double> plx(lmax+3);
    if (i==0){
      xm=xx[i];
      x=xx[i];
      xp=xx[i]+1e-5;
      Plm = &Plc[i];
      Pl  = &Plc[i];
      Plp = &legendre.cmp_single( GetLegendreArgument(xp,deltat,Nbin) ,plx);
    }else if (i==Nx-1){
      xm=xx[i]-1e-5;
      x=xx[i];
      xp=xx[i];
      Plm = &legendre.cmp_single( GetLegendreArgument(xm,deltat,Nbin), plx);
      Pl =  &Plc[i];
      Plp = &Plc[i];
    }else{
      xm = xx[i-1];
      x  = xx[i];
      xp = xx[i+1];
      Plm = &Plc[i-1];
      Pl  = &Plc[i];
      Plp = &Plc[i+1];
    }
    double zsum = InterpolateLegendreFunction( xm, x, xp, *Plm, *Pl, *Plp, dSml, Nbin,beta,lmax);
    cout<<x<<" "<<zsum<<endl;
  }
  
  return 0;

  int N=21;
  function1D<double> X(N);
  for (int i=0; i<X.size(); i++) X[i]= (i-10.)/10.;

  lmax=20;
  function2D<double> Pl(lmax+1,N);
  LegendreP(Pl, X, lmax);


  cout<<"#      ";
  for (int l=0; l<=lmax; l++) cout<<setw(14)<<l;
  cout<<endl;
  for (int i=0; i<X.size(); i++){
    cout<<"X="<<setw(4)<<X[i]<<" : ";
    for (int l=0; l<=lmax; l++)
      cout<<setw(14)<<Pl(l,i)<<" ";
    cout<<endl;
  }
}
*/
#endif // _LEGENDRE_
