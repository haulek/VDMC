// @Copyright 2018 Kristjan Haule 
#ifndef MY_UTIL
#define MY_UTIL

#include <vector>
#include <stdexcept>
#include <blitz/array.h>

namespace bl = blitz;
//using namespace blitz;

//********** Creates Python sign function **************/
template <typename T> int sign(T val) {
    return (T(0) < val) - (val < T(0));
};
template <typename T> double dsign(T val) {
  return (T(0) < val) - (val < T(0));
}
//********** Creates Python range function **************/
template <typename Scalar>
bl::Array<Scalar,1> arange(Scalar start, Scalar stop, Scalar step)
{
  if (step == Scalar(0))
  {
    throw std::invalid_argument("step for range must be non-zero");
  }
  int size = static_cast<int>((stop-start+step-0.5*sign(step))/step);
  if (size<=0) return bl::Array<Scalar,1>();
  bl::Array<Scalar,1> result( size );
  Scalar i = start;
  int ii=0;
  while ((step > 0) ? (i < stop) : (i > stop))
  {
    result(ii) = i;
    i += step;
    ++ii;
  }
  return result;
}
template <typename Scalar>
bl::Array<Scalar,1> arange(Scalar start, Scalar stop)
{
  return arange(start, stop, Scalar(1));
}
template <typename Scalar>
bl::Array<Scalar,1> arange(Scalar stop)
{
  return arange(Scalar(0), stop, Scalar(1));
}

inline int minus_1_2_n(int n)
{ return  n%2 ? -1 : 1;}

//********** TinyVector summation and subtraction **************/
bl::TinyVector<double,3> operator-(const bl::TinyVector<double,3>& a, const bl::TinyVector<double,3>& b)
{ return bl::TinyVector<double,3>(a[0]-b[0],a[1]-b[1],a[2]-b[2]);}
bl::TinyVector<double,3> operator+(const bl::TinyVector<double,3>& a, const bl::TinyVector<double,3>& b)
{ return bl::TinyVector<double,3>(a[0]+b[0],a[1]+b[1],a[2]+b[2]);}
bl::TinyVector<double,3> operator*(const bl::TinyVector<double,3>& a, double c)
{return bl::TinyVector<double,3>(a[0]*c,a[1]*c,a[2]*c);}

//********** Trivial function that has functionality of python bisection, but does not use real bisection **************/
template <int N_rank>
int tBisect(double x, const bl::TinyVector<double,N_rank>& Prs){
  for (int i=0; i<N_rank; i++){
    if (x<Prs[i]) return i;
  }
  return N_rank-1;
}
/*
double ipower(double base, int exp){
  if( exp == 0)
    return 1.;
  double temp = ipower(base, exp/2);
  if (exp%2 == 0)
    return temp*temp;
  else {
    if(exp > 0)
      return base*temp*temp;
    else
      return (temp*temp)/base; //negative exponent computation 
  }
} 
*/
inline double ipower(double base, int exp)
{
  switch (exp){
  case 0 : return 1.; break;
  case 1 : return base; break;
  case 2 : return base*base; break;
  case 3 : return base*base*base; break;
  default :
    if (exp<0){
      exp = -exp;
      base = 1./base;
    }
    double result = 1;
    while (exp){
      if (exp & 1) result *= base;
      exp >>= 1;
      base *= base;
    }
    return result;
  }
}


double pi = M_PI;

bool isPowerOfTwo (int x)
{
  /* First x in the below expression is for the case when x is 0 */
  return x && (!(x&(x-1)));
}
double romberg(const bl::Array<double,1>& ff, double dh)
{
  int m = ff.extent(0);
  if ( !isPowerOfTwo(m-1) ) std::cout<<"ERROR : for Romberg method the size of the array should be 2^k+1" << std::endl;
  int n=m-1, Nr=0;
  while (n!=1){
    n = n/2;
    Nr++;
  }
  int N=Nr+1;

  bl::Array<double,1> h(N+1);
  bl::Array<double,2> r(N+1,N+1);
  
  for (int i = 1; i < N + 1; ++i) {
    h(i) = dh / ( 1<<(i-1) ) ;
  }
  r(1,1) = h(1) / 2 * (ff(0) + ff(m-1));
  for (int i = 2; i < N + 1; ++i) {
    double coeff = 0;
    int dr =  1 << (N-i);
    for (int k = 1; k <= (1<<(i-2)); ++k) coeff += ff((2*k-1)*dr);
    r(i,1) = 0.5 * (r(i-1,1) + h(i-1)*coeff);
  }
  for (int i = 2; i < N + 1; ++i) {
    for (int j = 2; j <= i; ++j) {
      r(i,j) = r(i,j - 1) + (r(i,j-1) - r(i-1,j-1))/( (1<<(2*(j-1))) -1 );
    }
  }
  return r(N,N);
}

template<int m, int Nr>
double romberg2(const bl::TinyVector<double,m>& ff, double dh){
  const int N=Nr+1;
  int m2 = (1<<Nr) + 1;
  if (m2 != m){
    std::cout<<"ERROR : romberg should have number of points 2**k+1"<<std::endl;
  }
  double h[N+1], r[N+1][N+1];
  for (int i = 1; i < N + 1; ++i) {
    h[i] = dh / ( 1<<(i-1) ) ;
  }
  r[1][1] = h[1] / 2 * (ff[0] + ff[m-1]);
  for (int i = 2; i < N + 1; ++i) {
    double coeff = 0;
    int dr = 1 << (N-i);
    for (int k = 1; k <= (1<<(i-2)); ++k) coeff += ff[(2*k-1)*dr];
    r[i][1] = 0.5 * (r[i-1][1] + h[i-1]*coeff);
  }
  for (int i = 2; i < N + 1; ++i) {
    for (int j = 2; j <= i; ++j) {
      r[i][j] = r[i][j - 1] + (r[i][j-1] - r[i-1][j-1])/( (1<<(2*(j-1))) -1);
    }
  }
  return r[N][N];
}
/*
template<int m, int Nr>
double romberg2(const bl::TinyVector<double,m>& ff, double dh){
  const int N=Nr+1;
  int m2 = (1<<Nr) + 1;
  if (m2 != m){
    std::cout<<"ERROR : romberg should have number of points 2**k+1"<<std::endl;
  }
  double h[N+1], r[N+1][N+1];
  for (int i = 1; i < N + 1; ++i) {
    h[i] = dh / static_cast<int>(pow(2.0,i-1));
  }
  r[1][1] = h[1] / 2 * (ff[0] + ff[m-1]);
  for (int i = 2; i < N + 1; ++i) {
    double coeff = 0;
    int dr = static_cast<int>(pow(2.0,N-i));
    for (int k = 1; k <= static_cast<int>(pow(2.0, i-2)); ++k) coeff += ff[(2*k-1)*dr];
    r[i][1] = 0.5 * (r[i-1][1] + h[i-1]*coeff);
  }
  for (int i = 2; i < N + 1; ++i) {
    for (int j = 2; j <= i; ++j) {
      r[i][j] = r[i][j - 1] + (r[i][j-1] - r[i-1][j-1])/(static_cast<int>(pow(4., j-1))-1);
    }
  }
  return r[N][N];
}
*/

bl::Array<int,2> BinomialCoeff(int n){
  // Returns value of Binomial Coefficient C(n, k)
  bl::Array<int,2> C(n+1,n+1);
  // Calculate value of Binomial Coefficient in bottom up manner
  for (int i = 0; i <= n; i++){
    for (int j = 0; j <= i; j++){
      // Base Cases
      if (j == 0 || j == i) C(i,j) = 1;
      // Calculate value using previosly stored values
      else
        C(i,j) = C(i-1,j-1) + C(i-1,j);
    }
  }
  return C;
}

static unsigned int log2 (unsigned int val) {
  if (val == 0) return UINT_MAX;
  if (val == 1) return 0;
  unsigned int ret = 0;
  while (val > 1) {
    val >>= 1;
    ret++;
  }
  return ret;
}


#ifndef BZ_VECNORM_CC
#define BZ_VECNORM_CC
template<typename T, int N>
inline double norm(const bl::TinyVector<T,N>& vector)
{
  double sum = 0.0;
  for (int i=0; i < N; ++i){
    sum += vector[i]*vector[i];
  }
  return sqrt(sum);
}
#endif



#endif //MY_UTIL
