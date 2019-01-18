//#include <iostream>
#include <blitz/array.h>
//#include <algorithm>
//#define _TIME
//#include "timer.h"
//#include "legendre.h"
using namespace blitz;
// @Copyright 2018 Kristjan Haule and Kun Chen    

#ifdef NO_APPEND_FORTRAN
# define FNAME(x) x
#else
# define FNAME(x) x##_
#endif
extern "C" {
  void FNAME(dgesdd)(const char* jobz, const int* m, const int* n, double* A, const int* lda, double* S, double* U, const int* ldu, double* Vt, const int* ldvt, double* work, const int* lwork, int* iwork, int* info);
  void FNAME(zgesdd)(const char* jobz, const int* m, const int* n, complex<double>* A, const int* lda, double* S, complex<double>* U, const int* ldu, complex<double>* Vt, const int* ldvt, complex<double>* work, const int* lwork, double* rwork, int* iwork, int* info);
  void FNAME(dgemm)(const char* transa, const char* transb, const int* m, const int* n, const int* k, const double* alpha, const double* A, const int* lda, const double* B, const int* ldb, const double* beta, double* C, const int* ldc);
  void FNAME(zgemm)(const char* transa, const char* transb, const int* m, const int* n, const int* k, const complex<double>* alpha, const complex<double>* A, const int* lda, const complex<double>* B, const int* ldb, const complex<double>* beta, complex<double>* C, const int* ldc);
  //
  void FNAME(dgetrf)(int* n1, int* n2, double* a, int* lda, int* ipiv,int* info);
  void FNAME(zgetrf)(int* n1, int* n2, complex<double>* a, int* lda, int* ipiv,int* info);
  void FNAME(dgetrs)(const char* trans, int* n, int* nrhs, double* a, int* lda, int* ipiv, double* b, int* ldb, int* info);
  void FNAME(zgetrs)(const char* trans, int* n, int* nrhs, complex<double>* a, int* lda, int* ipiv, complex<double>* b, int* ldb, int* info);
  void FNAME(dtrtri)(const char* uplo, const char* diag, int* n, double* a, int* lda, int* info );
  void FNAME(ztrtri)(const char* uplo, const char* diag, int* n, complex<double>* a, int* lda, int* info );
}

inline int xgesdd(bool vect, int M, int N, double* A, int lda, double* S, double* U, int ldu, double* Vt, int ldvt, double* work, int lwork, double*, int *iwork)
{
  int info = 0;
  std::string job = vect ? "S" : "N";
  FNAME(dgesdd)(job.c_str(), &M, &N, A, &lda, S, U, &ldu, Vt, &ldvt, work, &lwork, iwork, &info);
  if (info) {
    std::cerr << "Can't compute SVD of the kernel! " << info << std::endl;
  }
  return info;
}
inline int xgesdd(bool vect, int M, int N, complex<double>* A, int lda, double* S, complex<double>* U, int ldu, complex<double>* Vt, int ldvt, complex<double>* work, int lwork, double* rwork, int *iwork)
{
  int info = 0;
  std::string job = vect ? "A" : "N";
  FNAME(zgesdd)(job.c_str(), &M, &N, A, &lda, S, U, &ldu, Vt, &ldvt, work, &lwork, rwork, iwork, &info);
  if (info) {
    std::cerr << "Can't compute SVD of the kernel! " << info << std::endl;
  }
  return info;
}
inline void xgemm(const std::string& transa, const std::string& transb, const int m, const int n,
                  const int k, const double alpha, const double* A,
                  const int lda, const double* B, const int ldb, const double beta,
                  double* C, const int ldc)
{
  FNAME(dgemm)(transa.c_str(), transb.c_str(), &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
}

inline void xgemm(const std::string& transa, const std::string& transb, const int m, const int n,
                  const int k, const complex<double>& alpha, const complex<double>* A,
                  const int lda, const complex<double>* B, const int ldb, const complex<double>& beta,
                  complex<double>* C, const int ldc)
{
  FNAME(zgemm)(transa.c_str(), transb.c_str(), &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
}


inline int xgetrf(int n, double* a, int lda, int* ipiv, int ldb)
{
  int info = 0;
  FNAME(dgetrf)(&n, &n, a, &lda, ipiv, &info);
  if (info){
    std::cerr << "Something wrong in LU (real) decomposition! " << info << std::endl;
  }  
  return info;
}
inline int xgetrf(int n, complex<double>* a, int lda, int* ipiv, int ldb)
{
  int info = 0;
  FNAME(zgetrf)(&n, &n, a, &lda, ipiv, &info);
  if (info){
    std::cerr << "Something wrong in LU (complex) decomposition! " << info << std::endl;
  }  
  return info;
}
inline int xgetrs(int n, int nrhs, double* a, int lda, int* ipiv, double* b, int ldb)
{
  int info = 0;
  FNAME(dgetrs)("T", &n, &nrhs, a, &lda, ipiv, b, &ldb, &info);
  if (info){
    std::cerr << "Something wrong with the system of (real) equations! " << info << std::endl;
  }  
  return info;
}
inline int xgetrs(int n, int nrhs, complex<double>* a, int lda, int* ipiv, complex<double>* b, int ldb)
{
  int info = 0;
  FNAME(zgetrs)("T", &n, &nrhs, a, &lda, ipiv, b, &ldb, &info);
  if (info){
    std::cerr << "Something wrong with the system of (complex) equations! " << info << std::endl;
  }
  return info;
}
template<typename T>
inline int xgesv(int n, int nrhs, T* a, int lda, int* ipiv, T* b, int ldb)
{
  int info = xgetrf(n, a, lda, ipiv, ldb);
  if (info) return info;
  return xgetrs(n, nrhs, a, lda, ipiv, b, ldb);
}

inline int xtrtri(const std::string& uplo, const std::string& diag, int n, double* a, int lda)
{
  int info = 0;
  FNAME(dtrtri)(uplo.c_str(), diag.c_str(), &n, a, &lda, &info);
  return info;
}

template<typename T>
inline void MProduct(Array<T,2>& C, const Array<T,2>& A, const Array<T,2>& B, const T& alpha=1.0, const T& beta=0.0)
{// Usual scalar product of two matrices : C = A * B
  if (C.extent(0)!=A.extent(0) || C.extent(1)!=B.extent(1)) C.resize(A.extent(0), B.extent(1));
  if (A.extent(1) != B.extent(0) || !B.extent(1) || !A.extent(0) || !A.extent(1) || !B.extent(0)) std::cerr << " Matrix sizes not alligned in MProduct" << std::endl;
  xgemm("N", "N", B.extent(1), A.extent(0), B.extent(0), alpha, B.data(), B.extent(1), A.data(), A.extent(1), beta, C.data(), C.extent(1));
}

template <typename T>
inline void Product(Array<T,2>& C, const Array<T,2>& A, const std::string& transa, const Array<T,2>& B, const std::string& transb, const T& alpha=1.0, const T& beta=0.0)
{ // Matrix product of matrices, where we can specify A or B to be transposed or conjugated.
  //if (transa!="N" && transa!="T" && transa!="C"){std::cerr<<"Did not recognize your task. Specify how to multiply matrices in dgemm!"<<std::endl; return;}
  if(transa=="N" && transb=="N"){
    if (C.extent(0)!=A.extent(0) || C.extent(1)!=B.extent(1)) C.resize(A.extent(0), B.extent(1));
    if (A.extent(1) != B.extent(0) || !B.extent(1) || !A.extent(0) || !A.extent(1) || !B.extent(0))
      std::cerr << " Matrix sizes not correct" << std::endl;
    xgemm(transb, transa, B.extent(1), A.extent(0), B.extent(0), alpha, B.data(), B.extent(1), A.data(), A.extent(1), beta, C.data(), C.extent(1));
  }else if((transa=="T"||transa=="C") && transb=="N"){
    if (C.extent(0)!=A.extent(1) || C.extent(1)!=B.extent(1)) C.resize(A.extent(1), B.extent(1));
    if (A.extent(0) != B.extent(0) || !B.extent(1) || !A.extent(1) || !A.extent(0) || !B.extent(0))
      std::cerr << " Matrix sizes not correct" << std::endl;
    xgemm(transb, transa, B.extent(1), A.extent(1), B.extent(0), alpha, B.data(), B.extent(1), A.data(), A.extent(1), beta, C.data(), C.extent(1));
  }else if (transa=="N" && (transb=="T"||transb=="C")){
    if (C.extent(0)!=A.extent(0) || C.extent(1)!=B.extent(0)) C.resize(A.extent(0), B.extent(0));
    if (A.extent(1) != B.extent(1) || !B.extent(0) || !A.extent(0) || !A.extent(1) || !B.extent(1))
      std::cerr << " Matrix sizes not correct" << std::endl;
    xgemm(transb, transa, B.extent(0), A.extent(0), B.extent(1), alpha, B.data(), B.extent(1), A.data(), A.extent(1), beta, C.data(), C.extent(1));
  }else if ((transa=="T"||transa=="C") && (transb=="T"||transb=="C")){
    if (C.extent(0)!=A.extent(1) || C.extent(1)!=B.extent(0)) C.resize(A.extent(1), B.extent(0));
    if (A.extent(0) != B.extent(1) || !B.extent(0) || !A.extent(0) || !A.extent(1) || !B.extent(1))
      std::cerr << " Matrix sizes not correct" << std::endl;
    xgemm(transb, transa, B.extent(0), A.extent(1), B.extent(1), alpha, B.data(), B.extent(1), A.data(), A.extent(1), beta, C.data(), C.extent(1));
  }
}

template<typename T>
inline void MProduct(Array<T,1>& c, const Array<T,2>& A, const Array<T,1>& b)
{ // scalar product of matrix with a vector
  if (c.extent(0)!=A.extent(0)) c.resize(A.extent(0));
  c=0;
  for (int l=0; l<A.extent(0); ++l){
    T dsum=0;
    for (int i=0; i<A.extent(1); ++i) dsum += A(l,i)*b(i);
    c(l) = dsum;
  }
}

void SVD(Array<double,2>& U, Array<double,1>& S, Array<double,2>& Vt, int& lmax, Array<double,2>& Ker)
{
  // Suppose K(om,tau)
  int Nom = Ker.extent(0);
  int Nt = Ker.extent(1);
  int Nl = min(Nom,Nt);
  S.resize(Nl);
  U.resize(Nom,Nom);
  Vt.resize(Nt,Nt);

  int lwork;
  {
    int mn = min(Nt,Nom);
    int mx = max(Nt,Nom);
    lwork =  4*mn*mn + 6*mn + mx + 4*mx;
  }
  Array<double,1> work(lwork);
  Array<int,1> iwork(8*min(Nom,Nt));
  int info=0;
  clock_t t0 = clock();
  info = xgesdd(true, Nt, Nom, Ker.data(), Ker.extent(1), S.data(), Vt.data(), Vt.extent(1), U.data(), U.extent(1), work.data(), lwork, work.data(), iwork.data());
  double dt = static_cast<double>( clock() - t0 )/CLOCKS_PER_SEC;
  clog<<"svd time="<<dt<<endl;

  lmax = Nl;
  for (int l=0; l<Nl; l++)
    if (fabs(S(l))<1e-13){
      lmax=l;
      clog<<"lmax reduced to "<<lmax<<endl;
      break;
    }
  clog<<"last singular value="<<S(lmax-1)<<endl;
}

inline int SolveSOLA(Array<double,2>& A, Array<double,2>& B, Array<int,1>& ipiv)
{ // Solves system of linear equations
  return xgesv(A.extent(1), B.extent(0), A.data(), A.extent(1), ipiv.data(), B.data(), B.extent(1));
}
template <typename T>
inline int Inverse(Array<T,2>& A)
{
  if (A.extent(0)!=A.extent(1)) {std::cerr<<"Can not invert nonquadratic matrix! "<<std::endl;return 1;}
  int N = A.extent(0);
  // B set to idenity
  Array<T,2> B(N,N);
  B = 0;
  for (int i=0; i<N; ++i) B(i,i)=1.0;
  Array<int,1> ipiv(N);
  
  int rr = SolveSOLA(A,B,ipiv);
  
  for (int i=0; i<N; ++i)
    for (int j=0; j<N; ++j)
      A(i,j) = B(j,i);
  return rr;
}
 
template <typename T>
inline int SolveSOLA(Array<T,2>& A, Array<T,1>& B, Array<int,1>& ipiv)
{
  return xgesv(A.extent(1), 1, A.data(), A.extent(1), ipiv.data(), B.data(), B.extent(0));
}
template <typename T>
inline int InverseTriangularMatrix(const std::string& uplo, Array<T,2>& A)
{
  if (A.extent(0)!=A.extent(1)) {std::cerr<<"Can not invert nonquadratic matrix! "<<std::endl;return 1;}
  std::string _uplo_="U";
  // Because things in fortran look transpose, we need to switch
  if (uplo=="U" || uplo=="u") _uplo_ = "L";
  else if (uplo=="L" || uplo=="l") _uplo_ = "U";
  else cerr<<"Did not recognize uplo"<<endl;
  std::string diag = "U";
  for (int i=0; i<A.extent(0); i++)
    if ( fabs(A(i,i)-1.0) > 1e-6) diag="N";
    
  return xtrtri(_uplo_, diag, A.extent(0), A.data(), A.extent(1));
}
