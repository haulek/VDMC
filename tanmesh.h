// @Copyright 2018 Kristjan Haule and Kun Chen    
#ifndef _TAN_MESH
#define _TAN_MESH

#include <gsl/gsl_vector.h>
#include <gsl/gsl_multiroots.h>

struct qparams{
  double x0, L;
  int Nw;
};

int tanmesh_g(const gsl_vector* x, void* params, gsl_vector* f){
  double x0 = ((struct qparams *) params)->x0;
  double  L = ((struct qparams *) params)->L;
  int Nw = ((struct qparams *) params)->Nw;
  const double d = gsl_vector_get (x, 0);
  const double w = gsl_vector_get (x, 1);
  gsl_vector_set (f, 0, L-w/tan(d) );
  gsl_vector_set (f, 1, x0-w*tan(M_PI/(2*Nw)-d/Nw) );
  return GSL_SUCCESS;
}

typedef int (*gsl_FNC)(const gsl_vector * x, void * params, gsl_vector * f);

class FindRootn{
  const gsl_multiroot_fsolver_type *T;
  gsl_multiroot_fsolver *s;
  size_t n;
  void* params;
  gsl_vector *x;
  gsl_multiroot_function f;
public:
  FindRootn(size_t n_, gsl_FNC fnc, double x_init[], void* params_) : n(n_), params(params_)
  {
    x = gsl_vector_alloc (n);
    f.f = fnc;
    f.n = n;
    f.params = params;
    for (int i=0; i<n; i++) gsl_vector_set (x, i, x_init[i]);
    T = gsl_multiroot_fsolver_hybrids;
    s = gsl_multiroot_fsolver_alloc (T, 2);
    gsl_multiroot_fsolver_set (s, &f, x);
  }
  void print_state (size_t iter, gsl_multiroot_fsolver * s){
    std::clog<<"iter = "<<iter<<" x=";
    for (int i=0; i<n; i++) std::clog<<gsl_vector_get(s->x,i)<<" ";
    std::clog<<" f(x)=";
    for (int i=0; i<n; i++) std::clog<<gsl_vector_get(s->f,i)<<" ";
    std::clog<<std::endl;
  };
  std::vector<double> call(){
    size_t iter = 0;
    //print_state (iter, s);
    int status;
    do{
      iter++;
      status = gsl_multiroot_fsolver_iterate (s);
      //print_state (iter, s);
      if (status)   /* check if solver is stuck */
	break;
      status = gsl_multiroot_test_residual (s->f, 1e-7);
    } while (status == GSL_CONTINUE && iter < 1000);
    //clog<<"status = "<<gsl_strerror (status)<<endl;
    
    std::vector<double> res(n);
    for (int i=0; i<n; i++) res[i] = gsl_vector_get (s->x, i);
    return res;
  }
  ~FindRootn(){
    gsl_multiroot_fsolver_free (s);
    gsl_vector_free (x);
  }
};

template<typename container>
void GiveTanMesh(container& om, container& dom, double& x0, double L, int Nw)
{
  double tnw = tan(M_PI/(2*Nw));
  if (x0 > L*0.25*Nw*tnw*tnw ){
    x0 = L*0.25*Nw*tnw*tnw-1e-15;
  }
  double d0 = Nw/2.*( tnw - sqrt( tnw*tnw - 4*x0/(L*Nw)));
  double w0 = L*d0;
  double x_init[2] = {d0, w0};
  struct qparams p = {x0,L,Nw};
  
  FindRootn fr(2, &tanmesh_g, x_init, &p);
  std::vector<double> dw = fr.call();
  double d = dw[0];
  double w = dw[1];
  
  om.resize(2*Nw+1);
  dom.resize(2*Nw+1);
  double dh = 1.0/static_cast<double>(2*Nw);
  for (int i=0; i<2*Nw+1; i++){
    double t0 = i/static_cast<double>(2*Nw);
    double t = t0*(M_PI-2*d) - M_PI/2 + d;
    om(i)  = w*tan(t);
    dom(i) = dh*w*(M_PI-2*d)/(cos(t)*cos(t));
  }
}
#endif // _TAN_MESH
