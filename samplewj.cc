// @Copyright 2018 Kristjan Haule and Kun Chen    
#include <cstdint>
#include <ostream>
#include <deque>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "pystreambuf.h"
#include "samplebj.h"
#include "analytic.h"
#include "interpolate.h"
#include "baymkadanoff.h"

namespace py = pybind11;
#define _RENAME_COLLISION
#include "ypy.h"

class ConvertDiags{
public:
  bl::Array<int,2> diagsG;
  bl::Array<double,1> diagSign;
  bl::Array<double,1> kx;
  bl::Array<double,1> epsx;
  bl::Array<int,2> Vtype;
  bl::Array<int,1> indx;
public:
  //ConvertDiags cd(diagsG,diagSign,kx,epsx);  
  ConvertDiags(py::array_t<int>& _diagsG_, py::array_t<double>& _diagSign_):
    diagsG( (int*)_diagsG_.request().ptr, bl::shape(_diagsG_.request().shape[0],_diagsG_.request().shape[1]), bl::neverDeleteData),
    diagSign( (double*)_diagSign_.request().ptr, _diagSign_.request().shape[0], bl::neverDeleteData)
  {
    py::buffer_info info_d = _diagsG_.request();
    py::buffer_info info_s = _diagSign_.request();
    if (info_d.ndim != 2) throw std::runtime_error("Number of dimensions for diagsG must be 2");
    if (info_s.ndim != 1) throw std::runtime_error("Number of dimensions for diagSign must be 1");
    if (info_d.shape[0] != info_s.shape[0]) throw std::runtime_error("Number of diagrams in diagsG and diagSign has to be the same.");
  }
  ConvertDiags(py::array_t<int>& _diagsG_, py::array_t<double>& _diagSign_, py::array_t<int>& _Vtype_, py::array_t<int>& _indx_):
    diagsG( (int*)_diagsG_.request().ptr, bl::shape(_diagsG_.request().shape[0],_diagsG_.request().shape[1]), bl::neverDeleteData),
    diagSign( (double*)_diagSign_.request().ptr, _diagSign_.request().shape[0], bl::neverDeleteData),
    Vtype( (int*)_Vtype_.request().ptr, bl::shape(_Vtype_.request().shape[0],_Vtype_.request().shape[1]), bl::neverDeleteData),
    indx( (int*)_indx_.request().ptr, _indx_.request().shape[0], bl::neverDeleteData)
  {
    py::buffer_info info_d = _diagsG_.request();
    py::buffer_info info_s = _diagSign_.request();
    if (info_d.ndim != 2) throw std::runtime_error("Number of dimensions for diagsG must be 2");
    if (info_s.ndim != 1) throw std::runtime_error("Number of dimensions for diagSign must be 1");
    if (info_d.shape[0] != info_s.shape[0]) throw std::runtime_error("Number of diagrams in diagsG and diagSign has to be the same.");
  }
  ConvertDiags(py::array_t<int>& _diagsG_, py::array_t<double>& _diagSign_, py::array_t<double>& _kx_, py::array_t<double>& _epsx_):
    diagsG( (int*)_diagsG_.request().ptr, bl::shape(_diagsG_.request().shape[0],_diagsG_.request().shape[1]), bl::neverDeleteData),
    diagSign( (double*)_diagSign_.request().ptr, _diagSign_.request().shape[0], bl::neverDeleteData),
    kx( (double*)_kx_.request().ptr, _kx_.request().shape[0], bl::neverDeleteData),
    epsx( (double*)_epsx_.request().ptr, _epsx_.request().shape[0], bl::neverDeleteData)
  {
    py::buffer_info info_d = _diagsG_.request();
    py::buffer_info info_s = _diagSign_.request();
    if (info_d.ndim != 2) throw std::runtime_error("Number of dimensions for diagsG must be 2");
    if (info_s.ndim != 1) throw std::runtime_error("Number of dimensions for diagSign must be 1");
    if (info_d.shape[0] != info_s.shape[0]) throw std::runtime_error("Number of diagrams in diagsG and diagSign has to be the same.");
    py::buffer_info info_k = _kx_.request();
    py::buffer_info info_e = _epsx_.request();
    if (info_k.ndim != 1) throw std::runtime_error("Number of dimensions for kx must be 1");
    if (info_e.ndim != 1) throw std::runtime_error("Number of dimensions for epsx must be 1");
  }
  ConvertDiags(py::array_t<int>& _diagsG_, py::array_t<double>& _diagSign_, py::array_t<double>& _kx_, py::array_t<double>& _epsx_, py::array_t<int>& _Vtype_, py::array_t<int>& _indx_):
    diagsG( (int*)_diagsG_.request().ptr, bl::shape(_diagsG_.request().shape[0],_diagsG_.request().shape[1]), bl::neverDeleteData),
    diagSign( (double*)_diagSign_.request().ptr, _diagSign_.request().shape[0], bl::neverDeleteData),
    kx( (double*)_kx_.request().ptr, _kx_.request().shape[0], bl::neverDeleteData),
    epsx( (double*)_epsx_.request().ptr, _epsx_.request().shape[0], bl::neverDeleteData),
    Vtype( (int*)_Vtype_.request().ptr, bl::shape(_Vtype_.request().shape[0],_Vtype_.request().shape[1]), bl::neverDeleteData),
    indx( (int*)_indx_.request().ptr, _indx_.request().shape[0], bl::neverDeleteData)
  {
    py::buffer_info info_d = _diagsG_.request();
    py::buffer_info info_s = _diagSign_.request();
    if (info_d.ndim != 2) throw std::runtime_error("Number of dimensions for diagsG must be 2");
    if (info_s.ndim != 1) throw std::runtime_error("Number of dimensions for diagSign must be 1");
    if (info_d.shape[0] != info_s.shape[0]) throw std::runtime_error("Number of diagrams in diagsG and diagSign has to be the same.");
    py::buffer_info info_k = _kx_.request();
    py::buffer_info info_e = _epsx_.request();
    if (info_k.ndim != 1) throw std::runtime_error("Number of dimensions for kx must be 1");
    if (info_e.ndim != 1) throw std::runtime_error("Number of dimensions for epsx must be 1");
  }
};
class ConvertDiags_{
public:
  bl::Array<unsigned short,2> diagsG;
  bl::Array<float,1> diagSign;
  bl::Array<double,1> kx;
  bl::Array<double,1> epsx;
  bl::Array<char,2> Vtype;
  bl::Array<int,1> indx;
  bl::Array<double,1> qx;
public:
  ConvertDiags_(py::array_t<int>& _diagsG_, py::array_t<double>& _diagSign_):
    diagsG(_diagsG_.request().shape[0],_diagsG_.request().shape[1]),
    diagSign(_diagSign_.request().shape[0])
  {
    py::buffer_info info_d = _diagsG_.request();
    bl::Array<int,2> t_diagsG( (int*)info_d.ptr, bl::shape(info_d.shape[0],info_d.shape[1]), bl::neverDeleteData);
    for (int i=0; i<diagsG.extent(0); i++)
      for (int j=0; j<diagsG.extent(1); j++)
	diagsG(i,j) = t_diagsG(i,j); // note that we are here converting int -> unsigned short (for efficiency)
    
    py::buffer_info info_s = _diagSign_.request();
    bl::Array<double,1> t_diagSign( (double*)info_s.ptr, info_s.shape[0], bl::neverDeleteData);
    for (int i=0; i<diagSign.extent(0); ++i)
      diagSign(i) = t_diagSign(i);
      
    if (info_d.ndim != 2) throw std::runtime_error("Number of dimensions for diagsG must be 2");
    if (info_s.ndim != 1) throw std::runtime_error("Number of dimensions for diagSign must be 1");
    if (info_d.shape[0] != info_s.shape[0]) throw std::runtime_error("Number of diagrams in diagsG and diagSign has to be the same.");
  }
  ConvertDiags_(py::array_t<int>& _diagsG_, py::array_t<double>& _diagSign_, py::array_t<int>& _Vtype_, py::array_t<int>& _indx_):
    diagsG(_diagsG_.request().shape[0],_diagsG_.request().shape[1]),
    diagSign(_diagSign_.request().shape[0]),
    Vtype(_Vtype_.request().shape[0],_Vtype_.request().shape[1]),
    indx( (int*)_indx_.request().ptr, _indx_.request().shape[0], bl::neverDeleteData)
  {
    py::buffer_info info_d = _diagsG_.request();
    bl::Array<int,2> t_diagsG( (int*)info_d.ptr, bl::shape(info_d.shape[0],info_d.shape[1]), bl::neverDeleteData);
    for (int i=0; i<diagsG.extent(0); i++)
      for (int j=0; j<diagsG.extent(1); j++)
	diagsG(i,j) = t_diagsG(i,j); // note that we are here converting int -> unsigned short (for efficiency)

    py::buffer_info info_s = _diagSign_.request();
    bl::Array<double,1> t_diagSign( (double*)info_s.ptr, info_s.shape[0], bl::neverDeleteData);
    for (int i=0; i<diagSign.extent(0); i++)
      diagSign(i) = t_diagSign(i); 
    
    py::buffer_info info_v = _Vtype_.request();
    bl::Array<int,2> t_Vtype( (int*)info_v.ptr, bl::shape(info_v.shape[0],info_v.shape[1]), bl::neverDeleteData);
    for (int i=0; i<Vtype.extent(0); i++)
      for (int j=0; j<Vtype.extent(1); j++)
	Vtype(i,j) = t_Vtype(i,j); // note that we are here converting int -> unsigned short (for efficiency)
    
    if (info_d.ndim != 2) throw std::runtime_error("Number of dimensions for diagsG must be 2");
    if (info_s.ndim != 1) throw std::runtime_error("Number of dimensions for diagSign must be 1");
    if (info_d.shape[0] != info_s.shape[0]) throw std::runtime_error("Number of diagrams in diagsG and diagSign has to be the same.");
  }
  ConvertDiags_(py::array_t<int>& _diagsG_, py::array_t<double>& _diagSign_, py::array_t<double>& _kx_, py::array_t<double>& _epsx_):
    diagsG(_diagsG_.request().shape[0],_diagsG_.request().shape[1]),
    diagSign(_diagSign_.request().shape[0]),
    kx( (double*)_kx_.request().ptr, _kx_.request().shape[0], bl::neverDeleteData),
    epsx( (double*)_epsx_.request().ptr, _epsx_.request().shape[0], bl::neverDeleteData)
  {
    py::buffer_info info_d = _diagsG_.request();
    bl::Array<int,2> t_diagsG( (int*)info_d.ptr, bl::shape(info_d.shape[0],info_d.shape[1]), bl::neverDeleteData);
    for (int i=0; i<diagsG.extent(0); i++)
      for (int j=0; j<diagsG.extent(1); j++)
	diagsG(i,j) = t_diagsG(i,j); // note that we are here converting int -> unsigned short (for efficiency)
    
    py::buffer_info info_s = _diagSign_.request();
    bl::Array<double,1> t_diagSign( (double*)info_s.ptr, info_s.shape[0], bl::neverDeleteData);
    for (int i=0; i<diagSign.extent(0); i++)
      diagSign(i) = t_diagSign(i); 
    
    if (info_d.ndim != 2) throw std::runtime_error("Number of dimensions for diagsG must be 2");
    if (info_s.ndim != 1) throw std::runtime_error("Number of dimensions for diagSign must be 1");
    if (info_d.shape[0] != info_s.shape[0]) throw std::runtime_error("Number of diagrams in diagsG and diagSign has to be the same.");
    py::buffer_info info_k = _kx_.request();
    py::buffer_info info_e = _epsx_.request();
    if (info_k.ndim != 1) throw std::runtime_error("Number of dimensions for kx must be 1");
    if (info_e.ndim != 1) throw std::runtime_error("Number of dimensions for epsx must be 1");
  }
  ConvertDiags_(py::array_t<int>& _diagsG_, py::array_t<double>& _diagSign_, py::array_t<double>& _kx_, py::array_t<double>& _epsx_, py::array_t<int>& _Vtype_, py::array_t<int>& _indx_):
    diagsG(_diagsG_.request().shape[0],_diagsG_.request().shape[1]),
    diagSign(_diagSign_.request().shape[0]),
    Vtype(_Vtype_.request().shape[0],_Vtype_.request().shape[1]),
    kx( (double*)_kx_.request().ptr, _kx_.request().shape[0], bl::neverDeleteData),
    epsx( (double*)_epsx_.request().ptr, _epsx_.request().shape[0], bl::neverDeleteData),
    indx( (int*)_indx_.request().ptr, _indx_.request().shape[0], bl::neverDeleteData)
  {
    py::buffer_info info_d = _diagsG_.request();
    bl::Array<int,2> t_diagsG( (int*)info_d.ptr, bl::shape(info_d.shape[0],info_d.shape[1]), bl::neverDeleteData);
    for (int i=0; i<diagsG.extent(0); i++)
      for (int j=0; j<diagsG.extent(1); j++)
	diagsG(i,j) = t_diagsG(i,j); // note that we are here converting int -> unsigned short (for efficiency)

    py::buffer_info info_s = _diagSign_.request();
    bl::Array<double,1> t_diagSign( (double*)info_s.ptr, info_s.shape[0], bl::neverDeleteData);
    for (int i=0; i<diagSign.extent(0); i++)
      diagSign(i) = t_diagSign(i); 
    
    py::buffer_info info_v = _Vtype_.request();
    bl::Array<int,2> t_Vtype( (int*)info_v.ptr, bl::shape(info_v.shape[0],info_v.shape[1]), bl::neverDeleteData);
    for (int i=0; i<Vtype.extent(0); i++)
      for (int j=0; j<Vtype.extent(1); j++)
	Vtype(i,j) = t_Vtype(i,j); // note that we are here converting int -> unsigned short (for efficiency)
    
    if (info_d.ndim != 2) throw std::runtime_error("Number of dimensions for diagsG must be 2");
    if (info_s.ndim != 1) throw std::runtime_error("Number of dimensions for diagSign must be 1");
    if (info_d.shape[0] != info_s.shape[0]) throw std::runtime_error("Number of diagrams in diagsG and diagSign has to be the same.");
    py::buffer_info info_k = _kx_.request();
    py::buffer_info info_e = _epsx_.request();
    if (info_k.ndim != 1) throw std::runtime_error("Number of dimensions for kx must be 1");
    if (info_e.ndim != 1) throw std::runtime_error("Number of dimensions for epsx must be 1");
  }
  ConvertDiags_(py::array_t<int>& _diagsG_, py::array_t<double>& _kx_, py::array_t<double>& _epsx_, py::array_t<int>& _Vtype_, py::array_t<int>& _indx_):
    diagsG(_diagsG_.request().shape[0],_diagsG_.request().shape[1]),
    Vtype(_Vtype_.request().shape[0],_Vtype_.request().shape[1]),
    kx( (double*)_kx_.request().ptr, _kx_.request().shape[0], bl::neverDeleteData),
    epsx( (double*)_epsx_.request().ptr, _epsx_.request().shape[0], bl::neverDeleteData),
    indx( (int*)_indx_.request().ptr, _indx_.request().shape[0], bl::neverDeleteData)
  {
    py::buffer_info info_d = _diagsG_.request();
    bl::Array<int,2> t_diagsG( (int*)info_d.ptr, bl::shape(info_d.shape[0],info_d.shape[1]), bl::neverDeleteData);
    for (int i=0; i<diagsG.extent(0); i++)
      for (int j=0; j<diagsG.extent(1); j++)
	diagsG(i,j) = t_diagsG(i,j); // note that we are here converting int -> unsigned short (for efficiency)

    py::buffer_info info_v = _Vtype_.request();
    bl::Array<int,2> t_Vtype( (int*)info_v.ptr, bl::shape(info_v.shape[0],info_v.shape[1]), bl::neverDeleteData);
    for (int i=0; i<Vtype.extent(0); i++)
      for (int j=0; j<Vtype.extent(1); j++)
	Vtype(i,j) = t_Vtype(i,j); // note that we are here converting int -> unsigned short (for efficiency)
    
    if (info_d.ndim != 2) throw std::runtime_error("Number of dimensions for diagsG must be 2");
    //if (info_s.ndim != 1) throw std::runtime_error("Number of dimensions for diagSign must be 1");
    //if (info_d.shape[0] != info_s.shape[0]) throw std::runtime_error("Number of diagrams in diagsG and diagSign has to be the same.");
    py::buffer_info info_k = _kx_.request();
    py::buffer_info info_e = _epsx_.request();
    if (info_k.ndim != 1) throw std::runtime_error("Number of dimensions for kx must be 1");
    if (info_e.ndim != 1) throw std::runtime_error("Number of dimensions for epsx must be 1");
  }
  ConvertDiags_(py::array_t<int>& _diagsG_, py::array_t<double>& _kx_, py::array_t<double>& _epsx_, py::array_t<int>& _Vtype_, py::array_t<int>& _indx_, py::array_t<double>& _qx_):
    diagsG(_diagsG_.request().shape[0],_diagsG_.request().shape[1]),
    Vtype(_Vtype_.request().shape[0],_Vtype_.request().shape[1]),
    kx( (double*)_kx_.request().ptr, _kx_.request().shape[0], bl::neverDeleteData),
    epsx( (double*)_epsx_.request().ptr, _epsx_.request().shape[0], bl::neverDeleteData),
    indx( (int*)_indx_.request().ptr, _indx_.request().shape[0], bl::neverDeleteData),
    qx( (double*)_qx_.request().ptr, _qx_.request().shape[0], bl::neverDeleteData)
  {
    py::buffer_info info_d = _diagsG_.request();
    bl::Array<int,2> t_diagsG( (int*)info_d.ptr, bl::shape(info_d.shape[0],info_d.shape[1]), bl::neverDeleteData);
    for (int i=0; i<diagsG.extent(0); i++)
      for (int j=0; j<diagsG.extent(1); j++)
	diagsG(i,j) = t_diagsG(i,j); // note that we are here converting int -> unsigned short (for efficiency)

    py::buffer_info info_v = _Vtype_.request();
    bl::Array<int,2> t_Vtype( (int*)info_v.ptr, bl::shape(info_v.shape[0],info_v.shape[1]), bl::neverDeleteData);
    for (int i=0; i<Vtype.extent(0); i++)
      for (int j=0; j<Vtype.extent(1); j++)
	Vtype(i,j) = t_Vtype(i,j); // note that we are here converting int -> unsigned short (for efficiency)
    
    if (info_d.ndim != 2) throw std::runtime_error("Number of dimensions for diagsG must be 2");
    //if (info_s.ndim != 1) throw std::runtime_error("Number of dimensions for diagSign must be 1");
    //if (info_d.shape[0] != info_s.shape[0]) throw std::runtime_error("Number of diagrams in diagsG and diagSign has to be the same.");
    py::buffer_info info_k = _kx_.request();
    py::buffer_info info_e = _epsx_.request();
    if (info_k.ndim != 1) throw std::runtime_error("Number of dimensions for kx must be 1");
    if (info_e.ndim != 1) throw std::runtime_error("Number of dimensions for epsx must be 1");
  }
};

class ConvertOutput{
  py::array_t<double, py::array::c_style> First;
  py::array_t<double, py::array::c_style> Second;
public:
  ConvertOutput(const bl::Array<double,2>& _First_, const bl::Array<double,2>& _Second_) :
    First({ _First_.extent(0), _First_.extent(1) }), Second({ _Second_.extent(0), _Second_.extent(1) })
  {
    auto r = First.mutable_unchecked<2>();
    for (int i = 0; i < r.shape(0); i++)
      for (int j = 0; j < r.shape(1); j++)
	r(i,j) = _First_(i,j);
  
    auto q = Second.mutable_unchecked<2>();
    for (int i = 0; i < q.shape(0); i++)
      for (int j = 0; j < q.shape(1); j++)
	q(i,j) = _Second_(i,j);
  }
  py::tuple operator()()
  { return py::make_tuple(First, Second); }
};
class ConvertArray{
  py::array_t<double, py::array::c_style> A;
public:
  ConvertArray(const bl::Array<double,1>& _A_) : A(  _A_.extent(0)  )
  {
    auto q = A.mutable_unchecked<1>();
    for (int i = 0; i < q.shape(0); i++) q(i) = _A_(i);
  }
  py::array_t<double> operator()(){return A;}
};

/*
py::tuple sample_static_NI(double lmbda, const params& p, py::array_t<int>& diagsG, py::array_t<double>& diagSign,
			   const std::vector<std::vector<std::vector<int> > >& Loop_index, const std::vector<std::vector<std::vector<int> > >& Loop_type,
			   py::array_t<int>& Vtype, py::array_t<int>& indx, my_mpi& mpi)
{
  ConvertDiags cd(diagsG,diagSign,Vtype,indx);

  const int Nlq=18;  // 10 legendre polynomials for q-dependence
  const int Nlt=24;  // 16 legendre polynomials for t-dependence
  
  bl::Array<double,2> C_Pln(Nlt+1,Nlq+1), Pbin(p.Nt,p.Nq);
  
  egass_Gk Gk(p.beta,p.kF);        // non-interacting green's function
  sample_static<Nlt,Nlq>(C_Pln, Pbin, lmbda, Gk, p, cd.diagsG, cd.diagSign, Loop_index, Loop_type, cd.Vtype, cd.indx, mpi);

  return ConvertOutput(C_Pln, Pbin)();
}
*/
/*
py::tuple sample_static_HF(double lmbda, const params& p, py::array_t<double>& kx, py::array_t<double>& epsx, 
			   py::array_t<int>& diagsG, py::array_t<double>& diagSign,
			   const std::vector<std::vector<std::vector<int> > >& Loop_index, const std::vector<std::vector<std::vector<int> > >& Loop_type,
			   py::array_t<int>& Vtype, py::array_t<int>& indx, my_mpi& mpi)
{
  ConvertDiags cd(diagsG,diagSign,kx,epsx,Vtype,indx);

  const int Nlq=18;  // 10 legendre polynomials for q-dependence
  const int Nlt=24;  // 16 legendre polynomials for t-dependence
  
  bl::Array<double,2> C_Pln(Nlt+1,Nlq+1), Pbin(p.Nt,p.Nq);

  Gk_HF Gk(p.beta, p.kF, cd.kx, cd.epsx);   // Hartree-Fock non-interacting green's function
  sample_static<Nlt,Nlq>(C_Pln, Pbin, lmbda, Gk, p, cd.diagsG, cd.diagSign, Loop_index, Loop_type, cd.Vtype, cd.indx, mpi);

  return ConvertOutput(C_Pln, Pbin)();
}
*/
/*
py::tuple sample_static_fast_HF(std::ostream& log, double lmbda, const std::vector<double>& lmbda_spct, const params& p, py::array_t<double>& kx, py::array_t<double>& epsx, 
				py::array_t<int>& diagsG, py::array_t<double>& diagSign,
				const std::vector<std::vector<std::vector<int> > >& Loop_index, const std::vector<std::vector<std::vector<int> > >& Loop_type,
				py::array_t<int>& Vtype, py::array_t<int>& indx, my_mpi& mpi)
{
  ConvertDiags_ cd(diagsG,diagSign,kx,epsx,Vtype,indx);

  py::array_t<double> _Pbin_( {p.Nt,p.Nq} );
  py::buffer_info info_N = _Pbin_.request();
  bl::Array<double,2> Pbin((double*)info_N.ptr, bl::shape(info_N.shape[0],info_N.shape[1]), bl::neverDeleteData);
  Gk_HF Gk(p.beta, p.kF, cd.kx, cd.epsx);   // Hartree-Fock non-interacting green's function
  
  py::array_t<double> _C_Pln_( {p.Nlt+1,p.Nlq+1} );
  py::buffer_info info_P = _C_Pln_.request();
  bl::Array<double,2> C_Pln((double*)info_P.ptr, bl::shape(info_P.shape[0],info_P.shape[1]), bl::neverDeleteData);
  StandardData BKdata(C_Pln);

  sample_static_fastC(BKdata, Pbin, log, lmbda, lmbda_spct, Gk, p, cd.diagsG, cd.diagSign, Loop_index, Loop_type, cd.Vtype, mpi);

  if (mpi.rank==mpi.master) 
    return py::make_tuple(_C_Pln_, _Pbin_);
  else{
    py::array_t<double> _C_Pln_( {1,1} );// fake empty output in order not to copy huge data on all cores.
    return py::make_tuple(_C_Pln_, _Pbin_);
  }
}
py::tuple sample_static_fast_VHF(std::ostream& log, double lmbda, const std::vector<double>& lmbda_spct, const params& p, py::array_t<double>& kx, py::array_t<double>& epsx, 
				py::array_t<int>& diagsG, py::array_t<double>& diagSign,
				const std::vector<std::vector<std::vector<int> > >& Loop_index, const std::vector<std::vector<std::vector<int> > >& Loop_type,
				py::array_t<int>& Vtype, py::array_t<int>& indx, py::array_t<double>& _kxb_, my_mpi& mpi)
{
  ConvertDiags_ cd(diagsG,diagSign,kx,epsx,Vtype,indx);

  py::array_t<double> _Pbin_( {p.Nt,p.Nq} );
  py::buffer_info info_N = _Pbin_.request();
  bl::Array<double,2> Pbin((double*)info_N.ptr, bl::shape(info_N.shape[0],info_N.shape[1]), bl::neverDeleteData);
  Gk_HF Gk(p.beta, p.kF, cd.kx, cd.epsx);   // Hartree-Fock non-interacting green's function
  
  py::buffer_info info_kxb = _kxb_.request();
  bl::Array<double,1> kxb((double*)info_kxb.ptr, info_kxb.shape[0], bl::neverDeleteData);
  py::array_t<double> _C_Pln_( {p.Nthbin,kxb.extent(0)-1,p.Nlt+1,p.Nlq+1} );
  py::buffer_info info_P = _C_Pln_.request();
  bl::Array<double,4> C_Pln((double*)info_P.ptr, bl::shape(info_P.shape[0],info_P.shape[1],info_P.shape[2],info_P.shape[3]), bl::neverDeleteData);
  
  BaymKadanoffData BKdata(C_Pln, kxb);

  sample_static_fastC(BKdata, Pbin, log, lmbda, lmbda_spct, Gk, p, cd.diagsG, cd.diagSign, Loop_index, Loop_type, cd.Vtype, mpi);

  if (mpi.rank==mpi.master) 
    return py::make_tuple(_C_Pln_, _Pbin_);
  else{
    py::array_t<double> _C_Pln_( {1,1,1,1} );// fake empty output in order not to copy huge data on all cores.
    return py::make_tuple(_C_Pln_, _Pbin_);
  }
}
*/

py::tuple sample_static_fast_HFD(std::ostream& log, double lmbda, const std::vector<double>& lmbda_spct, const params& p, py::array_t<double>& kx, py::array_t<double>& epsx, 
				py::array_t<int>& diagsG, py::array_t<double>& diagSign,
				const std::vector<std::vector<std::vector<int> > >& Loop_index, const std::vector<std::vector<std::vector<int> > >& Loop_type,
				py::array_t<int>& Vtype, py::array_t<int>& indx, my_mpi& mpi)
{
  ConvertDiags_ cd(diagsG,diagSign,kx,epsx,Vtype,indx);

  py::array_t<double> _Pbin_( {p.Nt,p.Nq} );
  py::buffer_info info_N = _Pbin_.request();
  bl::Array<double,2> Pbin((double*)info_N.ptr, bl::shape(info_N.shape[0],info_N.shape[1]), bl::neverDeleteData);
  Gk_HF Gk(p.beta, p.kF, cd.kx, cd.epsx);   // Hartree-Fock non-interacting green's function
  
  py::array_t<double> _C_Pln_( {p.Nlt+1,p.Nlq+1} );
  py::buffer_info info_P = _C_Pln_.request();
  bl::Array<double,2> C_Pln((double*)info_P.ptr, bl::shape(info_P.shape[0],info_P.shape[1]), bl::neverDeleteData);
  StandardData BKdata(C_Pln);

  sample_static_fastD(BKdata, Pbin, log, lmbda, lmbda_spct, Gk, p, cd.diagsG, cd.diagSign, Loop_index, Loop_type, cd.Vtype, mpi);

  if (mpi.rank==mpi.master) 
    return py::make_tuple(_C_Pln_, _Pbin_);
  else{
    py::array_t<double> _C_Pln_( {1,1} );// fake empty output in order not to copy huge data on all cores.
    return py::make_tuple(_C_Pln_, _Pbin_);
  }
}
py::tuple sample_static_fast_VHFD(std::ostream& log, double lmbda, const std::vector<double>& lmbda_spct, const params& p, py::array_t<double>& kx, py::array_t<double>& epsx, 
				py::array_t<int>& diagsG, py::array_t<double>& diagSign,
				const std::vector<std::vector<std::vector<int> > >& Loop_index, const std::vector<std::vector<std::vector<int> > >& Loop_type,
				py::array_t<int>& Vtype, py::array_t<int>& indx, py::array_t<double>& _kxb_, my_mpi& mpi)
{
  ConvertDiags_ cd(diagsG,diagSign,kx,epsx,Vtype,indx);
  
  py::array_t<double> _Pbin_( {p.Nt,p.Nq} );
  py::buffer_info info_N = _Pbin_.request();
  bl::Array<double,2> Pbin((double*)info_N.ptr, bl::shape(info_N.shape[0],info_N.shape[1]), bl::neverDeleteData);
  Gk_HF Gk(p.beta, p.kF, cd.kx, cd.epsx);   // Hartree-Fock non-interacting green's function
  
  py::buffer_info info_kxb = _kxb_.request();
  bl::Array<double,1> kxb((double*)info_kxb.ptr, info_kxb.shape[0], bl::neverDeleteData);
  py::array_t<double> _C_Pln_( {p.Nthbin,kxb.extent(0)-1,p.Nlt+1,p.Nlq+1} );
  py::buffer_info info_P = _C_Pln_.request();
  bl::Array<double,4> C_Pln((double*)info_P.ptr, bl::shape(info_P.shape[0],info_P.shape[1],info_P.shape[2],info_P.shape[3]), bl::neverDeleteData);
  
  BaymKadanoffData BKdata(C_Pln, kxb);

  sample_static_fastD(BKdata, Pbin, log, lmbda, lmbda_spct, Gk, p, cd.diagsG, cd.diagSign, Loop_index, Loop_type, cd.Vtype, mpi);
  
  if (mpi.rank==mpi.master) 
    return py::make_tuple(_C_Pln_, _Pbin_);
  else{
    py::array_t<double> _C_Pln_( {1,1,1,1} );// fake empty output in order not to copy huge data on all cores.
    return py::make_tuple(_C_Pln_, _Pbin_);
  }
}

py::tuple sample_static_fast_HFC(std::ostream& log, double lmbda, const std::vector<double>& lmbda_spct, const params& p, py::array_t<double>& kx, py::array_t<double>& epsx, 
				 py::array_t<int>& diagsG, 
				 const std::vector<std::vector<int> >& diagSign,
				 const std::vector<std::vector<std::vector<int> > >& Loop_index, const std::vector<std::vector<std::vector<int> > >& Loop_type,
				 py::array_t<int>& Vtype, py::array_t<int>& indx, my_mpi& mpi)
{
  ConvertDiags_ cd(diagsG,kx,epsx,Vtype,indx);

  py::array_t<double> _Pbin_( {p.Nt,p.Nq} );
  py::buffer_info info_N = _Pbin_.request();
  bl::Array<double,2> Pbin((double*)info_N.ptr, bl::shape(info_N.shape[0],info_N.shape[1]), bl::neverDeleteData);
  Gk_HF Gk(p.beta, p.kF, cd.kx, cd.epsx);   // Hartree-Fock non-interacting green's function
  
  py::array_t<double> _C_Pln_( {p.Nlt+1,p.Nlq+1} );
  py::buffer_info info_P = _C_Pln_.request();
  bl::Array<double,2> C_Pln((double*)info_P.ptr, bl::shape(info_P.shape[0],info_P.shape[1]), bl::neverDeleteData);
  StandardData BKdata(C_Pln);

  sample_static_fastC_combined(BKdata, Pbin, log, lmbda, lmbda_spct, Gk, p, cd.diagsG, diagSign, Loop_index, Loop_type, cd.Vtype, mpi);

  if (mpi.rank==mpi.master) 
    return py::make_tuple(_C_Pln_, _Pbin_);
  else{
    py::array_t<double> _C_Pln_( {1,1} );// fake empty output in order not to copy huge data on all cores.
    return py::make_tuple(_C_Pln_, _Pbin_);
  }
}

py::tuple sample_static_fast_VHFC(std::ostream& log, double lmbda, const std::vector<double>& lmbda_spct, const params& p, py::array_t<double>& kx, py::array_t<double>& epsx, 
				  py::array_t<int>& diagsG,
				  const std::vector<std::vector<int> >& diagSign,
				  const std::vector<std::vector<std::vector<int> > >& Loop_index, const std::vector<std::vector<std::vector<int> > >& Loop_type,
				  py::array_t<int>& Vtype, py::array_t<int>& indx, py::array_t<double>& _kxb_, my_mpi& mpi)
{
  ConvertDiags_ cd(diagsG,kx,epsx,Vtype,indx);
  
  py::array_t<double> _Pbin_( {p.Nt,p.Nq} );
  py::buffer_info info_N = _Pbin_.request();
  bl::Array<double,2> Pbin((double*)info_N.ptr, bl::shape(info_N.shape[0],info_N.shape[1]), bl::neverDeleteData);
  Gk_HF Gk(p.beta, p.kF, cd.kx, cd.epsx);   // Hartree-Fock non-interacting green's function
  
  py::buffer_info info_kxb = _kxb_.request();
  bl::Array<double,1> kxb((double*)info_kxb.ptr, info_kxb.shape[0], bl::neverDeleteData);
  py::array_t<double> _C_Pln_( {p.Nthbin,kxb.extent(0)-1,p.Nlt+1,p.Nlq+1} );
  py::buffer_info info_P = _C_Pln_.request();
  bl::Array<double,4> C_Pln((double*)info_P.ptr, bl::shape(info_P.shape[0],info_P.shape[1],info_P.shape[2],info_P.shape[3]), bl::neverDeleteData);
  
  BaymKadanoffData BKdata(C_Pln, kxb);

  sample_static_fastC_combined(BKdata, Pbin, log, lmbda, lmbda_spct, Gk, p, cd.diagsG, diagSign, Loop_index, Loop_type, cd.Vtype, mpi);
  
  if (mpi.rank==mpi.master) 
    return py::make_tuple(_C_Pln_, _Pbin_);
  else{
    py::array_t<double> _C_Pln_( {1,1,1,1} );// fake empty output in order not to copy huge data on all cores.
    return py::make_tuple(_C_Pln_, _Pbin_);
  }
}


py::tuple sample_static_Discrete_HFC(std::ostream& log, py::array_t<double>& qx,
				     double lmbda, const std::vector<double>& lmbda_spct, const params& p, py::array_t<double>& kx, py::array_t<double>& epsx, 
				     py::array_t<int>& diagsG, 
				     const std::vector<std::vector<int> >& diagSign,
				     const std::vector<std::vector<std::vector<int> > >& Loop_index, const std::vector<std::vector<std::vector<int> > >& Loop_type,
				     py::array_t<int>& Vtype, py::array_t<int>& indx, my_mpi& mpi)
{
  ConvertDiags_ cd(diagsG,kx,epsx,Vtype,indx,qx);

  py::array_t<double> _Pbin_( {p.Nt, cd.qx.extent(0)} );
  py::buffer_info info_N = _Pbin_.request();
  bl::Array<double,2> Pbin((double*)info_N.ptr, bl::shape(info_N.shape[0],info_N.shape[1]), bl::neverDeleteData);
  Gk_HF Gk(p.beta, p.kF, cd.kx, cd.epsx);   // Hartree-Fock non-interacting green's function
  
  py::array_t<double> _C_Pln_( {cd.qx.extent(0),p.Nlt+1} );
  py::buffer_info info_P = _C_Pln_.request();
  bl::Array<double,2> C_Pln((double*)info_P.ptr, bl::shape(info_P.shape[0],info_P.shape[1]), bl::neverDeleteData);
  Standard_Discrete BKdata(C_Pln);
  
  sample_static_DiscreteC_combined(BKdata, Pbin, log, cd.qx, lmbda, lmbda_spct, Gk, p, cd.diagsG, diagSign, Loop_index, Loop_type, cd.Vtype, mpi);

  if (mpi.rank==mpi.master) 
    return py::make_tuple(_C_Pln_, _Pbin_);
  else{
    py::array_t<double> _C_Pln_( {1,1} );// fake empty output in order not to copy huge data on all cores.
    return py::make_tuple(_C_Pln_, _Pbin_);
  }
}

py::tuple sample_static_Discrete_VHFC(std::ostream& log, py::array_t<double>& qx,
				      double lmbda, const std::vector<double>& lmbda_spct, const params& p, py::array_t<double>& kx, py::array_t<double>& epsx, 
				      py::array_t<int>& diagsG, 
				      const std::vector<std::vector<int> >& diagSign,
				      const std::vector<std::vector<std::vector<int> > >& Loop_index, const std::vector<std::vector<std::vector<int> > >& Loop_type,
				      py::array_t<int>& Vtype, py::array_t<int>& indx, py::array_t<double>& _kxb_, my_mpi& mpi)
{
  ConvertDiags_ cd(diagsG,kx,epsx,Vtype,indx,qx);
  
  py::array_t<double> _Pbin_( {p.Nt, cd.qx.extent(0)} );
  py::buffer_info info_N = _Pbin_.request();
  bl::Array<double,2> Pbin((double*)info_N.ptr, bl::shape(info_N.shape[0],info_N.shape[1]), bl::neverDeleteData);
  Gk_HF Gk(p.beta, p.kF, cd.kx, cd.epsx);   // Hartree-Fock non-interacting green's function
  
  py::buffer_info info_kxb = _kxb_.request();
  bl::Array<double,1> kxb((double*)info_kxb.ptr, info_kxb.shape[0], bl::neverDeleteData);
  py::array_t<double> _C_Pln_( {p.Nthbin,kxb.extent(0)-1,cd.qx.extent(0),p.Nlt+1} );
  py::buffer_info info_P = _C_Pln_.request();
  bl::Array<double,4> C_Pln((double*)info_P.ptr, bl::shape(info_P.shape[0],info_P.shape[1],info_P.shape[2],info_P.shape[3]), bl::neverDeleteData);
  BaymKadanoff_Discrete BKdata(C_Pln, kxb);

  sample_static_DiscreteC_combined(BKdata, Pbin, log, cd.qx, lmbda, lmbda_spct, Gk, p, cd.diagsG, diagSign, Loop_index, Loop_type, cd.Vtype, mpi);
  
  if (mpi.rank==mpi.master) 
    return py::make_tuple(_C_Pln_, _Pbin_);
  else{
    py::array_t<double> _C_Pln_( {1,1,1,1} );// fake empty output in order not to copy huge data on all cores.
    return py::make_tuple(_C_Pln_, _Pbin_);
  }
}


py::tuple sample_static_Q0W0_VHFD(std::ostream& log, double lmbda, const std::vector<double>& lmbda_spct, const params& p, py::array_t<double>& kx, py::array_t<double>& epsx, 
				  py::array_t<int>& diagsG, py::array_t<double>& diagSign,
				  const std::vector<std::vector<std::vector<int> > >& Loop_index, const std::vector<std::vector<std::vector<int> > >& Loop_type,
				  py::array_t<int>& Vtype, py::array_t<int>& indx, py::array_t<double>& _kxb_, my_mpi& mpi)
{
  ConvertDiags_ cd(diagsG,diagSign,kx,epsx,Vtype,indx);
  
  py::array_t<double> _Pbin_( p.Nt );
  py::buffer_info info_N = _Pbin_.request();
  bl::Array<double,2> Pbin((double*)info_N.ptr, bl::shape(p.Nt,1), bl::neverDeleteData);
  Gk_HF Gk(p.beta, p.kF, cd.kx, cd.epsx);   // Hartree-Fock non-interacting green's function
  
  py::buffer_info info_kxb = _kxb_.request();
  bl::Array<double,1> kxb((double*)info_kxb.ptr, info_kxb.shape[0], bl::neverDeleteData);
  py::array_t<double> _C_Pln_( {p.Nthbin,kxb.extent(0)-1} );
  py::buffer_info info_P = _C_Pln_.request();
  bl::Array<double,2> C_Pln((double*)info_P.ptr, bl::shape(info_P.shape[0],info_P.shape[1]), bl::neverDeleteData);
  
  BaymKadanoff_Q0W0_Data BKdata(C_Pln, kxb);

  sample_static_fastD_combined(BKdata, Pbin, log, lmbda, lmbda_spct, Gk, p, cd.diagsG, cd.diagSign, Loop_index, Loop_type, cd.Vtype, mpi);
  
  if (mpi.rank==mpi.master) 
    return py::make_tuple(_C_Pln_, _Pbin_);
  else{
    py::array_t<double> _C_Pln_( {1,1,1,1} );// fake empty output in order not to copy huge data on all cores.
    return py::make_tuple(_C_Pln_, _Pbin_);
  }
}

py::tuple sample_static_Q0W0_HFD(std::ostream& log, double lmbda, const std::vector<double>& lmbda_spct, const params& p, py::array_t<double>& kx, py::array_t<double>& epsx, 
				  py::array_t<int>& diagsG, py::array_t<double>& diagSign,
				  const std::vector<std::vector<std::vector<int> > >& Loop_index, const std::vector<std::vector<std::vector<int> > >& Loop_type,
				  py::array_t<int>& Vtype, py::array_t<int>& indx, my_mpi& mpi)
{
  ConvertDiags_ cd(diagsG,diagSign,kx,epsx,Vtype,indx);
  
  py::array_t<double> _Pbin_( p.Nt );
  py::buffer_info info_N = _Pbin_.request();
  bl::Array<double,2> Pbin((double*)info_N.ptr, bl::shape(p.Nt,1), bl::neverDeleteData);
  Gk_HF Gk(p.beta, p.kF, cd.kx, cd.epsx);   // Hartree-Fock non-interacting green's function
  
  py::array_t<double> _C_Pln_( {1,1} );
  py::buffer_info info_P = _C_Pln_.request();
  bl::Array<double,2> C_Pln((double*)info_P.ptr, bl::shape(info_P.shape[0],info_P.shape[1]), bl::neverDeleteData);
  Standard_Q0W0_Data BKdata(C_Pln);

  sample_static_fastD_combined(BKdata, Pbin, log, lmbda, lmbda_spct, Gk, p, cd.diagsG, cd.diagSign, Loop_index, Loop_type, cd.Vtype, mpi);
  
  return py::make_tuple(_C_Pln_, _Pbin_);
}

py::tuple sample_Discrete_Q0W0_VHFC(std::ostream& log, double Q_external,
				    double lmbda, const std::vector<double>& lmbda_spct, const params& p, py::array_t<double>& kx, py::array_t<double>& epsx, 
				    py::array_t<int>& diagsG, 
				    const std::vector<std::vector<int> >& diagSign,
				    const std::vector<std::vector<std::vector<int> > >& Loop_index, const std::vector<std::vector<std::vector<int> > >& Loop_type,
				    py::array_t<int>& Vtype, py::array_t<int>& indx, py::array_t<double>& _kxb_, my_mpi& mpi)
{
  ConvertDiags_ cd(diagsG,kx,epsx,Vtype,indx);
  
  py::array_t<double> _Pbin_( p.Nt );
  py::buffer_info info_N = _Pbin_.request();
  bl::Array<double,2> Pbin((double*)info_N.ptr, bl::shape(p.Nt,1), bl::neverDeleteData);
  Gk_HF Gk(p.beta, p.kF, cd.kx, cd.epsx);   // Hartree-Fock non-interacting green's function
  
  py::buffer_info info_kxb = _kxb_.request();
  bl::Array<double,1> kxb((double*)info_kxb.ptr, info_kxb.shape[0], bl::neverDeleteData);
  py::array_t<double> _C_Pln_( {p.Nthbin,kxb.extent(0)-1,1,1} );
  py::buffer_info info_P = _C_Pln_.request();
  bl::Array<double,4> C_Pln((double*)info_P.ptr, bl::shape(info_P.shape[0],info_P.shape[1],1,1), bl::neverDeleteData);
  BaymKadanoff_Discrete   BKdata(C_Pln, kxb);
  
  bl::Array<double,1> qx(1); qx(0)=Q_external;
  
  sample_static_DiscreteC_combined(BKdata, Pbin, log, qx, lmbda, lmbda_spct, Gk, p, cd.diagsG, diagSign, Loop_index, Loop_type, cd.Vtype, mpi);
  
  if (mpi.rank==mpi.master) 
    return py::make_tuple(_C_Pln_, _Pbin_);
  else{
    py::array_t<double> _C_Pln_( {1,1,1,1} );// fake empty output in order not to copy huge data on all cores.
    return py::make_tuple(_C_Pln_, _Pbin_);
  }
}
py::tuple sample_Discrete_Q0W0_HFC(std::ostream& log, double Q_external,
				   double lmbda, const std::vector<double>& lmbda_spct, const params& p, py::array_t<double>& kx, py::array_t<double>& epsx, 
				   py::array_t<int>& diagsG,
				   const std::vector<std::vector<int> >& diagSign,
				   const std::vector<std::vector<std::vector<int> > >& Loop_index, const std::vector<std::vector<std::vector<int> > >& Loop_type,
				   py::array_t<int>& Vtype, py::array_t<int>& indx, my_mpi& mpi)
{
  ConvertDiags_ cd(diagsG,kx,epsx,Vtype,indx);
  
  py::array_t<double> _Pbin_( p.Nt );
  py::buffer_info info_N = _Pbin_.request();
  bl::Array<double,2> Pbin((double*)info_N.ptr, bl::shape(p.Nt,1), bl::neverDeleteData);
  Gk_HF Gk(p.beta, p.kF, cd.kx, cd.epsx);   // Hartree-Fock non-interacting green's function
  
  py::array_t<double> _C_Pln_( {1,1} );
  py::buffer_info info_P = _C_Pln_.request();
  bl::Array<double,2> C_Pln((double*)info_P.ptr, bl::shape(info_P.shape[0],info_P.shape[1]), bl::neverDeleteData);
  Standard_Discrete BKdata(C_Pln );
  
  bl::Array<double,1> qx(1); qx(0)=Q_external;

  sample_static_DiscreteC_combined(BKdata, Pbin, log, qx, lmbda, lmbda_spct, Gk, p, cd.diagsG, diagSign, Loop_index, Loop_type, cd.Vtype, mpi);
  
  return py::make_tuple(_C_Pln_, _Pbin_);
}
py::tuple sample_Discrete_Q0W0_VSHFC(std::ostream& log, double Q_external,
				     double lmbda, const std::vector<double>& lmbda_spct, const params& p, py::array_t<double>& kx, py::array_t<double>& epsx, 
				     py::array_t<int>& diagsG,
				     const std::vector<std::vector<int> >& diagSign,
				     const std::vector<std::vector<std::vector<int> > >& Loop_index, const std::vector<std::vector<std::vector<int> > >& Loop_type,
				     py::array_t<int>& Vtype, py::array_t<int>& indx, py::array_t<double>& _kxb_, my_mpi& mpi)
{
  ConvertDiags_ cd(diagsG,kx,epsx,Vtype,indx);
  
  py::array_t<double> _Pbin_( p.Nt );
  py::buffer_info info_N = _Pbin_.request();
  bl::Array<double,2> Pbin((double*)info_N.ptr, bl::shape(p.Nt,1), bl::neverDeleteData);
  Gk_HF Gk(p.beta, p.kF, cd.kx, cd.epsx);   // Hartree-Fock non-interacting green's function
  
  py::buffer_info info_kxb = _kxb_.request();
  bl::Array<double,1> kxb((double*)info_kxb.ptr, info_kxb.shape[0], bl::neverDeleteData);
  py::array_t<double> _C_Pln_( {p.Nthbin,kxb.extent(0)-1,p.Nthbin,kxb.extent(0)-1,1,1} );
  py::buffer_info info_P = _C_Pln_.request();
  bl::Array<double,6> C_Pln((double*)info_P.ptr, bl::shape(info_P.shape[0],info_P.shape[1],info_P.shape[2],info_P.shape[3],1,1), bl::neverDeleteData);

  BaymKadanoff_Symmetric_Discrete BKdata(C_Pln, kxb);
  bl::Array<double,1> qx(1); qx(0)=Q_external;

  sample_static_DiscreteC_combined(BKdata, Pbin, log, qx, lmbda, lmbda_spct, Gk, p, cd.diagsG, diagSign, Loop_index, Loop_type, cd.Vtype, mpi);
  
  if (mpi.rank==mpi.master) 
    return py::make_tuple(_C_Pln_, _Pbin_);
  else{
    py::array_t<double> _C_Pln_( {1,1,1,1} );// fake empty output in order not to copy huge data on all cores.
    return py::make_tuple(_C_Pln_, _Pbin_);
  }
}



py::tuple sample_static_Q0W0_VHFC(double Q_external, std::ostream& log, double lmbda, const std::vector<double>& lmbda_spct, const params& p, py::array_t<double>& kx, py::array_t<double>& epsx, 
				  py::array_t<int>& diagsG, //py::array_t<double>& diagSign,
				  const std::vector<std::vector<int> >& diagSign,
				  const std::vector<std::vector<std::vector<int> > >& Loop_index, const std::vector<std::vector<std::vector<int> > >& Loop_type,
				  py::array_t<int>& Vtype, py::array_t<int>& indx, py::array_t<double>& _kxb_, my_mpi& mpi)
{
  ConvertDiags_ cd(diagsG,kx,epsx,Vtype,indx);
  
  py::array_t<double> _Pbin_( p.Nt );
  py::buffer_info info_N = _Pbin_.request();
  bl::Array<double,2> Pbin((double*)info_N.ptr, bl::shape(p.Nt,1), bl::neverDeleteData);
  Gk_HF Gk(p.beta, p.kF, cd.kx, cd.epsx);   // Hartree-Fock non-interacting green's function
  
  py::buffer_info info_kxb = _kxb_.request();
  bl::Array<double,1> kxb((double*)info_kxb.ptr, info_kxb.shape[0], bl::neverDeleteData);
  py::array_t<double> _C_Pln_( {p.Nthbin,kxb.extent(0)-1} );
  py::buffer_info info_P = _C_Pln_.request();
  bl::Array<double,2> C_Pln((double*)info_P.ptr, bl::shape(info_P.shape[0],info_P.shape[1]), bl::neverDeleteData);
  
  BaymKadanoff_Q0W0_Data BKdata(C_Pln, kxb, Q_external);

  sample_static_fastC_combined(BKdata, Pbin, log, lmbda, lmbda_spct, Gk, p, cd.diagsG, diagSign, Loop_index, Loop_type, cd.Vtype, mpi);
  
  if (mpi.rank==mpi.master) 
    return py::make_tuple(_C_Pln_, _Pbin_);
  else{
    py::array_t<double> _C_Pln_( {1,1,1,1} );// fake empty output in order not to copy huge data on all cores.
    return py::make_tuple(_C_Pln_, _Pbin_);
  }
}
py::tuple sample_static_Q0W0_VSHFC(double Q_external, std::ostream& log, double lmbda, const std::vector<double>& lmbda_spct, const params& p, py::array_t<double>& kx, py::array_t<double>& epsx, 
				  py::array_t<int>& diagsG, //py::array_t<double>& diagSign,
				  const std::vector<std::vector<int> >& diagSign,
				  const std::vector<std::vector<std::vector<int> > >& Loop_index, const std::vector<std::vector<std::vector<int> > >& Loop_type,
				  py::array_t<int>& Vtype, py::array_t<int>& indx, py::array_t<double>& _kxb_, my_mpi& mpi)
{
  ConvertDiags_ cd(diagsG,kx,epsx,Vtype,indx);
  
  py::array_t<double> _Pbin_( p.Nt );
  py::buffer_info info_N = _Pbin_.request();
  bl::Array<double,2> Pbin((double*)info_N.ptr, bl::shape(p.Nt,1), bl::neverDeleteData);
  Gk_HF Gk(p.beta, p.kF, cd.kx, cd.epsx);   // Hartree-Fock non-interacting green's function
  
  py::buffer_info info_kxb = _kxb_.request();
  bl::Array<double,1> kxb((double*)info_kxb.ptr, info_kxb.shape[0], bl::neverDeleteData);
  py::array_t<double> _C_Pln_( {p.Nthbin,kxb.extent(0)-1,p.Nthbin,kxb.extent(0)-1,1,1} );
  py::buffer_info info_P = _C_Pln_.request();
  bl::Array<double,4> C_Pln((double*)info_P.ptr, bl::shape(info_P.shape[0],info_P.shape[1],info_P.shape[2],info_P.shape[3]), bl::neverDeleteData);

  BaymKadanoff_Symmetric_Q0W0_Data BKdata(C_Pln, kxb, Q_external);

  sample_static_fastC_combined(BKdata, Pbin, log, lmbda, lmbda_spct, Gk, p, cd.diagsG, diagSign, Loop_index, Loop_type, cd.Vtype, mpi);
  
  if (mpi.rank==mpi.master) 
    return py::make_tuple(_C_Pln_, _Pbin_);
  else{
    py::array_t<double> _C_Pln_( {1,1,1,1} );// fake empty output in order not to copy huge data on all cores.
    return py::make_tuple(_C_Pln_, _Pbin_);
  }
}

py::tuple sample_static_Q0W0_HFC(double Q_external, std::ostream& log, double lmbda, const std::vector<double>& lmbda_spct, const params& p, py::array_t<double>& kx, py::array_t<double>& epsx, 
				 py::array_t<int>& diagsG, //py::array_t<double>& diagSign,
				 const std::vector<std::vector<int> >& diagSign,
				 const std::vector<std::vector<std::vector<int> > >& Loop_index, const std::vector<std::vector<std::vector<int> > >& Loop_type,
				 py::array_t<int>& Vtype, py::array_t<int>& indx, my_mpi& mpi)
{
  ConvertDiags_ cd(diagsG,kx,epsx,Vtype,indx);
  
  py::array_t<double> _Pbin_( p.Nt );
  py::buffer_info info_N = _Pbin_.request();
  bl::Array<double,2> Pbin((double*)info_N.ptr, bl::shape(p.Nt,1), bl::neverDeleteData);
  Gk_HF Gk(p.beta, p.kF, cd.kx, cd.epsx);   // Hartree-Fock non-interacting green's function
  
  py::array_t<double> _C_Pln_( {1,1} );
  py::buffer_info info_P = _C_Pln_.request();
  bl::Array<double,2> C_Pln((double*)info_P.ptr, bl::shape(info_P.shape[0],info_P.shape[1]), bl::neverDeleteData);
  Standard_Q0W0_Data BKdata(C_Pln, Q_external);

  sample_static_fastC_combined(BKdata, Pbin, log, lmbda, lmbda_spct, Gk, p, cd.diagsG, diagSign, Loop_index, Loop_type, cd.Vtype, mpi);
  
  return py::make_tuple(_C_Pln_, _Pbin_);
}

/*
py::tuple sample_static_fast_HF_old(std::ostream& log, double lmbda, const std::vector<double>& lmbda_spct, const params& p, py::array_t<double>& kx, py::array_t<double>& epsx, 
				py::array_t<int>& diagsG, py::array_t<double>& diagSign,
				const std::vector<std::vector<std::vector<int> > >& Loop_index, const std::vector<std::vector<std::vector<int> > >& Loop_type,
				py::array_t<int>& Vtype, py::array_t<int>& indx, my_mpi& mpi)
{
  const int Nlq=18;  // 10 legendre polynomials for q-dependence
  const int Nlt=24;  // 16 legendre polynomials for t-dependence

  ConvertDiags_ cd(diagsG,diagSign,kx,epsx,Vtype,indx);

  py::array_t<double> _Pbin_( {p.Nt,p.Nq} );
  py::buffer_info info_N = _Pbin_.request();
  bl::Array<double,2> Pbin((double*)info_N.ptr, bl::shape(info_N.shape[0],info_N.shape[1]), bl::neverDeleteData);
  Gk_HF Gk(p.beta, p.kF, cd.kx, cd.epsx);   // Hartree-Fock non-interacting green's function
  
  py::array_t<double> _C_Pln_( {Nlt+1,Nlq+1} );
  py::buffer_info info_P = _C_Pln_.request();
  bl::Array<double,2> C_Pln((double*)info_P.ptr, bl::shape(info_P.shape[0],info_P.shape[1]), bl::neverDeleteData);
  StandardData BKdata(C_Pln);

  //sample_static_fastC(BKdata, Pbin, log, lmbda, lmbda_spct, Gk, p, cd.diagsG, cd.diagSign, Loop_index, Loop_type, cd.Vtype, mpi);

  //bl::Array<double,2> C_Pln(Nlt+1,Nlq+1), Pbin(p.Nt,p.Nq);
  sample_static_fast<Nlt,Nlq>(C_Pln, Pbin, log, lmbda, Gk, p, cd.diagsG, cd.diagSign, Loop_index, Loop_type, cd.Vtype, mpi);
  //return ConvertOutput(C_Pln, Pbin)();
  
  if (mpi.rank==mpi.master) 
    return py::make_tuple(_C_Pln_, _Pbin_);
  else{
    py::array_t<double> _C_Pln_( {1,1} );// fake empty output in order not to copy huge data on all cores.
    return py::make_tuple(_C_Pln_, _Pbin_);
  }
}
*/
 /*
py::array_t<double> sample_static_Q0_HF(double Q_external, double lmbda, const params& p, py::array_t<double>& kx, py::array_t<double>& epsx, 
					py::array_t<int>& diagsG, py::array_t<double>& diagSign,
					const std::vector<std::vector<std::vector<int> > >& Loop_index, const std::vector<std::vector<std::vector<int> > >& Loop_type,
					py::array_t<int>& Vtype, py::array_t<int>& indx, my_mpi& mpi)
{
  ConvertDiags cd(diagsG,diagSign,kx,epsx,Vtype,indx);

  bl::Array<double,1> Pbin(p.Nt);
  
  Gk_HF Gk(p.beta, p.kF, cd.kx, cd.epsx);   // Hartree-Fock non-interacting green's function
  sample_static_Q0(Q_external, lmbda, Pbin, Gk, p, cd.diagsG, cd.diagSign, Loop_index, Loop_type, cd.Vtype, cd.indx, mpi);
  return ConvertArray(Pbin)();
}
*/
 
py::array_t<double> sample_static_Q0_fast_HF(std::ostream& log, double Q_external, double lmbda, const params& p, py::array_t<double>& kx, py::array_t<double>& epsx, 
					     py::array_t<int>& diagsG, py::array_t<double>& diagSign,
					     const std::vector<std::vector<std::vector<int> > >& Loop_index, const std::vector<std::vector<std::vector<int> > >& Loop_type,
					     py::array_t<int>& Vtype, py::array_t<int>& indx, my_mpi& mpi)
{
  ConvertDiags_ cd(diagsG,diagSign,kx,epsx,Vtype,indx);
  double lmbda_spct=0.0;
  bl::Array<double,1> Pbin(p.Nt);
  Gk_HF Gk(p.beta, p.kF, cd.kx, cd.epsx);   // Hartree-Fock non-interacting green's function
  //sample_static_Q0_fast(Q_external, lmbda, Pbin, Gk, p, cd.diagsG, cd.diagSign, ILoop_index, ILoop_type, cd.Vtype, mpi);
  sample_static_Q0_fast(log, Q_external, lmbda, lmbda_spct, Pbin, Gk, p, cd.diagsG, cd.diagSign, Loop_index, Loop_type, cd.Vtype, mpi);
  return ConvertArray(Pbin)();
}

/*
double sample_static_Q0_t0_fast_HF(double Q_external, double t_external, double lmbda, const params& p, py::array_t<double>& kx, py::array_t<double>& epsx, 
				   py::array_t<int>& diagsG, py::array_t<double>& diagSign,
				   const std::vector<std::vector<std::vector<int> > >& Loop_index, const std::vector<std::vector<std::vector<int> > >& Loop_type,
				   py::array_t<int>& Vtype, py::array_t<int>& indx, my_mpi& mpi)
{
  ConvertDiags cd(diagsG,diagSign,kx,epsx,Vtype,indx);
  Gk_HF Gk(p.beta, p.kF, cd.kx, cd.epsx);   // Hartree-Fock non-interacting green's function
  return sample_static_Q0_t0_fast(Q_external, t_external, lmbda, Gk, p, cd.diagsG, cd.diagSign, Loop_index, Loop_type, cd.Vtype, cd.indx, mpi);
}
*/
 /*
py::tuple sample_Density_HFC(std::ostream& log, double lmbda, const std::vector<double>& lmbda_spct, const params& p, py::array_t<double>& kx, py::array_t<double>& epsx, 
			    py::array_t<int>& diagsG, py::array_t<double>& diagSign,
			    const std::vector<std::vector<std::vector<int> > >& Loop_index, const std::vector<std::vector<std::vector<int> > >& Loop_type,
			    py::array_t<int>& Vtype, py::array_t<int>& indx, my_mpi& mpi)
{
  ConvertDiags_ cd(diagsG,diagSign,kx,epsx,Vtype,indx);
  
  Gk_HF Gk(p.beta, p.kF, cd.kx, cd.epsx);   // Hartree-Fock non-interacting green's function
  auto pw = sample_Density_C(log, lmbda, lmbda_spct, Gk, p, cd.diagsG, cd.diagSign, Loop_index, Loop_type, cd.Vtype, mpi);
  return py::make_tuple(std::get<0>(pw),std::get<1>(pw),std::get<2>(pw),std::get<3>(pw));
}
 */
py::tuple sample_Density_HFC(std::ostream& log, double lmbda, const std::vector<double>& lmbda_spct, const params& p, py::array_t<double>& kx, py::array_t<double>& epsx, 
			    py::array_t<int>& diagsG, const std::vector<std::vector<int>>& diagSign,
			    const std::vector<std::vector<std::vector<int> > >& Loop_index, const std::vector<std::vector<std::vector<int> > >& Loop_type,
			    py::array_t<int>& Vtype, py::array_t<int>& indx, my_mpi& mpi)
{
  ConvertDiags_ cd(diagsG,kx,epsx,Vtype,indx);
  
  Gk_HF Gk(p.beta, p.kF, cd.kx, cd.epsx);   // Hartree-Fock non-interacting green's function
  auto pw = sample_Density_C(log, lmbda, lmbda_spct, Gk, p, cd.diagsG, diagSign, Loop_index, Loop_type, cd.Vtype, mpi);
  return py::make_tuple(std::get<0>(pw),std::get<1>(pw),std::get<2>(pw),std::get<3>(pw));
}
py::tuple sample_Density_HFD(std::ostream& log, double lmbda, const std::vector<double>& lmbda_spct, const params& p, py::array_t<double>& kx, py::array_t<double>& epsx, 
			    py::array_t<int>& diagsG, py::array_t<double>& diagSign,
			    const std::vector<std::vector<std::vector<int> > >& Loop_index, const std::vector<std::vector<std::vector<int> > >& Loop_type,
			    py::array_t<int>& Vtype, py::array_t<int>& indx, my_mpi& mpi)
{
  ConvertDiags_ cd(diagsG,diagSign,kx,epsx,Vtype,indx);
  
  Gk_HF Gk(p.beta, p.kF, cd.kx, cd.epsx);   // Hartree-Fock non-interacting green's function

  auto pw = sample_Density_D(log, lmbda, lmbda_spct, Gk, p, cd.diagsG, cd.diagSign, Loop_index, Loop_type, cd.Vtype, mpi);
  return py::make_tuple(std::get<0>(pw),std::get<1>(pw),std::get<2>(pw),std::get<3>(pw));
}
/*
py::array_t<double> sample_static_Q0_NI(double Q_external, double lmbda, const params& p,
					py::array_t<int>& diagsG, py::array_t<double>& diagSign,
					const std::vector<std::vector<std::vector<int> > >& Loop_index, const std::vector<std::vector<std::vector<int> > >& Loop_type,
					py::array_t<int>& Vtype, py::array_t<int>& indx, my_mpi& mpi)
{
  ConvertDiags cd(diagsG,diagSign,Vtype,indx);
  
  bl::Array<double,1> Pbin(p.Nt);

  egass_Gk Gk(p.beta, p.kF);        // non-interacting green's function
  sample_static_Q0(Q_external, lmbda, Pbin, Gk, p, cd.diagsG, cd.diagSign, Loop_index, Loop_type, cd.Vtype, cd.indx, mpi);
  return ConvertArray(Pbin)();
}
*/
/*
py::tuple _sample_dynamic_(const params& p, py::array_t<int>& diagsG, py::array_t<double>& diagSign, py::array_t<int>& diagVertex,
			   const std::vector<std::vector<std::vector<int> > >& Loop_index, const std::vector<std::vector<std::vector<int> > >& Loop_type,
			   py::array_t<double>& _qx_, py::array_t<double>& _taux_, py::array_t<double>& _Wq0_, py::array_t<double>& _Wom0_, 
			   py::array_t<double>& _kx_, py::array_t<double>& _epsx_)
{
  const int Nlq=18;  // 10 legendre polynomials for q-dependence
  const int Nlt=24;  // 16 legendre polynomials for t-dependence

  ConvertDiags cd(diagsG,diagSign,_kx_,_epsx_);
  
  py::buffer_info info_v = diagVertex.request();
  int size_v1=1, size_v2=1, size_v3=1;
  if (info_v.ndim==2 && info_v.shape[1]==0){
    // empty vertex, still OK.
    size_v3=0;
  }else{
    if (info_v.ndim != 3) throw std::runtime_error("Number of dimensions for diagVertex must be 3");
    size_v1 = info_v.shape[0];
    size_v2 = info_v.shape[1];
    size_v3 = info_v.shape[2];
  }
  //cout<<"vertex dimensions is "<<info_v.ndim <<" "<<info_v.shape[0]<<" "<<info_v.shape[1]<<" but "<<size_v1<<" "<<size_v2<<" "<<size_v3<<endl;
  bl::Array<int,3> _diagVertex_( (int*)info_v.ptr, bl::shape(size_v1,size_v2,size_v3), bl::neverDeleteData);
  
  py::buffer_info info_qx = _qx_.request();
  if (info_qx.ndim != 1) throw std::runtime_error("Number of dimensions for qx should be 1.");
  bl::Array<double,1> qx( (double*)info_qx.ptr, info_qx.shape[0], bl::neverDeleteData);

  py::buffer_info info_Wom0 = _Wom0_.request();
  if (info_Wom0.ndim != 1) throw std::runtime_error("Number of dimensions for Wom0 should be 1.");
  bl::Array<double,1> Wom0( (double*)info_Wom0.ptr, info_Wom0.shape[0], bl::neverDeleteData);

  py::buffer_info info_tau = _taux_.request();
  if (info_tau.ndim != 1) throw std::runtime_error("Number of dimensions for tau should be 1.");
  bl::Array<double,1> taux( (double*)info_tau.ptr, info_tau.shape[0], bl::neverDeleteData);

  py::buffer_info info_Wq = _Wq0_.request();
  if (info_Wq.ndim != 2) throw std::runtime_error("Number of dimensions for Wq should be 2.");
  int Nq = info_Wq.shape[0];
  int Nt = info_Wq.shape[1];
  if (Nt != taux.extent(0) ) throw std::runtime_error("Number of time points in Wq and tau do not agree.");
  if (Nq != qx.extent(0)  ) throw std::runtime_error("Number of q points in Wq and qx do not agree.");
  bl::Array<double,2> Wq0((double*)info_Wq.ptr, bl::shape(info_Wq.shape[0],info_Wq.shape[1]), bl::neverDeleteData);
  
  bl::Array<double,2> C_Pln(Nlt+1,Nlq+1);
  bl::Array<double,2> Pbin(p.Nt,p.Nq);
  C_Pln=0.0; Pbin=0.0;
  //sample_dynamic<Nlt,Nlq>(C_Pln, Pbin, p, _diagsG_, _diagSign_, _diagVertex_, Loop_index, Loop_type, qx, taux, Wq0, Wom0, kx, epsx);
  sample_dynamic<Nlt,Nlq>(C_Pln, Pbin, p, cd.diagsG, cd.diagSign, _diagVertex_, Loop_index, Loop_type, qx, taux, Wq0, Wom0, cd.kx, cd.epsx);

  return ConvertOutput(C_Pln, Pbin)();
}
*/

py::array_t<std::complex<double>> FourierBoson(double beta, py::array_t<int>& _iom_, py::array_t<double>& _tau_, py::array_t<double>& Ptau)
{
  py::buffer_info info_iom = _iom_.request();
  if (info_iom.ndim != 1) throw std::runtime_error("Number of dimensions for iom should be 1.");
  bl::Array<int,1> iom( (int*)info_iom.ptr, bl::shape(info_iom.shape[0]), bl::neverDeleteData);
  py::buffer_info info_tau = _tau_.request();
  if (info_tau.ndim != 1) throw std::runtime_error("Number of dimensions for tau should be 1.");
  bl::Array<double,1> tau( (double*)info_tau.ptr, bl::shape(info_tau.shape[0]), bl::neverDeleteData);
  py::buffer_info info_Pt = Ptau.request();
  if (info_Pt.ndim != 2) throw std::runtime_error("Number of dimensions for Pt should be 2.");
  bl::Array<double,2> Pt( (double*)info_Pt.ptr, bl::shape(info_Pt.shape[0],info_Pt.shape[1]), bl::neverDeleteData);

  py::array_t<std::complex<double>, py::array::c_style> _Pqw_({ Pt.extent(0), iom.extent(0) });
  auto Pqw = _Pqw_.mutable_unchecked<2>();
 
  int Nt = tau.extent(0);
  for (int iq=0; iq<Pt.extent(0); iq++){
    Spline1D<double> wt(Nt);
    for (int it=0; it<Nt; it++) wt[it] = Pt(iq,it);
    wt.splineIt(tau);
    for (int in=0; in<iom.extent(0); in++){
      double om = 2*iom(in)*M_PI/beta;
      Pqw(iq,in) = wt.Fourier(om, tau);
    }
    
  }
  return _Pqw_;
}

py::array_t<double> InverseFourierBoson_new(double beta, py::array_t<double>& _tau_, int nom, py::array_t<int>& _iom_, py::array_t<double>& _Wq_, py::array_t<double>& _bi_, int order=1)
{ // nom : number of points before the tail.
  // if bi ne 0, we subtract c/(2*bi) * ( 1/(iw+bi) - 1/(iw-bi) ) == c/(w**2+bi**2) and treat it analytically
  py::buffer_info info_iom = _iom_.request();
  if (info_iom.ndim != 1) throw std::runtime_error("Number of dimensions for iom should be 1.");
  bl::Array<int,1> iom( (int*)info_iom.ptr, bl::shape(info_iom.shape[0]), bl::neverDeleteData);
  py::buffer_info info_tau = _tau_.request();
  if (info_tau.ndim != 1) throw std::runtime_error("Number of dimensions for tau should be 1.");
  bl::Array<double,1> tau( (double*)info_tau.ptr, bl::shape(info_tau.shape[0]), bl::neverDeleteData);
  py::buffer_info info_Wq = _Wq_.request();
  if (info_Wq.ndim != 2) throw std::runtime_error("Number of dimensions for Wq should be 2.");

  py::buffer_info info_b = _bi_.request();
  if (info_b.ndim != 1) throw std::runtime_error("Number of dimensions for bi should be 1.");
  bl::Array<double,1> bi( (double*)info_b.ptr, bl::shape(info_b.shape[0]), bl::neverDeleteData);
  
  int Nq = info_Wq.shape[0];
  int Nw = info_Wq.shape[1];
  int Nt = tau.extent(0);

  py::array_t<double,py::array::c_style> _Wqt_({ Nq, Nt });
  //auto Wqt = _Wqt_.mutable_unchecked<2>();
  py::buffer_info info_Wt = _Wqt_.request();
  for (int iq=0; iq<Nq; iq++){ // over all q-points
    bl::Array<double,1> W_iom( (double*)info_Wq.ptr + iq*Nw, Nw, bl::neverDeleteData);
    bl::Array<double,1> W_tau( (double*)info_Wt.ptr + iq*Nt, Nt, bl::neverDeleteData);
    double b = iq < bi.extent(0) ? bi(iq) : bi(0);
    //cout<<" b["<<iq<<"]= " << b << endl;
    inverse_fourier_boson(W_tau, beta, tau, nom, iom, W_iom, b, order);
  }
  return _Wqt_;
}
py::array_t<double> InverseFourierBoson_cmplx(double beta, py::array_t<double>& _tau_, int nom, py::array_t<int>& _iom_, py::array_t<std::complex<double>>& _Wq_, double b=0.0)
{ // nom : number of points before the tail.
  // if b ne 0, we subtract c/(2*b) * ( 1/(iw+b) - 1/(iw-b) ) == c/(w**2+b**2) and treat it analytically
  py::buffer_info info_iom = _iom_.request();
  if (info_iom.ndim != 1) throw std::runtime_error("Number of dimensions for iom should be 1.");
  bl::Array<int,1> iom( (int*)info_iom.ptr, bl::shape(info_iom.shape[0]), bl::neverDeleteData);
  py::buffer_info info_tau = _tau_.request();
  if (info_tau.ndim != 1) throw std::runtime_error("Number of dimensions for tau should be 1.");
  bl::Array<double,1> tau( (double*)info_tau.ptr, bl::shape(info_tau.shape[0]), bl::neverDeleteData);
  py::buffer_info info_Wq = _Wq_.request();
  if (info_Wq.ndim != 2) throw std::runtime_error("Number of dimensions for Wq should be 2.");
  int Nq = info_Wq.shape[0];
  int Nw = info_Wq.shape[1];
  int Nt = tau.extent(0);

  py::array_t<double,py::array::c_style> _Wqt_({ Nq, Nt });
  //auto Wqt = _Wqt_.mutable_unchecked<2>();
  py::buffer_info info_Wt = _Wqt_.request();
  
  for (int iq=0; iq<Nq; iq++){ // over all q-points
    bl::Array<std::complex<double>,1> W_iom( (std::complex<double>*)info_Wq.ptr + iq*Nw, Nw, bl::neverDeleteData);
    bl::Array<double,1> W_tau( (double*)info_Wt.ptr + iq*Nt, Nt, bl::neverDeleteData);
    inverse_fourier_boson(W_tau, beta, tau, nom, iom, W_iom, b);
  }
  return _Wqt_;
}
/*
py::array_t<double> InverseFourierBoson_cmplx_debug(double beta, py::array_t<double>& _tau_, int nom, py::array_t<int>& _iom_, py::array_t<std::complex<double>>& _Wq_, double b=0.0)
{ // nom : number of points before the tail.
  // if b ne 0, we subtract c/(2*b) * ( 1/(iw+b) - 1/(iw-b) ) == c/(w**2+b**2) and treat it analytically
  py::buffer_info info_iom = _iom_.request();
  if (info_iom.ndim != 1) throw std::runtime_error("Number of dimensions for iom should be 1.");
  bl::Array<int,1> iom( (int*)info_iom.ptr, bl::shape(info_iom.shape[0]), bl::neverDeleteData);
  py::buffer_info info_tau = _tau_.request();
  if (info_tau.ndim != 1) throw std::runtime_error("Number of dimensions for tau should be 1.");
  bl::Array<double,1> tau( (double*)info_tau.ptr, bl::shape(info_tau.shape[0]), bl::neverDeleteData);
  py::buffer_info info_Wq = _Wq_.request();
  if (info_Wq.ndim != 2) throw std::runtime_error("Number of dimensions for Wq should be 2.");
  int Nq = info_Wq.shape[0];
  int Nw = info_Wq.shape[1];
  int Nt = tau.extent(0);

  py::array_t<double,py::array::c_style> _Wqt_({ Nq, Nt });
  //auto Wqt = _Wqt_.mutable_unchecked<2>();
  py::buffer_info info_Wt = _Wqt_.request();
  
  for (int iq=0; iq<Nq; iq++){ // over all q-points
    bl::Array<std::complex<double>,1> W_iom( (std::complex<double>*)info_Wq.ptr + iq*Nw, Nw, bl::neverDeleteData);
    bl::Array<double,1> W_tau( (double*)info_Wt.ptr + iq*Nt, Nt, bl::neverDeleteData);
    inverse_fourier_boson_debug(W_tau, beta, tau, nom, iom, W_iom, b);
  }
  return _Wqt_;
}
*/

void MakeConcave(py::array_t<double>& Delta, double minDelta){
    // Checks for functions to be causal! Hence, if the functioin becomes very small at two points, it
    // should remain zero in all points between the two points.
    py::buffer_info info = Delta.request();
    if (info.ndim != 1) throw std::runtime_error("Number of dimensions for tau should be 1.");
    int Nt = info.shape[0];
    auto Dlt = Delta.mutable_unchecked<1>();
    int first=Nt;
    for (int it=0; it<Nt; it++){
      if (Dlt[it]>-minDelta){
	first=it;
	break;
      }
    }
    int last=-1;
    for (int it=Nt-1; it>=0; it--){
      if (Dlt[it]>-minDelta){
	last=it;
	break;
      }
    }
    for (int it=first; it<=last; it++) Dlt[it] = -minDelta; // changed 18.10.2008
    bool FoundPositive = (first<=last);
    for (int it=0; it<Nt; it++){ // check ones more
      if (Dlt[it]>-minDelta){
	FoundPositive=true;
	Dlt[it] = -minDelta;
      }
    }
}

PYBIND11_MODULE(samplewj,m) {
  m.doc() = "pybind11 wrap for Variational Monte Carlo code";

  py::class_<egass_Gk>(m, "egass_Gk", "Class to give electron gass non-interacting Green's function. Constructor needs (beta,kF)")
    .def(py::init<double,double>())
    .def("eps",&egass_Gk::eps)
    .def_readwrite("beta", &egass_Gk::beta, "Inverse temperature")
    .def_readwrite("kF",   &egass_Gk::kF,   "Fermi wave vector")
    .def("__call__", &egass_Gk::operator(), "Gives the electron gass non-interacting Green's function", py::is_operator(), py::arg("k"), py::arg("t"))
    ;

  py::class_<Gk_HF>(m, "Gk_HF", "Class to give electron gass Hartree-Fock Green's function. Constructor needs (beta,kF).Need to call SetUp")
    .def(py::init<double,double>())
    .def("eps", &Gk_HF::eps)
    .def_readwrite("beta", &Gk_HF::beta, "Inverse temperature")
    .def("SetUp", [](class Gk_HF& Gk, py::array_t<double>& _kx_, py::array_t<double>& _epsx_){
	py::buffer_info info_kx = _kx_.request(), info_epsx = _epsx_.request();
	if (info_kx.ndim != 1) throw std::runtime_error("Number of dimensions for kx must be 1");
	if (info_epsx.ndim != 1) throw std::runtime_error("Number of dimensions for epsx must be 1");
	bl::Array<double,1> kx( (double*)info_kx.ptr, info_kx.shape[0], bl::neverDeleteData);
	bl::Array<double,1> epsx( (double*)info_epsx.ptr, info_epsx.shape[0], bl::neverDeleteData);
	Gk.SetUp(kx,epsx);
      })
    .def_readwrite("kF", &Gk_HF::kF,   "Fermi wave vector")
    .def("__call__", &Gk_HF::operator(), "Gives the electron gass Hartree-Fock Green's function", py::is_operator(), py::arg("k"), py::arg("t"))
    ;

  py::class_<params>(m, "params", py::dynamic_attr(), "Class to give parameters to sampling routines")
    .def(py::init<>())
    .def_readwrite("kF",      &params::kF,       "The fermi wave vector")
    .def_readwrite("beta",    &params::beta,     "The inverse temperature")
    .def_readwrite("cutoffk", &params::cutoffk,  "Cutoff for momentum k")
    .def_readwrite("cutoffq", &params::cutoffq,  "Cutoff for independent momentum Q")
    .def_readwrite("Nitt",    &params::Nitt,     "Total number of MC -steps")
    .def_readwrite("V0norm",  &params::V0norm,   "The value of measuring diagram")
    .def_readwrite("V0exp",   &params::V0exp,    "The value of the exponent for measuring diagram")
    .def_readwrite("Pr",      &params::Pr,       "List of probabilities to make each of the MC steps")
    .def_readwrite("Nq",      &params::Nq,       "Number of independent momentum Q-points")
    .def_readwrite("Nt",      &params::Nt,       "Number of time points")
    .def_readwrite("Qring",   &params::Qring,    "Should we chose momentum in a ring around origin")
    .def_readwrite("dRk",     &params::dRk,      "For the above ring method, how large should be step")
    .def_readwrite("dkF",     &params::dkF,      "If not ring method, how much should we change momentum dk")
    .def_readwrite("iseed",   &params::iseed,    "Random number generator seed")
    .def_readwrite("tmeassure",&params::tmeassure,"How often to take measurements")
    .def_readwrite("Ncout",   &params::Ncout,    "How often to print logging information")
    .def_readwrite("Nwarm",   &params::Nwarm,    "How many warm-up steps")
    .def_readwrite("lmbdat",  &params::lmbdat,    "Parameter for reweighting time.")
    .def_readwrite("Nthbin",  &params::Nthbin,    "Number of bins for cos(theta) in computing vertex function.")// 8
    .def_readwrite("Nkbin",   &params::Nkbin,     "Number of bins for internal momentum k in computing vertex function.")// 50
    .def_readwrite("Nlt",     &params::Nlt,       "Number of Legendre Polynomials for expansion of time.")      // 20
    .def_readwrite("Nlq",     &params::Nlq,       "Numbef of Legendre Polynomials for expansion of external momentum q")// 18
    ;
  
  m.def("sample_static_fast_HFC", &sample_static_fast_HFC);   // DC-not dynamic, not-BaymKadanoff
  m.def("sample_static_fast_VHFC", &sample_static_fast_VHFC); // DC-not dynamic,     BaymKadanoff
  m.def("sample_static_fast_HFC", &sample_static_fast_HFC);   // DC-not dynamic, not-BaymKadanoff
  m.def("sample_static_fast_VHFC", &sample_static_fast_VHFC); // DC-not dynamic,     BaymKadanoff
  
  m.def("sample_static_fast_HFD", &sample_static_fast_HFD);  // DC dynamic,     not-BaymKadanoff
  m.def("sample_static_fast_VHFD", &sample_static_fast_VHFD);// DC dynamic,         BaymKadanoff

  m.def("sample_static_Q0W0_VHFD", &sample_static_Q0W0_VHFD); // dynamic, Baym Kadanoff
  m.def("sample_static_Q0W0_HFD", &sample_static_Q0W0_HFD);   // dynamic, non-Baym Kadanoff
  m.def("sample_static_Q0W0_VHFC", &sample_static_Q0W0_VHFC);   // DC-not dynamic,     BaymKadanoff
  m.def("sample_static_Q0W0_VSHFC", &sample_static_Q0W0_VSHFC); // DC-not dynamic,     BaymKadanoff-Symmetric from both sides
  m.def("sample_static_Q0W0_HFC", &sample_static_Q0W0_HFC);     // DC-not dynamic, not-BaymKadanoff

  m.def("sample_static_Discrete_HFC", &sample_static_Discrete_HFC);   // discrete q, non-Baym Kadanoff
  m.def("sample_static_Discrete_VHFC", &sample_static_Discrete_VHFC); // discrete q, Baym-Kadanoff
  m.def("sample_Discrete_Q0W0_HFC", &sample_Discrete_Q0W0_HFC);       // using discrete algorithm, q=0,w=0, non-Baym-Kadanof
  m.def("sample_Discrete_Q0W0_VHFC", &sample_Discrete_Q0W0_VHFC);     // using discrete algorithm, q=0,w=0, Baym-Kadanof
  m.def("sample_Discrete_Q0W0_VSHFC", &sample_Discrete_Q0W0_VSHFC);   // using discrete algorithm, q=0,w=0, symmetric Baym-Kadanof
  
  //m.def("sample_static_NI", &sample_static_NI);
  //m.def("sample_static_HF", &sample_static_HF);

  m.def("sample_static_Q0_fast_HF", &sample_static_Q0_fast_HF);
  m.def("sample_Density_HFC", &sample_Density_HFC);
  m.def("sample_Density_HFD", &sample_Density_HFD);

  //m.def("sample_static_fast_HF_old", &sample_static_fast_HF_old);
  //m.def("sample_dynamic", &_sample_dynamic_);
  //m.def("sample_static_Q0_HF", &sample_static_Q0_HF);
  //m.def("sample_static_Q0_NI", &sample_static_Q0_NI);
  //m.def("sample_static_Q0_t0_fast_HF", &sample_static_Q0_t0_fast_HF);
  
  m.def("P0_fast", [](double Q, py::array_t<double>& _tau_, double beta, double kF, double cutoffk) -> py::array_t<double>{
      py::buffer_info info_t = _tau_.request();
      if (info_t.ndim != 1) throw std::runtime_error("Number of dimensions for tau must be 1");
      bl::Array<double,1> tau( (double*)info_t.ptr, info_t.shape[0], bl::neverDeleteData);
      py::array_t<double, py::array::c_style> _p0_( info_t.shape[0] );
      py::buffer_info info_p = _p0_.request();
      bl::Array<double,1> p0( (double*)info_p.ptr, info_p.shape[0], bl::neverDeleteData);
      p0 = P0_fast(Q, tau, beta, kF, cutoffk);
      return _p0_;
    })
    ;
  
  py::class_<PO2>(m, "PO2", "Class to compute low-order diagrams by the double-integral")
    .def(py::init<double,double,double, double,double>())
    .def("P1", [](class PO2& a, double Q, py::array_t<double>& _tau_) ->  py::array_t<double>{
	py::buffer_info info_t = _tau_.request();
	if (info_t.ndim != 1) throw std::runtime_error("Number of dimensions for tau must be 1");
	bl::Array<double,1> tau( (double*)info_t.ptr, info_t.shape[0], bl::neverDeleteData);
	py::array_t<double, py::array::c_style> _p0_( info_t.shape[0] );
	py::buffer_info info_p = _p0_.request();
	bl::Array<double,1> p0( (double*)info_p.ptr, info_p.shape[0], bl::neverDeleteData);
	p0 = a.P1(Q, tau);
	return _p0_;
      }, "the actual value of the integral")
    ;
  
  m.def("FourierBoson", &FourierBoson, "Fourier Transform for bosonic set of functions");
  m.def("InverseFourierBoson_new", &InverseFourierBoson_new, "Inverse Fourier Transform for bosonic set of functions");
  m.def("InverseFourierBoson_cmplx", &InverseFourierBoson_cmplx, "Inverse Fourier Transform for bosonic set of functions");
  //m.def("InverseFourierBoson_cmplx_debug", &InverseFourierBoson_cmplx_debug, "Inverse Fourier Transform for bosonic set of functions");
  m.def("MakeConcave", &MakeConcave, "Makes the diagonal component of the imaginary-time function causal");

  //py::class_<HartreeFock>(m, "HartreeFock", "Produces Hartree-Fock dispersion")
  py::class_<HartreeFock>(m, "HartreeFock", py::buffer_protocol(), "Produces Hartree-Fock dispersion")
    //HartreeFock(double _kF_, double _cutoff_, double _beta_, double _lmbda_=0, double _dmu_=0, int Nq=128) :
    .def(py::init<double,double,double,double,double>())
    .def("get", [](class HartreeFock& a)-> py::tuple{
	a.cmp();
	py::array_t<double, py::array::c_style> _km_(a.kx.extent(0));
	py::array_t<double, py::array::c_style> _eps_(a.epsx.size());

	cout << "kx.size="<< a.kx.extent(0) << " eps.size="<< a.epsx.size() << endl;
	
	auto kr = _km_.mutable_unchecked<1>();
	auto er = _eps_.mutable_unchecked<1>();
	for (size_t i = 0; i < kr.shape(0); i++){
	  kr(i) = a.kx(i)/a.kF;
	  er(i) = a.epsx[i]/(a.kF*a.kF);
	}
	return py::make_tuple(_km_, _eps_);
      }, "access k/kF and eps/kF**2, but be careful, withouth units")
    .def("P0", [](class HartreeFock& a, double Q, py::array_t<double>& _tau_) -> py::array_t<double,py::array::c_style>{
	py::buffer_info info_t = _tau_.request();
	if (info_t.ndim != 1) throw std::runtime_error("Number of dimensions for tau must be 1");
	bl::Array<double,1> tau( (double*)info_t.ptr, info_t.shape[0], bl::neverDeleteData);
	py::array_t<double, py::array::c_style> _p0_( info_t.shape[0] );
	py::buffer_info info_p = _p0_.request();
	bl::Array<double,1> p0( (double*)info_p.ptr, info_p.shape[0], bl::neverDeleteData);
	//cout << "p0.size = " << info_t.shape[0] << " "<< info_p.shape[0] << endl;
	p0 = a.P0(Q, tau);
	//cout << "values in c="<< p0 << endl;
	return _p0_;
      }, "Polarization from Hartree-Fock")
    .def("Exchange", [](class HartreeFock& a) -> double{
	return a.Exchange();
      }," Exchange energy")
    .def("Exchange2", [](class HartreeFock& a) -> double{
	return a.Exchange2();
      }," Exchange energy")
    ;

  py::class_<my_mpi>(m, "my_mpi", "Class to contain mpi info")
    .def(py::init<>())
    .def_readwrite("size",   &my_mpi::size, "number of processes")
    .def_readwrite("rank",   &my_mpi::rank, "rank of this process")
    .def_readwrite("master", &my_mpi::master,"master for all processes")
    ;
  
  py::class_<Compute_Y_P_Y_>(m, "Compute_Y_P_Y_", "Computes <Y_{l0}|P(omega,k,q)|Y_{l'0}> where P is bubble polarization.")
    .def(py::init<double,double,double,double,double,int>())
    .def("Run", [](class Compute_Y_P_Y_& y_p_y, double k, double q, py::array_t<int>& iOm)->py::array_t<complex<double>>{
	py::buffer_info info = iOm.request();
	bl::Array<int,1> _iOm_((int*)info.ptr, info.shape[0], bl::neverDeleteData);
	py::array_t<complex<double>,py::array::c_style> _PpP_({y_p_y.lmax+1,y_p_y.lmax+1,_iOm_.extent(0)});
	py::buffer_info info2 = _PpP_.request();
	bl::Array<complex<double>,3> PpP((complex<double>*)info2.ptr, bl::shape(y_p_y.lmax+1,y_p_y.lmax+1,_iOm_.extent(0)), bl::neverDeleteData);
	y_p_y.Run(PpP, k, q, _iOm_);
	return _PpP_;
      }
      )
    .def("Pw0", [](class Compute_Y_P_Y_& y_p_y, double q, py::array_t<int>& iOm)-> py::array_t<double>{
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

}
