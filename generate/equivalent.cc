// @Copyright 2018 Kristjan Haule 
#include <cstdint>
#include <ostream>
#include <deque>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

std::ostream& operator<< (std::ostream& stream, const std::vector<int>& A)
{
  for (int i=0; i<A.size(); i++) stream<<A[i]<<",";
  stream<<"  ";
  return stream;
}

int FindEquivalent(const std::vector<int>& Gp, int istart, const std::vector<std::vector<int> >& Gall, const std::vector<std::vector<int> >& V0perm)
{
  std::vector<int> Gn(Gp); // copy constructor called;
  //std::cout<<" incomng Gp="<< Gp << " copy=" << Gn << " V0perm.size=" << V0perm.size() << std::endl;
  for (int i=0; i<V0perm.size(); i++){ // We go over all possible permutations of V0 interaction, and generate all equivalent diagrams
    const std::vector<int>& p = V0perm[i];
    //std::cout << "    cc i="<< i << " p.size="<< p.size() << std::endl;
    for (int j=0; j<p.size(); j++) // If we exchange vertices according to permutation V0perm, the electron propagators has to change 
      Gn[p[j]] = Gp[j+2];    // in the following way:
    
    std::vector<int> Gm(Gn); // copy again 
    for (int j=0; j<Gn.size(); j++) // because (0,1) interaction line is the measuring line
      if (Gn[j]>=2) Gm[j] = p[Gn[j]-2];
    // finally we have the correspondingly permuted Gp, and hence we will check if such diagram is hiding somewhere else
      
    //std::cout << "   c: Gp=" << Gp << " permuting it into "<< Gm << " with V0perm["<<i<<"]=" << p << std::endl;
    //std::cout << "Gp=" << Gp << " p=" << p << " Gm= " << Gm << std::endl;
    for (int iq=istart; iq<Gall.size(); iq++){  // over all other diagrams
      if (Gm==Gall[iq])    // is this permuted diagram somewhere in the remaining of the list?
	return iq;
    }
  }
  return -1;
}

std::vector<int> FindAllEquivalent(const std::vector<int>& Gp, int istart, const std::vector<std::vector<int> >& Gall, const std::vector<std::vector<int> >& V0perm)
{
  std::vector<int> iwhich(V0perm.size());
  #pragma omp parallel for
  for (int i=0; i<V0perm.size(); i++){ // We go over all possible permutations of V0 interaction, and generate all equivalent diagrams
    iwhich[i]=-1;
    std::vector<int> Gn(Gp); // copy constructor called;
    const std::vector<int>& p = V0perm[i];
    for (int j=0; j<p.size(); j++) // If we exchange vertices according to permutation V0perm, the electron propagators has to change 
      Gn[p[j]] = Gp[j+2];    // in the following way:
    
    std::vector<int> Gm(Gn); // copy again 
    for (int j=0; j<Gn.size(); j++) // because (0,1) interaction line is the measuring line
      if (Gn[j]>=2) Gm[j] = p[Gn[j]-2];
    // finally we have the correspondingly permuted Gp, and hence we will check if such diagram is hiding somewhere else
      
    for (int iq=istart; iq<Gall.size(); iq++){  // over all other diagrams
      if (Gm==Gall[iq]){    // is this permuted diagram somewhere in the remaining of the list?
	iwhich[i] = iq;
	break;
      }
    }
  }
  int n=0;
  for (int i=0; i<iwhich.size(); i++)
    if (iwhich[i]>=0) n++;
  std::vector<int> which(n);
  n=0;
  for (int i=0; i<iwhich.size(); i++)
    if (iwhich[i]>=0){
      which[n]=iwhich[i];
      n++;
    }
  return which;
}

//def FindEquivalent2(Gp, cVtyp, Gall, Vtyp, V0perm, Viiperm):
py::tuple FindEquivalent2(const std::vector<int>& Gp, const std::vector<int>& cVtyp,
		    const std::vector<std::vector<int> >& Gall, const std::vector<std::vector<int> >& Vtyp,
		    const std::vector<std::vector<int> >& V0perm, const std::vector<std::vector<int> >& Viiperm)
{
  std::vector<int> Gn(Gp); // copy constructor called;
  int nord = Gp.size()/2;
  std::vector<int> xVtyp(nord);
  xVtyp[0]=0;
  //std::cout<<" incomng Gp="<< Gp << " copy=" << Gn << " V0perm.size=" << V0perm.size() << std::endl;
  for (int i=0; i<V0perm.size(); i++){ // We go over all possible permutations of V0 interaction, and generate all equivalent diagrams
    const std::vector<int>& p = V0perm[i];
    //std::cout << "    cc i="<< i << " p.size="<< p.size() << std::endl;
    for (int j=0; j<p.size(); j++) // If we exchange vertices according to permutation V0perm, the electron propagators has to change 
      Gn[p[j]] = Gp[j+2];    // in the following way:
    
    std::vector<int> Gm(Gn); // copy again 
    for (int j=0; j<Gn.size(); j++) // because (0,1) interaction line is the measuring line
      if (Gn[j]>=2) Gm[j] = p[Gn[j]-2];
    // finally we have the correspondingly permuted Gp, and hence we will check if such diagram is hiding somewhere else

    for (int j=0; j<nord-1; j++){
      //xVtyp[j+1] = cVtyp[Viiperm[i][j]];
      xVtyp[Viiperm[i][j]] = cVtyp[j+1];
    }
    //std::cout << "   c: "<<i<<") Gp=" << Gp << " permuting it into "<< Gm << " with V0perm["<<i<<"]=" << p;
    //std::cout << "    : xVtyp="<< xVtyp << " cVtyp="<< cVtyp << std::endl;
    //std::cout << "Gp=" << Gp << " p=" << p << " Gm= " << Gm << std::endl;
    for (int iq=0; iq<Gall.size(); iq++){  // over all other diagrams
      if (Gm==Gall[iq] && xVtyp==Vtyp[iq])    // is this permuted diagram somewhere in the remaining of the list?
	return py::make_tuple(iq, i);
    }
  }
  return py::make_tuple(-1, 0);
}

PYBIND11_MODULE(equivalent,m) {
  m.doc() = "pybind11 wrap for FindEquivalent";
  m.def("FindEquivalent", &FindEquivalent); 
  m.def("FindAllEquivalent", &FindAllEquivalent); 
  m.def("FindEquivalent2", &FindEquivalent2); 
}
