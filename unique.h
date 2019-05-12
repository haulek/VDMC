// @Copyright 2018 Kristjan Haule 
#include <iostream>
#include <blitz/array.h>
#include <deque>
#include <vector>
#include <map>
#include <algorithm>
#include "bitset.h"
//#include "util.h"

namespace bl = blitz;
//using namespace std;
//using namespace blitz;

struct Class2Compare4
{
  bool operator()(const bl::TinyVector<int,4>& x, const bl::TinyVector<int,4>& y) const
   {
     if (x[0]!=y[0]) return x[0]<y[0];
     if (x[1]!=y[1]) return x[1]<y[1];
     if (x[2]!=y[2]) return x[2]<y[2];
     if (x[3]!=y[3]) return x[3]<y[3];
     return false;
   }
};
struct Class2Compare5
{
  bool operator()(const bl::TinyVector<int,5>& x, const bl::TinyVector<int,5>& y) const
   {
     if (x[0]!=y[0]) return x[0]<y[0];
     if (x[1]!=y[1]) return x[1]<y[1];
     if (x[2]!=y[2]) return x[2]<y[2];
     if (x[3]!=y[3]) return x[3]<y[3];
     if (x[4]!=y[4]) return x[4]<y[4];
     return false;
   }
};
struct Class2Compare6
{
  bool operator()(const bl::TinyVector<int,6>& x, const bl::TinyVector<int,6>& y) const
   {
     if (x[0]!=y[0]) return x[0]<y[0];
     if (x[1]!=y[1]) return x[1]<y[1];
     if (x[2]!=y[2]) return x[2]<y[2];
     if (x[3]!=y[3]) return x[3]<y[3];
     if (x[4]!=y[4]) return x[4]<y[4];
     if (x[5]!=y[5]) return x[5]<y[5];
     return false;
   }
};
struct Class2Compare7
{
  bool operator()(const bl::TinyVector<int,7>& x, const bl::TinyVector<int,7>& y) const
   {
     if (x[0]!=y[0]) return x[0]<y[0];
     if (x[1]!=y[1]) return x[1]<y[1];
     if (x[2]!=y[2]) return x[2]<y[2];
     if (x[3]!=y[3]) return x[3]<y[3];
     if (x[4]!=y[4]) return x[4]<y[4];
     if (x[5]!=y[5]) return x[5]<y[5];
     if (x[6]!=y[6]) return x[6]<y[6];
     return false;
   }
};
struct Class2Compare8
{
  bool operator()(const bl::TinyVector<int,8>& x, const bl::TinyVector<int,8>& y) const
   {
     if (x[0]!=y[0]) return x[0]<y[0];
     if (x[1]!=y[1]) return x[1]<y[1];
     if (x[2]!=y[2]) return x[2]<y[2];
     if (x[3]!=y[3]) return x[3]<y[3];
     if (x[4]!=y[4]) return x[4]<y[4];
     if (x[5]!=y[5]) return x[5]<y[5];
     if (x[6]!=y[6]) return x[6]<y[6];
     if (x[7]!=y[7]) return x[7]<y[7];
     return false;
   }
};
std::deque<int> FindPrimeNumbers(int lower, int upper)
{
  std::deque<int> pm;
  for (int num=lower; num<upper+1; num++){
    if (num > 1){
      bool prime=true;
      for (int i=2; i<num; i++){
	if ((num % i) == 0) {
	  prime=false;
	  break;
	}
      }
      if (prime) pm.push_back(num);
    }
  }
  return pm;
}

void findUnique(bl::Array<int,2>& Gindx, bl::Array<int,2>& Vindx, bl::Array<bl::TinyVector<int,2>,1>& gindx, bl::Array<bl::TinyVector<int,2>,1>& vindx,
		const bl::Array<int,2>& diagsG, const std::vector<std::vector<std::vector<int> > >& Loop_index, const std::vector<std::vector<std::vector<int> > >& Loop_type,
		const bl::Array<int,2>& Vtype, const bl::Array<int,1>& indx,
		bool debug=false)
{
  int Ndiags = diagsG.extent(0);
  int Norder = diagsG.extent(1)/2;
  // computing the inverse of all permutations, to provide fast lookup for the previous fermionic propagator
  bl::Array<int,2> i_diagsG(Ndiags,2*Norder);
  for (int id=0; id<Ndiags; id++)
    for (int i=0; i<2*Norder; i++)
      i_diagsG(id,diagsG(id,i))=i;

  std::deque<int> pm = FindPrimeNumbers(199,10000);
  
  //RanGSL drand(p.iseed); // GSL random number generator
  
  if (Loop_index.size()!=Ndiags) std::cerr<<"Wrong size of Loop_index "<<Loop_index.size()<<" instead of "<<Ndiags<<std::endl;
  int Nloops = Loop_index[0].size();
  if (Nloops != Norder+1) std::cerr<<"Expecting Nloops==Norder+1 in Loops_index, but got Nloops="<<Nloops<<" Norder="<<Norder<<std::endl;

  // This is momentum for loops
  bl::Array<bl::TinyVector<int,3>,1> momentum(Nloops); // momentum for all loops. Each diagram of the same order has the same number of loops.
  // This is momentum for all diagrams and all propagators
  bl::Array<bl::TinyVector<int,3>,2> mom_G(Ndiags,2*Norder); // momentum for 2*Norder fermionic propagators
  bl::Array<bl::TinyVector<int,3>,2> mom_V(Ndiags,  Norder); // momentum for Norder bosonic propagators
  
  // Starting point is set up to be random
  // For the times for all vertices in all the graphs
  bl::Array<bl::TinyVector<int,3>,1> times(2*Norder);
  times(0) = pm[0], pm[1], pm[2];
  times(1) = 0.0, 0.0, 0.0;
  int ip=3;
  for (int it=1; it<Norder; it++){
    bl::TinyVector<int,3> tv(pm[ip], pm[ip+1], pm[ip+2]);
    times(2*it  ) = tv;
    times(2*it+1) = tv;
    ip += 3;
  }
  // Next the momenta
  for (int ik=0; ik<Nloops; ik++){
    momentum(ik) = pm[ip], pm[ip+1], pm[ip+2];
    ip += 3;
  }
  // Momenta for all propagators in all graphs, which is needed to evaluate the diagrams.
  mom_G=0;
  mom_V=0;
  for (int id=0; id<Ndiags; id++){
    for (int iloop=0; iloop<Nloops; iloop++){
      const std::vector<int>& lindex = Loop_index[id][iloop];
      const std::vector<int>& ltype  = Loop_type[id][iloop];
      for (int i=0; i<lindex.size(); i++){
        if ( abs(ltype[i])==1 ){
	  mom_G(id, lindex[i]) += momentum(iloop) * sign(ltype[i]);
	}else{
	  mom_V(id,lindex[i]) += momentum(iloop) * sign(ltype[i]);
	}
      }
    }
  }

  if (debug){
    std::cout << "times=" << std::endl;
    for (int it=0; it<2*Norder; it++){
      std::cout << it << " " << times(it) << std::endl;
    }
    std::cout << "momenta=" << std::endl;
    for (int iq=0; iq<Nloops; iq++){
      std::cout<<iq<<" "<<momentum(iq) << std::endl;
    }
  }
  
  typedef bl::TinyVector<int,7> Key;
  typedef bl::TinyVector<int,5> key;
  std::map<Key,int,Class2Compare7> Gkt;
  std::map<key,int,Class2Compare5> Vk;
  
  //bl::Array<int,2> Gindx, Vindx;
  Gindx.resize(Ndiags,2*Norder); Vindx.resize(Ndiags,Norder);
  Vindx=-1;
  Gindx=-1;
  int ig=0;
  int iv=0;
  for (int id=0; id<Ndiags; id++){
    for (int i=0; i<2*Norder; i++){
      int i_final = diagsG(id,i);
      bl::TinyVector<int,3> K = mom_G(id,i);
      bl::TinyVector<int,3> dt = times(i_final)-times(i);
      Key kt( K(0), K(1), K(2), dt(0), dt(1), dt(2), indx(id) );
      if (Gkt.find(kt)!=Gkt.end()){
	Gindx(id,i) = Gkt[kt];
      }else{
	Gkt[kt]=ig;
	Gindx(id,i) = ig;
	ig += 1;
      }
    }
    for (int i=1; i<Norder; i++){ // do not add V for the meassuring line
      bl::TinyVector<double,3> Q = mom_V(id,i);
      key Qt( Q(0), Q(1), Q(2), Vtype(id,i), indx(id) );
      if (Vk.find(Qt)!=Vk.end()){
	Vindx(id,i) = Vk[Qt];
      }else{
	Vk[Qt] = iv;
	Vindx(id,i) = iv;
	iv += 1;
      }
    }
  }
  /*
  cout<<"diagrams"<<endl;
  for (int id=0; id<Ndiags; id++){
    cout<<"----- diag="<<id<<" ----"<<endl;
    for (int i=0; i<2*Norder; i++){
      bl::TinyVector<int,3> K = mom_G(id,i);
      int i_final = diagsG(id,i);
      bl::TinyVector<int,3> dt = times(i_final)-times(i);
      cout<<i<<" K="<< K << " dt="<<dt<<"  -> "<<Gindx(id,i)<<endl;
    }
  }
  */

  //bl::Array<bl::TinyVector<int,2>,1> gindx, vindx;
  int Ng = Gkt.size();
  gindx.resize(Ng);
  for (int ii=0; ii<gindx.extent(0); ++ii) gindx(ii) = -1,-1;
  
  for(int ii=0; ii<Ng; ++ii){
    if (debug) std::cout<<"G "<<ii<<" : ";
    for (int id=0; id<Ndiags; id++){
      for (int i=0; i<2*Norder; i++){
	if ( Gindx(id,i)==ii){
	  if (gindx(ii)[0]<0) gindx(ii) = id, i;
	  if (debug) std::cout<<"("<<id<<","<<i<<") ";
	}
      }
    }
    if (debug) std::cout<<std::endl;
  }
  if (debug) std::cout<<std::endl;

  int Nv = Vk.size();
  vindx.resize(Nv);
  for (int ii=0; ii<vindx.extent(0); ++ii) vindx(ii) = -1,-1;
  for(int ii=0; ii<Nv; ++ii){
    if (debug) std::cout<<"V "<<ii<<" : ";
    for (int id=0; id<Ndiags; id++){
      for (int i=0; i<Norder; i++){
	if ( Vindx(id,i)==ii){
	  if (vindx(ii)[0]<0) vindx(ii) = id,i;
	  if (debug) std::cout<<"("<<id<<","<<i<<") ";
	}
      }
    }
    if (debug) std::cout<<std::endl;
  }
  if (debug) std::cout<<std::endl;
  
  if (debug){
    for (int id=0; id<Ndiags; id++){
      std::cout<<std::setw(3)<<id<<" : ";
      for (int i=0; i<2*Norder; i++){
	std::cout<<std::setw(3)<<Gindx(id,i)<<" ";
      }
      std::cout<<std::endl;
    }
    for (int ii=0; ii<gindx.extent(0); ++ii){
      std::cout<<"G "<<ii<<" : ("<<gindx(ii)[0]<<","<<gindx(ii)[1]<<")"<<std::endl;
    }
    for (int ii=0; ii<vindx.extent(0); ++ii){
      std::cout<<"V "<<ii<<" : ("<<vindx(ii)[0]<<","<<vindx(ii)[1]<<")"<<std::endl;
    }
  }
}

void findUnique_noindx(bl::Array<int,2>& Gindx, bl::Array<int,2>& Vindx, bl::Array<bl::TinyVector<int,2>,1>& gindx, bl::Array<bl::TinyVector<int,2>,1>& vindx,
		       const bl::Array<int,2>& diagsG, const std::vector<std::vector<std::vector<int> > >& Loop_index, const std::vector<std::vector<std::vector<int> > >& Loop_type,
		       const bl::Array<int,2>& Vtype, int i_start_fermi=0,
		       bool debug=false)
{
  int Ndiags = diagsG.extent(0);
  int Norder = diagsG.extent(1)/2;
  // computing the inverse of all permutations, to provide fast lookup for the previous fermionic propagator
  bl::Array<int,2> i_diagsG(Ndiags,2*Norder);
  for (int id=0; id<Ndiags; id++)
    for (int i=0; i<2*Norder; i++)
      i_diagsG(id,diagsG(id,i))=i;

  std::deque<int> pm = FindPrimeNumbers(199,10000);
  
  //RanGSL drand(p.iseed); // GSL random number generator
  
  if (Loop_index.size()!=Ndiags) std::cerr<<"Wrong size of Loop_index "<<Loop_index.size()<<" instead of "<<Ndiags<<std::endl;
  int Nloops = Loop_index[0].size();
  if (Nloops != Norder+1-i_start_fermi) std::cerr<<"Expecting Nloops==Norder+1 in Loops_index, but got Nloops="<<Nloops<<" Norder="<<Norder<<std::endl;

  // This is momentum for loops
  bl::Array<bl::TinyVector<int,3>,1> momentum(Nloops); // momentum for all loops. Each diagram of the same order has the same number of loops.
  // This is momentum for all diagrams and all propagators
  bl::Array<bl::TinyVector<int,3>,2> mom_G(Ndiags,2*Norder); // momentum for 2*Norder fermionic propagators
  bl::Array<bl::TinyVector<int,3>,2> mom_V(Ndiags,  Norder); // momentum for Norder bosonic propagators
  
  // Starting point is set up to be random
  // For the times for all vertices in all the graphs
  bl::Array<bl::TinyVector<int,3>,1> times(2*Norder);
  times(0) = pm[0], pm[1], pm[2];
  times(1) = 0.0, 0.0, 0.0;
  int ip=3;
  for (int it=1; it<Norder; it++){
    bl::TinyVector<int,3> tv(pm[ip], pm[ip+1], pm[ip+2]);
    times(2*it  ) = tv;
    times(2*it+1) = tv;
    ip += 3;
  }
  // Next the momenta
  for (int ik=0; ik<Nloops; ik++){
    momentum(ik) = pm[ip], pm[ip+1], pm[ip+2];
    ip += 3;
  }
  // Momenta for all propagators in all graphs, which is needed to evaluate the diagrams.
  mom_G=0;
  mom_V=0;
  for (int id=0; id<Ndiags; id++){
    for (int iloop=0; iloop<Nloops; iloop++){
      const std::vector<int>& lindex = Loop_index[id][iloop];
      const std::vector<int>& ltype  = Loop_type[id][iloop];
      for (int i=0; i<lindex.size(); i++){
        if ( abs(ltype[i])==1 ){
	  mom_G(id, lindex[i]) += momentum(iloop) * sign(ltype[i]);
	}else{
	  mom_V(id,lindex[i]) += momentum(iloop) * sign(ltype[i]);
	}
      }
    }
  }

  if (debug){
    std::cout << "times=" << std::endl;
    for (int it=0; it<2*Norder; it++){
      std::cout << it << " " << times(it) << std::endl;
    }
    std::cout << "momenta=" << std::endl;
    for (int iq=0; iq<Nloops; iq++){
      std::cout<<iq<<" "<<momentum(iq)<<std::endl;
    }
  }
  
  typedef bl::TinyVector<int,6> Key;
  typedef bl::TinyVector<int,4> key;
  std::map<Key,int,Class2Compare6> Gkt;
  std::map<key,int,Class2Compare4> Vk;
  
  //bl::Array<int,2> Gindx, Vindx;
  Gindx.resize(Ndiags,2*Norder); Vindx.resize(Ndiags,Norder);
  Vindx=-1;
  Gindx=-1;
  int ig=0;
  int iv=0;
  for (int id=0; id<Ndiags; id++){
    for (int i=i_start_fermi; i<2*Norder; i++){
      int i_final = diagsG(id,i);
      bl::TinyVector<int,3> K = mom_G(id,i);
      bl::TinyVector<int,3> dt = times(i_final)-times(i);
      Key kt( K(0), K(1), K(2), dt(0), dt(1), dt(2) );
      if (Gkt.find(kt)!=Gkt.end()){
	Gindx(id,i) = Gkt[kt];
      }else{
	Gkt[kt]=ig;
	Gindx(id,i) = ig;
	ig += 1;
      }
    }
    for (int i=1; i<Norder; i++){ // do not add V for the meassuring line
      bl::TinyVector<double,3> Q = mom_V(id,i);
      key Qt( Q(0), Q(1), Q(2), Vtype(id,i) );
      if (Vk.find(Qt)!=Vk.end()){
	Vindx(id,i) = Vk[Qt];
      }else{
	Vk[Qt] = iv;
	Vindx(id,i) = iv;
	iv += 1;
      }
    }
  }
  /*
  cout<<"diagrams"<<endl;
  for (int id=0; id<Ndiags; id++){
    cout<<"----- diag="<<id<<" ----"<<endl;
    for (int i=0; i<2*Norder; i++){
      bl::TinyVector<int,3> K = mom_G(id,i);
      int i_final = diagsG(id,i);
      bl::TinyVector<int,3> dt = times(i_final)-times(i);
      cout<<i<<" K="<< K << " dt="<<dt<<"  -> "<<Gindx(id,i)<<endl;
    }
  }
  */

  //bl::Array<bl::TinyVector<int,2>,1> gindx, vindx;
  int Ng = Gkt.size();
  gindx.resize(Ng);
  for (int ii=0; ii<gindx.extent(0); ++ii) gindx(ii) = -1,-1;
  
  for(int ii=0; ii<Ng; ++ii){
    if (debug) std::cout<<"G "<<ii<<" : ";
    for (int id=0; id<Ndiags; id++){
      for (int i=0; i<2*Norder; i++){
	if ( Gindx(id,i)==ii){
	  if (gindx(ii)[0]<0) gindx(ii) = id, i;
	  if (debug) std::cout<<"("<<id<<","<<i<<") ";
	}
      }
    }
    if (debug) std::cout<<std::endl;
  }
  if (debug) std::cout<<std::endl;

  int Nv = Vk.size();
  vindx.resize(Nv);
  for (int ii=0; ii<vindx.extent(0); ++ii) vindx(ii) = -1,-1;
  for(int ii=0; ii<Nv; ++ii){
    if (debug) std::cout<<"V "<<ii<<" : ";
    for (int id=0; id<Ndiags; id++){
      for (int i=0; i<Norder; i++){
	if ( Vindx(id,i)==ii){
	  if (vindx(ii)[0]<0) vindx(ii) = id,i;
	  if (debug) std::cout<<"("<<id<<","<<i<<") ";
	}
      }
    }
    if (debug) std::cout<<std::endl;
  }
  if (debug) std::cout<<std::endl;
  
  if (debug){
    for (int id=0; id<Ndiags; id++){
      std::cout<<std::setw(3)<<id<<" : ";
      for (int i=0; i<2*Norder; i++){
	std::cout<<std::setw(3)<<Gindx(id,i)<<" ";
      }
      std::cout<<std::endl;
    }
    for (int ii=0; ii<gindx.extent(0); ++ii){
      std::cout<<"G "<<ii<<" : ("<<gindx(ii)[0]<<","<<gindx(ii)[1]<<")"<<std::endl;
    }
    for (int ii=0; ii<vindx.extent(0); ++ii){
      std::cout<<"V "<<ii<<" : ("<<vindx(ii)[0]<<","<<vindx(ii)[1]<<")"<<std::endl;
    }
  }
  /*
  if (istart_fermi>0){
    for (int id=0; id<Ndiags; id++){
      for (int i=0; i<i_start_fermi; i++){
	int i_final = diagsG(id,i);
	bl::TinyVector<int,3> K = mom_G(id,i);
	bl::TinyVector<int,3> dt = times(i_final)-times(i);
	Key kt( K(0), K(1), K(2), dt(0), dt(1), dt(2) );
	if (Gkt.find(kt)!=Gkt.end()){
	  Gindx(id,i) = Gkt[kt];
	}else{
	  Gkt[kt]=ig;
	  Gindx(id,i) = ig;
	  ig += 1;
	}
      }
    }
  }
  */
}
template <typename sint, typename ushort, typename uchar>
void findUnique_noindx_spct(bl::Array<sint,2>& Gindx, bl::Array<sint,2>& Vindx, bl::Array<bl::TinyVector<int,2>,1>& gindx, bl::Array<bl::TinyVector<int,2>,1>& vindx,
			    int& N0v,
			    const bl::Array<ushort,2>& diagsG, const bl::Array<uchar,2>& Vtype,
			    const std::vector<std::vector<std::vector<int> > >& Loop_index, const std::vector<std::vector<std::vector<int> > >& Loop_type,
			    int i_start_fermi=0, bool dynamic_counter=false,
			    bool debug=false, ostream& log=std::cout)
{
  int Ndiags = diagsG.extent(0);
  int Norder = diagsG.extent(1)/2;
  // computing the inverse of all permutations, to provide fast lookup for the previous fermionic propagator
  bl::Array<int,2> i_diagsG(Ndiags,2*Norder);
  for (int id=0; id<Ndiags; id++)
    for (int i=0; i<2*Norder; i++)
      i_diagsG(id, static_cast<int>(diagsG(id,i)) )=i;

  std::deque<int> pm = FindPrimeNumbers(199,10000);
  
  //RanGSL drand(p.iseed); // GSL random number generator
  
  //if (Loop_index.extent(0)!=Ndiags) log<<"ERROR : Wrong size of Loop_index "<<Loop_index.size()<<" instead of "<<Ndiags<<endl;
  if (Loop_index.size()!=Ndiags) log<<"ERROR : Wrong size of Loop_index "<<Loop_index.size()<<" instead of "<<Ndiags<<std::endl;
  //int Nloops = Loop_index.extent(1);
  int Nloops = Loop_index[0].size();
  if (Nloops != Norder+1-i_start_fermi) log<<"ERROR : Expecting Nloops==Norder+1 in Loops_index, but got Nloops="<<Nloops<<" Norder="<<Norder<<std::endl;

  // This is momentum for loops
  bl::Array<bl::TinyVector<int,3>,1> momentum(Nloops); // momentum for all loops. Each diagram of the same order has the same number of loops.
  // This is momentum for all diagrams and all propagators
  bl::Array<bl::TinyVector<int,3>,2> mom_G(Ndiags,2*Norder); // momentum for 2*Norder fermionic propagators
  bl::Array<bl::TinyVector<int,3>,2> mom_V1(Ndiags,  Norder); // momentum for Norder bosonic propagators
  bl::Array<bl::TinyVector<int,3>,2> mom_V2(Ndiags,  Norder); // momentum for Norder bosonic propagators
  
  // Starting point is set up to be random
  // For the times for all vertices in all the graphs
  bl::Array<bl::TinyVector<int,3>,1> times(2*Norder);
  times(0) = pm[0], pm[1], pm[2];
  times(1) = 0.0, 0.0, 0.0;
  int ip=3;
  for (int it=1; it<Norder; it++){
    bl::TinyVector<int,3> tv(pm[ip], pm[ip+1], pm[ip+2]);
    times(2*it  ) = tv;
    times(2*it+1) = tv;
    ip += 3;
  }
  // Next the momenta
  for (int ik=0; ik<Nloops; ik++){
    momentum(ik) = pm[ip], pm[ip+1], pm[ip+2];
    ip += 3;
  }
  // Momenta for all propagators in all graphs, which is needed to evaluate the diagrams.
  mom_G=0;
  mom_V1=0;
  for (int id=0; id<Ndiags; id++){
    for (int iloop=0; iloop<Nloops; iloop++){
      const std::vector<int>& lindex = Loop_index[id][iloop];
      const std::vector<int>& ltype  = Loop_type[id][iloop];
      for (int i=0; i<lindex.size(); i++){
	int lindex_i = lindex[i];
	int ltype_i  = ltype[i];
        if ( abs(ltype_i)==1 ){
	  mom_G(id, lindex_i) += momentum(iloop) * sign(ltype_i);
	}else{
	  mom_V1(id,lindex_i) += momentum(iloop) * sign(ltype_i);
	}
      }
    }
    for (int i=0; i<Norder; ++i){
      int i_previ = i_diagsG(id,2*i);
      bl::TinyVector<int,3> k_in   = mom_G(id,i_previ);
      bl::TinyVector<int,3> k_out  = mom_G(id,2*i);
      bl::TinyVector<int,3> k_out2 = mom_G(id,2*i+1);
      bl::TinyVector<int,3> q =  k_in - k_out;
      bl::TinyVector<int,3> q1 = mom_V1(id,i);
      bl::TinyVector<int,3> dq = q1-q;
      if (fabs(norm(dq))>1e-6) log<<" ERROR : Vq1 is wrong q="<< q << " mom_V1=" << mom_V1(id,i) << std::endl;
      mom_V2(id,i) = -k_in + k_out2;
    }
  }

  if (debug){
    //log << "times=" << std::endl;
    //for (int it=0; it<2*Norder; it++){
    //  log << it << " " << times(it) << std::endl;
    //}
    //log << "momenta=" << std::endl;
    //for (int iq=0; iq<Nloops; iq++){
    //  log<<iq<<" "<<momentum(iq)<<std::endl;
    //}
    /// DEBUG
    //log<< "momenta for all interactions="<<endl;
    //for (int id=0; id<Ndiags; id++){
    //  for (int i=0; i<Norder; ++i){
    //	log << setw(3)<< id << " "<< setw(2) << i << " : ("<< mom_V1(id,i)[0]<<","<<mom_V1(id,i)[1]<<","<<mom_V1(id,i)[2]<<") ";
    //	log << " ("<<mom_V2(id,i)[0]<<","<<mom_V2(id,i)[1]<<","<<mom_V2(id,i)[2]<<")"<<endl;
    //  }
    //}
  }
  
  typedef bl::TinyVector<int,6> Key;
  typedef bl::TinyVector<int,8> key8;
  std::map<Key,int,Class2Compare6> Gkt;
  std::map<key8,int,Class2Compare8> Vk;

  //int max_element = std::max_element(Gkt.begin(), Gkt.end(), Gkt.value_comp());
  //if (max_element > numeric_limits<sint>().max()){
  //  log<<"ERROR : You must change the type for Gindx, because the needed value is larger than allowed by this short int type"<<endl;
  //}
  //bl::Array<int,2> Gindx, Vindx;
  Gindx.resize(Ndiags,2*Norder); Vindx.resize(Ndiags,Norder);
  Vindx = std::numeric_limits<sint>().max();
  Gindx = std::numeric_limits<sint>().max();
  int ig=0;
  int iv=0;
  for (int id=0; id<Ndiags; id++){
    for (int i=i_start_fermi; i<2*Norder; i++){
      int i_final = diagsG(id,i);
      bl::TinyVector<int,3> K = mom_G(id,i);
      bl::TinyVector<int,3> dt = times(i_final)-times(i);
      Key kt( K(0), K(1), K(2), dt(0), dt(1), dt(2) );
      if (Gkt.find(kt)!=Gkt.end()){
	int ii = Gkt[kt];
	if (ii >= std::numeric_limits<sint>().max()){
	  log<<"ERROR : You must change the type for Gindx, because the needed value is larger than allowed by this short int type"<<std::endl;
	  exit(1);
	}
	Gindx(id,i) = ii;
      }else{
	Gkt[kt]=ig;
	if (ig >= std::numeric_limits<sint>().max()){
	  log<<"ERROR : You must change the type for Gindx, because the needed value is larger than allowed by this short int type"<<std::endl;
	  exit(1);
	}	  
	Gindx(id,i) = ig;
	ig += 1;
      }
    }
    for (int i=1; i<Norder; i++){ // do not add V for the meassuring line
      bl::TinyVector<int,3> Q = mom_V1(id,i);
      bl::TinyVector<int,3> Q2(0, 0, 0);
      
      int vtyp = Vtype(id,i);
      if (vtyp>=10) continue; // this is Hugenholtz-type, and do not do anything just yet. Will treat it below
      if (vtyp<0){// for single-particle counter term, the final value depends also on n_{k+q}, hence we need to be much more careful.
	if (diagsG(id,2*i)==2*i+1){ // G_propagator goes from 2i->2i+1
	  vtyp = vtyp - Gindx(id,2*i)*10;     // G_propagtaors for n_{k+q}. We add its value, so that we combine only those terms, which have equal Vq and also equal n_{k+q}
	} else if (diagsG(id,2*i+1)==2*i){ // alternatively it goes from 2i+1->2i
	  vtyp = vtyp - Gindx(id,2*i+1)*10;   // G_propagator for n_{k+q}
	}
      }
      int dtyp = 0;
      if ( dynamic_counter && Vtype(id,i)!= 0 ){
	// Here we have dynamic counter term, hence 2*i+1 vertex requires Vertex object, and can be moved.
	// Consequently, the interaction in two Feynman diagrams is equal only if both vertices : diagsG(id,2*i+1) and i_diagsG(id,2*i+1) are equal in the two diagramns.
	// i_diagsG(id,2*i+1) takes care of incoming vertex being the same
	//   diagsG(id,2*i+1) following vertex being the same
	//   At the point 2i+1 we have vertex, where the incoming (k_i) and outgoing G momentum (k_o) needs to be the same.
	//   This is satisfied if Gindx(id,2*i+1) is the same
	dtyp = i_diagsG(id,2*i+1) + diagsG(id,2*i+1)*(2*Norder) + Gindx(id,2*i+1)*(2*Norder*2*Norder);
      }
      
      key8 Qt( Q(0), Q(1), Q(2), vtyp, dtyp, Q2(0), Q2(1), Q2(2) );
      if (Vk.find(Qt)!=Vk.end()){
	Vindx(id,i) = Vk[Qt];
	//log<<"ERROR : setting1 Vindx("<<id<<","<<i<<")="<<Vindx(id,i)<<endl;
      }else{
	Vk[Qt] = iv;
	Vindx(id,i) = iv;
	//log<<"ERROR : setting2 Vindx("<<id<<","<<i<<")="<<Vindx(id,i)<<endl;
	iv += 1;
      }
    }
  }

  N0v = iv;
  for (int id=0; id<Ndiags; id++){
    for (int i=1; i<Norder; i++){ // do not add V for the meassuring line
      bl::TinyVector<int,3> Q = mom_V1(id,i);
      bl::TinyVector<int,3> Q2(0, 0, 0);
      int vtyp = Vtype(id,i);
      if (vtyp<10) continue; // this is not Hugenholtz-type, hence it was treated above
      Q2 = mom_V2(id,i);
      int dtyp = 0;
      key8 Qt( Q(0), Q(1), Q(2), vtyp, dtyp, Q2(0), Q2(1), Q2(2) );
      if (Vk.find(Qt)!=Vk.end()){
	Vindx(id,i) = Vk[Qt];
	//log<<"ERROR : setting1 Vindx("<<id<<","<<i<<")="<<Vindx(id,i)<<endl;
      }else{
	Vk[Qt] = iv;
	Vindx(id,i) = iv;
	//log<<"ERROR : setting2 Vindx("<<id<<","<<i<<")="<<Vindx(id,i)<<endl;
	iv += 1;
      }
    }
  }
  
  /*
  log<<"diagrams"<<endl;
  for (int id=0; id<Ndiags; id++){
    log<<"----- diag="<<id<<" ----"<<endl;
    for (int i=0; i<2*Norder; i++){
      bl::TinyVector<int,3> K = mom_G(id,i);
      int i_final = diagsG(id,i);
      bl::TinyVector<int,3> dt = times(i_final)-times(i);
      log<<i<<" K="<< K << " dt="<<dt<<"  -> "<<Gindx(id,i)<<endl;
    }
  }
  */

  //bl::Array<bl::TinyVector<int,2>,1> gindx, vindx;
  int Ng = Gkt.size();
  gindx.resize(Ng);
  for (int ii=0; ii<gindx.extent(0); ++ii) gindx(ii) = -1,-1;
  
  for(int ii=0; ii<Ng; ++ii){
    if (debug) log<<"G "<<ii<<" : ";
    for (int id=0; id<Ndiags; id++){
      for (int i=0; i<2*Norder; i++){
	if ( static_cast<int>(Gindx(id,i))==ii){
	  if (gindx(ii)[0]<0) gindx(ii) = id, i;
	  if (debug) log<<"("<<id<<","<<i<<") ";
	}
      }
    }
    if (debug) log<<std::endl;
  }
  if (debug) log<<std::endl;

  int Nv = Vk.size();
  vindx.resize(Nv);
  for (int ii=0; ii<vindx.extent(0); ++ii) vindx(ii) = -1,-1;
  for(int ii=0; ii<Nv; ++ii){
    if (debug) log<<"V "<<ii<<" : ";
    for (int id=0; id<Ndiags; id++){
      for (int i=0; i<Norder; i++){
	if ( static_cast<int>(Vindx(id,i))==ii){
	  if (vindx(ii)[0]<0) vindx(ii) = id,i;
	  if (debug) log<<"("<<id<<","<<i<<") ";
	}
      }
    }
    if (debug) log<<std::endl;
  }
  if (debug) log<<std::endl;
  
  if (debug){
    log<< "id : Gindx="<< endl;
    for (int id=0; id<Ndiags; id++){
      log<<std::setw(3)<<id<<" : ";
      for (int i=0; i<2*Norder; i++){
	log<<std::setw(3)<<static_cast<int>(Gindx(id,i))<<" ";
      }
      log<<std::endl;
    }
    log<< "id : Vindx="<< endl;
    for (int id=0; id<Ndiags; id++){
      log<<std::setw(3)<<id<<" : ";
      for (int i=1; i<Norder; i++){
	log<<std::setw(3)<<static_cast<int>(Vindx(id,i))<<" ";
      }
      log<<std::endl;
    }
    /*
    for (int ii=0; ii<gindx.extent(0); ++ii){
      log<<"G "<<ii<<" : ("<<gindx(ii)[0]<<","<<gindx(ii)[1]<<")"<<std::endl;
    }
    for (int ii=0; ii<vindx.extent(0); ++ii){
      log<<"V "<<ii<<" : ("<<vindx(ii)[0]<<","<<vindx(ii)[1]<<")"<<std::endl;
    }
    */
  }
  /*
  if (istart_fermi>0){
    for (int id=0; id<Ndiags; id++){
      for (int i=0; i<i_start_fermi; i++){
	int i_final = diagsG(id,i);
	bl::TinyVector<int,3> K = mom_G(id,i);
	bl::TinyVector<int,3> dt = times(i_final)-times(i);
	Key kt( K(0), K(1), K(2), dt(0), dt(1), dt(2) );
	if (Gkt.find(kt)!=Gkt.end()){
	  Gindx(id,i) = Gkt[kt];
	}else{
	  Gkt[kt]=ig;
	  Gindx(id,i) = ig;
	  ig += 1;
	}
      }
    }
  }
  */
}

template <typename sint, typename ushort, typename uchar>
void findUnique_noindx(bl::Array<sint,2>& Gindx, bl::Array<sint,2>& Vindx, bl::Array<bl::TinyVector<int,2>,1>& gindx, bl::Array<bl::TinyVector<int,2>,1>& vindx,
		       const bl::Array<ushort,2>& diagsG,
		       //const vector<vector<vector<int> > >& Loop_index, const vector<vector<vector<int> > >& Loop_type,
		       const bl::Array<bl::Array<unsigned char,1>,2>& Loop_index, const bl::Array<bl::Array<char,1>,2>& Loop_type,
		       const bl::Array<uchar,2>& Vtype, int i_start_fermi=0,
		       bool debug=false, ostream& log=std::cout)
{
  int Ndiags = diagsG.extent(0);
  int Norder = diagsG.extent(1)/2;
  // computing the inverse of all permutations, to provide fast lookup for the previous fermionic propagator
  bl::Array<int,2> i_diagsG(Ndiags,2*Norder);
  for (int id=0; id<Ndiags; id++)
    for (int i=0; i<2*Norder; i++)
      i_diagsG(id, static_cast<int>(diagsG(id,i)) )=i;

  std::deque<int> pm = FindPrimeNumbers(199,10000);
  
  //RanGSL drand(p.iseed); // GSL random number generator
  
  if (Loop_index.extent(0)!=Ndiags) log<<"ERROR: Wrong size of Loop_index "<<Loop_index.size()<<" instead of "<<Ndiags<<std::endl;
  //if (Loop_index.size()!=Ndiags) log<<"ERROR : Wrong size of Loop_index "<<Loop_index.size()<<" instead of "<<Ndiags<<endl;
  int Nloops = Loop_index.extent(1);
  //int Nloops = Loop_index[0].size();
  if (Nloops != Norder+1-i_start_fermi) log<<"ERROR: Expecting Nloops==Norder+1 in Loops_index, but got Nloops="<<Nloops<<" Norder="<<Norder<<std::endl;

  // This is momentum for loops
  bl::Array<bl::TinyVector<int,3>,1> momentum(Nloops); // momentum for all loops. Each diagram of the same order has the same number of loops.
  // This is momentum for all diagrams and all propagators
  bl::Array<bl::TinyVector<int,3>,2> mom_G(Ndiags,2*Norder); // momentum for 2*Norder fermionic propagators
  bl::Array<bl::TinyVector<int,3>,2> mom_V(Ndiags,  Norder); // momentum for Norder bosonic propagators
  
  // Starting point is set up to be random
  // For the times for all vertices in all the graphs
  bl::Array<bl::TinyVector<int,3>,1> times(2*Norder);
  times(0) = pm[0], pm[1], pm[2];
  times(1) = 0.0, 0.0, 0.0;
  int ip=3;
  for (int it=1; it<Norder; it++){
    bl::TinyVector<int,3> tv(pm[ip], pm[ip+1], pm[ip+2]);
    times(2*it  ) = tv;
    times(2*it+1) = tv;
    ip += 3;
  }
  // Next the momenta
  for (int ik=0; ik<Nloops; ik++){
    momentum(ik) = pm[ip], pm[ip+1], pm[ip+2];
    ip += 3;
  }
  // Momenta for all propagators in all graphs, which is needed to evaluate the diagrams.
  mom_G=0;
  mom_V=0;
  for (int id=0; id<Ndiags; id++){
    for (int iloop=0; iloop<Nloops; iloop++){
      const bl::Array<unsigned char,1>& lindex = Loop_index(id,iloop);
      const bl::Array<char,1>& ltype  = Loop_type(id,iloop);
      //const vector<int>& lindex = Loop_index[id][iloop];
      //const vector<int>& ltype  = Loop_type[id][iloop];
      for (int i=0; i<lindex.extent(0); i++){
	//for (int i=0; i<lindex.size(); i++){
	int lindex_i = lindex(i);
	int ltype_i  = ltype(i);
	//int lindex_i = lindex[i];
	//int ltype_i  = ltype[i];
        if ( abs(ltype_i)==1 ){
	  mom_G(id, lindex_i) += momentum(iloop) * sign(ltype_i);
	}else{
	  mom_V(id,lindex_i) += momentum(iloop) * sign(ltype_i);
	}
      }
    }
  }

  if (debug){
    log << "times=" << std::endl;
    for (int it=0; it<2*Norder; it++){
      log << it << " " << times(it) << std::endl;
    }
    log << "momenta=" << std::endl;
    for (int iq=0; iq<Nloops; iq++){
      log<<iq<<" "<<momentum(iq)<<std::endl;
    }
  }
  
  typedef bl::TinyVector<int,6> Key;
  typedef bl::TinyVector<int,4> key;
  std::map<Key,int,Class2Compare6> Gkt;
  std::map<key,int,Class2Compare4> Vk;

  Gindx.resize(Ndiags,2*Norder); Vindx.resize(Ndiags,Norder);
  Vindx = std::numeric_limits<sint>().max();
  Gindx = std::numeric_limits<sint>().max();
  int ig=0;
  int iv=0;
  for (int id=0; id<Ndiags; id++){
    for (int i=i_start_fermi; i<2*Norder; i++){
      int i_final = diagsG(id,i);
      bl::TinyVector<int,3> K = mom_G(id,i);
      bl::TinyVector<int,3> dt = times(i_final)-times(i);
      Key kt( K(0), K(1), K(2), dt(0), dt(1), dt(2) );
      if (Gkt.find(kt)!=Gkt.end()){
	int ii = Gkt[kt];
	if (ii >= std::numeric_limits<sint>().max()){
	  log<<"ERROR : You must change the type for Gindx, because the needed value is larger than allowed by this short int type"<<std::endl;
	  exit(1);
	}
	Gindx(id,i) = ii;
      }else{
	Gkt[kt]=ig;
	if (ig >= std::numeric_limits<sint>().max()){
	  log<<"ERROR : You must change the type for Gindx, because the needed value is larger than allowed by this short int type"<<std::endl;
	  exit(1);
	}	  
	Gindx(id,i) = ig;
	ig += 1;
      }
    }
    for (int i=1; i<Norder; i++){ // do not add V for the meassuring line
      bl::TinyVector<double,3> Q = mom_V(id,i);
      key Qt( Q(0), Q(1), Q(2), Vtype(id,i) );
      if (Vk.find(Qt)!=Vk.end()){
	Vindx(id,i) = Vk[Qt];
      }else{
	Vk[Qt] = iv;
	Vindx(id,i) = iv;
	iv += 1;
      }
    }
  }

  int Ng = Gkt.size();
  gindx.resize(Ng);
  for (int ii=0; ii<gindx.extent(0); ++ii) gindx(ii) = -1,-1;
  
  for(int ii=0; ii<Ng; ++ii){
    if (debug) log<<"G "<<ii<<" : ";
    for (int id=0; id<Ndiags; id++){
      for (int i=0; i<2*Norder; i++){
	if ( static_cast<int>(Gindx(id,i))==ii){
	  if (gindx(ii)[0]<0) gindx(ii) = id, i;
	  if (debug) log<<"("<<id<<","<<i<<") ";
	}
      }
    }
    if (debug) log<<std::endl;
  }
  if (debug) log<<std::endl;

  int Nv = Vk.size();
  vindx.resize(Nv);
  for (int ii=0; ii<vindx.extent(0); ++ii) vindx(ii) = -1,-1;
  for(int ii=0; ii<Nv; ++ii){
    if (debug) log<<"V "<<ii<<" : ";
    for (int id=0; id<Ndiags; id++){
      for (int i=0; i<Norder; i++){
	if ( static_cast<int>(Vindx(id,i))==ii){
	  if (vindx(ii)[0]<0) vindx(ii) = id,i;
	  if (debug) log<<"("<<id<<","<<i<<") ";
	}
      }
    }
    if (debug) log<<std::endl;
  }
  if (debug) log<<std::endl;
  
  if (debug){
    for (int id=0; id<Ndiags; id++){
      log<<std::setw(3)<<id<<" : ";
      for (int i=0; i<2*Norder; i++){
	log<<std::setw(3)<<static_cast<int>(Gindx(id,i))<<" ";
      }
      log<<std::endl;
    }
    for (int ii=0; ii<gindx.extent(0); ++ii){
      log<<"G "<<ii<<" : ("<<gindx(ii)[0]<<","<<gindx(ii)[1]<<")"<<std::endl;
    }
    for (int ii=0; ii<vindx.extent(0); ++ii){
      log<<"V "<<ii<<" : ("<<vindx(ii)[0]<<","<<vindx(ii)[1]<<")"<<std::endl;
    }
  }
}

void invert_indx(std::vector<std::vector<int>>& idiag, bl::Array<int,1>& indx)
{
  //for (int i=0; i<indx.extent(0); ++i) cout<<i<<" -> "<<indx(i)<<endl;
  std::map<int,std::deque<int>> widiag;
  for (int i=0; i<indx.extent(0); ++i){
    int k = indx(i);
    widiag[k].push_back(i);
  }
  idiag.resize(widiag.size());
  std::map<int,std::deque<int>>::const_iterator il=widiag.begin();
  int l=0;
  for (; il!=widiag.end(); ++il){
    idiag[l].resize(widiag[il->first].size());
    for (int i=0; i<widiag[il->first].size(); i++){
      int id = widiag[il->first][i];
      idiag[l][i] = id;
      indx(id)=l;
    }
    /*
    cout<<il->first<<" : ";
    for (int i=0; i<widiag[il->first].size(); i++){
      cout<<widiag[il->first][i]<<" ";
    }
    cout<<endl;
    */
    l+=1;
  }
  /*
  for (int l=0; l<idiag.size(); l++){
    cout<<l<<" : ";
    for (int i=0; i<idiag[l].size(); i++){
      cout<<idiag[l][i]<<" ";
    }
    cout<<endl;
  }
  for (int i=0; i<indx.extent(0); ++i) cout<<i<<" -> "<<indx(i)<<endl;
  */
}






template <typename sint, typename ushort>
void findCommonOccurence(bl::Array<sint,2>& Gindx, bl::Array<sint,2>& Vindx,
			 bl::Array<bl::TinyVector<int,2>,1>& gindx, bl::Array<bl::TinyVector<int,2>,1>& vindx,
			 const bl::Array<ushort,2>& diagsG,
			 const std::vector<std::vector<std::vector<int> > >& Loop_index, const std::vector<std::vector<std::vector<int> > >& Loop_type)
{

  int Ndiags = diagsG.extent(0);
  int Norder = diagsG.extent(1)/2;
  int Nloops = Loop_index[0].size();

  bl::Array<int,1> Noccur(gindx.extent(0));
  Noccur = 0;
  for (int id=0; id<Ndiags; id++){
    for (int i=0; i<2*Norder; i++){
      int ii = Gindx(id,i);
      Noccur(ii) += 1;
    }
  }

  // initialize original index locations
  std::vector<size_t> idx(Noccur.extent(0));
  iota(idx.begin(), idx.end(), 0);

  // sort indexes based on comparing values in v
  sort(idx.begin(), idx.end(), [&Noccur](size_t i1, size_t i2) {return Noccur(i1)>Noccur(i2);});

  std::map<long,std::pair<int,int>> CommonMomenta;
  for (int j=0; j<Noccur.extent(0); j++){
    std::cout<<std::setw(3)<<j<<" "<<std::setw(3)<<idx[j]<<" "<<std::setw(5)<<Noccur(idx[j])<<"  : ";

    int ii = idx[j];         // this gives unique G-propagtor. This propagator has certain momentum and certain time argument.
    int id = gindx(ii)[0];   // For now we are interested only what is the momentum variable, i.e., k5-k2, etc...
    int ig = gindx(ii)[1];   // So we need to find proper Feyn-diagram==id and propagator ig.

    BitSet a; // will contain the momentum combination, like k5-k3, k4-k1, ...
    for (int iloop=0; iloop<Nloops; iloop++){                  // now we go over all loops of this diagram
      const std::vector<int>& lindex = Loop_index[id][iloop];
      const std::vector<int>& ltype  = Loop_type[id][iloop];
      for (int il=0; il<lindex.size(); il++){
	int ltype_i = ltype[il];
	int lindex_i = lindex[il];
	if ( abs(ltype_i)==1 && ig==lindex_i){                 // g-propagator in this diagram has propagator with exactly this momentum and time variable.
	  //mom_G(id, lindex_i) += momentum(iloop) * sign(ltype_i);
	  if (ltype_i>0) std::cout<<"+"<<iloop<<", ";
	  else std::cout<<"-"<<iloop<<", ";
	  a.Add(sign(ltype_i), iloop);                        // now we know which momentum and its sign, hence storing it in BitSet
	}
      }
    }
    std::cout<<std::endl;
    // Now bitset contains information of what is the momentum, i.e, G(k5-k3,...) contains k5-k3 combination
    if (a.how_many_values()>1){ // We skip arguments, which contain single momentum (like k1 ), because they will be added anyway
      if ( CommonMomenta.find(a) == CommonMomenta.end() ) { // does this combination of momenta already exists?
	CommonMomenta[a] = std::pair<int,int>(ii,Noccur(idx[j]));  // propagators can have the same momentum, but different time variables, and we want to know how many are those with the same momentum
      } else {  // keep the same index ii (for G-propagator), but update the number of occurences.
	CommonMomenta[a] = std::make_pair( CommonMomenta[a].first, CommonMomenta[a].second+Noccur(idx[j]) );
      }
    }
  }

  using mypair = std::pair<long,std::pair<int,int>>;
  std::vector<mypair> v(begin(CommonMomenta), end(CommonMomenta));
  sort(begin(v), end(v), [](const mypair& a, const mypair& b) { return a.second.second > b.second.second; });
  std::cout<<"How often: "<<std::endl;
  for(auto const &p : v)
    std::cout << "m[" << BitSet(p.first).repr() << "] = " << p.second.first << " : "<< p.second.second << std::endl;
  
  //  for (auto it=CommonMomenta.begin(); it!=CommonMomenta.end(); ++it){
  //    std::cout<< BitSet(it->first).repr() << " "<< it->second << std::endl;
  //  }
  exit(0);
}
