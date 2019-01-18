// @Copyright 2018 Kristjan Haule and Kun Chen    
#include <iostream>
#include <blitz/array.h>
#include <deque>
#include <vector>
#include <algorithm>

void findBaymKadanoffGroups(bl::Array<int,1>& BKgroups, bl::Array<unsigned short,1>& BKindex,
			    const bl::Array<unsigned short,2>& diagsG,
			    const std::vector<std::vector<std::vector<int> > >& Loop_index, const std::vector<std::vector<std::vector<int> > >& Loop_type)
{
  int Ndiags = diagsG.extent(0);
  int Nloops = Loop_index[0].size();
  std::deque<int> bkgroups;
  std::deque< std::deque<int> > how_many;
  for (int id=0; id<Ndiags; id++){
    std::deque<int> k_momenta;
    for (int iloop=0; iloop<Nloops; ++iloop){
      const std::vector<int>& lindex = Loop_index[id][iloop];
      const std::vector<int>& ltype  = Loop_type[id][iloop];
      for (int i=0; i<lindex.size(); i++){
	// this shows you which momenta contribute to the zeroth propagator, and what is the loops orientation.
	// Note that external momentum (iloop==0) can have only a single orientation (i.e.,negative) and therefore 0 can serve the purpose
	if ( abs(ltype[i])==1  && lindex[i]==0 )
	  k_momenta.push_back( sign(ltype[i])*iloop );
      }
    }
    auto where = std::find(how_many.begin(), how_many.end(), k_momenta );
    int iwhere = how_many.size();
    if (where == how_many.end()){
      how_many.push_back(k_momenta);
      bkgroups.push_back(id);
    }else{
      iwhere = distance(how_many.begin(),where);
    }
    BKindex(id) = iwhere;
    //std::cout << "id="<<id <<" "<< "iwhere="<<iwhere << " ";
    //for (int i=0; i<k_momenta.size(); i++) std::cout<< k_momenta[i]<<",";
    //std::cout << std::endl;
  }
  BKgroups.resize(bkgroups.size());
  for (int i=0; i<bkgroups.size(); i++) BKgroups(i) = bkgroups[i];

  /*
  std::cout<<std::endl;
  std::cout<<"BKgroups:"<<std::endl;
  for (int i=0; i<BKgroups.extent(0); i++){
    std::cout<<i<<" "<<BKgroups(i)<<std::endl;
  }
  std::cout<<"BKindx:"<<std::endl;
  for (int i=0; i<BKindex.extent(0); i++){
    std::cout<<i<<" "<<BKindex(i)<<std::endl;
  }
  */
}
