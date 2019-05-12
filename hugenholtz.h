// @Copyright 2018 Kristjan Haule 
void Get_GVind_Hugenholtz(bl::Array<unsigned short,1>& hh_indx, int& nhid,
			  bl::Array<bl::Array<unsigned short,1>,1>& hugh_diags,
			  bl::Array<bl::Array<unsigned short,1>,1>& loop_Gkind, bl::Array<bl::Array<char,1>,1>& loop_Gksgn,
			  bl::Array<bl::Array<unsigned short,1>,1>& loop_Vqind, bl::Array<bl::Array<char,1>,1>& loop_Vqsgn,
			  bl::Array<bl::Array<unsigned short,1>,1>& loop_Vqind2, bl::Array<bl::Array<char,1>,1>& loop_Vqsgn2,
			  int& Ngp, int& Nvp,  int& Nvp2,
			  const bl::Array<unsigned short,2>& diagsG, const bl::Array<unsigned short,2>& i_diagsG,
			  const std::vector<std::vector<std::vector<int> > >& Loop_index, const std::vector<std::vector<std::vector<int> > >& Loop_type,
			  const bl::Array<unsigned short,2>& Gindx, const bl::Array<unsigned short,2>& Vindx,
			  const bl::Array<int,1>& single_counter_index, const std::vector<double>& lmbda_spct,
			  const bl::Array<char,2>& Vtype,
			  bool Print=false,
			  bool dynamic_counter=false,
			  ostream& log=std::cout)
{
  hh_indx = 0;
  nhid=0;
  int Ndiags = diagsG.extent(0);
  int Norder = diagsG.extent(1)/2;
  int Nloops = hugh_diags.extent(0); // hugh_diags(Nloops)(some_diagrams);
  std::vector<std::set<unsigned short>> _hugh_diags_(Nloops);
  if (Print) log << "Vq2 : "<<std::endl;
  bl::Array<vecList,2> Vqh2(Ndiags,Norder);
  for (int id=0; id<Ndiags; id++){
    bl::Array<vecList,1> Gkh(2*Norder);
    bl::Array<vecList,1> Vqh(Norder);
    for (int iloop=0; iloop<Nloops; iloop++){
      const std::vector<int>& lindex = Loop_index[id][iloop];
      const std::vector<int>& ltype  = Loop_type[id][iloop];
      for (int i=0; i<lindex.size(); i++){
	int ltype_i = ltype[i];
	int lindex_i = lindex[i];
	if ( abs(ltype_i)==1 ){
	  Gkh(lindex_i).add(iloop, sign(ltype_i));
	}else{
	  if (lindex_i>=Norder) log<<"ERROR : writting beyond boundary"<<std::endl;
	  Vqh(lindex_i).add(iloop, sign(ltype_i) );
	}
      }
    }
    bl::Array<vecList,1> Vqh1(Norder);
    bool isHugenholtz=false;
    for (int i=0; i<Norder; i++)
      if ( abs(Vtype(id,i))>=10 )
	isHugenholtz=true;
    for (int i=0; i<Norder; i++){
      if (abs(Vtype(id,i))>=10){
	int i_previ = i_diagsG(id,2*i);
	vecList Vqh1 = Gkh(i_previ) - Gkh(2*i); // momentum in the absence of hugenholtz : k(i_G(2*i))-k(G(2*i))
	vecList dVq = Vqh1 - Vqh(i);            // while hugenholtz momentum is          : k(G(2*i+1))-k(i_G(2*i))
	if (dVq.m.size()!=0) log << "WARNING expecting Vqh0="<< Vqh(i) << " == "<< Vqh1 <<" but they are different"<<endl;
	Vqh2(id,i) = Gkh(2*i+1) - Gkh(i_previ);
      }
    }
    if (isHugenholtz){
      for (int i=0; i<Norder; i++){
	if (abs(Vtype(id,i))>=10){

	  // THIS MUST BE WRONG => BUG IN March 2019
	  //_hugh_diags_[i].insert(id); // This is because V12(0) is changed

	  for (auto j=Vqh(i).m.begin(); j!=Vqh(i).m.end(); ++j){ // This is because V12(0) is changed.
	    _hugh_diags_[j->first].insert(id); // BUG corrected in March 2019. 
	    //log<<" adding id="<<id<<" to loop"<<j->first<<" because of V12(0)"<<endl;
	  }
	  
	  for (auto j=Vqh2(id,i).m.begin(); j!=Vqh2(id,i).m.end(); ++j){ // This is because V12(1) is changed.
	    _hugh_diags_[j->first].insert(id);
	    //log<<" adding id="<<id<<" to loop"<<j->first<<" because of V12(1)"<<endl;
	  }
	}
      }
      hh_indx(id) = nhid;
      nhid++;
      if (Print){
	log << "id="<<setw(3)<<id<<" (";
	for (int i=0; i<2*Norder; i++) log << diagsG(id,i) <<",";
	log << ") ; (";
	for (int i=0; i<Norder; i++) log <<setw(2)<<static_cast<int>(Vtype(id,i)) <<",";
	log<< ") : ";
	for (int i=0; i<Norder; i++){
	  if (abs(Vtype(id,i))>=10) log << Vqh2(id,i) << "| ";
	  else log << "   |";
	}
	log << std::endl;
      }
    }
    for (int iloop=0; iloop<Nloops; iloop++){
      hugh_diags(iloop).resize(_hugh_diags_[iloop].size());
      int i=0;
      for (auto it=_hugh_diags_[iloop].begin(); it!=_hugh_diags_[iloop].end(); ++it,++i) hugh_diags(iloop)(i)=(*it);
    }
  }
  // Find index-tables to all momentum loops, so that we do not need to loop over all diagrams, but rather only over all differet propagators corresponding to certain loop.
  // Here we also take into account that single-particle counter terms are sometimes added.
  Get_GVind(loop_Gkind, loop_Gksgn, loop_Vqind, loop_Vqsgn, loop_Vqind2, loop_Vqsgn2, Ngp, Nvp, Nvp2, Vqh2, Loop_index, Loop_type, Gindx, Vindx, single_counter_index, lmbda_spct, Vtype, false, log);
}

double Compute_V_Hugenholtz(const bl::Array<double,2>& V12, int nvh, const Tsign& sign, bl::Array<double,1>& Vtree, ostream& log)
{
  int nall = (1<<(nvh+1)) -1;
  int nleaves = 1<<nvh;
  int first_leaf = nleaves-1;
  //log << "nvh="<< nvh << " nall="<< nall << "  nleaves=" << nleaves << endl;
  //log<< "V12=";
  //for (int i=0; i<nvh; i++) log <<  "("<<V12(i,0)<<", "<< V12(i,1) << ") ";
  //log << endl;
  //log << "signs=";
  //for (int i=0; i<nleaves; i++) log << sign[i] << ", ";
  //log << endl;
  if (sign.size()!=nleaves) log << "ERROR in Compute_V_Hugenholtz diagSign.size="<< sign.size() << " while nleaves=" << nleaves << " at nvh=" << nvh << endl;
  if (Vtree.extent(0) < nall ) log << "ERROR in Compute_V_Hugenholtz Vtree is too small : Vtree.size="<< Vtree.extent(0) << " while nall="<< nall << endl;
  double result=0;
  Vtree(0)=1.0;
  for (int ilevel=1; ilevel<=nvh; ilevel++){
    for (int i=(1<<ilevel)-1; i<(1<<(ilevel+1))-1; i++){
      if (i%2==1) // left son
	Vtree(i) = Vtree((i-1)/2) * V12(ilevel-1,0);
      else // right son
	Vtree(i) = Vtree((i-1)/2) * V12(ilevel-1,1);
      if (ilevel==nvh){
	result += Vtree(i) * sign[i-first_leaf];
      }
    }
  }
  //log<< "result=" << result << endl;
  return result;
}
