// @Copyright 2018 Kristjan Haule 
void VerifyCurrentState10(int itt, double beta, double lmbda_spct, double PQ_original, const bl::Array<bl::TinyVector<double,3>,1>& momentum, bl::Array<double,1>& times,
			  //const bl::Array<bl::TinyVector<double,3>,1>& mom_g, const bl::Array<bl::TinyVector<double,3>,1>& mom_v,
			  const bl::Array<unsigned short,2>& diagsG, const bl::Array<unsigned short,2>& i_diagsG,
			  const bl::Array<char,2>& Vtype, const bl::Array<unsigned short,2>& Gindx, const bl::Array<unsigned short,2>& Vindx,
			  const Gk_HF& Gk, const bl::Array<float,1>& diagSign, const std::vector<std::vector<std::vector<int> > >& Loop_index,
			  const std::vector<std::vector<std::vector<int> > >& Loop_type, const bl::Array<int,1>& BKgroups,
			  const bl::Array<unsigned short,1>& BKindex, const CounterCoulombV& Vqc, 
			  const bl::Array<double,1>& G_current_orig, const bl::Array<double,1>& V_current_orig)
{
  // Momenta for all propagators in all graphs, which is needed to evaluate the diagrams.
  // We put this inside brackets, so that mom_G and mom_V are temporary variables, that do not waste memory.
  // We will use more optimized mom_g and mom_v below.
  int Ndiags = diagsG.extent(0);
  int Norder = diagsG.extent(1)/2;
  int Nloops = Loop_index[0].size();
  
  bl::Array<bl::TinyVector<double,3>,2> mom_G(Ndiags,2*Norder); // momentum for 2*Norder fermionic propagators
  bl::Array<bl::TinyVector<double,3>,2> mom_V(Ndiags,  Norder); // momentum for Norder bosonic propagators
  mom_G=0.0;
  mom_V=0.0;
  for (int id=0; id<Ndiags; id++){
    for (int iloop=0; iloop<Nloops; iloop++){
      const std::vector<int>& lindex = Loop_index[id][iloop];
      const std::vector<int>& ltype  = Loop_type[id][iloop];
      for (int i=0; i<lindex.size(); i++){
	int ltype_i = ltype[i];
	int lindex_i = lindex[i];
	if ( abs(ltype_i)==1 ){
	  mom_G(id, lindex_i) += momentum(iloop) * dsign(ltype_i);
	}else{
	  if (lindex_i>=Norder) std::cerr<<"ERROR : writting beyond boundary"<<std::endl;
	  mom_V(id,lindex_i) += momentum(iloop) * dsign(ltype_i);
	}
      }
    }
  }
  // START DEBUGGING
  for (int id=0; id<Ndiags; id++){
    for (int i=0; i<2*Norder; i++){
      //int i_final = diagsG(id,i);
      int i_previ = i_diagsG(id,i);
      bl::TinyVector<double,3> k_in = mom_G(id,i_previ);
      bl::TinyVector<double,3> k_out = mom_G(id,i);
      bl::TinyVector<double,3> q = mom_V(id,i/2)*(1.-2*(i%2));
      bl::TinyVector<double,3> dk = k_in-k_out;
      double n_dk = norm(dk);
      double n_q = norm(q);
      if (fabs(n_dk-n_q)>1e-6){
	std::cerr<<"ERROR: diagram "<<id<<" vertex i="<<i<<" k_in="<<k_in<<" k_out="<<k_out<<" q="<<q<<std::endl;
	std::cerr<<"   i_previ="<<i_previ<<" q_i="<< (i/2) <<" mom_V="<<mom_V(id,i/2)<<std::endl;
      }
    }
  }
  bl::Array<double,1> k_nrm(BKgroups.extent(0)), k_cth(BKgroups.extent(0));
  for (int ig=0; ig<BKgroups.extent(0); ig++){
    int id_representative = BKgroups(ig);
    bl::TinyVector<double,3> k = mom_G(id_representative,0);
    k_nrm(ig) = norm(k);
    k_cth(ig) = k(2)/k_nrm(ig);
  }
  for (int id=0; id<Ndiags; id++){
    bl::TinyVector<double,3> k = mom_G(id,0);
    double nk = norm(k);
    double cos_theta = k(2)/nk;
    int ig = BKindex(id);
    if (fabs(k_nrm(ig)-nk)>1e-7){std::cerr<<"ERROR : 1) It seems BKgroups or BKindex was not properly computed!"<<std::endl; exit(1);}
    if (fabs(k_cth(ig)-cos_theta)>1e-7){ std::cerr<<"ERROR : 2) It seems BKgroups or BKindex was not properly computed! "<<std::endl; exit(1);}
  }
  // STOP DEBUGGING
  // Finally evaluating the polarizations for all diagrams

  bl::Array<double,2> G_current(Ndiags,2*Norder), V_current(Ndiags,Norder);
  G_current=0;
  for (int id=0; id<Ndiags; id++){
    for (int i=0; i<2*Norder; i++){
      //int id = gindx(ii)[0];
      //int  i = gindx(ii)[1];
      double aK = norm(mom_G(id,i));
      int i_final = diagsG(id,i);
      double dt = times(i_final)-times(i);
      G_current(id,i) = Gk(aK, dt);
      if ( fabs(G_current(id,i)-G_current_orig(Gindx(id,i)))>1e-6)
	std::cerr<<"WARNING: G_current original and new are different: original="<<G_current_orig(Gindx(id,i))<<" G_new="<<G_current(id,i)<<std::endl;
    }
  }
  V_current=0;  
  for (int id=0; id<Ndiags; id++){
    for (int i=1; i<Norder; i++){
      //int id = vindx(ii)[0];
      //int  i = vindx(ii)[1];
      double aQ = norm(mom_V(id,i));
      double Vq = Vqc(aQ,abs(Vtype(id,i)));
      double g_kq=1;
      if (Vtype(id,i)>=0){
	V_current(id,i) = Vq;
      }else{// we also add the single-particle counter term in addition to two-particle counter term.
	if (diagsG(id,2*i)==2*i+1){ // G_propagator goes from 2j->2j+1
	  g_kq = G_current(id,2*i);     // G_propagtaors is stored in 2j
	} else if (diagsG(id,2*i+1)==2*i){ // alternatively it goes from 2j+1->2j
	  g_kq = G_current(id,2*i+1);   // and is hence stored in 2j+1
	}else{
	  std::cerr<<"ERROR : Should not happen as we should set Vtype to negative above"<<std::endl;
	}
	if (g_kq < 0) std::cerr<<"ERROR G_{k+q}(tau=0) should be positive, because it is density. It is "<<g_kq<<std::endl;
	V_current(id,i) = (Vq + lmbda_spct/g_kq);
      }
      if ( fabs( (V_current(id,i)-V_current_orig(Vindx(id,i)))/V_current(id,i) ) > 1e-6 )
	std::cerr<<"WARNING: V_current original and new are different: original="<<V_current_orig(Vindx(id,i))<<" V_new="<<V_current(id,i)<<std::endl;
    }
  }
  double PQ=0;
  for (int id=0; id<Ndiags; id++){
    double PQd = 1.0;
    for (int i=0; i<2*Norder; i++) PQd *= G_current(id,i);
    for (int i=1; i<Norder; i++) PQd *= V_current(id,i);
    PQd *= diagSign(id);
    PQ += PQd;
  }
  if (fabs(PQ_original-PQ)>1e-6){
    std::cerr<<"WARNING: PQ_original and PQ are different: original="<<PQ_original<<" while PQ="<<PQ<<std::endl;
  }
}


template<class GK>
void CarefullyVerifyCurrentState3(int itt, double beta, int Qweight, double PQ, const bl::Array<bl::TinyVector<double,3>,1>& momentum, const bl::Array<double,1>& times, const bl::Array<bl::TinyVector<double,3>,2>& mom_G,
				  const bl::Array<bl::TinyVector<double,3>,2>& mom_V, const bl::Array<int,2>& diagsG, const bl::Array<int,2>& i_diagsG, const GK& Gk, const bl::Array<double,1>& qx, const bl::Array<double,1>& taux, 
				  const Spline2D<double>& rW, const bl::Array<double,1>& diagSign, const std::vector<std::vector<std::vector<int> > >& Loop_index, const std::vector<std::vector<std::vector<int> > >& Loop_type,
				  const bl::Array<double,2>& G_current, const bl::Array<double,2>& V_current, const bl::Array<double,2>& W_current, const bl::Array<double,2>& V_eqv_current, const bl::Array<double,2>& Vrtx_current,
				  const bl::Array<double,2>& Vrtx_current2, const bl::Array<int,3>& diagVertex, const bl::Array<int,2>& pure_Exchange, const Spline1D<double>&rWom0,
				  const bl::Array<int,2>& vertex_companion, const bl::Array<int,2>& vertex_incoming, const bl::Array<double,1>& PQs)

{
  int Ndiags = diagsG.extent(0);
  int Norder = diagsG.extent(1)/2;
  //int Nloops = Loop_index[0].size();

  //std::cout<<"momentum="<<momentum<<std::endl;
  //std::cout<<"times="<<times<<std::endl;

  bl::Array<double,2> G_trial(Ndiags,2*Norder), V_trial(Ndiags,Norder), W_trial(Ndiags,Norder), Vrtx_trial(Ndiags, Norder), Vrtx_trial2(Ndiags,Norder);
  bl::Array<double,2> V_eqv_trial(Ndiags,Norder);
  
  int ncases = 1<<(Norder-1);
  bl::Array<double,1> ttimes(times.extent(0));
  G_trial=0; V_trial=0; W_trial=0; Vrtx_trial=0;Vrtx_trial2=0;
  V_eqv_trial=0;
  double PQ_new=0;
  for (int id=0; id<Ndiags; id++){
    //std::cout<<"------ diagram "<<id<<std::endl;
    double PQd = 1.0;
    for (int i=0; i<2*Norder; i++){
      double aK = norm(mom_G(id,i));
      int i_final = diagsG(id,i);
      double dt = times(i_final)-times(i);
      G_trial(id,i) = Gk(aK, dt);
      if (fabs(G_trial(id,i)-G_current(id,i))>1e-6){
	std::cout<<itt<<" diag="<<id<<" WARNING: G wrong at id="<<id<<" i="<<i<<" dt="<<dt<<" G_should="<<G_trial(id,i)<<" G_curr="<<G_current(id,i)<<std::endl;
      }
      //std::cout<<"G_c("<<i<<")="<<G_current(id,i)<<"  dt="<<dt<<" k="<<aK<<" eps="<<Gk.eps(aK)<<std::endl;
      PQd *= G_trial(id,i);
    }
    double PQd_GG=PQd;
    //std::cout<<"PQd is now "<<PQd<<std::endl;
    for (int ii=1; ii<Norder; ii++){
      int i     = diagVertex(id,ii-1,0);
      int itype = diagVertex(id,ii-1,1);
      double t1 =  times(i);
      double t2 =  times( (i/2)*2 + (1-i%2) ); // partner in the interaction, i.e., (2,3),(4,5)...
      int i_m = i_diagsG(id,i);
      double ti = times(i_m);
      int i_p = diagsG(id,i);
      double to = times(i_p);
      double ki = norm(mom_G(id,i_m));
      double ko = norm(mom_G(id,i));
      Vrtx_trial(id,ii) = MoveVertex(t2, t1, ti, to, ki, ko, Gk);
      if (itype==2){
	Vrtx_trial2(id,ii) = Vrtx_trial(id,ii); // saving regular vertex to another array
	if (vertex_incoming(id,ii-1)){
	  int i_m_partner = (i_m/2)*2 + (1-i_m%2);
	  double ti_new = times(i_m_partner);
	  Vrtx_trial(id,ii) = MoveVertex(t2, t1, ti_new, to, ki, ko, Gk);
	}else{
	  int i_p_partner = (i_p/2)*2 + (1-i_p%2);
	  double to_new = times(i_p_partner);
	  Vrtx_trial(id,ii) = MoveVertex(t2, t1, ti, to_new, ki, ko, Gk);
	}
	if (fabs(Vrtx_trial2(id,ii)-Vrtx_current2(id,ii))>1e-6){
	  std::cout<<itt<<" diag="<<id<<" WARNING: Vrtx wrong at id="<<id<<" i="<<i<<" t1="<<t1<<" t2="<<t2<<" V_should="<<Vrtx_trial(id,ii)<<" V_curr="<<V_current(id,ii)<<std::endl;
	}
	//std::cout<<"Vrtx2_c("<<ii<<")="<<Vrtx_current2(id,ii)<<std::endl;
      }
      if (fabs(Vrtx_trial(id,ii)-Vrtx_current(id,ii))>1e-6){
	std::cout<<itt<<" diag="<<id<<" WARNING: Vrtx wrong at id="<<id<<" i="<<i<<" t1="<<t1<<" t2="<<t2<<" V_should="<<Vrtx_trial(id,ii)<<" V_curr="<<V_current(id,ii)<<std::endl;
      }
      //std::cout<<"Vrtx_c("<<ii<<")="<<Vrtx_current(id,ii)<<std::endl;
    }

    //std::cout<<"----------------- 8 cases ----------------"<<std::endl;
    double PQr=0;
    // setting equal time for instantenous interaction
    for (int ic=0; ic<ncases; ic++){
      //std::cout<<"---------------- case "<<ic<<"  -------------"<<std::endl;
      ttimes = times;
      for (int ii=1; ii<Norder; ii++){
	int icase = (ic & (1<<(ii-1)));
	if (icase){
	  int i = diagVertex(id,ii-1,0);
	  int i_partner = (i/2)*2 + (1-i%2);
	  ttimes(i) = times(i_partner);
	}
      }
      //std::cout<<"times="<<times<<std::endl;
      //std::cout<<"ttimes="<<ttimes<<std::endl;
      double PQ0 = 1.0;
      for (int i=0; i<2*Norder; i++){
	double aK = norm(mom_G(id,i));
	int i_final = diagsG(id,i);
	double dt = ttimes(i_final)-ttimes(i);
	G_trial(id,i) = Gk(aK, dt);
	// std::cout<<"G_eqt("<<i<<")="<<G_trial(id,i)<<" k="<<aK<<" dt="<<dt<<std::endl;
	PQ0 *= G_trial(id,i);
      }
      // std::cout<<"Product of G's for equal times="<<PQ0<<std::endl;
      // std::cout<<"Product of G's for different times="<<PQd_GG<<std::endl;
      double PQ1 = PQd_GG;
      for (int ii=1; ii<Norder; ii++){
	int itype = diagVertex(id,ii-1,1);
	int icase = (ic & (1<<(ii-1)));
	if (itype==1 && icase)
	  PQ1 *= Vrtx_current(id,ii);
	if (itype==2){
	  int jj = vertex_companion(id,ii-1);
	  int jcase = (ic & (1<<(jj-1)));
	  if (icase && jcase){
	    PQ1 *= Vrtx_current(id,ii) * Vrtx_current(id,jj);
	  }else if (icase){
	    PQ1 *= Vrtx_current2(id,ii);
	  }else if (jcase){
	    PQ1 *= Vrtx_current(id,jj);
	  }
	}
	
      }
      //std::cout<<"With vertices it becomes "<<PQ1<<std::endl;
      if ( fabs((PQ0-PQ1)/PQ0)>1e-5){
	std::cout<<itt<<" diag="<<id<<" ERROR PQ0 and PQ1 different PQ0="<<PQ0<<" PQ1="<<PQ1<<std::endl;
      }
      /*
      for (int ii=1; ii<Norder; ii++){
	double Wi = W_current(id,ii);
	double vi = (V_eqv_current(id,ii) - Wi);
	std::cout<<"vi["<<ii<<"]="<<vi<<" Wi["<<ii<<"]="<<Wi<<" pure_Exchange["<<ii<<"]="<<pure_Exchange(id,ii)<<std::endl;
      }
      */
      double PQW=1;
      for (int ii=1; ii<Norder; ii++){
	int icase = (ic & (1<<(ii-1)));
	if (!pure_Exchange(id,ii)){
	  double Wi = W_current(id,ii);
	  double vi = (V_eqv_current(id,ii) - Wi);
	  PQW *= icase ? vi : Wi;
	}else{
	  PQW *= icase ? 0 : W_current(id,ii);
	}
      }
      //std::cout<<" PQ0="<<PQ0<<" PQ1="<<PQ1<<" PQW="<<PQW<<" PQ0*PQW*sign="<<(PQ0*PQW*diagSign(id))<<std::endl;
      PQr += PQ0*PQW*diagSign(id);
    }
    //std::cout<<"****** Results for diagram "<<id<<" PQs="<<PQs(id)<<" while PQs_new="<<PQr<<std::endl;
    if ( fabs((PQs(id)-PQr)/PQs(id)) > 1e-5){
      std::cout<<"ERROR at it="<<itt<<" and diagram "<<id<<" PQ_new("<<id<<")="<<PQr<<" while PQ("<<id<<")="<<PQs(id)<<std::endl;
    }
    PQ_new += PQr;
  }
  if ( fabs((PQ_new-PQ)/PQ)>1e-5){
    std::cout<<"ERROR at it="<<itt<<" PQ_new="<<PQ_new<<" while PQ="<<PQ<<std::endl;
  }
}



template<class GK>
void VerifyCurrentState(int itt, double beta, int Qweight, double PQ, const bl::Array<int,2>& diagsG, const bl::Array<int,2>& i_diagsG, const bl::Array<double,1>& diagSign,
			const GK& Gk, const CounterCoulombV& Vq, const std::vector<std::vector<std::vector<int> > >& Loop_index, const std::vector<std::vector<std::vector<int> > >& Loop_type,
			const bl::Array<bl::TinyVector<double,3>,1>& momentum, const bl::Array<double,1>& times,
			const bl::Array<bl::TinyVector<double,3>,2>& mom_G, const bl::Array<bl::TinyVector<double,3>,2>& mom_V,
			const bl::Array<double,1>& G_current, const bl::Array<double,1>& V_current,
			const bl::Array<int,2>& Gindx, const bl::Array<int,2>& Vindx, const bl::Array<int,2>& Vtype)

{
  int Ndiags = diagsG.extent(0);
  int Norder = diagsG.extent(1)/2;
  int Nloops = Loop_index[0].size();

  // PROBABLY NOT NEEDED
  bl::Array<bl::TinyVector<double,3>,2> tmom_G(Ndiags,2*Norder), tmom_V(Ndiags,  Norder); // momentum for Norder bosonic propagators
  tmom_G=0.0;
  tmom_V=0.0;
  for (int id=0; id<Ndiags; id++){
    for (int iloop=0; iloop<Nloops; iloop++){
      const std::vector<int>& lindex = Loop_index[id][iloop];
      const std::vector<int>& ltype  = Loop_type[id][iloop];
      for (int i=0; i<lindex.size(); i++){
	if ( abs(ltype[i])==1 ){
	  tmom_G(id, lindex[i]) += momentum(iloop) * dsign(ltype[i]);
	}else{
	  tmom_V(id,lindex[i]) += momentum(iloop) * dsign(ltype[i]);
	}
      }
    }
  }
  for (int id=0; id<Ndiags; id++){
    for (int i=0; i<2*Norder; i++){
      double aK = norm(tmom_G(id,i)-mom_G(id,i));
      if (fabs(aK)>1e-5) std::cerr<<itt<<" WARNING: G-Momenta for id="<<id<<" i="<<i<<" are not correct k_should="<<tmom_G(id,i)<<" k_exist="<<mom_G(id,i)<<std::endl;
    }
    for (int i=0; i<Norder; i++){
      double aQ = norm(tmom_V(id,i)-mom_V(id,i));
      if (fabs(aQ)>1e-5) std::cerr<<itt<<" WARNING: V-Momenta for id="<<id<<" i="<<i<<" are not correct Q_should="<<tmom_V(id,i)<<" Q_exist="<<mom_V(id,i)<<std::endl;
    }
  }
  for (int id=0; id<Ndiags; id++){
    for (int i=0; i<2*Norder; i++){
      //int i_final = diagsG(id,i);
      int i_previ = i_diagsG(id,i);
      bl::TinyVector<double,3> k_in = mom_G(id,i_previ);
      bl::TinyVector<double,3> k_out = mom_G(id,i);
      bl::TinyVector<double,3> q = mom_V(id,i/2)*(1.-2*(i%2));
      if (norm(k_in-k_out-q)>1e-10)
	if (norm(k_in-k_out+q)>1e-10)
	  std::cerr<<itt<<" WARNING: diagram id="<<id<<" vertex i="<<i<<" k_in="<<k_in<<" k_out="<<k_out<<" q="<<q<<std::endl;
    }
  }
  // PROBABLY NOT NEEDED
  
  bl::Array<double,2> G_trial(Ndiags,2*Norder), V_trial(Ndiags,Norder);
  G_trial=0; V_trial=0;
  double PQ_new=0;
  for (int id=0; id<Ndiags; id++){
    double PQd = 1.0;
    for (int i=0; i<2*Norder; i++){
      double aK = norm(mom_G(id,i));
      int i_final = diagsG(id,i);
      double dt = times(i_final)-times(i);
      G_trial(id,i) = Gk(aK, dt);
      if (fabs(G_trial(id,i)-G_current(Gindx(id,i)))>1e-6){
	std::cerr<<itt<<" WARNING: G wrong at id="<<id<<" i="<<i<<" dt="<<dt<<" G_should="<<G_trial(id,i)<<" G_curr="<<G_current(Gindx(id,i))<<std::endl;
      }
      PQd *= G_trial(id,i);
    }
    for (int i=1; i<Norder; i++){ // do not add V for the meassuring line
      double aQ = norm(mom_V(id,i));
      V_trial(id,i) = Vq(aQ, Vtype(id,i));
      if (fabs(V_trial(id,i)-V_current(Vindx(id,i)))>1e-6){
	std::cerr<<itt<<" WARNING V_q wrong at id="<<id<<" i="<<i<<" V_should="<<V_trial(id,i)<<" V_curr="<<V_current(Vindx(id,i))<<std::endl;
      }
      PQd *= V_trial(id,i);
    }
    PQ_new += PQd * diagSign(id);
  }
  if (!Qweight && fabs(PQ_new-PQ)/fabs(PQ)>1e-6){
    std::cerr<<itt<<" WARNING PQ_should="<<PQ_new<<" while PQ_curr="<<PQ<<std::endl;
  }
}


