// @Copyright 2018 Kristjan Haule 
#define CNTR_VN

// The latest version based on "sample_static_DiscreteC_combined"
template<typename GK, typename Data>
void sample_combined_wBK(Data& BKdata, bl::Array<double,2>& Pbin, bl::Array<double,2>& Pq0, std::ostream& log, const bl::Array<double,1>& qx,
			 double lmbda, const std::vector<double>& lmbda_spct, const GK& Gk, const params& p,
			 const bl::Array<unsigned short,2>& diagsG,
			 const TdiagSign& diagSign,
			 const std::vector<std::vector<std::vector<int> > >& Loop_index, const std::vector<std::vector<std::vector<int> > >& Loop_type,
			 bl::Array<char,2>& Vtype, my_mpi& mpi)
{
  typedef double real;
  log.precision(12);
  
  if (mpi.rank==mpi.master){
    for (int id=0; id<diagSign.size(); id++){
      log << setw(3) << id <<" : (";
      for (int j=0; j<diagsG.extent(1); j++) log << diagsG(id,j)<<",";
      log << ")  : ";
      for (int j=0; j<diagSign[id].size(); j++) log << diagSign[id][j]<<", ";
      log<<endl;
    }
  }
  bool Q0w0 = BKdata.Q0w0;
  double Q_external = qx(0);
  if (Q0w0){
    if (mpi.rank==mpi.master) log << "Q_external=" << Q_external << endl;
    if (Pbin.extent(0)!=p.Nt) Pbin.resize(p.Nt,1);
  } else {
    if ( BKdata.Nq!=qx.extent(0)){
      log<<"ERROR : Dimensions of C_Pln is wrong : either "<<BKdata.Nlt<<" != "<<p.Nlt<<" or "<<BKdata.Nq<<" != "<<qx.extent(0)<<std::endl;
      exit(1);
    }
    if (Pbin.extent(0)!=p.Nt || Pbin.extent(1)!=qx.extent(0)) Pbin.resize(p.Nt,qx.extent(0));
    if (mpi.rank==mpi.master && BKdata.Nlt!=p.Nlt){
      log<<"WARNING : Dimensions BKdata.Nlt ("<<BKdata.Nlt<<") != "<<p.Nlt<<"("<<p.Nlt<<") "<<std::endl;
    }
  }
  Pbin=0.0;
  
  const int Nr=3; // three different steps in this algorithm
  double Pr_norm = (p.Pr[0]+p.Pr[1]+p.Pr[2]); // probability for all four has to be normalized
  bl::TinyVector<double,Nr> Prs;  // cumulative probability for taking a step, needed for bisection below.
  Prs[0] = p.Pr[0]/Pr_norm;
  for (int i=1; i<Nr; i++) Prs[i] = Prs[i-1] + p.Pr[i]/Pr_norm; // prepares cumulative probability from p.Pr

  int Ndiags = diagsG.extent(0);
  if (Ndiags==0) {log<<"No diagrams to simulate...."<<std::endl; return;}
  int Norder = diagsG.extent(1)/2;
  // computing the inverse of all permutations, to provide fast lookup for the previous fermionic propagator
  bl::Array<unsigned short,2> i_diagsG(Ndiags,2*Norder);
  for (int id=0; id<Ndiags; id++)
    for (int i=0; i<2*Norder; i++)
      i_diagsG(id, static_cast<int>(diagsG(id,i)) )=i;
  
  RanGSL drand(p.iseed); // GSL random number generator
  Timer t_mes, t_all, t_Pl, t_k, t_Q, t_w, t_t;
  //Timer t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14;
  
  vector<int> times_2_change;
  std::set<int> times_to_change;
  int Vcmax=0;
  {
    //std::set<int> counter_types;
    for (int i=0; i<Norder; i++) times_to_change.insert(2*i);
    // default vertices to change time will be : [0,2,...,2*(Norder-1)]
    times_2_change.resize(times_to_change.size());
    int j=0;
    for (auto it=times_to_change.begin(); it!=times_to_change.end(); ++it, ++j) times_2_change[j] = (*it);
    if (mpi.rank==mpi.master){
      log<<"times_to_change : ";
      for (auto it=times_to_change.begin(); it!=times_to_change.end(); ++it) log << (*it) <<",  ";
      log<<std::endl;
      log << "times_2_change: "<<endl;
      for (int i=0; i<times_2_change.size(); i++) log << times_2_change[i] << ",";
      log << endl;
    }
  }
  
  int tmeassure=p.tmeassure, Ncout=p.Ncout, Nwarm=p.Nwarm;
  double V0norm = p.V0norm, dkF = p.dkF, kF=p.kF;
  double cutoffk = p.cutoffk, dRk = p.dRk;
  double beta = p.beta;
  double cutoffq = p.cutoffq;
  int Nt = p.Nt, Nq = p.Nq;


  // Everything for Legendre Polynomials in time and momentum
  LegendrePl Pln;
  //bl::Array<double,1> pl_Q;  // Will contain legendre Pl(2*q/qmax-1)
  bl::Array<double,1> pl_t(1);  // Will contain legendre Pl(2*t//beta-1)
  pl_t(0)=1.0; // for Q0w0 it should be 1, because om=0.
  if (!Q0w0){
    Pln.resize(BKdata.Nlt);
    //pl_Q.resize(BKdata.Nlq+1);  // Will contain legendre Pl(2*q/qmax-1)
    pl_t.resize(BKdata.Nlt+1);  // Will contain legendre Pl(2*t//beta-1)
  }
  BKdata.C_Pln2 = 0.0;            // cummulative result for legendre expansion
  BKdata.C_Pln1 = 0.0;            // cummulative result for legendre expansion
  BKdata.C_Pln0 = 0.0;            // cummulative result for legendre expansion
  
  double Pnorm = 0.0;            // normalization diagram value
  
  // Here we set Vtype(id,i) => -Vtype(id,i) for those interactions which contain single-particle counter terms.
  Where_to_Add_Single_Particle_Counter_Term(Vtype, lmbda_spct, Ndiags, Norder, diagsG);
  // Now that Vtype is properly adjusted for single-particle counter term, we can proceed to find unique propagators.
  typedef unsigned short sint;
  bl::Array<sint,2> Gindx, Vindx;
  bl::Array<bl::TinyVector<int,2>,1> gindx, vindx;
  // Finding Unique propagators for G and V
  bool debug=true;
  int N0v=0;
  findUnique_noindx_spct(Gindx, Vindx, gindx, vindx, N0v, diagsG, Vtype, Loop_index, Loop_type, 0, false, mpi.rank==mpi.master, log);

  // Once more we go through single-particle counter terms and create index single_counter_index
  bl::Array<int,1> single_counter_index;
  Find_Single_Particle_Counter_G_index(single_counter_index, Ndiags, Norder, vindx.extent(0), diagsG, Vtype, Gindx, Vindx);
  
  // This is related to Baym-Kadanoff approach. For efficiency, we want to find which diagram have common values of momenta at the right hand side of the diagram.
  BKdata.FindGroups(diagsG, Loop_index, Loop_type, log, mpi.rank==mpi.master, Q_external);
  
  if (Loop_index.size()!=Ndiags){ log<<"Wrong size of Loop_index "<<Loop_index.size()<<" instead of "<<Ndiags<<std::endl; exit(1);}
  int Nloops = Loop_index[0].size();
  if (Nloops != Norder+1){ log<<"Expecting Nloops==Norder+1 in Loops_index, but got Nloops="<<Nloops<<" Norder="<<Norder<<std::endl; exit(1);}
  int Ng = gindx.extent(0);
  int Nv = vindx.extent(0);
  int _Nv_ = std::max(Nv,1); // can not assign 0 to empty file

  bl::Array<bl::Array<sint,1>,1> loop_Gkind, loop_Vqind, loop_Vqind2;
  bl::Array<bl::Array<char,1>,1> loop_Gksgn, loop_Vqsgn, loop_Vqsgn2;
  int Ngp, Nvp, Nvp2;
  bl::Array<bl::Array<sint,1>,1> hugh_diags(Nloops);
  bl::Array<sint,1> hh_indx(Ndiags);

  int nhid=0;
  Get_GVind_Hugenholtz(hh_indx,nhid,hugh_diags,loop_Gkind,loop_Gksgn,loop_Vqind,loop_Vqsgn,loop_Vqind2,loop_Vqsgn2,Ngp,Nvp,Nvp2,diagsG,i_diagsG,Loop_index,Loop_type,Gindx,Vindx,single_counter_index,lmbda_spct,Vtype,mpi.rank==mpi.master,false,log);
  int Ngp_tr = GetMax_NGp_trial(i_diagsG, Gindx, Ng, Ngp);

  if (mpi.rank==mpi.master){
    log << "hugh_diags="<<endl;
    for (int iloop=0; iloop<Nloops; iloop++){
      log << "iloop="<< iloop << " ("<< setw(3)<< hugh_diags(iloop).extent(0) << "): ";
      for (int it=0; it<hugh_diags(iloop).extent(0); ++it) log << hugh_diags(iloop)(it) << ", "; 
      log << endl;
    }
    log << "hh_indx="<<endl;
    for (int id=0; id<Ndiags; id++){
      log << setw(2) << id << "  " << setw(2) << hh_indx(id) << endl;
    }
  }
  // DEBUG
  if (mpi.rank==mpi.master){
    bool debug=false;
    log<<"Single particle counter-terms : ";
    for (int i=0; i<lmbda_spct.size(); ++i) log<<lmbda_spct[i]<<",";
    log<<std::endl;
    for (int id=0; id<Ndiags; id++){
      for (int j=1; j<Norder; j++){
	if (Vtype(id,j)<0){
	  int ii_v = Vindx(id,j);
	  int ii_g = single_counter_index(ii_v);
	  log<<"id="<<std::setw(3)<<id<<" diag=(";
	  for (int k=0; k<Norder*2; k++) log<< diagsG(id,k) <<", ";
	  log<<")  V_between=("<<2*j<<","<<2*j+1<<") with ii_v="<<ii_v<<" and Vtype="<<static_cast<int>(Vtype(id,j))<<" and ii_g="<<ii_g<<" which comes from id="<<gindx(ii_g)[0]<<" and i=("<<gindx(ii_g)[1]<<"->"<<diagsG(gindx(ii_g)[0],gindx(ii_g)[1])<< ")"<<std::endl;
	}
      }
    }
    if (debug){
      log << "loop_Gkind:"<<std::endl;
      for (int iloop=0; iloop<Nloops; ++iloop){
	for (int ip=0; ip<loop_Gkind(iloop).extent(0); ++ip){ // all G-propagators, which are changed when momentum in certain loop is changed.
	  int ii   = loop_Gkind(iloop)(ip);
	  int isgn = loop_Gksgn(iloop)(ip);
	  int id = gindx(ii)[0];           // find first diagram with this propagator
	  int  i = gindx(ii)[1];           // and which propagator is this in the diagram
	  int i_final = diagsG(id,i);      // times in the propagators.
	  log<<"iloop="<<iloop<<" ip="<<setw(2)<<ip<<" ii_g="<<setw(2)<<ii<<" isgn="<<setw(2)<<isgn<<" which comes from id="<< setw(2)<<id <<" and pair =("<<i<<","<<i_final<<") "<<std::endl;
	}
      }
      log<<"loop_Vqind:"<<std::endl;
      for (int iloop=0; iloop<Nloops; ++iloop){
	for (int iq=0; iq<loop_Vqind(iloop).extent(0); ++iq){ // Now do the same for the interaction
	  int ii   = loop_Vqind(iloop)(iq);
	  int isgn = loop_Vqsgn(iloop)(iq);
	  int id = vindx(ii)[0];
	  int  i = vindx(ii)[1];
	  int vtyp = Vtype(id,i);
	  log<<"iloop="<<iloop<<" iq="<<setw(2)<<iq<<" ii_v="<<setw(2)<<ii<<" isgn="<<setw(2)<<isgn<<" which comes from id="<< setw(2)<<id <<" and pair =("<<2*i<<","<<2*i+1<<") of type="<<vtyp<<std::endl;
	}
      }
      log<<"loop_Vqind2:"<<std::endl;
      for (int iloop=0; iloop<Nloops; ++iloop){
	for (int iq=0; iq<loop_Vqind2(iloop).extent(0); ++iq){ // Now do the same for the interaction
	  int ii   = loop_Vqind2(iloop)(iq);
	  int isgn = loop_Vqsgn2(iloop)(iq);
	  int id = vindx(ii)[0];
	  int  i = vindx(ii)[1];
	  int vtyp = Vtype(id,i);
	  log<<"iloop="<<iloop<<" iq="<<setw(2)<<iq<<" ii_v="<<setw(2)<<ii<<" isgn="<<setw(2)<<isgn<<" which comes from id="<<setw(2)<<id <<" and pair =("<<2*i<<","<<2*i+1<<") of type="<<vtyp<<std::endl;
	}
      }
    }
  }
  // END DEBUG
  
  Reweight_time rw(beta,p.lmbdat);
  //int Nbin = 129;
  int Nbin = 513;
  bl::Array<double,1> lmbda_spct_cutoffk(lmbda_spct.size());
  // Instead of using : lmbda_spct, which is q-independent, we will introduce "artificial" q-dependence, which gives identical integral.
  // When lmbda_spct is q-independent, we should normalize it by  lmbda_spct * 1/Integrate[d^3 1/(2*pi)^3].
  // The "artificial" dependence we choose is lmbda_spct * (Vq)^2, in which case the normalization constant is
  //     lmbda_spct * (Vq)^2 / Integrate[d^3q Vq^2/(2*pi)^3 ]
  //  which is
  //     lmbda_spct * (Vq)^2 / ( 2/sqrt(lmbda) *atan(ck/sqrt(lmbda)) - 2*ck/(ck^2+lmbda) )
  std::vector<double> nrm = VNrm(lmbda, cutoffk, lmbda_spct.size()+2);
  for (int i=0; i<lmbda_spct.size(); ++i){
#ifdef CNTR_VN
    // [Vq^2, Vq^3, Vq^4,..]
    //lmbda_spct_cutoffk(i) = lmbda_spct[i]*nrm[i+2];
    lmbda_spct_cutoffk(i) = lmbda_spct[i]*nrm[2];
#else    
    // exp(-2*q/kF)
    lmbda_spct_cutoffk(i) = lmbda_spct[i] * pi*pi*8/(kF*kF*kF*(1. - exp(-2*cutoffk/kF)*ipower(cutoffk/kF,2)*(1. + ipower(1.+kF/cutoffk,2) )));
#endif    
    // Vq^2
    //double ck2 = cutoffk*cutoffk;
    //lmbda_spct_cutoffk(i) = lmbda_spct[i]/( 16./sqrt(lmbda)*atan(cutoffk/sqrt(lmbda)) - 16.*cutoffk/(cutoffk*cutoffk+lmbda));
    // Vq^4
    //lmbda_spct_cutoffk(i) = lmbda_spct[i]/( 128*pi*pi*(atan(cutoffk/sqrt(lmbda))/(lmbda*lmbda*sqrt(lmbda)) - cutoffk*(lmbda*lmbda-ck2*(ck2+8./3.*lmbda))/(lmbda*lmbda*ipower(ck2+lmbda,3))) );
    // 1.0
    //lmbda_spct_cutoffk(i) = lmbda_spct[i]*(2*pi*2*pi*2*pi)/(4*pi*ipower(cutoffk,3)/3.);
  }
  meassureWeight mweight(p.V0exp, p.cutoffk, kF, Nbin, Nloops);

  bl::Array<double,2> K_Ghist, K_Vhist;
  bool GetHistogram=false;
  if (GetHistogram){
    K_Ghist.resize(Ng,Nbin);
    K_Ghist=0;
    K_Vhist.resize(Nv,Nbin);
    K_Vhist=0;
  }
  
  double dk_hist = 1.0;
  
  // This is momentum for loops
  bl::Array<bl::TinyVector<double,3>,1> momentum(Nloops); // momentum for all loops. Each diagram of the same order has the same number of loops.
  bl::Array<double,1> amomentm(Nloops);               // absolute value of momentum for each loop
  
  // Starting point is set up to be random
  // For the times for all vertices in all the graphs
  bl::Array<double,1> times(2*Norder);
  times(0) = beta*drand(); // external time
  times(1) = 0;            // beginning time
  for (int it=1; it<Norder; it++){
    double tv = beta*drand();
    times(2*it  ) = tv;
  }
  for (int it=Norder-1; it>0; --it){
    if ( times_to_change.count(2*it+1) ) // 2*it+1 is in set times_to_change
      times(2*it+1) = beta*drand48();//beta*drand(); we use different random-number generator to compare results with static interaction code.
    else
      times(2*it+1) = std::nan("1");// so that we make sure it is newer referenced.
  }
  // Next the momenta. For a start with nonzero weight, we want to start close to kF.
  int iiQ=0; // the index of the external momentum in the mesh qx(iiQ);
  {
    //double Qa = Q0w0 ? Q_external : p.kF*drand(); // absolute value of external momentum
    iiQ = drand()*qx.extent(0);
    double Qa = qx(iiQ); // absolute value of external momentum
    momentum(0) = 0.0,0.0,Qa; // external momentum Q
    amomentm(0) = Qa;
    for (int ik=1; ik<Nloops; ik++){
      double th = pi*(1-drand()), phi = 2*pi*(1-drand());
      momentum(ik) = kF*sin(th)*cos(phi), kF*sin(th)*sin(phi), kF*cos(th);
      amomentm(ik) = kF;
    }
  }
  log.precision(12);
  if (mpi.rank==mpi.master) log<<"Starting times="<<times<<" and starting momenta="<<momentum<<std::endl;

  bl::Array<double,1> G_current(Ng);
  bl::Array<long double,1> V_current(_Nv_);
  bl::Array<long double,1> V_current2(_Nv_-N0v);
  // This is momentum for all diagrams and all propagators
  bl::Array<bl::TinyVector<real,3>,1> mom_g(Ng), mom_v(Nv), mom_v2(Nv-N0v);
  int nvh_max=0;
  for (int id=0; id<Ndiags; id++)
    nvh_max = std::max(nvh_max, static_cast<int>(diagSign[id].size()));
  nvh_max = log2(nvh_max);
  bl::Array<double,2> V12(nvh_max,2);
  bl::Array<double,1> Vtree( (1<<(nvh_max+1))-1 ); 
  bl::Array<double,1> V_Hugh(nhid);
  bl::Array<double,1> V_Hugh_trial(nhid);
  BitArray V_Hugh_changed(nhid);
    
  double PQ=0;
  {  
    // Momenta for all propagators in all graphs, which is needed to evaluate the diagrams.
    // We put this inside brackets, so that mom_G and mom_V are temporary variables, that do not waste memory.
    // We will use more optimized mom_g and mom_v below.
    bl::Array<bl::TinyVector<real,3>,2> mom_G(Ndiags,2*Norder); // momentum for 2*Norder fermionic propagators
    bl::Array<bl::TinyVector<real,3>,2> mom_V(Ndiags,  Norder); // momentum for Norder bosonic propagators
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
	    if (lindex_i>=Norder) log<<"ERROR : writting beyond boundary"<<std::endl;
	    mom_V(id,lindex_i) += momentum(iloop) * dsign(ltype_i);
	  }
	}
      }
    }
    // START DEBUGGING
    for (int id=0; id<Ndiags; id++){
      for (int i=0; i<Norder; i++){
	int i_previ = i_diagsG(id,2*i);
	bl::TinyVector<double,3> q = mom_G(id,i_previ) - mom_G(id,2*i);
	if (fabs(norm(q-mom_V(id,i)))>1e-6){
	  log<<"ERROR: diagram "<<id<<" vertex i="<<i<<" k_in="<<mom_G(id,i_previ)<<" k_out="<<mom_G(id,2*i)<<" q="<<q<<std::endl;
	  log<<"   i_previ="<<i_previ<<" q_i="<< (i/2) <<" mom_V="<<mom_V(id,i/2)<<std::endl;
	}
      }
    }
    BKdata.TestGroups(mom_G);
    
    // STOP DEBUGGING
    // Finally evaluating the polarizations for all diagrams
    G_current=0;
    for (int ii=0; ii<Ng; ++ii){
      int id = gindx(ii)[0];
      int  i = gindx(ii)[1];
      double aK = norm(mom_G(id,i));
      int i_final = diagsG(id,i);
      double dt = times( et(i_final) )-times( et(i) ); // BB 
      G_current(ii) = Gk(aK, dt);
    }
    //log<<"G_current="<<G_current<<endl;
    V_current=0;
    V_current2=0;
    for (int ii=0; ii<Nv; ++ii){
      int id = vindx(ii)[0];
      int  i = vindx(ii)[1];
      double aQ = norm(mom_V(id,i));
      double Vq = 8*pi/(aQ*aQ+lmbda);
      int vtyp = Vtype(id,i);
      int Nc = abs(vtyp % 10);
      V_current(ii) = (Nc==0) ? Vq : Vq * ipower(Vq * lmbda/(8*pi), Nc);
      if (vtyp>=10){ // Also Hugenholtz Vq2 needed
	bl::TinyVector<double,3> q2 = mom_G(id,2*i+1) - mom_G(id,static_cast<int>(i_diagsG(id,2*i)));
	mom_v2(ii-N0v) = q2;
	double aQ2 = norm(q2);
	double Vq2 = 8*pi/(aQ2*aQ2+lmbda);
	V_current2(ii-N0v) = (Nc==0) ? Vq2 : Vq2 * ipower(Vq2 * lmbda/(8*pi), Nc);
      }
      if (vtyp < 0){ // we also add the single-particle counter term in addition to two-particle counter term.
	int ii_g = single_counter_index(ii);
	long double g_kq = G_current(ii_g);
#ifdef CNTR_VN
	//long double Vn = ipower(Vq, Nc+1);
	long double Vn = ipower(Vq, 2);
	if (g_kq!=0) V_current(ii) += Vn*lmbda_spct_cutoffk(-vtyp-1)/g_kq;
#else	  
	if (g_kq!=0) V_current(ii) += exp(-aQ*2/kF)*lmbda_spct_cutoffk(-vtyp-1)/g_kq;
#endif	  
      }
    }
    //log<<"V_current="<< V_current <<endl<<"V_current2="<< V_current2 << endl;
    PQ=0;
    BKdata.PQg_Initialize();
    for (int id=0; id<Ndiags; id++){
      long double PQd = 1.0;
      for (int i=0; i<2*Norder; i++) PQd *= G_current(Gindx(id,i));
      int nvh=0;
      for (int i=1; i<Norder; i++){
	int ii=Vindx(id,i);
	if (Vtype(id,i)>=10){ // collect all Hugenholtz interactions in this diagram
	  V12(nvh,0) = V_current(ii);
	  V12(nvh,1) = V_current2(ii-N0v);
	  nvh++;
	}else{ // not Hugenholtz, than we can directly multiply
	  PQd *= V_current(ii);
	}
      }
      if (nvh>0){ // at least one interaction is Hugenholtz type. Than we need to compute the sum of 2^n Hugenholtz terms.
	double hh = Compute_V_Hugenholtz(V12, nvh, diagSign[id], Vtree, log);
	V_Hugh(hh_indx(id)) = hh;
	PQd *= hh;
      }else{  // not a single Hugenholtz, hence we can just multiply with the overal sign
	PQd *= diagSign[id][0];
      }
      PQ += PQd;
      BKdata.PQg_Add(id, PQd);
    }
    // Now we create smaller arrays mom_g and mom_v, which contain the same information as mom_G and mom_V.
    for (int id=0; id<Ndiags; id++){
      for (int i=0; i<2*Norder; i++){
	int ii = Gindx(id,i);
	mom_g(ii) = mom_G(id,i);
      }
      for (int i=1; i<Norder; i++){
	int ii = Vindx(id,i);
	mom_v(ii) = mom_V(id,i);
      }
    }
  }

  // START DEBUGGING
  for (int id=0; id<Ndiags; id++){
    for (int i=0; i<2*Norder; i++){
      if (fabs(G_current(Gindx(id,i)))==0){
	log<<"G_current("<<id<<","<<i<<") not set!"<<std::endl;
      }
    }
    for (int i=1; i<Norder; i++)
      if (fabs(V_current(Vindx(id,i)))==0){
	log<<"V_current("<<id<<","<<i<<") not set!"<<std::endl;
      }
  }
  // STOP DEBUGGING

  if (!Q0w0){
    // Now computing legendre polinomials in q and tau for the initial configuration
    //Pln.cmp_single(2*amomentm(0)/cutoffq-1., pl_Q, BKdata.Nlq);          // Legendre polynomials at this Q
    Pln.cmp_single(2*(times(0)-times(1))/beta-1., pl_t, BKdata.Nlt);     // Legendre polynomials at this time
  }


  bl::Array<double,1> times_trial(2*Norder);
  int _Nvp_ = std::max(Nvp,1); // can not assign 0 to empty file
  bl::Array<bl::TinyVector<real,3>,1> tmom_g(Ngp), tmom_v(_Nvp_+Nvp2);
  bl::Array<sint,1> ip_ind(Ng), iq_ind(Nv), iq_ind2(Nv-N0v);
  bl::Array<double,1> g_trial(Ngp_tr);
  bl::Array<long double,1> v_trial(_Nvp_+Nvp2);
  BitArray changed_G(Ng), changed_V(Nv), changed_V2(Nv-N0v);
  int Qweight = 0;      // if Qweight==1 then we have the normalization diagram
  double Nweight = 0;   // how many times did we see measuring diagram.
  int NlowT = 0;        // how many times do we see external time < 0.05*beta
  int Nmeassure = 0;    // how many times did we measure quantities.
  int Nchanged  = 0;    // when was V0norm last changed?
  double Wchanged = 0;  // what was Nweight when we last time changed V0norm?
  int Nacc_k = 0, Nacc_q = 0, Nacc_w = 0, Nacc_t = 0;
  int Nall_k = 0, Nall_q = 0, Nall_w = 0, Nall_t = 0;
  int iiQ_trial=0;
  if (mpi.rank==mpi.master){
    log<<"Toccurence="<<p.Toccurence<<endl;
    PrintInfo_(0, Qweight, amomentm(0), amomentm(1), times(0)-times(1), PQ, PQ, 0, 0, log);
  }
  //Check(0, log, -1, Nv, PQ, lmbda, times, times_to_change, Gk, G_current, V_current, diagsG, Gindx, Vindx, mom_g, mom_v, diagSign, Vtype, bPqt, qx, tx, vindx, Vrtx_current);

  //log << "tmom_v.size="<< tmom_v << " v_trial.size="<< v_trial.extent(0) << " iq_ind2.size="<< iq_ind2.extent(0) << "changed_V2="<< changed_V2.size() << endl;
  //CheckAccept(-1, log, i_diagsG, Vtype,  Vindx, mom_v2, mom_g, Gindx, N0v);
  t_all.start();
  for (int itt=0; itt<p.Nitt; itt++){
    double PQ_new=0;
    int icase = tBisect(drand(), Prs); // which steps should we attempt?
    if (icase==0){ // changing momentum Q or k-point
      int iloop = (!Q0w0) ? static_cast<int>(Nloops*drand()) : 1+static_cast<int>((Nloops-1)*drand());
      bl::TinyVector<double,3> K_new; double Ka_new; double trial_ratio=1;
      bool accept=false;
      //log << itt << " changing momentum " << iloop << endl;
      if (iloop==0){ // this is the loop for the external momentum Q
	Nall_q += 1;
	//accept = Find_new_Q_point(K_new, Ka_new, trial_ratio, momentum(iloop), amomentm(iloop), cutoffq, drand);
	accept = Find_new_discrete_Q_point(iiQ_trial, K_new, Ka_new, trial_ratio, momentum(iloop), amomentm(iloop), qx, drand);
      }else{         // any other momentum point
	Nall_k += 1;
	accept = Find_new_k_point(K_new, Ka_new, trial_ratio, momentum(iloop), amomentm(iloop), cutoffk, drand, drand()<p.Qring, dkF, dRk);
      }
      if (accept){
	if (!Qweight){
	  bl::TinyVector<double,3> dK = K_new - momentum(iloop); // how much momentum will be changed.
	  //t1.start(); // takes 20.3% of the time
	  changed_G = 0;
	  for (int ip=0; ip<loop_Gkind(iloop).extent(0); ++ip){ // all G-propagators, which are changed when momentum in certain loop is changed.
	    int ii   = loop_Gkind(iloop)(ip);  // index of G-propagators, which we are now changing
	    double isgn = loop_Gksgn(iloop)(ip);  // momentum can be added or subtracted on this propagator.
	    bl::TinyVector<double,3> k = mom_g(ii) + dK * isgn; // actually changing the momentum
	    tmom_g(ip) = k;                    // remember the momentum
	    changed_G.set(ii,1);               // remember that it is changed
	    double aK = norm(k);
	    int id = gindx(ii)[0];           // find first diagram with this propagator
	    int  i = gindx(ii)[1];           // and which propagator is this in the diagram
	    int i_final = diagsG(id,i);      // times in the propagators.
	    double dt = times( et(i_final) )-times( et(i) );
	    g_trial(ip) = Gk(aK, dt);        // remeber the value of the propagator
	    ip_ind(ii)=ip;                   // remeber where it is stored
	  }
	  //t1.stop();
	  //t2.start(); // takes 0.6% of the time
	  changed_V = 0;
	  for (int iq=0; iq<loop_Vqind(iloop).extent(0); ++iq){ // Now do the same for the interaction
	    int ii   = loop_Vqind(iloop)(iq);
	    double isgn = loop_Vqsgn(iloop)(iq);
	    bl::TinyVector<double,3> q =  mom_v(ii) + dK * isgn;
	    tmom_v(iq) = q;
	    changed_V.set(ii,1);
	    double aQ = norm(q);
	    int id = vindx(ii)[0];
	    int  i = vindx(ii)[1];
	    int vtyp = Vtype(id,i);
	    double Vq = 8*pi/(aQ*aQ+lmbda);
	    int Nc = abs(vtyp % 10);
	    v_trial(iq) = (Nc==0) ? Vq : Vq * ipower(Vq * lmbda/(8*pi),Nc);
	    if (vtyp < 0){ // single-particle is negative, but not Higenholtz. If Hugenholtz is negative, it means that we stored the diagram with more loops.
	      // we also add the single-particle counter term in addition to two-particle counter term.
	      int ii_g = single_counter_index(ii);
	      long double g_kq = changed_G[ii_g] ? g_trial(ip_ind(ii_g)) : G_current(ii_g);
#ifdef CNTR_VN		
	      //long double Vn = ipower(Vq, Nc+1);
	      long double Vn = ipower(Vq, 2);
	      if (g_kq!=0) v_trial(iq) += Vn*lmbda_spct_cutoffk(-vtyp-1)/g_kq;
#else		
	      if (g_kq!=0) v_trial(iq) += exp(-aQ*2/kF)*lmbda_spct_cutoffk(-vtyp-1)/g_kq;
#endif		
	    }
	    iq_ind(ii)=iq;
	  }
	  //t2.stop();
	  //t3.start(); // takes 0.7% of the time
	  int dq = loop_Vqind(iloop).extent(0);
	  changed_V2 = 0;
	  for (int iq=0; iq<loop_Vqind2(iloop).extent(0); ++iq){ // Now do the same for the Hugenholtz interaction for the second interaction
	    int ii   = loop_Vqind2(iloop)(iq);
	    double isgn = loop_Vqsgn2(iloop)(iq);
	    bl::TinyVector<double,3> q = mom_v2(ii-N0v) + dK * isgn;
	    tmom_v(iq+dq) = q;
	    changed_V2.set(ii-N0v,1); // we want to change only the second interaction, and not the first
	    double aQ = norm(q);
	    int id = vindx(ii)[0];
	    int  i = vindx(ii)[1];
	    int vtyp = Vtype(id,i);
	    if (vtyp<10){ log << "ERROR : We should have Hugenholtz here!" << std::endl; exit(1);}
	    double Vq = 8*pi/(aQ*aQ+lmbda);
	    int Nc = vtyp % 10;
	    v_trial(iq+dq) = (Nc==0) ? Vq : Vq * ipower(Vq * lmbda/(8*pi),Nc); // Note that Hugenholtz does not have single-particle counter-term.
	    iq_ind2(ii-N0v)=iq+dq;
	  }
	  //t3.stop();
	  // Hugenholtz diagrams are evaluated
	  //t14.start(); // takes 27% of time
	  V_Hugh_changed=0;
	  for (int iid=0; iid<hugh_diags(iloop).extent(0); iid++){
	    int id = hugh_diags(iloop)(iid);
	    int nvh=0;
	    for (int i=1; i<Norder; i++){
	      if (Vtype(id,i)>=10){// Hugenholtz
		int ii = Vindx(id,i);
		V12(nvh,0) = changed_V[ii]      ?  v_trial(iq_ind(ii))  : V_current(ii);
		V12(nvh,1) = changed_V2[ii-N0v] ?  v_trial(iq_ind2(ii-N0v)) : V_current2(ii-N0v);
		nvh++;
	      }
	    }
	    {
	      int ih = hh_indx(id);
	      V_Hugh_trial(ih) = Compute_V_Hugenholtz(V12, nvh, diagSign[id], Vtree, log);
	      V_Hugh_changed.set(ih,1);
	    }
	  }
	  //t14.stop();
	  // we computed the polarization diagram
	  //t4.start();// takes 13.6% of the time
	  PQ_new=0;
	  BKdata.PQg_new_Initialize();
	  for (int id=0; id<Ndiags; id++){
	    long double PQd = 1.0;
	    for (int i=0; i<2*Norder; i++){
	      int ii = Gindx(id,i);
	      PQd *= ( changed_G[ii] ? g_trial(ip_ind(ii)) : G_current(ii) );
	    }
	    bool isHugenholtz=false;
	    for (int i=1; i<Norder; i++){
	      if (Vtype(id,i)<10){// non-Hugenholtz
		int ii = Vindx(id,i);
		PQd *= ( changed_V[ii] ? v_trial(iq_ind(ii)) : V_current(ii) );
	      }else{
		isHugenholtz=true;
	      }
	    }
	    if (isHugenholtz){// It contains Hugenholtz type interactions
	      int ih = hh_indx(id);
	      double hh = V_Hugh_changed[ih] ? V_Hugh_trial(ih) : V_Hugh(ih);
	      PQd *= hh;
	    }else{// It is not Hugenholtz type
	      PQd *=  diagSign[id][0];
	    }
	    // DEBUGGING just to check if there is a problem
	    if (false){
	      //if (isHugenholtz && (itt%10000==0) ){ // DEBUG Check again carefully                                                                                                         
              long double PQt = 1.0;
              for (int i=0; i<2*Norder; i++){ // CAREFUL : density does not contain G(0)                                                                               
                int ii = Gindx(id,i);
                PQt *= ( changed_G[ii] ? g_trial(ip_ind(ii)) : G_current(ii) );
              }
              int nvh=0;
              for (int i=1; i<Norder; i++){
                int ii = Vindx(id,i);
                if (Vtype(id,i)<10){// non-Hugenholtz                                                                                                                  
                  PQt *= ( changed_V[ii] ? v_trial(iq_ind(ii)) : V_current(ii) );
                }else{
                  V12(nvh,0) = changed_V[ii]      ?  v_trial(iq_ind(ii))  : V_current(ii);
                  V12(nvh,1) = changed_V2[ii-N0v] ?  v_trial(iq_ind2(ii-N0v)) : V_current2(ii-N0v);
                  nvh++;
                }
              }
              if (nvh>0){
                PQt *= Compute_V_Hugenholtz(V12, nvh, diagSign[id], Vtree, log);
              }else{// It is not Hugenholtz type                                                                                                                       
                PQt *=  diagSign[id][0];
              }
              if (fabs(PQt-PQd)>1e-7){
                log<<"ERROR Computing the Hugenholtz diagram in two different ways leads to different result PQd="<<PQd<<" and PQt="<<PQt<<endl;
                exit(1);
              }
            }
	    
	    PQ_new += PQd;
	    BKdata.PQg_new_Add(id,PQd);
	  }
	  //t4.stop();
	}else{
	  //t5.start(); // takes 0.1%
	  double Ka_old = amomentm(iloop);
	  amomentm(iloop) = Ka_new;
	  bl::TinyVector<double,3> K_old = momentum(iloop);
	  momentum(iloop) = K_new;
	  PQ_new = V0norm * mweight(amomentm, momentum);
	  amomentm(iloop) = Ka_old;
	  momentum(iloop) = K_old;
	  //t5.stop();
	}
	double ratio = PQ_new*trial_ratio / PQ;
	accept = fabs(ratio) > 1-drand() && PQ_new!=0;
      }
      if ((itt+1)%Ncout==0 && mpi.rank==mpi.master) _PrintInfo_(itt+1, Qweight, icase, amomentm(1), amomentm(Nloops-1), times(0), PQ_new, PQ, dk_hist, Nweight/(Nmeassure+0.0), log);
      if (accept){
	if (iloop==0){
	  Nacc_q += 1;
	  iiQ = iiQ_trial;
	} else Nacc_k += 1;
	if (!Qweight){
	  //t6.start(); // takes 0.3%
	  momentum(iloop) = K_new;  // this momentum was changed
	  amomentm(iloop) = Ka_new;
	  for (int ip=0; ip<loop_Gkind(iloop).extent(0); ++ip){
	    int ii   = loop_Gkind(iloop)(ip); // index of G-propagators, which we are now changing
	    mom_g(ii) = tmom_g(ip);           // momentum has changed
	    G_current(ii)=g_trial(ip);        // all G-propagators, which are changed when momentum in certain loop is changed.
	  }
	  for (int iq=0; iq<loop_Vqind(iloop).extent(0); ++iq){ // Now do the same for the interaction
	    int ii   = loop_Vqind(iloop)(iq);
	    mom_v(ii) = tmom_v(iq);
	    V_current(ii) = v_trial(iq);
	  }
	  int dq = loop_Vqind(iloop).extent(0);
	  for (int iq=0; iq<loop_Vqind2(iloop).extent(0); ++iq){ // Now do the same for the interaction
	    int ii   = loop_Vqind2(iloop)(iq);
	    mom_v2(ii-N0v) = tmom_v(iq+dq);
	    V_current2(ii-N0v) = v_trial(iq+dq);//iq+dq==iq_ind2(ii-N0v);
	  }
	  for (int iid=0; iid<hugh_diags(iloop).extent(0); iid++){
	    int id = hugh_diags(iloop)(iid);
	    int ih = hh_indx(id);
	    V_Hugh(ih) = V_Hugh_trial(ih);
	  }
	  //t6.stop();
	}else{
	  //t7.start(); // takes 7.3%
	  // since we have the measuring diagram, we did not calculate G_trial and V_trial before, hence we need to evaluate them now.
	  bl::TinyVector<double,3> dK = K_new - momentum(iloop); // how much momentum will be changed.
	  momentum(iloop) = K_new;  // this momentum was changed
	  amomentm(iloop) = Ka_new;
	  // We have meassuring line, but we still need to update many quantities when accepting the step....
	  for (int ip=0; ip<loop_Gkind(iloop).extent(0); ++ip){ // all G-propagators, which are changed when momentum in certain loop is changed.
	    int ii   = loop_Gkind(iloop)(ip);  // index of G-propagators, which we are now changing
	    double isgn = loop_Gksgn(iloop)(ip);  // momentum can be added or subtracted on this propagator.
	    bl::TinyVector<double,3> k = mom_g(ii) + dK * isgn; // actually changing the momentum
	    mom_g(ii) = k;                    // remember the momentum
	    double aK = norm(k);
	    int id = gindx(ii)[0];           // find first diagram with this propagator
	    int  i = gindx(ii)[1];           // and which propagator is this in the diagram
	    int i_final = diagsG(id,i);      // times in the propagators.
	    double dt = times( et(i_final) )-times( et(i) );
	    G_current(ii) = Gk(aK, dt);        // remeber the value of the propagator
	  }
	  for (int iq=0; iq<loop_Vqind(iloop).extent(0); ++iq){ // Now do the same for the interaction
	    int ii   = loop_Vqind(iloop)(iq);
	    double isgn = loop_Vqsgn(iloop)(iq);
	    bl::TinyVector<double,3> q =  mom_v(ii) + dK * isgn;
	    mom_v(ii) = q;
	    double aQ = norm(q);
	    int id = vindx(ii)[0];
	    int  i = vindx(ii)[1];
	    int vtyp = Vtype(id,i);
	    double Vq = 8*pi/(aQ*aQ+lmbda);
	    int Nc = abs(vtyp % 10);
	    V_current(ii) = (Nc==0) ? Vq : Vq * ipower(Vq * lmbda/(8*pi),Nc);
	    if (vtyp<0){
	      // we also add the single-particle counter term in addition to two-particle counter term.
	      int ii_g = single_counter_index(ii);
	      long double g_kq = G_current(ii_g);
#ifdef CNTR_VN		
	      //long double Vn = ipower(Vq, Nc+1);
	      long double Vn = ipower(Vq, 2);
	      if (g_kq!=0) V_current(ii) += Vn*lmbda_spct_cutoffk(-vtyp-1)/g_kq;
#else		
	      if (g_kq!=0) V_current(ii) += exp(-aQ*2/kF)*lmbda_spct_cutoffk(-vtyp-1)/g_kq;
#endif		
	    }
	  }
	  int dq = loop_Vqind(iloop).extent(0);
	  for (int iq=0; iq<loop_Vqind2(iloop).extent(0); ++iq){ // Now do the same for the Hugenholtz interaction for the second interaction
	    int ii   = loop_Vqind2(iloop)(iq);
	    double isgn = loop_Vqsgn2(iloop)(iq);
	    bl::TinyVector<double,3> q = mom_v2(ii-N0v) + dK * isgn;
	    mom_v2(ii-N0v) = q;
	    double aQ = norm(q);
	    int id = vindx(ii)[0];
	    int  i = vindx(ii)[1];
	    int vtyp = Vtype(id,i);
	    if (vtyp<10){ log << "ERROR : We should have Hugenholtz here!" << std::endl; exit(1);}
	    if (false){
	      int iig1 = Gindx(id,2*i+1);
	      int iig2 = Gindx(id,static_cast<int>(i_diagsG(id,2*i)));
	      bl::TinyVector<double,3> q2 = mom_g(iig1) - mom_g(iig2);
	      if (fabs(norm(q2-q))>1e-6){
		log << itt << ") ERROR 2 in computing q2 for Hugenholtz interaction q2="<< q2 << " while q2_old+dq=" << q << std::endl;
		exit(1);
	      }
	    }
	    double Vq = 8*pi/(aQ*aQ+lmbda);
	    int Nc = vtyp % 10;
	    V_current2(ii-N0v) = (Nc==0) ? Vq : Vq * ipower(Vq * lmbda/(8*pi),Nc); // Note that Hugenholtz does not have single-particle counter-term.
	  }
	  // Hugenholtz diagrams are evaluated
	  for (int iid=0; iid<hugh_diags(iloop).extent(0); iid++){
	    int id = hugh_diags(iloop)(iid);
	    int nvh=0;
	    for (int i=1; i<Norder; i++){
	      if (Vtype(id,i)>=10){// Hugenholtz
		int ii = Vindx(id,i);
		V12(nvh,0) = V_current(ii);
		V12(nvh,1) = V_current2(ii-N0v);
		nvh++;
	      }
	    }
	    V_Hugh(hh_indx(id)) = Compute_V_Hugenholtz(V12, nvh, diagSign[id], Vtree, log);
	  }
	  //t7.stop();
	}
	//t8.start(); // takes 0.14%
	PQ = PQ_new;
	if (!Qweight) BKdata.Set_PQg_new_to_PQg();
	//if (iloop==0 && !Q0w0) Pln.cmp_single(2*amomentm(iloop)/cutoffq-1., pl_Q, BKdata.Nlq);  // update Legendre Polynomials
	//t8.stop();
      }
      //if (!Qweight) CheckAccept(itt, log, i_diagsG, Vtype,  Vindx, mom_v2, mom_g, Gindx, N0v);
    } else if (icase==1){  // changing one of the time variables
      Nall_t += 1;
      int itime = times_2_change[ static_cast<int>( times_2_change.size()*drand() ) ];
      changed_G=0;              // which propagators are being changed?
      times_trial = times;      // times_trial will contain the trial step times.
      if (itime==0){            // this is the measuring time with vertex=0
	double t_new = rw.gtau(drand());
	if (t_new<1e-15) t_new = 1e-15; // to be on the right branchcut
	times_trial(0) = t_new;
	int ivertex=0;
	for (int id=0; id<Ndiags; id++){
	  int i_pre_vertex = i_diagsG(id,ivertex);
	  changed_G.set(Gindx(id,ivertex),1);         // these two propagators contain vertex=0.
	  changed_G.set(Gindx(id,i_pre_vertex),1);
	}
      }else{
	double t_new = beta*drand();
	// unscreened interaction, hence changing two times simultaneously!
	int ivertex = itime;
	times_trial(itime) = t_new;
	if (itime%2==0){ // this is the time for static interaction
	  for (int ivertex=itime; ivertex<itime+2; ivertex++){
	    for (int id=0; id<Ndiags; id++){
	      int i_pre_vertex = i_diagsG(id,ivertex);
	      changed_G.set(Gindx(id,ivertex),1);     // these two propagators are changed because of the new time.
	      changed_G.set(Gindx(id,i_pre_vertex),1);
	    }
	  }
	}
      }
      if (! Qweight){
      	//t9.start(); // takes 5% of the time
	int ip=0;
	for (int ii=0; ii<Ng; ++ii){
	  if (changed_G[ii]){
	    int id = gindx(ii)[0];
	    int  i = gindx(ii)[1];
	    double aK = norm(mom_g(ii));
	    int i_final = diagsG(id,i);
	    double dt = times_trial( et(i_final) )-times_trial( et(i) );
	    g_trial(ip) = Gk(aK, dt);
	    ip_ind(ii)=ip;
	    ip+=1;
	  }
	}
	//t9.stop();
	//t10.start(); // takes 4.4% of the time
	PQ_new=0;   // recalculating PQ, taking into account one change of time.
	BKdata.PQg_new_Initialize();
	for (int id=0; id<Ndiags; id++){
	  long double PQd = 1.0;
	  for (int i=0; i<2*Norder; i++){
	    int ii = Gindx(id,i);
	    PQd *= ( changed_G[ii] ? g_trial(ip_ind(ii)) : G_current(ii) );
	  }
	  bool isHugenholtz=false;
	  for (int i=1; i<Norder; i++)
	    if (Vtype(id,i) < 10)// non-Hugenholtz
	      PQd *= V_current(Vindx(id,i));
	    else // this diagram is Hugenholtz-type
	      isHugenholtz=true;
	  if (isHugenholtz)
	    PQd *= V_Hugh(hh_indx(id));
	  else
	    PQd *=  diagSign[id][0];
	  PQ_new += PQd;
	  BKdata.PQg_new_Add(id,PQd);
	}
	//t10.stop();
      }else{
	PQ_new = V0norm * mweight(amomentm, momentum);
      }
      double ratio = PQ_new/PQ;
      bool accept = fabs(ratio) > 1-drand() && PQ_new!=0;
      if ((itt+1)%Ncout==0 && mpi.rank==mpi.master)
	_PrintInfo_(itt+1, Qweight, icase, amomentm(1), amomentm(Nloops-1), times(itime), PQ_new, PQ, dk_hist, Nweight/(Nmeassure+0.0),log);
      if (accept){
	//t11.start(); // takes 1.3% of time
	Nacc_t +=1 ;
	times = times_trial;  // the new times are now accepted. We change the independent time variable, or two internal times.
	if (!Qweight){
	  for (int ii=0; ii<Ng; ++ii)
	    if (changed_G[ii])
	      G_current(ii) = g_trial(ip_ind(ii)); // The green's functions in the loop have been recalculated, and are now stored.
	}else{
	  // since we have the measuring diagram, we did not calculate G_trial and V_trial before, hence we need to evaluate them now.
	  for (int ii=0; ii<Ng; ++ii){
	    if (changed_G[ii]){
	      int id = gindx(ii)[0];
	      int  i = gindx(ii)[1];
	      int i_final = diagsG(id,i);
	      G_current(ii) = Gk( norm(mom_g(ii)),  times( et(i_final) )-times( et(i)) ); // these propagators were changed, but they were not computed before.
	    }
	  }
	}
	//t11.stop();
	PQ = PQ_new;
	//t12.start(); // takes 0.1%
	if (!Qweight) BKdata.Set_PQg_new_to_PQg();
	if (itime==0 && !Q0w0) Pln.cmp_single(2*(times(0)-times(1))/beta-1., pl_t, BKdata.Nlt);  // update Legendre Polynomials
	//t12.stop();
      }
      //log<<itt<<" G_current="<<G_current<<" V_current="<<V_current<<endl;
      //if (!Qweight) CheckAccept(itt, log, i_diagsG, Vtype,  Vindx, mom_v2, mom_g, Gindx, N0v);
    }else{  // normalization diagram step
      Nall_w += 1;
      //t13.start(); // takes 0.7% of the time
      if (Qweight){ // trying to jump back into physical hilbert space
	PQ_new=0;
	BKdata.PQg_new_Initialize();
	for (int id=0; id<Ndiags; id++){
	  long double PQd = 1.0;
	  for (int i=0; i<2*Norder; i++) PQd *= G_current(Gindx(id,i)); // we have all propagators precomputed, hence it is easy to compute the diagrams.
	  bool isHugenholtz=false;
	  for (int i=1; i<Norder; i++)
	    if (Vtype(id,i) < 10) // non-Hugenholtz
	      PQd *= V_current(Vindx(id,i));
	    else // this diagram is Hugenholtz-type
	      isHugenholtz=true;
	  if (isHugenholtz) 
	    PQd *= V_Hugh(hh_indx(id));
	  else
	    PQd *=  diagSign[id][0];
	  PQ_new += PQd;
	  BKdata.PQg_new_Add(id,PQd);
	}
      }else{        // or if we are in the physical Hilbert space, we try to jump into the normalization diagram
	PQ_new = V0norm * mweight(amomentm, momentum);
      }
      double ratio = PQ_new/PQ;
      bool accept = fabs(ratio) > 1-drand() && PQ_new!=0;
      if ((itt+1)%Ncout==0 && mpi.rank==mpi.master)
	_PrintInfo_(itt+1, Qweight, icase, amomentm(1), amomentm(Nloops-1), times(0), PQ_new, PQ, dk_hist, Nweight/(Nmeassure+0.0), log);
      if (accept){
	Nacc_w += 1;
	Qweight = 1-Qweight;
	PQ = PQ_new;
	if (!Qweight) BKdata.Set_PQg_new_to_PQg();
      }
      //t13.stop();
    }
    
    // Measuring
    if (itt>=Nwarm && itt%tmeassure==0){
      t_mes.start(); // takes 13% time for BK=True
      Nmeassure += 1;
      double Qa = amomentm(0);
      if (fabs(Qa-qx(iiQ))>1e-7) log << "ERROR : Either iiQ="<<iiQ<<" is wrong or Qa="<< Qa << " is wrong qx(iiQ)="<< qx(iiQ)<<endl;
      double t = times(0)-times(1);
      double cw = 1./rw.wt(t);
      double sp = sign(PQ) * cw;
      int iq = std::min(int(Qa/cutoffq*Nq), Nq-1);
      if (Q0w0) iq=0;
      int it = std::min(int(t/beta*Nt), Nt-1);
      if (Qweight){
	Pnorm += Q0w0 ? cw : Qa*Qa * cw;                 // |Q|^2 * reweighting
	Nweight+=1;
      }else{
	Pbin(it,iiQ) += sp;
      //BKdata.Meassure(itt,tmeassure,PQ,sp,pl_Q,pl_t,momentum(0),amomentm(0),mom_g,Gindx);
	BKdata.Meassure(itt,tmeassure,PQ,sp,iiQ,pl_t,momentum(0),amomentm(0),mom_g,Gindx);
      }
      if (t < 0.05*beta){
	NlowT +=1 ;
      }
      if (!Qweight){
	mweight.Add_to_K_histogram(dk_hist, momentum, cutoffq, cutoffk);
	if (GetHistogram){
	  for (int ii=0; ii<Ng; ii++){
	    double k = norm(mom_g(ii));
	    int ik = static_cast<int>(k/cutoffk*Nbin);
	    if (ik>=Nbin) ik=Nbin-1;
	    K_Ghist(ii,ik) += 1e-5;
	  }
	  for (int ii=0; ii<Nv; ii++){
	    double k = norm(mom_v(ii));
	    int ik = static_cast<int>(k/cutoffk*Nbin);
	    if (ik>=Nbin) ik=Nbin-1;
	    K_Vhist(ii,ik) += 1e-5;
	  }
	}
      }
      if ( itt>2e5*p.Toccurence/mpi.size && itt%(static_cast<int>(100000*p.Toccurence/mpi.size+1000)*tmeassure) == 0){
	double last_occurence = (Nweight-Wchanged)/(Nmeassure-Nchanged); // occurence from the last time we changed it
#ifdef _MPI
	double total_occurence=0;
	int ierr = MPI_Allreduce(&last_occurence, &total_occurence, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	if (ierr!=0) { log << "MPI_Allreduce(total_occurence) returned error="<< ierr << endl; exit(1);}
	last_occurence = total_occurence/mpi.size;
#endif	
	int change_V0 = 0;
	if ( last_occurence > 0.20){ // decrease by two
	  change_V0 = -1;
	  V0norm /= 2.;
	  Pnorm /= 2.;
	  //Nchanged = Nmeassure;
	  //Wchanged = Nweight;
	  //if (mpi.rank==mpi.master) log<<std::left<<(itt/1.0e6)<<"M occurence="<<std::setw(8)<<last_occurence<<" V0 reduced to "<<V0norm<<std::endl;
	}
	if ( last_occurence < 0.03){ // increase by two
	  change_V0 = 1;
	  V0norm *= 2.;
	  Pnorm *= 2.;
	  //Nchanged = Nmeassure;
	  //Wchanged = Nweight;
	  //if (mpi.rank==mpi.master) log<<std::left<<(itt/1.0e6)<<"M occurence="<<std::setw(8)<<last_occurence<<" V0 increased to "<<V0norm<<std::endl;
	}
	if (change_V0 != 0){ 
	  std::string schange = (change_V0<0) ? " V0 reduced to " : " V0 increased to ";
	  if (itt < 0.4*p.Nitt){ // if still enough time to meassure, then just completely reset the measurements
	    Nmeassure = 0;       // all measurements are thrown away
	    Nweight = 0;
	    BKdata.ResetMeasure();
	    Pbin          = 0.0;
	    Pnorm         = 0.0;
	    if (mpi.rank==mpi.master) log<<std::left<<(itt/1.0e6)<<"M occurence="<<std::setw(8)<<last_occurence<<schange<<V0norm<<" and meassure reset"<<std::endl;
	  }else{
	    if (mpi.rank==mpi.master) log<<std::left<<(itt/1.0e6)<<"M occurence="<<std::setw(8)<<last_occurence<<schange<<V0norm<<std::endl;
	  }
	  Nchanged = Nmeassure;
	  Wchanged = Nweight;
	}
      }
      if (itt%(500000*tmeassure) == 0){
#ifdef _MPI	  
	int ierr = MPI_Allreduce(MPI_IN_PLACE, mweight.K_hist.data(), mweight.K_hist.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	if (ierr!=0) {log << "MPI_Allreduce(K_hist) returned error="<< ierr << endl; exit(1);}
#endif
	double dsum = sum(mweight.K_hist);
	if (dsum > 1000*dk_hist){
	  // We can get overflow during MPI for a long run. We should use a constant for normalization.
	  // We could here normalize K_hist (with knorm), and than when we measure, we would
	  // add instead of adding unity, we would add 1./knorm
	  dk_hist *= mweight.Normalize_K_histogram();
	  if (dk_hist < 1e-8) dk_hist = 1.0;
	  //mweight.Recompute(mpi.rank==mpi.master);
	  mweight.Recompute(false);
	  // If we are in a meassuring space, we have to update now the value of the meassuring diagram as the weight has changed.
	  if (Qweight) PQ = V0norm * mweight(amomentm, momentum);
	}
      }
      t_mes.stop();
    }
  }
  t_all.stop();
  BKdata.C_Pln2 *= 1.0/Nmeassure;
  BKdata.C_Pln1 *= 1.0/Nmeassure;
  BKdata.C_Pln0 *= 1.0/Nmeassure;
  Pbin         *= 1.0/Nmeassure;
  Pnorm        *= 1.0/Nmeassure;
  double occurence = static_cast<double>(Nweight)/static_cast<double>(Nmeassure);

  if (GetHistogram){
    K_Ghist *= 1.0/Nmeassure;
    K_Vhist *= 1.0/Nmeassure;
  }

  BKdata.BeforeReduce();
  
  for (int iQ=0; iQ<qx.extent(0); iQ++){
    Pq0(iQ,0) = bl::sum(Pbin(bl::Range::all(),iQ));
    Pq0(iQ,1) = Pq0(iQ,0)*Pq0(iQ,0);
  }

#ifdef _MPI
  double dat[2] = {Pnorm, occurence};
  if (mpi.rank==mpi.master){
    MPI_Reduce(MPI_IN_PLACE, BKdata.C_Pln2.data(), BKdata.C_Pln2.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, BKdata.C_Pln1.data(), BKdata.C_Pln1.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, BKdata.C_Pln0.data(), BKdata.C_Pln0.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE,  Pbin.data(),  Pbin.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, dat, 2, MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, mweight.K_hist.data(), mweight.K_hist.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);

    MPI_Reduce(MPI_IN_PLACE,  Pq0.data(),  Pq0.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    
    if (GetHistogram){
      MPI_Reduce(MPI_IN_PLACE, K_Ghist.data(), K_Ghist.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
      MPI_Reduce(MPI_IN_PLACE, K_Vhist.data(), K_Vhist.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    }
  }else{
    MPI_Reduce(BKdata.C_Pln2.data(), BKdata.C_Pln2.data(), BKdata.C_Pln2.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    MPI_Reduce(BKdata.C_Pln1.data(), BKdata.C_Pln1.data(), BKdata.C_Pln1.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    MPI_Reduce(BKdata.C_Pln0.data(), BKdata.C_Pln0.data(), BKdata.C_Pln0.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    MPI_Reduce( Pbin.data(),  Pbin.data(),  Pbin.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    MPI_Reduce(dat, dat, 2, MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    MPI_Reduce(mweight.K_hist.data(), mweight.K_hist.data(), mweight.K_hist.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);

    MPI_Reduce( Pq0.data(),  Pq0.data(),  Pq0.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    
    if (GetHistogram){
      MPI_Reduce(K_Ghist.data(), K_Ghist.data(), K_Ghist.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
      MPI_Reduce(K_Vhist.data(), K_Vhist.data(), K_Vhist.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    }
  }
  if (mpi.rank==mpi.master){
    BKdata.C_Pln2 *= 1./mpi.size;
    BKdata.C_Pln1 *= 1./mpi.size;
    BKdata.C_Pln0 *= 1./mpi.size;
    Pbin         *= 1./mpi.size;
    Pnorm        = dat[0]/mpi.size;
    occurence    = dat[1]/mpi.size;

    BKdata.AfterReduce(mpi.size);
    
    Pq0 *= 1./mpi.size;
    for (int iQ=0; iQ<qx.extent(0); iQ++)
      Pq0(iQ,1) = sqrt( fabs(Pq0(iQ,1) - Pq0(iQ,0)*Pq0(iQ,0))/mpi.size );
    //
    mweight.K_hist *= 1./mpi.size;
  }
#endif
  if (mpi.rank==mpi.master){
    log<<"total acceptance rate="<<(Nacc_k+Nacc_q+Nacc_t+Nacc_w)/(p.Nitt+0.0)<<" "<<" k-acceptance="<<(Nacc_k/(Nall_k+0.0))<<" q-acceptance="<<(Nacc_q/(Nall_q+0.0))<<" t-acceptanece="<<(Nacc_t/(Nall_t+0.0))<<" w-acceptanece="<<(Nacc_w/(Nall_w+0.0))<<std::endl;
    log<<" k-trials="<<(Nall_k/(p.Nitt+0.0))<<" q-trials="<<(Nall_q/(p.Nitt+0.0))<<" t-trials="<<(Nall_t/(p.Nitt+0.0))<<" w-trials="<<(Nall_w/(p.Nitt+0.0))<<std::endl;
    //log<<"t1="<<t1.elapsed()<<" t2="<<t2.elapsed()<<" t3="<<t3.elapsed()<<" t4="<<t4.elapsed()<<" t5="<<t5.elapsed()<<" t6="<<t6.elapsed()<<" t7="<<t7.elapsed();
    //log<<" t8="<<t8.elapsed()<<" t9="<<t9.elapsed()<<" t10="<<t10.elapsed()<<" t11="<<t11.elapsed()<<" t12="<<t12.elapsed()<<" t13="<<t13.elapsed()<<" t14="<<t14.elapsed()<<std::endl;
    log<<"total time="<<t_all.elapsed()<<" measuring time="<<t_mes.elapsed()<<std::endl;

    double dOmega = (!Q0w0) ? 1.0/(4*pi) : 1.0;
    BKdata.C_Pln2 *= dOmega * (fabs(V0norm)/Pnorm); // 4*pi because we must divide by 1/(4*pi*q**2), and 1/q**2 was already done in sampling
    BKdata.C_Pln1 *= dOmega * (fabs(V0norm)/Pnorm); // 4*pi because we must divide by 1/(4*pi*q**2), and 1/q**2 was already done in sampling
    BKdata.C_Pln0 *= dOmega * (fabs(V0norm)/Pnorm); // 4*pi because we must divide by 1/(4*pi*q**2), and 1/q**2 was already done in sampling
    Pbin          *= dOmega * (fabs(V0norm)/Pnorm);
    Pq0           *= dOmega * (fabs(V0norm)/Pnorm);
    log<<"  meassuring diagram occurence frequency="<<occurence<<" and its norm Pnorm="<<Pnorm<<std::endl;
    
    // Proper normalization of the resulting Monte Carlo data
    //double dOmega2 = (!Q0w0) ? (4*pi*cutoffq*cutoffq*cutoffq/3) : 1.0;
    double dOmega2 = (!Q0w0) ? 4*pi*sum(qx*qx) : 1.0;
      
    double norm = ipower( beta/((2*pi)*(2*pi)*(2*pi)), Norder) * dOmega2;
    BKdata.C_Pln2 *= norm;
    BKdata.C_Pln1 *= norm;
    BKdata.C_Pln0 *= norm;
    Pbin          *= norm;
    Pq0           *= norm;
    // Later we will use Nthbin/2.0*Nkbin/2.0  normalization in the Kernel to transform from binned data to function values.
    BKdata.Normalize(beta);
    
    //double dq_binning = cutoffq/Nq;
    //double dt_binning = beta/Nt;
    //if (!Q0w0)
    //Pbin *= 1.0/(dt_binning);
    //Pbin *= 1.0/(dq_binning * dt_binning);
    
    for (int ik=0; ik<mweight.K_hist.extent(0); ik++){
      double dsum=0;
      for (int i=0; i<Nbin; i++) dsum += mweight.K_hist(ik,i);
      mweight.K_hist(ik,bl::Range::all()) *= 1./dsum;
      std::ofstream hout((std::string("Q_hist.")+std::to_string(ik)).c_str());
      double cutoff = (ik>0) ? cutoffk : cutoffq;
      for (int i=0; i<Nbin; i++){
	hout<< (i+0.5)/Nbin*cutoff/kF << " " << mweight.K_hist(ik,i)<<std::endl;
      }
    }
    
    if (GetHistogram){
      for (int ik=0; ik<K_Ghist.extent(0); ik++){
	double dsum = sum(K_Ghist(ik,bl::Range::all()));
	K_Ghist(ik,bl::Range::all()) *= 1./dsum;
	std::ofstream hout((std::string("K_G_hist.")+std::to_string(ik)).c_str());
	for (int i=0; i<Nbin; i++)
	  hout << (i+0.5)/Nbin*cutoffk/kF << " " << K_Ghist(ik,i) << std::endl;
      }
      for (int ik=0; ik<K_Vhist.extent(0); ik++){
	double dsum = sum(K_Vhist(ik,bl::Range::all()));
	K_Vhist(ik,bl::Range::all()) *= 1./dsum;
	std::ofstream hout((std::string("K_V_hist.")+std::to_string(ik)).c_str());
	for (int i=0; i<Nbin; i++)
	  hout << (i+0.5)/Nbin*cutoffk/kF << " " << K_Vhist(ik,i) << std::endl;
      }
    }
  }
}
