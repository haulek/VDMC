// @Copyright 2018 Kristjan Haule 
inline double Type2VertexContribution(int id, int i, bool changed_Vi,
				      const bl::Array<int,2>& vertex_companion, const bl::Array<int,2>& changed_V,
				      const bl::Array<double,2>& W_trial, const bl::Array<double,2>& V_eqv_trial,
				      const bl::Array<double,2>& W_current, const bl::Array<double,2>& V_eqv_current,
				      const bl::Array<double,2>& Vrtx_trial, const bl::Array<double,2>& Vrtx_trial2)
{
  double Wi, vi;
  if (changed_Vi){
    Wi = W_trial(id,i);
    vi = (V_eqv_trial(id,i) - Wi);
  }else{
    Wi = W_current(id,i);
    vi = (V_eqv_current(id,i) - Wi);
  }
  int j = vertex_companion(id,i-1);
  double Wj, vj;
  if (changed_V(id,j)){
    Wj = W_trial(id,j);
    vj = V_eqv_trial(id,j) - Wj;
  }else{
    Wj = W_current(id,j);
    vj = V_eqv_current(id,j) - Wj;
  }
  return  ( vi*Vrtx_trial(id,i) + Wi ) * vj*Vrtx_trial(id,j) + ( vi*Vrtx_trial2(id,i) + Wi ) * Wj;
}

inline double Type2VertexContribution(int id, int i, const bl::Array<int,2>& vertex_companion, 
				      const bl::Array<double,2>& W_current, const bl::Array<double,2>& V_eqv_current,
				      const bl::Array<double,2>& Vrtx_current, const bl::Array<double,2>& Vrtx_current2)
{
  double Wi = W_current(id,i);
  double vi = (V_eqv_current(id,i) - Wi);
  int j = vertex_companion(id,i-1);
  double Wj = W_current(id,j);
  double vj = V_eqv_current(id,j) - Wj;
  return  ( vi*Vrtx_current(id,i) + Wi ) * vj*Vrtx_current(id,j) + ( vi*Vrtx_current2(id,i) + Wi ) * Wj;
}

template<int Nlt, int Nlq>
void sample_dynamic(bl::Array<double,2>& C_Pln, bl::Array<double,2>& Pbin, 
		    const params& p, const bl::Array<int,2>& diagsG, const bl::Array<double,1>& diagSign, bl::Array<int,3>& diagVertex,
		    const std::vector<std::vector<std::vector<int> > >& Loop_index, const std::vector<std::vector<std::vector<int> > >& Loop_type,
		    const bl::Array<double,1>& qx, const bl::Array<double,1>& taux, const bl::Array<double,2>& _rW_, const bl::Array<double,1>& _rWom0_,
		    const bl::Array<double,1>& kx, const bl::Array<double,1>& epsx)
{
  const int Nr=3; // three different steps in this algorithm
  double Pr_norm = (p.Pr[0]+p.Pr[1]+p.Pr[2]); // probability for all four has to be normalized
  bl::TinyVector<double,Nr> Prs;  // cumulative probability for taking a step, needed for bisection below.
  Prs[0] = p.Pr[0]/Pr_norm;
  for (int i=1; i<Nr; i++) Prs[i] = Prs[i-1] + p.Pr[i]/Pr_norm; // prepares cumulative probability from p.Pr

  if (Pbin.extent(0)!=p.Nt || Pbin.extent(1)!=p.Nq) Pbin.resize(p.Nt,p.Nq); Pbin=0.0;

  if (diagVertex.extent(2)>0){
    std::cout << "diagVertex=" << diagVertex << std::endl;
  }

  ScreenedCoulombV Vq(1e-5);
  
  Spline2D<double> rW(qx.extent(0),taux.extent(0));
  for (int iq=0; iq<qx.extent(0); iq++)
    for (int it=0; it<taux.extent(0); it++)
      rW(iq,it) = _rW_(iq,it);
  rW.splineIt(qx, taux);
  Spline1D<double> rWom0(qx.extent(0));
  for (int iq=0; iq<qx.extent(0); iq++) rWom0[iq] = _rWom0_(iq);
  rWom0.splineIt(qx);
  
  int Ndiags = diagsG.extent(0);
  int Norder = diagsG.extent(1)/2;
  // computing the inverse of all permutations, to provide fast lookup for the previous fermionic propagator
  bl::Array<int,2> i_diagsG(Ndiags,2*Norder);
  for (int id=0; id<Ndiags; id++)
    for (int i=0; i<2*Norder; i++)
      i_diagsG(id,diagsG(id,i))=i;

  // Checking in which diagrams one should not add exchange, as it is already present due to self-consistency
  bl::Array<int,2> pure_Exchange(Ndiags,Norder);
  pure_Exchange=0;
  for (int id=0; id<Ndiags; id++){
    for (int i=1; i<Norder; i++){
      if (  diagsG(id,2*i)   == (2*i+1) ) pure_Exchange(id,i)=1;
      if (  diagsG(id,2*i+1) == 2*i )     pure_Exchange(id,i)=1;
      if (i_diagsG(id,2*i)   == 2*i+1)    pure_Exchange(id,i)=1;
      if (i_diagsG(id,2*i+1) == 2*i)      pure_Exchange(id,i)=1;
    }
  }
  std::cout << " pure_Exchange=" << pure_Exchange << std::endl;

  bl::Array<int,2> vertex_companion(Ndiags,Norder-1); vertex_companion=0;
  bl::Array<int,2> vertex_incoming(Ndiags,Norder-1); vertex_incoming=0;
  bl::Array<int,2> is_companion(Ndiags,2*Norder); is_companion=0;
  ComputeVertexCompanion(vertex_companion, vertex_incoming, diagVertex, is_companion, diagsG, i_diagsG);

  std::cout<<"is_companion="<<std::endl;
  for (int id=0; id<Ndiags; id++){
    std::cout<<id<<" ";
    for (int i=0; i<2*Norder; i++)
      if (is_companion(id,i))
	std::cout << " (" << i << "," << is_companion(id,i) << ") ";
    std::cout<<std::endl;
  }
  
  RanGSL drand(p.iseed); // GSL random number generator
  Timer t_mes, t_all, t_Pl, t_k, t_Q, t_w, t_t;
  
  int tmeassure=p.tmeassure, Ncout=p.Ncout, Nwarm=p.Nwarm;
  double V0norm = p.V0norm, dkF = p.dkF;
  double cutoffq = p.cutoffq, cutoffk = p.cutoffk, dRk = p.dRk;
  double beta = p.beta;
  int Nt = p.Nt, Nq = p.Nq;
  
  LegendreQ<Nlq> Plq(0);
  LegendreQ<Nlt> Plt(0);
  bl::TinyVector<double,Nlq+1> pl_Q;  // Will contain legendre Pl(2*q/qmax-1)
  bl::TinyVector<double,Nlt+1> ul_t;  // Will contain legendre Pl(2*t/beta-1)

  C_Pln = 0.0;                   // cummulative result for legendre expansion
  double Pnorm = 0.0;            // normalization diagram value

  // Green's function: we can use non-interacting or Hartree-Fock.
  Gk_HF Gk(beta,p.kF,kx,epsx);        // Hartree-Fock non-interacting green's function
 //egass_Gk Gk(beta,p.kF);             // free particle non-interacting green's function

  // DEBUG
  //Gk.debug("debg7.dat");
  // DEBUG
  
  if (Loop_index.size()!=Ndiags) std::cerr<<"Wrong size of Loop_index "<<Loop_index.size()<<" instead of "<<Ndiags<<std::endl;
  int Nloops = Loop_index[0].size();
  if (Nloops != Norder+1) std::cerr<<"Expecting Nloops==Norder+1 in Loops_index, but got Nloops="<<Nloops<<" Norder="<<Norder<<std::endl;

  Reweight_time rw(beta,p.lmbdat);
  int Nbin = 129;
  meassureWeight mweight(p.V0exp, p.cutoffk, p.kF, Nbin, Nloops); // Weight of normalization diagram. We self-consist the shape of this diagram.
  //bl::Array<double,2> K_hist(Nloops,Nbin); K_hist=0;              // For self-consistent weight we need some histograms, created here/.
  bl::Array<double,2> T_hist((Norder-1)*2,Nbin); T_hist=0;        // also histogram in time, which unfortunately does not work yet.
  double dk_hist = 1.0;
  
  // This is momentum for loops
  bl::Array<bl::TinyVector<double,3>,1> momentum(Nloops); // momentum for all loops. Each diagram of the same order has the same number of loops.
  bl::Array<double,1> amomentm(Nloops);               // absolute value of momentum for each loop
  // This is momentum for all diagrams and all propagators
  bl::Array<bl::TinyVector<double,3>,2> mom_G(Ndiags,2*Norder); // momentum for 2*Norder fermionic propagators
  bl::Array<bl::TinyVector<double,3>,2> mom_V(Ndiags,  Norder); // momentum for Norder bosonic propagators
  
  // Starting point is set up to be random
  // For the times for all vertices in all the graphs
  bl::Array<double,1> times(2*Norder);
  times(0) = beta*drand(); // external time
  times(1) = 0;            // beginning time
  for (int it=2; it<2*Norder; it++) times(it) = beta*drand();
  // Next the momenta
  {
    double Qa = p.kF*drand(); // absolute value of external momentum is radom in the interval [0-kF,0,0]
    momentum(0) = Qa,0.0,0.0; // external momentum Q
    amomentm(0) = Qa;
    double kf_3 = p.kF/sqrt(3.);
    for (int ik=1; ik<Nloops; ik++){
      momentum(ik) = drand()*kf_3, drand()*kf_3, drand()*kf_3; // the rest of the momenta start as [0-1,0-1,0-1]/sqrt(3)
      amomentm(ik) = norm(momentum(ik));
    }
  }
  // Momenta for all propagators in all graphs, which is needed to evaluate the diagrams.
  mom_G=0.0;
  mom_V=0.0;
  for (int id=0; id<Ndiags; id++){
    for (int iloop=0; iloop<Nloops; iloop++){
      const std::vector<int>& lindex = Loop_index[id][iloop];
      const std::vector<int>& ltype  = Loop_type[id][iloop];
      for (int i=0; i<lindex.size(); i++){
        if ( abs(ltype[i])==1 ){
	  mom_G(id, lindex[i]) += momentum(iloop) * dsign(ltype[i]);
	}else{
	  mom_V(id,lindex[i]) += momentum(iloop) * dsign(ltype[i]);
	}
      }
    }
  }
  // DEBUGGING: Checking the conservation of momenta in every vertex.
  for (int id=0; id<Ndiags; id++){
    for (int i=0; i<2*Norder; i++){
      //int i_final = diagsG(id,i);
      int i_previ = i_diagsG(id,i);
      bl::TinyVector<double,3> k_in = mom_G(id,i_previ);
      bl::TinyVector<double,3> k_out = mom_G(id,i);
      bl::TinyVector<double,3> q = mom_V(id,i/2)*(1.-2*(i%2));
      if (norm(k_in-k_out-q)>1e-10){
	std::cerr<<"ERROR : diagram "<<id<<" vertex i="<<i<<" k_in="<<k_in<<" k_out="<<k_out<<" q="<<q<<std::endl;
      }
    }
  }
  // DEBUGGING

  // Finally evaluating the value of Feynman diagrams at this configuration.
  bl::Array<double,2> G_current(Ndiags,2*Norder), V_current(Ndiags,Norder), W_current(Ndiags,Norder);
  G_current=0; V_current=0; W_current=0;
  bl::Array<double,2>  Vrtx_current(Ndiags,Norder), V_eqv_current(Ndiags,Norder);
  Vrtx_current=0.0;  V_eqv_current=0;
  bl::Array<double,2>  Vrtx_current2(Ndiags,Norder);
  Vrtx_current2=0;
  bl::Array<double,1> PQs(Ndiags);
  PQs=0;

  double PQ=0;
  for (int id=0; id<Ndiags; id++){
    double PQd = 1.0;
    // Fermionic propagators
    for (int i=0; i<2*Norder; i++){
      double aK = norm(mom_G(id,i));
      int i_final = diagsG(id,i);
      double dt = times(i_final)-times(i);
      G_current(id,i) = Gk(aK, dt);
      PQd *= G_current(id,i);
    }
    // Vertices, which are needed to express screened interaction with the same time difference as the unscreened part.
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
      Vrtx_current(id,ii) = MoveVertex(t2, t1, ti, to, ki, ko, Gk);
      if (itype==2){
	Vrtx_current2(id,ii) = Vrtx_current(id,ii); // saving regular vertex to another array
	if (vertex_incoming(id,ii-1)){
	  int i_m_partner = (i_m/2)*2 + (1-i_m%2);
	  double ti_new = times(i_m_partner);
	  Vrtx_current(id,ii) = MoveVertex(t2, t1, ti_new, to, ki, ko, Gk);
	}else{
	  int i_p_partner = (i_p/2)*2 + (1-i_p%2);
	  double to_new = times(i_p_partner);
	  Vrtx_current(id,ii) = MoveVertex(t2, t1, ti, to_new, ki, ko, Gk);
	}
      }
    }
    // The screened interaction, both the regular and the delta-part.
    for (int i=1; i<Norder; i++){ // do not add V for the meassuring line
      double aQ = norm(mom_V(id,i));
      V_current(id,i) = Vq(aQ);
      double dt = fabs(times(2*i)-times(2*i+1));
      intpar pq = Interp(aQ,qx);
      intpar pt = Interp(dt,taux);
      W_current(id,i) = rW(pq,pt) * V_current(id,i);
      int itype = diagVertex(id,i-1,1);
      if (!pure_Exchange(id,i)){
	V_eqv_current(id,i) = (rWom0(pq) + 1)*V_current(id,i)/beta;
	if (itype==1){
	  PQd *= ( (V_eqv_current(id,i) - W_current(id,i))*Vrtx_current(id,i) + W_current(id,i) );
	}else if (itype==2){
	  PQd *= Type2VertexContribution(id,i,vertex_companion, W_current, V_eqv_current, Vrtx_current, Vrtx_current2);
	}
      }else{
	PQd *= W_current(id,i);
	if (itype!=1){
	  std::cerr<<"ERROR: vertex of pure exchange should not be type 2 or companion of type 2"<<std::endl;
	  exit(0);
	}
      }
    }
    PQs(id) = PQd * diagSign(id);
    PQ += PQs(id);
  }

  CarefullyVerifyCurrentState3(0, beta, 0, PQ, momentum, times, mom_G, mom_V, diagsG, i_diagsG, Gk, qx, taux, rW, diagSign, Loop_index, Loop_type, G_current, V_current, W_current, V_eqv_current, Vrtx_current, Vrtx_current2, diagVertex, pure_Exchange, rWom0, vertex_companion, vertex_incoming, PQs);
  
  // DEBUGGING
  for (int id=0; id<Ndiags; id++){
    for (int i=0; i<2*Norder; i++)
      if (fabs(G_current(id,i))==0){
	std::cerr<<"G_current("<<id<<","<<i<<") not set!"<<std::endl;
      }
    for (int i=1; i<Norder; i++){
      if (fabs(V_current(id,i))==0){
	std::cerr<<"V_current("<<id<<","<<i<<") not set!"<<std::endl;
      }
      if (fabs(W_current(id,i))==0){
	std::cerr<<"W_current("<<id<<","<<i<<") not set!"<<std::endl;
      }
    }
  }
  std::cout<<"times="<<times<<std::endl;
  std::cout<<"momentum="<<momentum<<std::endl;
  
  Plq.cmp_single(2*amomentm(0)/cutoffq-1., pl_Q); // Legendre polynomials at this Q
  Plt.cmp_single(2*(times(0)-times(1))/beta-1., ul_t);     // Legendre polynomials at this time


  int Nhist = 71;
  bl::Array<double,1> W_hist(Nhist);
  bl::Array<double,1> W2_hist(Nhist);
  bl::Array<double,1> Q_hist(Nhist);
  bl::Array<double,1> t_hist(Nhist);
  bl::Array<double,2> diag_contribution(Ndiags,Nhist);
  W_hist=0;
  W2_hist=0;
  Q_hist=0;
  t_hist=0;
  diag_contribution=0;
  
  bl::Array<bl::TinyVector<double,3>,2> tmom_G(Ndiags,2*Norder); // momentum for 2*Norder fermionic propagators
  bl::Array<bl::TinyVector<double,3>,2> tmom_V(Ndiags,  Norder); // momentum for Norder bosonic propagators
  bl::Array<int,2> changed_G(Ndiags,2*Norder), changed_V(Ndiags,Norder);
  bl::Array<int,2> changed_Vrtx(Ndiags,2*Norder);
  bl::Array<double,2> G_trial(Ndiags,2*Norder), V_trial(Ndiags,Norder), W_trial(Ndiags,Norder), Vrtx_trial(Ndiags, Norder);
  bl::Array<double,2> Vrtx_trial2(Ndiags, Norder);
  bl::Array<double,2> V_eqv_trial(Ndiags,Norder);
  bl::Array<double,1> PQs_trial(Ndiags);
  bl::Array<double,1> times_trial(2*Norder);
  
  int Qweight = 0;      // if Qweight==1 then we have the normalization diagram
  double Nweight = 0;   // how many times did we see measuring diagram.
  int NlowT = 0;        // how many times do we see external time < 0.05*beta
  int Nmeassure = 0;    // how many times did we measure quantities.
  double aver_sign = 0; // sign
  int Nacc_k = 0, Nacc_q = 0, Nacc_w = 0, Nacc_t = 0;
  int Nall_k = 0, Nall_q = 0, Nall_w = 0, Nall_t = 0;
  //if (mpi.rank==mpi.master)
    PrintInfo_(0, Qweight, amomentm(0), amomentm(1), times(0)-times(1), PQ, PQ, 0, 0);
  const double C2H=4;

  t_all.start();
  for (int itt=0; itt<p.Nitt; itt++){
    double PQ_new=0;
    int icase = tBisect(drand(), Prs); // which steps should we attempt?
    bool accept=false;
    if (icase==0){ // changing momentum Q or k-point
      int iloop = static_cast<int>(Nloops*drand());
      bl::TinyVector<double,3> K_new; double Ka_new, trial_ratio=1;
      /*bool*/ accept=false;
      if (iloop==0){ // this is the loop for the external momentum Q
	Nall_q += 1;
	accept = Find_new_Q_point(K_new, Ka_new, trial_ratio, momentum(iloop), amomentm(iloop), cutoffq, drand);
      }else{         // any other momentum point
	Nall_k += 1;
	accept = Find_new_k_point(K_new, Ka_new, trial_ratio, momentum(iloop), amomentm(iloop), cutoffk, drand, drand()<p.Qring, dkF, dRk);
      }
      if (accept){
	bl::TinyVector<double,3> dK = K_new - momentum(iloop); // how much momentum will be changed.
	tmom_G = mom_G; tmom_V = mom_V;
	changed_G = 0; changed_V = 0, changed_Vrtx = 0;
	for (int id=0; id<Ndiags; id++){ // updating momentum in all propagators
	  const std::vector<int>& lindex = Loop_index[id][iloop];
	  const std::vector<int>& ltype  = Loop_type[id][iloop];
	  for (int i=0; i<lindex.size(); i++){
	    if ( abs(ltype[i])==1 ){
	      int iv1 = lindex[i];
	      tmom_G(id, iv1) += dK * dsign(ltype[i]);
	      changed_G(id, iv1)=1;
	      // To correct for the unscreened part of the interaction, we need to update time-dependent interaction
	      changed_Vrtx(id, iv1)=1;
	      changed_Vrtx(id, diagsG(id,iv1) )=1;
	    }else{
	      int iv1 = lindex[i];
	      tmom_V(id, iv1) += dK * dsign(ltype[i]);
	      changed_V(id, iv1)=1;
	    }
	  }
	}
	//std::cout<<"------ changing momentum of loop "<<iloop<<" to "<<K_new<<" from "<<momentum(iloop)<<std::endl;
	//std::cout<<"------ changed_Vrtx="<<changed_Vrtx<<std::endl;
	if (! Qweight){ // we computed the polarization diagram
	  PQ_new=0;
	  for (int id=0; id<Ndiags; id++){
	    double PQd = 1.0;
	    for (int i=0; i<2*Norder; i++){
	      if (changed_G(id,i)){
		double aK = norm(tmom_G(id,i));
		int i_final = diagsG(id,i);
		double dt = times(i_final)-times(i);
		G_trial(id,i) = Gk(aK, dt);
		PQd *= G_trial(id,i);
	      }else{
		PQd *= G_current(id,i);
	      }
	    }
	    for (int ii=1; ii<Norder; ii++){
	      int i     = diagVertex(id,ii-1,0);
	      int itype = diagVertex(id,ii-1,1);
	      if (changed_Vrtx(id,i)){
		double t1 =  times(i);
		double t2 =  times( (i/2)*2 + (1-i%2) ); // partner in the interaction, i.e., (2,3),(4,5)...
		int i_m = i_diagsG(id,i);
		double ti = times(i_m);
		int i_p = diagsG(id,i);
		double to = times(i_p);
		double ki = norm(tmom_G(id,i_m));
		double ko = norm(tmom_G(id,i));
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
		}
	      }else{
		Vrtx_trial(id,ii) = Vrtx_current(id,ii);
		if (itype==2) Vrtx_trial2(id,ii) = Vrtx_current2(id,ii);
	      }
	    }
	    for (int i=1; i<Norder; i++){ // do not add V for the meassuring line
	      if (changed_V(id,i)){
		double aQ = norm(tmom_V(id,i));
		V_trial(id,i) = Vq(aQ);
		double dt = fabs(times(2*i)-times(2*i+1));
		intpar pq = Interp(aQ,qx);
		intpar pt = Interp(dt,taux);
		W_trial(id,i) = rW(pq,pt) * V_trial(id,i);
		int itype = diagVertex(id,i-1,1);
		if (!pure_Exchange(id,i)){
		  V_eqv_trial(id,i) = (rWom0(pq) + 1)*V_trial(id,i)/beta;
		  if (itype==1){
		    PQd *= ( (V_eqv_trial(id,i) - W_trial(id,i))*Vrtx_trial(id,i) +  W_trial(id,i) );
		  }else if (itype==2){
		    PQd *= Type2VertexContribution(id,i,true, vertex_companion, changed_V, W_trial, V_eqv_trial, W_current, V_eqv_current, Vrtx_trial, Vrtx_trial2);
		  }
		}else{
		  PQd *= W_trial(id,i);
		}
	      }else{
		if (!pure_Exchange(id,i)){
		  int itype = diagVertex(id,i-1,1);
		  if (itype==1){
		    PQd *= ( (V_eqv_current(id,i) - W_current(id,i))*Vrtx_trial(id,i) + W_current(id,i) );
		  }else if (itype==2){
		    PQd *= Type2VertexContribution(id,i,false, vertex_companion, changed_V, W_trial, V_eqv_trial, W_current, V_eqv_current, Vrtx_trial, Vrtx_trial2);
		  }
		}else{
		  PQd *=  W_current(id,i);
		}
	      }
	    }
	    PQd *= diagSign(id);
	    PQ_new += PQd;
	    PQs_trial(id) = PQd;
	  }
	}else{ // Qweight, hence we are in meassuring Hilbert space
	  double Ka_old = amomentm(iloop);
	  amomentm(iloop) = Ka_new;
	  bl::TinyVector<double,3> K_old = momentum(iloop);
	  momentum(iloop) = K_new;
	  PQ_new = V0norm * mweight(amomentm, momentum);
	  amomentm(iloop) = Ka_old;
	  momentum(iloop) = K_old;
	}
	double ratio = PQ_new*trial_ratio / PQ;
	accept = fabs(ratio) > 1-drand() || PQ==0;
      }
      
      if ((itt+1)%Ncout==0)
	//if (mpi.rank==mpi.master)
	  PrintInfo_(itt+1, Qweight, amomentm(iloop), amomentm((iloop+1)%Nloops), times(0), PQ_new, PQ, NlowT/(Nmeassure+0.0), Nweight/(Nmeassure+0.0));
      if (accept){
	if (iloop==0) Nacc_q += 1; else Nacc_k += 1;
	momentum(iloop)  = K_new;  // this momentum was changed
	amomentm(iloop) = Ka_new;
	mom_G = tmom_G;           // and therefore many momenta in diagrams have changed. We could optimize this, and 
	mom_V = tmom_V;           // change only those momenta that actually change....
	if (!Qweight){
	  for (int id=0; id<Ndiags; id++){
	    for (int i=0; i<2*Norder; i++)
	      if (changed_G(id,i))
		G_current(id,i) = G_trial(id,i); // The green's functions in the loop have been recalculated, and are now stored.
	    for (int ii=1; ii<Norder; ii++){
	      int i     = diagVertex(id,ii-1,0);
	      int itype = diagVertex(id,ii-1,1);
	      if (changed_Vrtx(id,i)){
		Vrtx_current(id,ii) = Vrtx_trial(id,ii);
		if (itype==2) Vrtx_current2(id,ii) = Vrtx_trial2(id,ii);
	      }
	    }
	    for (int i=1; i<Norder; i++) // do not add V for the meassuring line
	      if (changed_V(id,i)){
		V_current(id,i) = V_trial(id,i); // The interactions in the loop have been recalculated and stored.
		W_current(id,i) = W_trial(id,i);
		if (!pure_Exchange(id,i)) V_eqv_current(id,i) = V_eqv_trial(id,i);
	      }
	  }
	  PQs = PQs_trial;
	}else{
	  // since we have the measuring diagram, we did not calculate G_trial and V_trial before, hence we need to evaluate them now.
	  for (int id=0; id<Ndiags; id++){
	    for (int i=0; i<2*Norder; i++)
	      if (changed_G(id,i)){
		double aK = norm(tmom_G(id,i));
		int i_final = diagsG(id,i);
		double dt = times(i_final)-times(i);
		G_current(id,i) = Gk(aK, dt);
	      }
	    for (int ii=1; ii<Norder; ii++){
	      int i     = diagVertex(id,ii-1,0);
	      int itype = diagVertex(id,ii-1,1);
	      if (changed_Vrtx(id,i)){
		double t1 = times(i);
		double t2 = times( (i/2)*2 + (1-i%2) ); // partner in the interaction, i.e., (2,3),(4,5)...
		int i_m   = i_diagsG(id,i);
		double ti = times(i_m);
		int i_p   = diagsG(id,i);
		double to = times(i_p);
		double ki = norm(tmom_G(id,i_m));
		double ko = norm(tmom_G(id,i));
		Vrtx_current(id,ii) = MoveVertex(t2, t1, ti, to, ki, ko, Gk);
		if (itype==2){
		  Vrtx_current2(id,ii) = Vrtx_current(id,ii);
		  if (vertex_incoming(id,ii-1)){
		    int i_m_partner = (i_m/2)*2 + (1-i_m%2);
		    double ti_new = times(i_m_partner);
		    Vrtx_current(id,ii) = MoveVertex(t2, t1, ti_new, to, ki, ko, Gk);
		  }else{
		    int i_p_partner = (i_p/2)*2 + (1-i_p%2);
		    double to_new = times(i_p_partner);
		    Vrtx_current(id,ii) = MoveVertex(t2, t1, ti, to_new, ki, ko, Gk);
		  }
		}
	      }
	    }
	    for (int i=1; i<Norder; i++)
	      if (changed_V(id,i)){
		double aQ = norm(tmom_V(id,i));
		V_current(id,i) = Vq(aQ);
		double dt = fabs(times(2*i)-times(2*i+1));
		intpar pq = Interp(aQ,qx);
		intpar pt = Interp(dt,taux);
		W_current(id,i) = rW(pq,pt) * V_current(id,i);
		if (!pure_Exchange(id,i))
		  V_eqv_current(id,i) = (rWom0(pq) + 1)*V_current(id,i)/beta;
	      }
	  }
	}
	PQ = PQ_new;
	if (iloop==0) Plq.cmp_single(2*amomentm(iloop)/cutoffq-1., pl_Q);  // update Legendre Polynomials
      }
    } else if (icase==1){  // changing one of the time variables
      Nall_t += 1;
      int itime = static_cast<int>((2*Norder-1)*drand()); // which time to change? For bare interaction, there are only Norder different times.
      changed_G=0; changed_V=0;  // which propagators are being changed?
      changed_Vrtx=0;            // and vertices
      times_trial = times;       // times_trial will contain the trial step times.
      if (itime==0){             // this is the measuring time with vertex=0
	double t_new = rw.gtau(drand());
	if (t_new<1e-15) t_new = 1e-15; // to be on the right branchcut
	times_trial(0) = t_new;
	for (int id=0; id<Ndiags; id++){
	  int i_pre_vertex = i_diagsG(id,itime);
	  changed_G(id,itime)=1;         // these two propagators contain vertex=0.
	  changed_G(id,i_pre_vertex)=1;
	  int i_post_vertex = diagsG(id,itime);
	  changed_Vrtx(id,i_pre_vertex)=1; // might bot be needed if Order(times(itime),time(i_pre_vertex),time(i_pre_vertex_partner)) does not change
	  changed_Vrtx(id,i_post_vertex)=1;// might bot be needed if Order(times(itime),time(i_post_vertex),time(i_post_vertex_partner)) does not change
	}
      }else{
	itime += 1; // we do not change itime=1, because it is set to zero. We change itime=0,2,3,4,...2*N-1
	double t_new = beta*drand();
	if (t_new<1e-15) t_new = 1e-15; // to be on the right branchcut
	// screened interaction, hence changing only a single vertex.
	times_trial(itime) = t_new;
	for (int id=0; id<Ndiags; id++){
	  int i_pre_vertex = i_diagsG(id,itime);
	  changed_G(id,itime)=1;        // these two propagators are changed because of the new time.
	  changed_G(id,i_pre_vertex)=1;
	  changed_V(id, itime/2)=1;
	  int i_post_vertex = diagsG(id,itime);
	  int i_partner = (itime/2)*2 + (1-itime%2);
	  changed_Vrtx(id,itime)=1;
	  changed_Vrtx(id,i_partner)=1;
	  changed_Vrtx(id,i_pre_vertex)=1; // might bot be needed if Order(times(itime),time(i_pre_vertex),time(i_pre_vertex_partner)) does not change
	  changed_Vrtx(id,i_post_vertex)=1;// might bot be needed if Order(times(itime),time(i_post_vertex),time(i_post_vertex_partner)) does not change
	  int j = is_companion(id,i_partner);
	  if (j) changed_Vrtx(id,j)=1;
	  //if (j){
	  //std::cout<<" diag="<<id<<" Found companion "<<j<<" to itime="<<itime<<std::endl;
	  //}
	}
      }
      if (! Qweight){
	PQ_new=0; // recalculating PQ, taking into account one change of time.
	//std::cout<<"..... itt="<<itt<<" icase="<<icase<<" itime="<<itime<<std::endl;
	for (int id=0; id<Ndiags; id++){
	  double PQd = 1.0;
	  for (int i=0; i<2*Norder; i++){
	    if (changed_G(id,i)){  // this propagator is changed due to change of time.
	      double aK = norm(mom_G(id,i));
	      int i_final = diagsG(id,i);
	      double dt = times_trial(i_final)-times_trial(i);
	      G_trial(id,i) = Gk(aK, dt);
	      PQd *= G_trial(id,i);
	    }else{
	      PQd *= G_current(id,i); // this is unchanged, hence just using current value.
	    }
	  }
	  for (int ii=1; ii<Norder; ii++){
	    int i     = diagVertex(id,ii-1,0); // Now go trhrough all the vertices that need to be changed.
	    int itype = diagVertex(id,ii-1,1);
	    //std::cout<<"id="<<id<<" vertex change i="<<i<<" changed="<<changed_Vrtx(id,i)<<" itype="<<itype<<std::endl;
	    if (changed_Vrtx(id,i)){
	      double t1 = times_trial(i);
	      double t2 = times_trial( (i/2)*2 + (1-i%2) ); // partner in the interaction, i.e., (2,3),(4,5)...
	      int i_m   = i_diagsG(id,i);
	      double ti = times_trial(i_m);
	      int i_p   = diagsG(id,i);
	      double to = times_trial(i_p);
	      double ki = norm(mom_G(id,i_m));
	      double ko = norm(mom_G(id,i));
	      Vrtx_trial(id,ii) = MoveVertex(t2, t1, ti, to, ki, ko, Gk);
	      if (itype==2){
		Vrtx_trial2(id,ii) = Vrtx_trial(id,ii); // saving regular vertex to another array
		if (vertex_incoming(id,ii-1)){
		  int i_m_partner = (i_m/2)*2 + (1-i_m%2);
		  double ti_new = times_trial(i_m_partner);
		  Vrtx_trial(id,ii) = MoveVertex(t2, t1, ti_new, to, ki, ko, Gk);
		}else{
		  int i_p_partner = (i_p/2)*2 + (1-i_p%2);
		  double to_new = times_trial(i_p_partner);
		  Vrtx_trial(id,ii) = MoveVertex(t2, t1, ti, to_new, ki, ko, Gk);
		}
	      }
	    }else{
	      Vrtx_trial(id,ii) = Vrtx_current(id,ii);
	      if (itype==2) Vrtx_trial2(id,ii) = Vrtx_current2(id,ii);
	    }
	  }
	  for (int i=1; i<Norder; i++){ // The bare interaction does not depend on time, hence it is not changed here. But W is.
	    if (changed_V(id,i)){
	      double aQ = norm(mom_V(id,i));
	      double dt = fabs(times_trial(2*i)-times_trial(2*i+1));
	      intpar pq = Interp(aQ,qx);
	      intpar pt = Interp(dt,taux);
	      W_trial(id,i) = rW(pq,pt) * V_current(id,i);
	      int itype = diagVertex(id,i-1,1);
	      if (!pure_Exchange(id,i)){
		if (itype==1){
		  PQd *= ( (V_eqv_current(id,i) - W_trial(id,i)) *Vrtx_trial(id,i) + W_trial(id,i) );
		}else if (itype==2){
		  PQd *= Type2VertexContribution(id,i,true, vertex_companion, changed_V, W_trial, V_eqv_current, W_current, V_eqv_current, Vrtx_trial, Vrtx_trial2);
		}
	      }else{
		PQd *= W_trial(id,i);
	      }
	    }else{
	      if (!pure_Exchange(id,i)){
		int itype = diagVertex(id,i-1,1);
		if (itype==1){
		  PQd *= ( (V_eqv_current(id,i) - W_current(id,i))*Vrtx_trial(id,i) + W_current(id,i) );
		}else if (itype==2){
		  PQd *= Type2VertexContribution(id,i,false, vertex_companion, changed_V, W_trial, V_eqv_current, W_current, V_eqv_current, Vrtx_trial, Vrtx_trial2);
		}
	      }else{
		PQd *= W_current(id,i);
	      }
	    }
	  }
	  PQd *= diagSign(id);
	  PQ_new += PQd;
	  PQs_trial(id) = PQd;
	}
      }else{
	// measuring diagram.
	PQ_new = V0norm * mweight(amomentm, momentum /*times_trial*/);
      }
      double ratio = PQ_new/PQ;
      /*bool*/ accept = fabs(ratio) > 1-drand() || PQ==0;
      if ((itt+1)%Ncout==0)
	//if (mpi.rank==mpi.master)
	  PrintInfo_(itt+1, Qweight, amomentm(0), amomentm(1), times(itime), PQ_new, PQ, NlowT/(Nmeassure+0.0), Nweight/(Nmeassure+0.0));
      if (accept){
	Nacc_t +=1 ;
	times = times_trial;  // the new times are now accepted. We change the independent time variable, or two internal times.
	if (!Qweight){
	  for (int id=0; id<Ndiags; id++){
	    for (int i=0; i<2*Norder; i++)
	      if (changed_G(id,i))
		G_current(id,i) = G_trial(id,i);  // just saving the propagators which were changed.
	    for (int ii=1; ii<Norder; ii++){
	      int i     = diagVertex(id,ii-1,0);
	      int itype = diagVertex(id,ii-1,1);
	      if (changed_Vrtx(id,i)){
		Vrtx_current(id,ii) = Vrtx_trial(id,ii);
		if (itype==2) Vrtx_current2(id,ii) = Vrtx_trial2(id,ii);
	      }
	    }
	    for (int i=1; i<Norder; i++) // do not add V for the meassuring line
	      if (changed_V(id,i))
		W_current(id,i) = W_trial(id,i);
	  }
	  PQs = PQs_trial;
	}else{
	  for (int id=0; id<Ndiags; id++){
	    for (int i=0; i<2*Norder; i++)
	      if (changed_G(id,i)){
		int i_final = diagsG(id,i);
		G_current(id,i) = Gk( norm(mom_G(id,i)),  times(i_final)-times(i)); // these propagators were changed, but they were not computed before.
	      }
	    for (int ii=1; ii<Norder; ii++){
	      int i     = diagVertex(id,ii-1,0);
	      int itype = diagVertex(id,ii-1,1);
	      if (changed_Vrtx(id,i)){
		double t1 =  times(i);
		double t2 =  times( (i/2)*2 + (1-i%2) ); // partner in the interaction, i.e., (2,3),(4,5)...
		int i_m = i_diagsG(id,i);
		double ti = times(i_m);
		int i_p   = diagsG(id,i);
		double to = times(i_p);
		double ki = norm(mom_G(id,i_m));
		double ko = norm(mom_G(id,i));
		Vrtx_current(id,ii) = MoveVertex(t2, t1, ti, to, ki, ko, Gk);
		if (itype==2){
		  Vrtx_current2(id,ii) = Vrtx_current(id,ii);
		  if (vertex_incoming(id,ii-1)){
		    int i_m_partner = (i_m/2)*2 + (1-i_m%2);
		    double ti_new = times(i_m_partner);
		    Vrtx_current(id,ii) = MoveVertex(t2, t1, ti_new, to, ki, ko, Gk);
		  }else{
		    int i_p_partner = (i_p/2)*2 + (1-i_p%2);
		    double to_new = times(i_p_partner);
		    Vrtx_current(id,ii) = MoveVertex(t2, t1, ti, to_new, ki, ko, Gk);
		  }
		}
	      }
	    }
	    for (int i=1; i<Norder; i++){
	      if (changed_V(id,i)){
		double aQ = norm(mom_V(id,i));
		double dt = fabs(times(2*i)-times(2*i+1));
		intpar pq = Interp(aQ,qx); // ATTENTION: Note that we could store pq, because it is equal to the old aQ.
		intpar pt = Interp(dt,taux);
		W_current(id,i) = rW(pq,pt)*V_current(id,i);
	      }
	    }
	  }
	}
	PQ = PQ_new;
	if (itime==0) Plt.cmp_single(2*(times(0)-times(1))/beta-1., ul_t);  // update Legendre Polynomials
      }
    }else{  // normalization diagram step
      Nall_w += 1;
      if (Qweight){ // trying to jump back into physical hilbert space
	PQ_new=0;
	for (int id=0; id<Ndiags; id++){
	  double PQd = 1.0;
	  for (int i=0; i<2*Norder; i++) PQd *= G_current(id,i); // we have all propagators precomputed, hence it is easy to compute the diagrams.
	  for (int i=1; i<Norder; i++)
	    if (!pure_Exchange(id,i)){
	      int itype = diagVertex(id,i-1,1);
	      if (itype==1)
		PQd *= ( (V_eqv_current(id,i) - W_current(id,i))*Vrtx_current(id,i) + W_current(id,i) );
	      else if (itype==2)
		PQd *= Type2VertexContribution(id,i,vertex_companion, W_current, V_eqv_current, Vrtx_current, Vrtx_current2);
	    }else{
	      PQd *= W_current(id,i);
	    }
	  PQd *= diagSign(id);
	  PQs_trial(id) = PQd;
	  PQ_new += PQd;
	}
      }else{        // or if we are in the physical Hilbert space, we try to jump into the normalization diagram
	PQ_new = V0norm * mweight(amomentm, momentum/*, times*/);
      }
      double ratio = PQ_new/PQ;
      /*bool*/ accept = fabs(ratio) > 1-drand() || PQ==0;
      if (accept){
	Nacc_w += 1;
	PQ = PQ_new;
	if (Qweight) PQs = PQs_trial;
	Qweight = 1-Qweight;
      }
    }

    //std::cout<<" At itt="<<itt<<" icase="<<icase<<" Qweight="<<Qweight<<" accept="<<accept<<std::endl;
    if (itt>=Nwarm && itt%10000==0){
      if (!Qweight) CarefullyVerifyCurrentState3(itt, beta, Qweight, PQ, momentum, times, mom_G, mom_V, diagsG, i_diagsG, Gk, qx, taux, rW, diagSign, Loop_index, Loop_type, G_current, V_current, W_current, V_eqv_current, Vrtx_current, Vrtx_current2, diagVertex, pure_Exchange, rWom0, vertex_companion, vertex_incoming, PQs);
    }
    // Measuring
    if (itt>=Nwarm && itt%tmeassure==0){
      t_mes.start();
      Nmeassure += 1;
      double Qa = amomentm(0);
      double t = times(0)-times(1);
      //double sp = sign(PQ);
      //double cw = 1.0;
      double cw = 1./rw.wt(t);
      double sp = sign(PQ) * cw;
      int iq = std::min(int(Qa/cutoffq*Nq), Nq-1);
      int it = std::min(int(t/beta*Nt), Nt-1);
      if (Qweight){
	Pnorm += Qa*Qa * cw;                 // |Q|^2 * reweighting
	Nweight+=1;
      }else{
#ifdef DGER_BLAS
	int inct=1, incq=1, lda=Nlq+1, n1=Nlq+1, n2=Nlt+1;
	dger_(&n1, &n2, &sp, pl_Q.data(), &incq, ul_t.data(), &inct, C_Pln.data(), &lda);
#else
	for (int lt=0; lt<=Nlt; lt++)
	  for (int lq=0; lq<=Nlq; lq++)
	    C_Pln(lt,lq) += ul_t[lt]*pl_Q[lq]*sp;
#endif
	Pbin(it,iq) += sp; 
      }
      if (t < 0.05*beta){
	NlowT +=1 ;
      }
      aver_sign += sp;
      
      if (!Qweight){
	mweight.Add_to_K_histogram(dk_hist, momentum, cutoffq, cutoffk);
	/*
	int iik = static_cast<int>( amomentm(0)/cutoffq * Nbin);
	if (iik>=Nbin) iik=Nbin-1;
	K_hist(0,iik) += 1.0;
	for (int ik=1; ik<Nloops; ik++){
	  double k = amomentm(ik);
	  int iik = static_cast<int>(k/cutoffk * Nbin);
	  if (iik>=Nbin) iik=Nbin-1;
	  K_hist(ik,iik) += 1./(k*k);
	}
	*/
	for (int it=2; it<2*Norder; it++){
	  int iit = static_cast<int>(times(it)/beta * Nbin);
	  if (iit>=Nbin) iit=Nbin-1;
	  T_hist(it-2,iit) += 1;
	}
      }

      if ( itt%(10000*tmeassure) == 0){
	double occurence = static_cast<double>(Nweight)/static_cast<double>(Nmeassure);
	if ( occurence > 0.25){ // decrease by two
	  V0norm /= 1.02;
	  Pnorm /= 1.02;
	  //if (mpi.rank==mpi.master)
	    std::cout<<"occurence="<<occurence<<" V0 reduced to "<<V0norm<<std::endl;
	}
	if ( occurence < 0.01){ // increase by two
	  V0norm *= 1.02;
	  Pnorm *= 1.02;
	  //if (mpi.rank==mpi.master)
	    std::cout<<"occurence="<<occurence<<" V0 increased to "<<V0norm<<std::endl;
	}
      }
      if (!Qweight){
	/// The code below measures various histograms.
	double unity = 1.0/(Ndiags*(Norder-1.));
	for (int id=0; id<Ndiags; id++){
	  for (int i=1; i<Norder; i++){
	    double x = W_current(id,i)/V_current(id,i);
	    int ii = static_cast<int>( (2*x+1)*Nhist/2. );
	    if (ii>=Nhist) ii=Nhist-1;
	    if (ii<0) ii=0;
	    W_hist(ii)  += unity;
	  
	    double y = W_current(id,i)/(V_current(id,i)*Vrtx_current(id,i)/beta + W_current(id,i));
	    int jj = static_cast<int>( (y/C2H + 1) * Nhist/2. );
	    if (jj>=Nhist) jj=Nhist-1;
	    if (jj<0) jj=0;
	    W2_hist(jj) += unity;

	    double aQ = norm(mom_V(id,i));
	    int iiq = static_cast<int>(aQ/cutoffk * Nhist);
	    if (iiq>=Nhist) iiq=Nhist-1;
	    if (iiq<0) iiq=0;
	    double dt = fabs(times(2*i)-times(2*i+1));
	    int iit = static_cast<int>(dt/beta * Nhist);
	    if (iit>=Nhist) iit=Nhist-1;
	    if (iit<0) iit=0;
	    Q_hist(iiq) += unity;
	    t_hist(iit) += unity;
	  }
	  if (fabs(Qa)<0.1){
	    double y = PQs(id)/fabs(PQ);
	    int ih = static_cast<int>((y+1.5)/3.*Nhist);
	    if (ih<0) ih=0;
	    if (ih>=Nhist) ih=Nhist-1;
	    diag_contribution(id, ih) += 1;
	  }
	}
      }
      if (itt%(500000*tmeassure) == 0){
	dk_hist *= mweight.Normalize_K_histogram();
	mweight.Recompute(/*K_hist, T_hist, mpi.rank==mpi.master*/);
	if (Qweight) PQ = V0norm * mweight(amomentm, momentum/*, times*/);
      }
      t_mes.stop();
    }
  }

  t_all.stop();
  Pnorm     *= 1.0/Nmeassure;
  aver_sign *= 1.0/Nmeassure;
  Nweight   *= 1.0/Nmeassure;
  C_Pln     *= 1.0/Nmeassure;
  Pbin      *= 1.0/Nmeassure;

  W_hist    *= 1.0/Nmeassure;
  W2_hist   *= 1.0/Nmeassure;
  Q_hist    *= 1.0/Nmeassure;
  t_hist    *= 1.0/Nmeassure;

  diag_contribution *= 1.0/Nmeassure;
  
  C_Pln *= 1.0/(4*pi); // 4*pi because we must divide by 1/(4*pi*q**2), and 1/q**2 was already done in sampling
  C_Pln *= (fabs(V0norm)/Pnorm);
  Pbin  *= 1.0/(4*pi);
  Pbin  *= (fabs(V0norm)/Pnorm);
  std::clog<<"total acceptance rate="<<(Nacc_k+Nacc_q+Nacc_t+Nacc_w)/(p.Nitt+0.0)<<" "<<" k-acceptance="<<(Nacc_k/(Nall_k+0.0))<<" q-acceptance="<<(Nacc_q/(Nall_q+0.0))<<" t-acceptanece="<<(Nacc_t/(Nall_t+0.0))<<" w-acceptanece="<<(Nacc_w/(Nall_w+0.0))<<std::endl;
  std::clog<<" k-trials="<<(Nall_k/(p.Nitt+0.0))<<" q-trials="<<(Nall_q/(p.Nitt+0.0))<<" t-trials="<<(Nall_t/(p.Nitt+0.0))<<" w-trials="<<(Nall_w/(p.Nitt+0.0))<<std::endl;
  std::clog<<"  meassuring diagram occurence frequency="<<Nweight<<" and its norm Pnorm="<<Pnorm<<std::endl;
  std::clog<<"total time="<<t_all.elapsed()<<" measuring time="<<t_mes.elapsed()<<std::endl;


  std::ofstream ohist1("histogramWr.dat");
  for (int i=0; i<Nhist; i++)
    ohist1<< (i+0.5)/Nhist-1./2. <<" "<<W_hist(i)<<std::endl;
  std::ofstream ohist2("histogramWrW.dat");
  for (int i=0; i<Nhist; i++)
    ohist2<< (2*(i+0.5)/Nhist-1)*(C2H) <<" "<<W2_hist(i)<<std::endl;
  std::ofstream ohist3("histogramQ.dat");
  for (int i=0; i<Nhist; i++)
    ohist3<< (i+0.5)/Nhist * cutoffk/p.kF <<" "<<Q_hist(i)<<std::endl;
  std::ofstream ohist4("histogramT.dat");
  for (int i=0; i<Nhist; i++)
    ohist4<< (i+0.5)/Nhist <<" "<<t_hist(i)<<std::endl;
  for (int id=0; id<Ndiags; id++){
    std::ofstream ohist5( (std::string("dhist.")+std::to_string(id)).c_str());
    for (int i=0; i<Nhist; i++)
      ohist5<< 3*(i+0.5)/Nhist-1.5 << " "<< log10(diag_contribution(id, i)) <<std::endl;
    ohist5.close();
  }
  
  //K_hist *= 1.0/Nmeassure;
  T_hist *= 1.0/Nmeassure;
  for (int ik=0; ik<Nloops; ik++){
    std::ofstream hout((std::string("Q_hist.")+std::to_string(ik)).c_str());
    double cutoff = (ik>0)? cutoffk : cutoffq;
    for (int i=0; i<Nbin; i++){
      hout<< (i+0.5)/Nbin*cutoff/p.kF << " " << mweight.K_hist(ik,i)<<std::endl;
    }
  }
  for (int it=2; it<2*Norder; it++){
    std::ofstream hout((std::string("T_hist.")+std::to_string(it)).c_str());
    for (int i=0; i<Nbin; i++){
      hout<< (i+0.5)/Nbin*beta << " " << T_hist(it-2,i)<<std::endl;
    }
  }
}
