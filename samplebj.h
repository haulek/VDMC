// @Copyright 2018 Kristjan Haule and Kun Chen    
#include <iostream>
#include <blitz/array.h>
#include <algorithm>
#include "random.h"
#include "sample0.h"
#include "legendre.h"
#include "unique.h"
#include "bkgroups.h"
#include "baymkadanoff.h"
#include "debug.h"
#include "dynamic.h"
#define _TIME
#include "timer.h"
#include "mmpi.h"
#include "vectlist.h"
#include <cmath>

#define CNTR_VN    

template<typename GK, typename Data>
void sample_static_fastC(Data& BKdata, bl::Array<double,2>& Pbin, std::ostream& log,
			 double lmbda, const std::vector<double>& lmbda_spct, const GK& Gk, const params& p,
			 const bl::Array<unsigned short,2>& diagsG, const bl::Array<float,1>& diagSign,
			 const std::vector<std::vector<std::vector<int> > >& Loop_index, const std::vector<std::vector<std::vector<int> > >& Loop_type,
			 bl::Array<char,2>& Vtype, my_mpi& mpi)
{
  typedef double real;
//#define MPI_VNREAL MPI_DOUBLE
//  typedef float real;
//#define MPI_VNREAL MPI_FLOAT

  log.precision(12);
  if ( BKdata.Nlt!=p.Nlt || BKdata.Nlq!=p.Nlq){
    log<<"ERROR : Dimensions of C_Pln is wrong : either "<<BKdata.Nlt<<" != "<<p.Nlt<<" or "<<BKdata.Nlq<<" != "<<p.Nlq<<std::endl;
    exit(1);
  }
  
  if (Pbin.extent(0)!=p.Nt || Pbin.extent(1)!=p.Nq) Pbin.resize(p.Nt,p.Nq); Pbin=0.0;

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
  
  CounterCoulombV Vqc(lmbda);
  
  int tmeassure=p.tmeassure, Ncout=p.Ncout, Nwarm=p.Nwarm;
  double V0norm = p.V0norm, dkF = p.dkF, kF=p.kF;// V0exp= p.V0exp;
  double cutoffk = p.cutoffk, dRk = p.dRk;
  double beta = p.beta;
  double cutoffq = p.cutoffq;
  int Nt = p.Nt, Nq = p.Nq;


  // THIS PART IS DIFFERENT
  // Everything for Legendre Polynomials in time and momentum
  LegendrePl Pln(std::max(BKdata.Nlq,BKdata.Nlt));
  bl::Array<double,1> pl_Q(BKdata.Nlq+1);  // Will contain legendre Pl(2*q/qmax-1)
  bl::Array<double,1> pl_t(BKdata.Nlt+1);  // Will contain legendre Pl(2*t//beta-1)
  BKdata.C_Pln = 0.0;            // cummulative result for legendre expansion
  // THIS PART IS DIFFERENT
  
  double Pnorm = 0.0;            // normalization diagram value
  // Here we set Vtype(id,i) => -Vtype(id,i) for those interactions which contain single-particle counter terms.
  Where_to_Add_Single_Particle_Counter_Term(Vtype, lmbda_spct, Ndiags, Norder, diagsG);
  // Now that Vtype is properly adjusted for single-particle counter term, we can proceed to find unique propagators.
  typedef unsigned short sint;
  bl::Array<sint,2> Gindx, Vindx;
  bl::Array<bl::TinyVector<int,2>,1> gindx, vindx;
  // Finding Unique propagators for G and V
  int N0v=0;
  findUnique_noindx_spct(Gindx, Vindx, gindx, vindx, N0v, diagsG, Vtype, Loop_index, Loop_type, 0, false, false);

  //findCommonOccurence(Gindx, Vindx, gindx, vindx, diagsG, Loop_index, Loop_type);
  
  // Once more we go through single-particle counter terms and create index single_counter_index
  bl::Array<int,1> single_counter_index;
  Find_Single_Particle_Counter_G_index(single_counter_index, Ndiags, Norder, vindx.extent(0), diagsG, Vtype, Gindx, Vindx);
  
  // This is related to Baym-Kadanoff approach. For efficiency, we want to find which diagram have common values of momenta at the right hand side of the diagram.
  BKdata.FindGroups(diagsG, Loop_index, Loop_type, log);
  
  if (Loop_index.size()!=Ndiags){ log<<"Wrong size of Loop_index "<<Loop_index.size()<<" instead of "<<Ndiags<<std::endl; exit(1);}
  int Nloops = Loop_index[0].size();
  if (Nloops != Norder+1){ log<<"Expecting Nloops==Norder+1 in Loops_index, but got Nloops="<<Nloops<<" Norder="<<Norder<<std::endl; exit(1);}
  int Ng = gindx.extent(0);
  int Nv = vindx.extent(0);
  int _Nv_ = std::max(Nv,1); // can not assign 0 to empty file

  bl::Array<bl::Array<sint,1>,1> loop_Gkind, loop_Vqind;
  bl::Array<bl::Array<char,1>,1> loop_Gksgn, loop_Vqsgn;
  int Ngp, Nvp;
  // Find index-tables to all momentum loops, so that we do not need to loop over all diagrams, but rather only over all differet propagators corresponding to certain loop.
  // Here we also take into account that single-particle counter terms are sometimes added.
  Get_GVind(loop_Gkind, loop_Gksgn, loop_Vqind, loop_Vqsgn, Ngp, Nvp, Loop_index, Loop_type, Gindx, Vindx, single_counter_index, lmbda_spct, Vtype);
  int Ngp_tr = GetMax_NGp_trial(i_diagsG, Gindx, Ng, Ngp);

  // DEBUG
  if (mpi.rank==mpi.master){
    log<<"lmbda_spct=";
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
    log<<"loop_Vqind:"<<std::endl;
    for (int iloop=0; iloop<Nloops; ++iloop){
      for (int iq=0; iq<loop_Vqind(iloop).extent(0); ++iq){ // Now do the same for the interaction
	int ii   = loop_Vqind(iloop)(iq);
	int isgn = loop_Vqsgn(iloop)(iq);

	int id = vindx(ii)[0];
	int  i = vindx(ii)[1];
	int vtyp = Vtype(id,i);

	log<<"iloop="<<iloop<<" iq="<<iq<<" ii_v="<<ii<<" isgn="<<isgn<<" which comes from id="<< id <<" and pair =("<<2*i<<","<<2*i+1<<") of type="<<vtyp<<std::endl;
      }
    }
  }
  // END DEBUG
  
  Reweight_time rw(beta,p.lmbdat);
  int Nbin = 513;
  //int Nbin = 129;
  bl::Array<double,1> lmbda_spct_cutoffk(lmbda_spct.size());
  for (int i=0; i<lmbda_spct.size(); ++i)
    lmbda_spct_cutoffk(i) = lmbda_spct[i]*(2*pi*2*pi*2*pi)/(4*pi*ipower(cutoffk,3)/3.);
  
  meassureWeight mweight(p.V0exp, p.cutoffk, kF, Nbin, Nloops);
  //bl::Array<double,2> K_hist(Nloops,Nbin); K_hist=0;
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
    times(2*it+1) = tv;
  }
  // Next the momenta. For a start with nonzero weight, we want to start close to kF.
  {
    double Qa = p.kF*drand(); // absolute value of external momentum
    momentum(0) = 0.0,0.0,Qa; // external momentum Q
    amomentm(0) = Qa;
    double kf_3 = p.kF/sqrt(3.);
    for (int ik=1; ik<Nloops; ik++){
      momentum(ik) = drand()*kf_3, drand()*kf_3, drand()*kf_3;
      amomentm(ik) = norm(momentum(ik));
    }
  }

  log<<"Starting times="<<times<<" and starting momenta="<<momentum<<std::endl;

  bl::Array<double,1> G_current(Ng), V_current(_Nv_);
  // This is momentum for all diagrams and all propagators
  bl::Array<bl::TinyVector<real,3>,1> mom_g(Ng), mom_v(Nv);
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
      for (int i=0; i<2*Norder; i++){
	//int i_final = diagsG(id,i);
	int i_previ = i_diagsG(id,i);
	bl::TinyVector<double,3> k_in = mom_G(id,i_previ);
	bl::TinyVector<double,3> k_out = mom_G(id,i);
	bl::TinyVector<double,3> q = mom_V(id,i/2)*(1.-2*(i%2));
	if (fabs(norm(k_in-k_out)-norm(q))>1e-6){
	  log<<"ERROR: diagram "<<id<<" vertex i="<<i<<" k_in="<<k_in<<" k_out="<<k_out<<" q="<<q<<std::endl;
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
      double dt = times(i_final)-times(i);
      G_current(ii) = Gk(aK, dt);
    }
    V_current=0;  
    for (int ii=0; ii<Nv; ++ii){
      int id = vindx(ii)[0];
      int  i = vindx(ii)[1];
      double aQ = norm(mom_V(id,i));
      int vtyp = Vtype(id,i);
      double Vq = Vqc(aQ,abs(vtyp));
      if (vtyp>=0){
	V_current(ii) = Vq;
      }else{// we also add the single-particle counter term in addition to two-particle counter term.
	int ii_g = single_counter_index(ii);
	double g_kq = G_current(ii_g);
	V_current(ii) = Vq;
	if (g_kq!=0) V_current(ii) += lmbda_spct_cutoffk(-vtyp-1)/g_kq;
      }
    }
    //log<<"V_current="<<V_current<<endl;
    PQ=0;
    BKdata.PQg_Initialize();
    for (int id=0; id<Ndiags; id++){
      double PQd = 1.0;
      for (int i=0; i<2*Norder; i++) PQd *= G_current(Gindx(id,i));
      for (int i=1; i<Norder; i++) PQd *= V_current(Vindx(id,i));
      PQd *= diagSign(id);
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

  // Now computing legendre polinomials in q and tau for the initial configuration
  Pln.cmp_single(2*amomentm(0)/cutoffq-1., pl_Q, BKdata.Nlq);          // Legendre polynomials at this Q
  Pln.cmp_single(2*(times(0)-times(1))/beta-1., pl_t, BKdata.Nlt);     // Legendre polynomials at this time

  bl::Array<double,1> times_trial(2*Norder);
  int _Nvp_ = std::max(Nvp,1); // can not assign 0 to empty file
  bl::Array<bl::TinyVector<real,3>,1> tmom_g(Ngp), tmom_v(_Nvp_);
  bl::Array<sint,1> ip_ind(Ng), iq_ind(Nv);
  bl::Array<double,1> g_trial(Ngp_tr), v_trial(_Nvp_);
  BitArray changed_G(Ng), changed_V(Nv);

  int Qweight = 0;      // if Qweight==1 then we have the normalization diagram
  double Nweight = 0;   // how many times did we see measuring diagram.
  int NlowT = 0;        // how many times do we see external time < 0.05*beta
  int Nmeassure = 0;    // how many times did we measure quantities.
  int Nchanged  = 0;    // when was V0norm last changed?
  double Wchanged = 0;  // what was Nweight when we last time changed V0norm?
  int Nacc_k = 0, Nacc_q = 0, Nacc_w = 0, Nacc_t = 0;
  int Nall_k = 0, Nall_q = 0, Nall_w = 0, Nall_t = 0;
  if (mpi.rank==mpi.master) PrintInfo_(0, Qweight, amomentm(0), amomentm(1), times(0)-times(1), PQ, PQ, 0, 0, log);
  
  t_all.start();
  for (int itt=0; itt<p.Nitt; itt++){
    double PQ_new=0;
    int icase = tBisect(drand(), Prs); // which steps should we attempt?
    if (icase==0){ // changing momentum Q or k-point
      int iloop = static_cast<int>(Nloops*drand());
      bl::TinyVector<double,3> K_new; double Ka_new; double trial_ratio=1;
      bool accept=false;
      if (iloop==0){ // this is the loop for the external momentum Q
	Nall_q += 1;
	accept = Find_new_Q_point(K_new, Ka_new, trial_ratio, momentum(iloop), amomentm(iloop), cutoffq, drand);
      }else{         // any other momentum point
	Nall_k += 1;
	accept = Find_new_k_point(K_new, Ka_new, trial_ratio, momentum(iloop), amomentm(iloop), cutoffk, drand, drand()<p.Qring, dkF, dRk);
      }
      if (accept){
	bl::TinyVector<double,3> dK = K_new - momentum(iloop); // how much momentum will be changed.
	//t1.start(); // takes 40% of the time
	changed_G = 0;
	for (int ip=0; ip<loop_Gkind(iloop).extent(0); ++ip){ // all G-propagators, which are changed when momentum in certain loop is changed.
	  int ii   = loop_Gkind(iloop)(ip);  // index of G-propagators, which we are now changing
	  double isgn = loop_Gksgn(iloop)(ip);  // momentum can be added or subtracted on this propagator.
	  bl::TinyVector<double,3> k = mom_g(ii) + dK * isgn; // actually changing the momentum
	  tmom_g(ip) = k;                    // remember the momentum
	  changed_G.set(ii,1);                   // remember that it is changed
	  if (!Qweight){
	    double aK = norm(k);
	    int id = gindx(ii)[0];           // find first diagram with this propagator
	    int  i = gindx(ii)[1];           // and which propagator is this in the diagram
	    int i_final = diagsG(id,i);      // times in the propagators.
	    double dt = times(i_final)-times(i);
	    g_trial(ip) = Gk(aK, dt);        // remeber the value of the propagator
	    ip_ind(ii)=ip;                   // remeber where it is stored
          }
	}
	//t1.stop();
	//t2.start(); // takes 1.5% of the time
	changed_V = 0;
	for (int iq=0; iq<loop_Vqind(iloop).extent(0); ++iq){ // Now do the same for the interaction
	  int ii   = loop_Vqind(iloop)(iq);
	  double isgn = loop_Vqsgn(iloop)(iq);
	  bl::TinyVector<double,3> q =  mom_v(ii) + dK * isgn;
	  tmom_v(iq) = q;
	  changed_V.set(ii,1);
	  if (!Qweight){
	    double aQ = norm(q);
	    int id = vindx(ii)[0];
	    int  i = vindx(ii)[1];
	    int vtyp = Vtype(id,i);
	    double Vq = Vqc(aQ, abs(vtyp)); // This is Vq^2*lambda/(8*pi)
	    if (vtyp>=0){
	      v_trial(iq) = Vq;
	    }else{
	      // we also add the single-particle counter term in addition to two-particle counter term.
	      int ii_g = single_counter_index(ii);
	      double g_kq = changed_G[ii_g] ? g_trial(ip_ind(ii_g)) : G_current(ii_g);
	      v_trial(iq) = Vq;
	      if (g_kq!=0) v_trial(iq) += lmbda_spct_cutoffk(-vtyp-1)/g_kq;
	    }
	    iq_ind(ii)=iq;
	  }
	}
	//t2.stop();
	if (! Qweight){ // we computed the polarization diagram
	  //t3.start();// takes 22% of the time
	  PQ_new=0;
	  BKdata.PQg_new_Initialize();
	  for (int id=0; id<Ndiags; id++){
	    double PQd = 1.0;
	    for (int i=0; i<2*Norder; i++){
	      int ii = Gindx(id,i);
	      PQd *= ( changed_G[ii] ? g_trial(ip_ind(ii)) : G_current(ii) );
	    }
	    for (int i=1; i<Norder; i++){
	      int ii = Vindx(id,i);
	      PQd *= ( changed_V[ii] ? v_trial(iq_ind(ii)) : V_current(ii) );
	    }
	    PQd *=  diagSign(id);
	    PQ_new += PQd;
	    BKdata.PQg_new_Add(id,PQd);
	  }
	  //t3.stop();
	}else{
	  double Ka_old = amomentm(iloop);
	  amomentm(iloop) = Ka_new;
	  bl::TinyVector<double,3> K_old = momentum(iloop);
	  momentum(iloop) = K_new;
	  PQ_new = V0norm * mweight(amomentm, momentum);
	  amomentm(iloop) = Ka_old;
	  momentum(iloop) = K_old;
	}
	double ratio = PQ_new*trial_ratio / PQ;
	accept = fabs(ratio) > 1-drand() && PQ_new!=0;
      }
      if ((itt+1)%Ncout==0 && mpi.rank==mpi.master) _PrintInfo_(itt+1, Qweight, icase, amomentm(1), amomentm(Nloops-1), times(0), PQ_new, PQ, dk_hist, Nweight/(Nmeassure+0.0), log);
      if (accept){
	if (iloop==0) Nacc_q += 1; else Nacc_k += 1;
	momentum(iloop) = K_new;  // this momentum was changed
	amomentm(iloop) = Ka_new;
	for (int ip=0; ip<loop_Gkind(iloop).extent(0); ++ip){
	  int ii   = loop_Gkind(iloop)(ip); 
	  mom_g(ii) = tmom_g(ip);
	}
	for (int iq=0; iq<loop_Vqind(iloop).extent(0); ++iq){ // Now do the same for the interaction
	  int ii   = loop_Vqind(iloop)(iq);
	  mom_v(ii) = tmom_v(iq);
	}
	if (!Qweight){
	  for (int ip=0; ip<loop_Gkind(iloop).extent(0); ++ip){ // all G-propagators, which are changed when momentum in certain loop is changed.
	    int ii   = loop_Gkind(iloop)(ip);  // index of G-propagators, which we are now changing
	    G_current(ii)=g_trial(ip);
	  }
	  for (int iq=0; iq<loop_Vqind(iloop).extent(0); ++iq){ // Now do the same for the interaction
	    int ii   = loop_Vqind(iloop)(iq);
	    V_current(ii) = v_trial(iq);
	  }
	}else{
	  // since we have the measuring diagram, we did not calculate G_trial and V_trial before, hence we need to evaluate them now.
	  //t5.start(); // takes 3% of the time
	  for (int ip=0; ip<loop_Gkind(iloop).extent(0); ++ip){ // all G-propagators, which are changed when momentum in certain loop is changed.
	    int ii    = loop_Gkind(iloop)(ip);  // index of G-propagators, which we are now changing
	    double aK = norm(tmom_g(ip));
	    int id = gindx(ii)[0];           // find first diagram with this propagator
	    int  i = gindx(ii)[1];           // and which propagator is this in the diagram
	    int i_final = diagsG(id,i);      // times in the propagators.
	    double dt = times(i_final)-times(i);
	    G_current(ii) = Gk(aK, dt);        // remeber the value of the propagator
	  }
	  for (int iq=0; iq<loop_Vqind(iloop).extent(0); ++iq){ // Now do the same for the interaction
	    int ii   = loop_Vqind(iloop)(iq);
	    double aQ = norm(tmom_v(iq));
	    int id = vindx(ii)[0];
	    int  i = vindx(ii)[1];
	    int vtyp = Vtype(id,i);         // is it counterterm (vtyp!=0) of just coulomb (vtyp==0)
	    double Vq = Vqc(aQ, abs(vtyp)); // This is Vq*(Vq*lambda/(8*pi))^n
	    if (vtyp>=0){
	      V_current(ii) = Vq;
	    }else{
	      // we also add the single-particle counter term in addition to two-particle counter term.
	      int ii_g = single_counter_index(ii);
	      double g_kq = G_current(ii_g);
	      V_current(ii) = Vq;
	      if (g_kq!=0) V_current(ii) += lmbda_spct_cutoffk(-vtyp-1)/g_kq;
	    }
	  }
	  //t5.stop();
	}
	PQ = PQ_new;
	if (!Qweight) BKdata.Set_PQg_new_to_PQg();
	//t6.start(); // takes 0.5% time
	if (iloop==0) Pln.cmp_single(2*amomentm(iloop)/cutoffq-1., pl_Q, BKdata.Nlq);  // update Legendre Polynomials
	//t6.stop();
      }
    } else if (icase==1){  // changing one of the time variables
      Nall_t += 1;
      int itime = static_cast<int>(Norder*drand()); // which time to change? For bare interaction, there are only Norder different times.
      //log<<"    itime="<<itime<<endl;
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
	for (int ivertex=2*itime; ivertex<=2*itime+1; ++ivertex){ // 2*itime and 2*itime+1 are changed at the same time, because they are equal
	  times_trial(ivertex) = t_new;
	  for (int id=0; id<Ndiags; id++){
	    int i_pre_vertex = i_diagsG(id,ivertex);
	    changed_G.set(Gindx(id,ivertex),1);     // these two propagators are changed because of the new time.
	    changed_G.set(Gindx(id,i_pre_vertex),1);
	  }
	}
      }
      if (! Qweight){
	//t7.start(); // takes 11% of the time
	int ip=0;
	for (int ii=0; ii<Ng; ++ii){
	  if (changed_G[ii]){
	    int id = gindx(ii)[0];
	    int  i = gindx(ii)[1];
	    double aK = norm(mom_g(ii));
	    int i_final = diagsG(id,i);
	    double dt = times_trial(i_final)-times_trial(i);
	    g_trial(ip) = Gk(aK, dt);
	    ip_ind(ii)=ip;
	    ip+=1;
	  }
	}
	//t7.stop();
	//t8.start(); // takes 8% of the time
	PQ_new=0;   // recalculating PQ, taking into account one change of time.
	BKdata.PQg_new_Initialize();
	for (int id=0; id<Ndiags; id++){
	  double PQd = 1.0;
	  for (int i=0; i<2*Norder; i++){
	    int ii = Gindx(id,i);
	    PQd *= ( changed_G[ii] ? g_trial(ip_ind(ii)) : G_current(ii) );
	  }
	  for (int i=1; i<Norder; i++) PQd *= V_current(Vindx(id,i)); // interaction does not depend on time, hence it is not changed here.
	  PQd *= diagSign(id);
	  PQ_new += PQd;
	  BKdata.PQg_new_Add(id,PQd);
	}
	//t8.stop();
      }else{
	PQ_new = V0norm * mweight(amomentm, momentum);
      }
      double ratio = PQ_new/PQ;
      bool accept = fabs(ratio) > 1-drand() && PQ_new!=0;
      if ((itt+1)%Ncout==0 && mpi.rank==mpi.master)
	_PrintInfo_(itt+1, Qweight, icase, amomentm(1), amomentm(Nloops-1), times(2*itime), PQ_new, PQ, dk_hist, Nweight/(Nmeassure+0.0),log);
      if (accept){
	//t9.start(); // takes 1.7% of time
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
	      G_current(ii) = Gk( norm(mom_g(ii)),  times(i_final)-times(i)); // these propagators were changed, but they were not computed before.
	    }
	  }
	}
	//t9.stop();
	PQ = PQ_new;
	if (!Qweight) BKdata.Set_PQg_new_to_PQg();
	//t10.start(); // takes 0.5% of the time
	if (itime==0) Pln.cmp_single(2*(times(0)-times(1))/beta-1., pl_t, BKdata.Nlt);  // update Legendre Polynomials
	//t10.stop();
      }
    }else{  // normalization diagram step
      Nall_w += 1;
      //t11.start(); // takes 0.3% of the time
      if (Qweight){ // trying to jump back into physical hilbert space
	PQ_new=0;
	BKdata.PQg_new_Initialize();
	for (int id=0; id<Ndiags; id++){
	  double PQd = 1.0;
	  for (int i=0; i<2*Norder; i++) PQd *= G_current(Gindx(id,i)); // we have all propagators precomputed, hence it is easy to compute the diagrams.
	  for (int i=1; i<Norder; i++) PQd *= V_current(Vindx(id,i));
	  PQd *= diagSign(id);
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
      //t11.stop();
    }

    //if (itt>=Nwarm && itt%10000==0){
    //  if (!Qweight) VerifyCurrentState10(itt,beta,lmbda_spct,PQ,momentum,times,diagsG,i_diagsG,Vtype,Gindx,Vindx,Gk,diagSign,Loop_index,Loop_type,BKgroups,BKindex,Vqc,G_current,V_current);
    //}
    
    // Measuring
    if (itt>=Nwarm && itt%tmeassure==0){
      t_mes.start(); // takes 1% time
      Nmeassure += 1;
      double Qa = amomentm(0);
      double t = times(0)-times(1);
      double cw = 1./rw.wt(t);
      double sp = sign(PQ) * cw;
      int iq = std::min(int(Qa/cutoffq*Nq), Nq-1);
      int it = std::min(int(t/beta*Nt), Nt-1);
      if (Qweight){
	Pnorm += Qa*Qa * cw;                 // |Q|^2 * reweighting
	Nweight+=1;
      }else{
	Pbin(it,iq) += sp;
	BKdata.Meassure(itt,tmeassure,PQ,sp,pl_Q,pl_t,momentum(0),amomentm(0),mom_g,Gindx);
      }
      if (t < 0.05*beta){
	NlowT +=1 ;
      }
      if (!Qweight){
	mweight.Add_to_K_histogram(dk_hist, momentum, cutoffq, cutoffk);
	/*
	int iik = static_cast<int>( amomentm(0)/cutoffq * Nbin);
	if (iik>=Nbin) iik=Nbin-1;
	K_hist(0,iik) += dk_hist;
	for (int ik=1; ik<Nloops; ik++){
	  double k = amomentm(ik);
	  int iik = static_cast<int>(k/cutoffk * Nbin);
	  if (iik>=Nbin) iik=Nbin-1;
	  K_hist(ik,iik) += dk_hist/(k*k);
	}
	*/
      }
      //if ( itt>0 && itt%(2000*tmeassure) == 0){
      if ( itt>2e5/mpi.size && itt%(static_cast<int>(100000/mpi.size+1000)*tmeassure) == 0){
	double last_occurence = (Nweight-Wchanged)/(Nmeassure-Nchanged); // occurence from the last time we changed it
#ifdef _MPI
	double total_occurence=0;
	MPI_Allreduce(&last_occurence, &total_occurence, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	last_occurence = total_occurence/mpi.size;
#endif	
	if ( last_occurence > 0.15){ // decrease by two
	  V0norm /= 2.;
	  Pnorm /= 2.;
	  Nchanged = Nmeassure;
	  Wchanged = Nweight;
	  if (mpi.rank==mpi.master) log<<std::left<<(itt/1.0e6)<<"M occurence="<<std::setw(8)<<last_occurence<<" V0 reduced to "<<V0norm<<std::endl;
	}
	if ( last_occurence < 0.03){ // increase by two
	  V0norm *= 2.;
	  Pnorm *= 2.;
	  Nchanged = Nmeassure;
	  Wchanged = Nweight;
	  if (mpi.rank==mpi.master) log<<std::left<<(itt/1.0e6)<<"M occurence="<<std::setw(8)<<last_occurence<<" V0 increased to "<<V0norm<<std::endl;
	}
      }
      if (itt%(500000*tmeassure) == 0){
	if (Nmeassure-Nweight>1000){
#ifdef _MPI	  
	  MPI_Allreduce(MPI_IN_PLACE, mweight.K_hist.data(), mweight.K_hist.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
	  // We can get overflow during MPI for a long run. We should use a constant for normalization.
	  // We could here normalize K_hist (with knorm), and than when we measure, we would
	  // add instead of adding unity, we would add 1./knorm
	  dk_hist *= mweight.Normalize_K_histogram();
	  if (dk_hist < 1e-16) dk_hist = 1.0;
	  /*
	  {
	    double dsum=0;
	    for (int ik=0; ik<Nloops; ik++)
	      for (int i=0; i<Nbin; i++)
		dsum += K_hist(ik,i);
	    double dnrm = Nloops/dsum;
	    K_hist *= dnrm;
	    dk_hist *= dnrm;
	  }
	  */
	  mweight.Recompute(/*K_hist,*/ mpi.rank==mpi.master);
	  // OLD_BUG : If we are in a meassuring space, we have to update now the value of the meassuring diagram as the weight has changed.
	  if (Qweight) PQ = V0norm * mweight(amomentm, momentum);
	}
      }
      t_mes.stop();
    }
  }
  t_all.stop();
  BKdata.C_Pln *= 1.0/Nmeassure;
  Pbin         *= 1.0/Nmeassure;
  Pnorm        *= 1.0/Nmeassure;
  double occurence = static_cast<double>(Nweight)/static_cast<double>(Nmeassure);

  //  K_hist *= 1.0/Nmeassure;

#ifdef _MPI  
  double dat[2] = {Pnorm, occurence};
  if (mpi.rank==mpi.master){
    MPI_Reduce(MPI_IN_PLACE, BKdata.C_Pln.data(), BKdata.C_Pln.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE,  Pbin.data(),  Pbin.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, dat, 2, MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, mweight.K_hist.data(), mweight.K_hist.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
  }else{
    MPI_Reduce(BKdata.C_Pln.data(), BKdata.C_Pln.data(), BKdata.C_Pln.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    MPI_Reduce( Pbin.data(),  Pbin.data(),  Pbin.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    MPI_Reduce(dat, dat, 2, MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    MPI_Reduce(mweight.K_hist.data(), mweight.K_hist.data(), mweight.K_hist.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
  }
  if (mpi.rank==mpi.master){
    BKdata.C_Pln *= 1./mpi.size;
    Pbin  *= 1./mpi.size;
    Pnorm = dat[0]/mpi.size;
    occurence = dat[1]/mpi.size;
    //
    mweight.K_hist *= 1./mpi.size;
  }
#endif

  if (mpi.rank==mpi.master){
    log<<"total acceptance rate="<<(Nacc_k+Nacc_q+Nacc_t+Nacc_w)/(p.Nitt+0.0)<<" "<<" k-acceptance="<<(Nacc_k/(Nall_k+0.0))<<" q-acceptance="<<(Nacc_q/(Nall_q+0.0))<<" t-acceptanece="<<(Nacc_t/(Nall_t+0.0))<<" w-acceptanece="<<(Nacc_w/(Nall_w+0.0))<<std::endl;
    log<<" k-trials="<<(Nall_k/(p.Nitt+0.0))<<" q-trials="<<(Nall_q/(p.Nitt+0.0))<<" t-trials="<<(Nall_t/(p.Nitt+0.0))<<" w-trials="<<(Nall_w/(p.Nitt+0.0))<<std::endl;
    //cout<<"t1="<<t1.elapsed()<<" t2="<<t2.elapsed()<<" t3="<<t3.elapsed()<<" t4="<<t4.elapsed()<<" t5="<<t5.elapsed()<<" t6="<<t6.elapsed()<<" t7="<<t7.elapsed()<<" t8="<<t8.elapsed()<<" t9="<<t9.elapsed()<<" t10="<<t10.elapsed()<<" t11="<<t11.elapsed()<<std::endl;
    log<<"total time="<<t_all.elapsed()<<" measuring time="<<t_mes.elapsed()<<std::endl;

    
    BKdata.C_Pln *= 1.0/(4*pi) * (fabs(V0norm)/Pnorm); // 4*pi because we must divide by 1/(4*pi*q**2), and 1/q**2 was already done in sampling
    Pbin  *= 1.0/(4*pi) * (fabs(V0norm)/Pnorm);
    log<<"  meassuring diagram occurence frequency="<<occurence<<" and its norm Pnorm="<<Pnorm<<std::endl;
    
    // Proper normalization of the resulting Monte Carlo data
    double norm = ipower( beta/((2*pi)*(2*pi)*(2*pi)), Norder) * (4*pi*cutoffq*cutoffq*cutoffq/3);
    BKdata.C_Pln *= norm;

    // Later we will use Nthbin/2.0*Nkbin/2.0  normalization in the Kernel to transform from binned data to function values.
    BKdata.Normalize(beta, cutoffq);
    
    double dq_binning = cutoffq/Nq;
    double dt_binning = beta/Nt;
    Pbin *= (norm/(dq_binning * dt_binning));
    
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
  }
}


void ReadCounterTermPolarization(vector<Spline2D<double> >& bPqt, bl::Array<double,1>& qx, bl::Array<double,1>& tx, double beta)
{
  int Vcmax = bPqt.size();
  {
    ifstream f0("Pqt0.tau");
    int Nq, Nt;
    f0 >> Nt >> Nq;
    std::string str;
    std::getline(f0, str);// skip the rest for now...
    tx.resize(Nt);
    for (int it=0; it<Nt; ++it) f0 >> tx(it);
    qx.resize(Nq);
    for (int iv=0; iv<Vcmax; iv++){
      bPqt[iv].resize(Nq,Nt);
      //bl::Array<double,1> qx(Nq);
      ifstream f1("Pqt"+std::to_string(iv+1)+".tau");
      for (int iq=0; iq<Nq; ++iq){
	f1 >> qx(iq);
	for (int it=0; it<Nt; ++it) f1 >> bPqt[iv](iq,it);
      }
      bPqt[iv].splineIt(qx, tx);
    }

    const bool debug=false;
    if (debug){
      for (int iv=0; iv<Vcmax; iv++){
	ofstream fo("__Pqt"+std::to_string(iv+1)+".tau");
	fo.precision(12);
	
	int iq=0;//Nq-2;
	int Nn = 1000;

	intpar pq(iq,0);
	for (int it=0; it<Nn; ++it){
	  double t = (it*beta)/Nn;
	  intpar pt = Interp(t,tx);
	  fo << t << "  "<< bPqt[iv](pq,pt) << std::endl;
	}
	fo.close();
      }
      exit(0);
    }
  }
}

inline int et(int i){
  // for equal time we will use t[3]=t[2], t[5]=t[4], ...
  // but we should not use t[1]!=t[0]
  return i==1 ? 1 : 2*(i/2);
}
inline int partner(int i){
  if (i%2==0)
    return i+1;
  else
    return i-1;
}

//int ID=1;
//int ID2=1;

template<typename GK>
void Check_trial_K(int which, ostream& log, int itt, int Nv, double PQ, double PQ_current, double lmbda, const bl::Array<double,1>& times, const std::set<int>& times_to_change, const GK& Gk,
		   const bl::Array<double,1>& G_current, const bl::Array<double,1>& V_current,
		   const bl::Array<double,1>& g_trial, const bl::Array<unsigned short,1>& ip_ind, const bl::Array<double,1>& v_trial, const bl::Array<unsigned short,1>& iq_ind,
		   const bl::Array<bl::TinyVector<double,3>,1>& tmom_g, const BitArray& changed_G, const bl::Array<bl::TinyVector<double,3>,1>& tmom_v, const BitArray& changed_V,
		   const bl::Array<unsigned short,2>& diagsG, const bl::Array<unsigned short,2>& Gindx, const bl::Array<unsigned short,2>& Vindx,
		   const bl::Array<bl::TinyVector<double,3>,1>& mom_g, const bl::Array<bl::TinyVector<double,3>,1>& mom_v,
		   const bl::Array<float,1>& diagSign, const bl::Array<char,2>& Vtype,
		   const vector<Spline2D<double> >& bPqt,
		   const bl::Array<double,1>& qx, const bl::Array<double,1>& tx,
		   const bl::Array<bl::TinyVector<int,2>,1>& vindx, bl::Array<double,1>& Vrtx_current)
{
  int Ndiags = diagsG.extent(0);
  int Norder = diagsG.extent(1)/2;
  double PQ_new=0;
  for (int id=0; id<Ndiags; id++){
    double PQd = 1.0;
    for (int i=0; i<2*Norder; i++){
      int ii = Gindx(id,i);
      double aK = changed_G[ii] ? norm(tmom_g(ip_ind(ii))) : norm(mom_g(ii));
      //double aK = norm(mom_g(Gindx(id,i)));
      int j = diagsG(id,i);
      double t1 = times_to_change.count(i) ? times(i) : times( et(i) );
      double t2 = times_to_change.count(j) ? times(j) : times( et(j) );
      double gk = Gk(aK, t2-t1);
      PQd *= gk;
    }

    double PQd_old=1.0;
    for (int i=0; i<2*Norder; i++){
      int ii = Gindx(id,i);
      PQd_old *= ( changed_G[ii] ? g_trial(ip_ind(ii)) : G_current(ii) );
    }
    double PQd_old_backup = PQd_old;
    for (int ii=0; ii<Nv; ++ii){
      int idn = vindx(ii)[0];
      if (idn!=id) continue;
      int  i = vindx(ii)[1];
      int vtyp = Vtype(id,i);
      if (vtyp!=0){
	//log<<"PQ was before="<<PQd_old<< " and ii="<<ii<<" and vertex="<< Vrtx_current(ii) << endl;
	PQd_old *= Vrtx_current(ii);
	//log<<" but now is Pq="<<PQd_old << endl;
      }
    }
    if (itt==23359484) log<<itt<<" PQd_new="<<PQd<<" PQd_old="<<PQd_old<<" PQ_no_vrtx="<<PQd_old_backup<<" Vrtx="<<Vrtx_current(0)<<endl;
    
    
    for (int i=1; i<Norder; i++){
      int ii = Vindx(id,i);
      double aQ = changed_V[ii] ? norm(tmom_v(iq_ind(ii))) : norm(mom_v(ii));
      //double aQ =  norm(mom_v(ii));
      double Vq = 8*pi/(aQ*aQ+lmbda);
      int vtyp = Vtype(id,i);
      double Veff=0;
      if (vtyp==0){
	Veff = Vq;
      }else{
	intpar pq = Interp(aQ,qx);
	intpar pt = Interp(fabs(times(2*i)-times(2*i+1)), tx);
	//if (pq.p<0 || pq.p>1) log<<"  Warning pq=" << pq.p << ", " << pq.i << ", " << qx.extent(0) << endl;
	//if (pt.p<0 || pt.p>1) log<<"  Warning pt=" << pt.p << ", " << pt.i << ", " << tx.extent(0) << endl;
	if (abs(vtyp)==1){
	  double ct = bPqt[0](pq,pt);
	  double Vn = Vq*Vq;
	  Veff = Vn * ct;
	}else if (abs(vtyp)==2){
	  double ct = bPqt[1](pq,pt) + 2*bPqt[0](pq,pt)*lmbda/(8*pi);
	  double Vn = Vq*Vq*Vq;
	  Veff = Vn * ct;
	  if (itt==23359484) log<<" ct=" << ct <<" Vn=" << Vn << " Veff=" << Veff << " ct1="<< bPqt[1](pq,pt)  << " ct2="<< bPqt[0](pq,pt) << endl;
	}else{
	  exit(1);
	}
      }
      PQd *= Veff;
      if (itt==23359484) log<<"  ... veff inside Check="<< Veff << " PQd now "<< PQd << endl;
    }
    PQd *=  diagSign(id);
    if (itt==23359484) log<<"    after multiply with diagSign PQd=" << PQd << endl;
    //if (id==ID || id==ID2)
    PQ_new += PQd;
  }
  
  if (fabs((PQ-PQ_new)/PQ_new) > 1e-4 && (fabs(PQ)>1e-50 || fabs(PQ_new)>1e-50)) {
    log << "changed_V = "<< changed_V << " iq_ind=" << iq_ind << " v_trial=" << v_trial << endl;
    
    double PQ_old=0;
    PQ_old=0;
    for (int id=0; id<Ndiags; id++){
      double PQd = 1.0;
      for (int i=0; i<2*Norder; i++){
	int ii = Gindx(id,i);
	PQd *= ( changed_G[ii] ? g_trial(ip_ind(ii)) : G_current(ii) );
      }
      log<<"   .... product gg="<< PQd << endl;
      for (int i=1; i<Norder; i++){
	int ii = Vindx(id,i);
	PQd *= ( changed_V[ii] ? v_trial(iq_ind(ii)) : V_current(ii) );
	if (itt==23359484){
	  log<<"  changed_V["<<ii<<"]="<<changed_V[ii]<<" v="<<( changed_V[ii] ? v_trial(iq_ind(ii)) : V_current(ii) )<<endl;
	}
      }
      log<<"   .... product gg*vv=" << PQd << endl; 
      PQd *= diagSign(id);
      log<<"   .... product gg*vv*sign="<< PQd << endl;
      PQ_old += PQd;
      log<<"   .... PQ_old="<< PQ_old << endl;
    }
    
    log<< itt<< "._k_"<<which<<" ERROR PQ=" << PQ << " while PQ_new="<<PQ_new<<" PQ_old="<<PQ_old<<" ratio="<< (PQ_new/PQ_current) << endl;
    if (fabs(PQ_new/PQ_current)>1e-16 || fabs(PQ/PQ_current)>1e-16 ) exit(1);
  }else{
    //log<< itt<<"."<<which<<"  OK PQ="<< PQ << " PQ_new=" << PQ_new << endl;
  }
}

template<typename GK>
void Check_trial_T(int which, ostream& log, int itt, int Nv, double PQ, double PQ_current, double lmbda, const bl::Array<double,1>& times, const std::set<int>& times_to_change, const GK& Gk,
		   const bl::Array<double,1>& G_current, const bl::Array<double,1>& V_current,
		   const bl::Array<double,1>& g_trial, const bl::Array<unsigned short,1>& ip_ind, const bl::Array<double,1>& v_trial, const bl::Array<unsigned short,1>& iq_ind,
		   const BitArray& changed_G, const BitArray& changed_V,
		   const bl::Array<unsigned short,2>& diagsG, const bl::Array<unsigned short,2>& Gindx, const bl::Array<unsigned short,2>& Vindx,
		   const bl::Array<bl::TinyVector<double,3>,1>& mom_g, const bl::Array<bl::TinyVector<double,3>,1>& mom_v,
		   const bl::Array<float,1>& diagSign, const bl::Array<char,2>& Vtype,
		   const vector<Spline2D<double> >& bPqt,
		   const bl::Array<double,1>& qx, const bl::Array<double,1>& tx,
		   const bl::Array<bl::TinyVector<int,2>,1>& vindx, bl::Array<double,1>& Vrtx_current)
{
  int Ndiags = diagsG.extent(0);
  int Norder = diagsG.extent(1)/2;
  double PQ_new=0;
  for (int id=0; id<Ndiags; id++){
    double PQd = 1.0;
    for (int i=0; i<2*Norder; i++){
      int ii = Gindx(id,i);
      //double aK = changed_G[ii] ? norm(tmom_g(ip_ind(ii))) : norm(mom_g(ii));
      double aK = norm(mom_g(ii));
      int j = diagsG(id,i);
      double t1 = times_to_change.count(i) ? times(i) : times( et(i) );
      double t2 = times_to_change.count(j) ? times(j) : times( et(j) );
      double gk = Gk(aK, t2-t1);
      PQd *= gk;
    }

    double PQd_old=1.0;
    for (int i=0; i<2*Norder; i++){
      int ii = Gindx(id,i);
      PQd_old *= ( changed_G[ii] ? g_trial(ip_ind(ii)) : G_current(ii) );
    }
    //double PQd_old_backup = PQd_old;
    for (int ii=0; ii<Nv; ++ii){
      int idn = vindx(ii)[0];
      if (idn!=id) continue;
      int  i = vindx(ii)[1];
      int vtyp = Vtype(id,i);
      if (vtyp!=0){
	//log<<"PQ was before="<<PQd_old<< " and ii="<<ii<<" and vertex="<< Vrtx_current(ii) << endl;
	PQd_old *= Vrtx_current(ii);
	//log<<" but now is Pq="<<PQd_old << endl;
      }
    }
    //log<<itt<<" PQd_new="<<PQd<<" PQd_old="<<PQd_old<<" PQ_no_vrtx="<<PQd_old_backup<<" Vrtx="<<Vrtx_current(0)<<endl;

    for (int i=1; i<Norder; i++){
      int ii = Vindx(id,i);
      //double aQ = changed_V[ii] ? norm(tmom_v(iq_ind(ii))) : norm(mom_v(ii));
      double aQ =  norm(mom_v(ii));
      double Vq = 8*pi/(aQ*aQ+lmbda);
      int vtyp = Vtype(id,i);
      double Veff=0;
      if (vtyp==0){
	Veff = Vq;
      }else{
	intpar pq = Interp(aQ,qx);
	intpar pt = Interp(fabs(times(2*i)-times(2*i+1)), tx);
	//if (pq.p<0 || pq.p>1) log<<"  Warning pq=" << pq.p << ", " << pq.i << ", " << qx.extent(0) << endl;
	//if (pt.p<0 || pt.p>1) log<<"  Warning pt=" << pt.p << ", " << pt.i << ", " << tx.extent(0) << endl;
	if (abs(vtyp)==1){
	  double ct = bPqt[0](pq,pt);
	  double Vn = Vq*Vq;
	  Veff = Vn * ct;
	}else if (abs(vtyp)==2){
	  double ct = bPqt[1](pq,pt) + 2*bPqt[0](pq,pt)*lmbda/(8*pi);
	  double Vn = Vq*Vq*Vq;
	  Veff = Vn * ct;
	}else{
	  exit(1);
	}
      }
      PQd *= Veff;
      //log<<"  ... veff inside Check="<< Veff << " PQd now "<< PQd << endl;
    }
    PQd *=  diagSign(id);
    //log<<"    after multiply with diagSign PQd=" << PQd << endl;
    //if (id==ID || id==ID2)
    PQ_new += PQd;
  }
  
  if (fabs((PQ-PQ_new)/PQ_new) > 1e-4 && (fabs(PQ)>1e-50 || fabs(PQ_new)>1e-50)) {

    double PQ_old=0;
    PQ_old=0;
    for (int id=0; id<Ndiags; id++){
      double PQd = 1.0;
      for (int i=0; i<2*Norder; i++){
	int ii = Gindx(id,i);
	PQd *= ( changed_G[ii] ? g_trial(ip_ind(ii)) : G_current(ii) );
      }
      log<<"   .... product gg="<< PQd << endl;
      for (int i=1; i<Norder; i++){
	int ii = Vindx(id,i);
	PQd *= ( changed_V[ii] ? v_trial(iq_ind(ii)) : V_current(ii) );
      }
      log<<"   .... product gg*vv=" << PQd << endl; 
      PQd *= diagSign(id);
      log<<"   .... product gg*vv*sign="<< PQd << endl;
      PQ_old += PQd;
      log<<"   .... PQ_old="<< PQ_old << endl;
    }
    
    log<< itt<< "._t_"<<which<<" ERROR PQ=" << PQ << " while PQ_new="<<PQ_new<<" PQ_old="<<PQ_old<<" ratio="<< (PQ_new/PQ_current) << endl;
    if (fabs(PQ_new/PQ_current)>1e-16 || fabs(PQ/PQ_current)>1e-16 ) exit(1);
  }else{
    //log<< itt<<"."<<which<<"  OK PQ="<< PQ << " PQ_new=" << PQ_new << endl;
  }
}

template<typename GK>
void Check(int which, ostream& log, int itt, int Nv, double PQ, double lmbda, const bl::Array<double,1>& times, const std::set<int>& times_to_change, const GK& Gk,
	   const bl::Array<double,1>& G_current, const bl::Array<double,1>& V_current,
	   const bl::Array<unsigned short,2>& diagsG, const bl::Array<unsigned short,2>& Gindx, const bl::Array<unsigned short,2>& Vindx,
	   const bl::Array<bl::TinyVector<double,3>,1>& mom_g, const bl::Array<bl::TinyVector<double,3>,1>& mom_v,
	   const bl::Array<float,1>& diagSign, const bl::Array<char,2>& Vtype,
	   const vector<Spline2D<double> >& bPqt,
	   const bl::Array<double,1>& qx, const bl::Array<double,1>& tx,
	   const bl::Array<bl::TinyVector<int,2>,1>& vindx, bl::Array<long double,1>& Vrtx_current)
{
  int Ndiags = diagsG.extent(0);
  int Norder = diagsG.extent(1)/2;

  //int Ng = max(Gindx);
  //for (int ii=0; ii<Ng; ii++){
  //  log<< "k_"<<ii<<" = ("<< mom_g(ii)[0]<<", "<<mom_g(ii)[1]<<", "<<mom_g(ii)[2]<<")"<<endl;
  //}
  
  //log<< "  Vrtx_current=" << Vrtx_current << endl;
  
  double PQ_new=0;
  for (int id=0; id<Ndiags; id++){
    double PQd = 1.0;
    for (int i=0; i<2*Norder; i++){
      double aK = norm(mom_g(Gindx(id,i)));
      int j = diagsG(id,i);
      //double t1 = times_to_change.count(i) ? times(i) : times( et(i) );
      //double t2 = times_to_change.count(j) ? times(j) : times( et(j) );
      double t1 =  Vtype(id,i/2)!=0 ? times(i) : times( et(i) );
      double t2 =  Vtype(id,j/2)!=0 ? times(j) : times( et(j) );
      double gk = Gk(aK, t2-t1);
      //log<<"    id=" << id << " gk="<< gk<< " k=" << aK << " t1="<< t1 << " t2="<< t2 << endl;
      PQd *= gk;
    }
    
    double PQd_old=1.0;
    for (int i=0; i<2*Norder; i++) PQd_old *= G_current(Gindx(id,i));
    //double PQd_old_backup = PQd_old;
    double vrtx=1.0;
    for (int i=0; i<Norder; i++){
      int ii = Vindx(id,i);
      int vtyp = Vtype(id,i);
      if (vtyp!=0){
	//log<<"PQ was before="<<PQd_old<< " and ii="<<ii<<" and vertex="<< Vrtx_current(ii) << endl;
	PQd_old *= Vrtx_current(ii);
	vrtx *= Vrtx_current(ii);
	//log<<" but now is Pq="<<PQd_old << endl;
      }
    }
    //log<<itt<<" PQd_new="<<PQd<<" PQd_old="<<PQd_old<<" PQ_no_vrtx="<<PQd_old_backup<<" Vrtx="<<vrtx<<endl;
    
      
    for (int i=1; i<Norder; i++){
      //int ii = Vindx(id,i);
      double aQ =  norm(mom_v(Vindx(id,i)));
      double Vq = 8*pi/(aQ*aQ+lmbda);
      int vtyp = Vtype(id,i);
      double Veff=0;
      if (vtyp==0){
	Veff = Vq;
      }else{
	intpar pq = Interp(aQ,qx);
	intpar pt = Interp(fabs(times(2*i)-times(2*i+1)), tx);
	//if (pq.p<0 || pq.p>1) log<<"  Warning pq=" << pq.p << ", " << pq.i << ", " << qx.extent(0) << endl;
	//if (pt.p<0 || pt.p>1) log<<"  Warning pt=" << pt.p << ", " << pt.i << ", " << tx.extent(0) << endl;
	// ORDER ONE
	if (abs(vtyp)==1){
	  double ct = bPqt[0](pq,pt);
	  double Vn = Vq*Vq;
	  Veff = Vn * ct;
	}else if (abs(vtyp)==2){
	  // ORDER TWO
	  double ct = bPqt[1](pq,pt) + 2*bPqt[0](pq,pt)*lmbda/(8*pi);
	  double Vn = Vq*Vq*Vq;
	  Veff = Vn * ct;
	}else{
	  exit(1);
	}
      }
      PQd *= Veff;
      //log<<"  ... veff inside Check="<< Veff << " PQd now "<< PQd << endl;
    }
    PQd *=  diagSign(id);
    //log<<"    after multiply with diagSign PQd=" << PQd << endl;
    PQ_new += PQd;
  }
  
  if (fabs((PQ-PQ_new)/PQ) > 1e-7 && (fabs(PQ)>1e-50 || fabs(PQ_new)>1e-50) ) {
    log<<" .... Now getting PQ_old: "<<endl;
    double PQ_old=0;
    PQ_old=0;
    for (int id=0; id<Ndiags; id++){
      double PQd = 1.0;
      for (int i=0; i<2*Norder; i++) PQd *= G_current(Gindx(id,i));
      log<<"   .... product gg="<< PQd << endl;
      for (int i=1; i<Norder; i++) PQd *= V_current(Vindx(id,i));
      log<<"   .... product gg*vv=" << PQd << endl; 
      PQd *= diagSign(id);
      log<<"   .... product gg*vv*sign="<< PQd << endl;
      PQ_old += PQd;
      log<<"   .... PQ_old="<< PQ_old << endl;
    }
    
    log<< itt<< "."<<which<<" ERROR PQ=" << PQ << "while PQ_new="<<PQ_new<<" PQ_old="<<PQ_old<<endl;
  }else{
    //log<< itt<<"."<<which<<"  OK PQ="<< PQ << " PQ_new=" << PQ_new << endl;
  }
}

std::vector<double> Nrm(double lmbda, double cutoffk, int n)
{
  // This calculates the integral
  //
  //  Nrm[n] = 1/Integrate[d^3q/(2 Pi)^3 Vq[q]^n, {q, 0, cuttofq}]
  //
  //  where  Vq = 8*pi/(q^2+lmbda)
  //  for  n=2,3,4 we calculate the integral exactly, while for 5,... we calculate in the limit cutoffk->infinity
  double cl = cutoffk/sqrt(lmbda);
  double cl2 = cl*cl;
  std::vector<double> nrm(n+1);
  for (int i=0; i<=n; i++) nrm[i]=0.;
  // integral with Vq^2
  if (n>=2) nrm[2] = sqrt(lmbda)/(16.* (atan(cl) - cl/(1+cl2)) );
  // integral with Vq^3
  //double V3 = lmbda*sqrt(lmbda)*ipower(1.+cl2,2)/(32.*pi*( cl*(cl2-1) + ipower(cl2+1,2)*atan(cl)));
  if (n>=3) nrm[3] = lmbda*sqrt(lmbda) /(32.*pi*( atan(cl) - cl*(1-cl2)/ipower(1.+cl2,2) ));
  // integral with Vq^4
  if (n>=4) nrm[4] = lmbda*lmbda*sqrt(lmbda)/( 128*pi*pi*(  atan(cl) - cl*(1-cl2*(cl2+8./3.))/ipower(1+cl2,3)) );
  // for the rest we set cutoffk->infinity
  double Vn = sqrt(lmbda)/(8*pi);
  //cout<<"i=2"<<" Vn="<<Vn << endl;
  for (int i=3; i<n+1; i++){
    Vn *= (lmbda/(8*pi)) * ( i-1.)/(i-5/2.);
    if (i>4) nrm[i] = Vn;
    //cout<<"i="<<i<<" Vn="<<Vn<< endl;
  }
  return nrm;
}

template<typename GK, typename Data>
void sample_static_fastD(Data& BKdata, bl::Array<double,2>& Pbin, std::ostream& log,
			 double lmbda, const std::vector<double>& lmbda_spct, const GK& Gk, const params& p,
			 const bl::Array<unsigned short,2>& diagsG, const bl::Array<float,1>& diagSign,
			 const std::vector<std::vector<std::vector<int> > >& Loop_index, const std::vector<std::vector<std::vector<int> > >& Loop_type,
			 bl::Array<char,2>& Vtype, my_mpi& mpi)
{
  int bubble_only = 0;
  typedef double real;
  log.precision(12);
  if ( BKdata.Nlt!=p.Nlt || BKdata.Nlq!=p.Nlq){
    log<<"ERROR : Dimensions of C_Pln is wrong : either "<<BKdata.Nlt<<" != "<<p.Nlt<<" or "<<BKdata.Nlq<<" != "<<p.Nlq<<std::endl;
    exit(1);
  }
  
  if (Pbin.extent(0)!=p.Nt || Pbin.extent(1)!=p.Nq) Pbin.resize(p.Nt,p.Nq); Pbin=0.0;
  
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
  
  //CounterCoulombV Vqc(lmbda);

  // Set up which times need change when counter term is dynamic
  vector<int> times_2_change;
  std::set<int> times_to_change;
  int Vcmax=0;
  {
    std::set<int> counter_types;
    times_to_change.insert(0);
    for (int i=0; i<Norder; i++) times_to_change.insert(2*i);
    // default vertices to change time will be : [0,2,...,2*(Norder-1)]
    for (int id=0; id<Ndiags; id++)    // Checking where can we add single-particle counter terms. 
      for (int j=1; j<Norder; j++)     // skip meassuring line
	if (Vtype(id,j)>0){
	  counter_types.insert(Vtype(id,j));
	  times_to_change.insert(2*j+1);  // since the counter-term is dynamic, we need to make vertex 2j+1 different from 2j.
	}
    if (counter_types.size()>0){
      Vcmax = *counter_types.rbegin();
    }
    times_2_change.resize(times_to_change.size());
    int j=0;
    for (auto it=times_to_change.begin(); it!=times_to_change.end(); ++it, ++j) times_2_change[j] = (*it);
    log<<"times_to_change : ";
    for (auto it=times_to_change.begin(); it!=times_to_change.end(); ++it) log << (*it) <<",  ";
    log<<std::endl;
    log<<"counter_types : ";
    for (auto it=counter_types.begin(); it!=counter_types.end(); ++it) log << (*it) <<",  ";
    log<<std::endl;
    log << "times_2_change: "<<endl;
    for (int i=0; i<times_2_change.size(); i++) log << times_2_change[i] << ",";
    log << endl;
  }
  // Reads numerically evaluated Bubble diagram.
  vector<Spline2D<double> > bPqt(Vcmax);
  bl::Array<double,1> qx, tx;  
  if (Vcmax>0) ReadCounterTermPolarization(bPqt, qx, tx, p.beta);
  bl::Array<int,2> binomial = BinomialCoeff(Vcmax);

  /*
  for (int iq=0; iq<100; iq++){
    double aQ = iq/100. * p.cutoffk;
    ofstream clog( std::string("debug.")+std::to_string(iq));
    clog.precision(14);
    clog << "# aQ=" << aQ << endl;
    for (int it=0; it<1000; it++){
      double dt = it/(1000-1.)*p.beta;
      intpar pq = Interp(aQ,qx);
      intpar pt = Interp(dt, tx);
      clog << dt << "  " << bPqt[0](pq,pt) << endl;
    }
  }
  exit(0);
  */
  
  int tmeassure=p.tmeassure, Ncout=p.Ncout, Nwarm=p.Nwarm;
  double V0norm = p.V0norm, dkF = p.dkF, kF=p.kF;// V0exp= p.V0exp;
  double cutoffk = p.cutoffk, dRk = p.dRk;
  double beta = p.beta;
  double cutoffq = p.cutoffq;
  int Nt = p.Nt, Nq = p.Nq;


  // THIS PART IS DIFFERENT
  // Everything for Legendre Polynomials in time and momentum
  LegendrePl Pln(std::max(BKdata.Nlq,BKdata.Nlt));
  bl::Array<double,1> pl_Q(BKdata.Nlq+1);  // Will contain legendre Pl(2*q/qmax-1)
  bl::Array<double,1> pl_t(BKdata.Nlt+1);  // Will contain legendre Pl(2*t//beta-1)
  BKdata.C_Pln = 0.0;            // cummulative result for legendre expansion
  // THIS PART IS DIFFERENT
  
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
  findUnique_noindx_spct(Gindx, Vindx, gindx, vindx, N0v, diagsG, Vtype, Loop_index, Loop_type, 0, true, debug, log);

  // Once more we go through single-particle counter terms and create index single_counter_index
  bl::Array<int,1> single_counter_index;
  Find_Single_Particle_Counter_G_index(single_counter_index, Ndiags, Norder, vindx.extent(0), diagsG, Vtype, Gindx, Vindx);
  
  // This is related to Baym-Kadanoff approach. For efficiency, we want to find which diagram have common values of momenta at the right hand side of the diagram.
  BKdata.FindGroups(diagsG, Loop_index, Loop_type, log);
  
  if (Loop_index.size()!=Ndiags){ log<<"Wrong size of Loop_index "<<Loop_index.size()<<" instead of "<<Ndiags<<std::endl; exit(1);}
  int Nloops = Loop_index[0].size();
  if (Nloops != Norder+1){ log<<"Expecting Nloops==Norder+1 in Loops_index, but got Nloops="<<Nloops<<" Norder="<<Norder<<std::endl; exit(1);}
  int Ng = gindx.extent(0);
  int Nv = vindx.extent(0);
  int _Nv_ = std::max(Nv,1); // can not assign 0 to empty file

  bl::Array<bl::Array<sint,1>,1> loop_Gkind, loop_Vqind;
  bl::Array<bl::Array<char,1>,1> loop_Gksgn, loop_Vqsgn;
  int Ngp, Nvp;
  // Find index-tables to all momentum loops, so that we do not need to loop over all diagrams, but rather only over all differet propagators corresponding to certain loop.
  // Here we also take into account that single-particle counter terms are sometimes added.
  Get_GVind(loop_Gkind, loop_Gksgn, loop_Vqind, loop_Vqsgn, Ngp, Nvp, Loop_index, Loop_type, Gindx, Vindx, single_counter_index, lmbda_spct, Vtype, true, log);
  int Ngp_tr = GetMax_NGp_trial(i_diagsG, Gindx, Ng, Ngp);

  // DEBUG
  if (mpi.rank==mpi.master){
    log<<"lmbda_spct=";
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
    log<<"loop_Vqind:"<<std::endl;
    for (int iloop=0; iloop<Nloops; ++iloop){
      for (int iq=0; iq<loop_Vqind(iloop).extent(0); ++iq){ // Now do the same for the interaction
	int ii   = loop_Vqind(iloop)(iq);
	int isgn = loop_Vqsgn(iloop)(iq);
	
	int id = vindx(ii)[0];
	int  i = vindx(ii)[1];
	int vtyp = Vtype(id,i);
	
	log<<"iloop="<<iloop<<" iq="<<iq<<" ii_v="<<ii<<" isgn="<<isgn<<" which comes from id="<< id <<" and pair =("<<2*i<<","<<2*i+1<<") of type="<<vtyp<<std::endl;
      }
    }
  }
  // END DEBUG
  
  Reweight_time rw(beta,p.lmbdat);
  int Nbin = 513;
  //int Nbin = 129;
  bl::Array<double,1> lmbda_spct_cutoffk(lmbda_spct.size());
  // Instead of using : lmbda_spct, which is q-independent, we will introduce "artificial" q-dependence, which gives identical integral.
  // When lmbda_spct is q-independent, we should normalize it by  lmbda_spct * 1/Integrate[d^3 1/(2*pi)^3].
  // The "artificial" dependence we choose is lmbda_spct * (Vq)^2, in which case the normalization constant is
  //     lmbda_spct * (Vq)^2 / Integrate[d^3q Vq^2/(2*pi)^3 ]
  //  which is
  //     lmbda_spct * (Vq)^2 / ( 2/sqrt(lmbda) *atan(ck/sqrt(lmbda)) - 2*ck/(ck^2+lmbda) )
  std::vector<double> nrm = Nrm(lmbda, cutoffk, lmbda_spct.size()+2);
  for (int i=0; i<lmbda_spct.size(); ++i){
#ifdef CNTR_VN
    // [Vq^2, Vq^3, Vq^4,..]
    lmbda_spct_cutoffk(i) = lmbda_spct[i]*nrm[i+2];
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
  bool GetHistogram=true;
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
  {
    double Qa = p.kF*drand(); // absolute value of external momentum
    momentum(0) = 0.0,0.0,Qa; // external momentum Q
    amomentm(0) = Qa;
    double kf_3 = p.kF/sqrt(3.);
    for (int ik=1; ik<Nloops; ik++){
      momentum(ik) = drand()*kf_3, drand()*kf_3, drand()*kf_3;
      amomentm(ik) = norm(momentum(ik));
    }
  }
  log.precision(12);
  log<<"Starting times="<<times<<" and starting momenta="<<momentum<<std::endl;

  bl::Array<double,1> G_current(Ng);
  bl::Array<long double,1> V_current(_Nv_);
  // This is momentum for all diagrams and all propagators
  bl::Array<bl::TinyVector<real,3>,1> mom_g(Ng), mom_v(Nv);
  bl::Array<long double,1> Vrtx_current(1);
  if (Vcmax>0) Vrtx_current.resize(_Nv_);
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
      for (int i=0; i<2*Norder; i++){
	//int i_final = diagsG(id,i);
	int i_previ = i_diagsG(id,i);
	bl::TinyVector<double,3> k_in = mom_G(id,i_previ);
	bl::TinyVector<double,3> k_out = mom_G(id,i);
	bl::TinyVector<double,3> q = mom_V(id,i/2)*(1.-2*(i%2));
	if (fabs(norm(k_in-k_out)-norm(q))>1e-6){
	  log<<"ERROR: diagram "<<id<<" vertex i="<<i<<" k_in="<<k_in<<" k_out="<<k_out<<" q="<<q<<std::endl;
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
    Vrtx_current=0;
    // Vertices, which are needed to express screened interaction with the same time difference as the unscreened part.
    for (int ii=0; ii<Nv; ++ii){
      int id = vindx(ii)[0];
      int  i = vindx(ii)[1];
      int vtyp = Vtype(id,i);
      if (vtyp!=0){
	double t1 = times(2*i);     // t1 is old time, because for equal time case t[2i+1] was actually t[2i]
	double t2 = times(2*i+1);   // t2 is the new time, because now we want t[2i+1] to be different.
	// For now we will neglect possibility of type-II vertex. Should correct later!
	int i_m = i_diagsG(id,2*i+1);
	double ti = times( et(i_m) );
	int i_p = diagsG(id,2*i+1);
	double to = times( et(i_p) );
	double ki = norm(mom_G(id,i_m));
	double ko = norm(mom_G(id,2*i+1));
	Vrtx_current(ii) = MoveVertex(t2, t1, ti, to, ki, ko, Gk);
	//double direct_vertex = Gk(ki, t2-ti)*Gk(ko, to-t2)/(Gk(ki, t1-ti)*Gk(ko, to-t1) );
	//log<<"t1="<<t1<<" t2="<<t2<<" i_m="<<i_m<<" ti="<<ti<<" i_p="<<i_p<<" to="<<to<<" ki="<<ki<<" ko="<<ko<<" vrtx="<<Vrtx_current(ii)<<"  direct_vrtx="<<direct_vertex<<endl;
	if ((Vtype(id,i_m/2)!=0 && (i_m%2==1)) || (Vtype(id,i_p/2)!=0 && (i_p%2==1))){
	  log<<"ERROR : We have type2, which is not yet implemented!  id="<<id<<" i="<<i<<" vtyp="<<vtyp<<endl;
	  log<<"..... i_m="<<i_m<<" i_p="<<i_p<<" 2*i="<< 2*i << endl;
	  log<<" 1 : "<< (Vtype(id,i_m/2)!=0 && (i_m != 2*i)) << " 2 : "<< (Vtype(id,i_p/2)!=0 && (i_p != 2*i)) << endl;
	}
      }
    }
    //log<<"Vrtx_current="<<Vrtx_current<<endl;
    V_current=0;  
    for (int ii=0; ii<Nv; ++ii){
      int id = vindx(ii)[0];
      int  i = vindx(ii)[1];
      double aQ = norm(mom_V(id,i));
      int vtyp = Vtype(id,i);
      double Vq = 8*pi/(aQ*aQ+lmbda);
      if (vtyp==0){
	V_current(ii) = Vq;
      }else{
	intpar pq = Interp(aQ,qx);
	intpar pt = Interp(fabs(times(2*i)-times(2*i+1)), tx);
	if ( fabs(pt.p)>1 ) log << "ERROR tx.i=" << pt.i << " tx.p="<< pt.p << endl;
	int Nc = abs(vtyp);
	double ct=0;
	double lmbda_k = 1.0;
	for (int k=0; k<Nc; k++){
	  ct += binomial(Nc,k) * lmbda_k * bPqt[Nc-k-1](pq,pt);
	  lmbda_k *= (lmbda/(8*pi));
	}
	long double Vn = ipower(Vq, Nc+1);
	V_current(ii) = Vn * ct * Vrtx_current(ii);
	V_current(ii) += Vn * lmbda_k *(1-bubble_only);
	if (vtyp < 0){
	  // we also add the single-particle counter term in addition to two-particle counter term.
	  int ii_g = single_counter_index(ii);
	  long double g_kq = G_current(ii_g);
#ifdef CNTR_VN
	  // vtype=-1, Nc=1 : lmbda[0] -> Vn = Vq^2
	  // vtype=-2, Nc=2 : lmbda[1] -> Vn = Vq^3
	  if (g_kq!=0) V_current(ii) += Vn*lmbda_spct_cutoffk(-vtyp-1)/g_kq;
#else	  
	  if (g_kq!=0) V_current(ii) += exp(-aQ*2/kF)*lmbda_spct_cutoffk(-vtyp-1)/g_kq;
#endif	  
	}
      }
    }
    //log<<"V_current="<<V_current<<endl;
    PQ=0;
    BKdata.PQg_Initialize();
    for (int id=0; id<Ndiags; id++){
      long double PQd = 1.0;
      for (int i=0; i<2*Norder; i++) PQd *= G_current(Gindx(id,i));
      for (int i=1; i<Norder; i++) PQd *= V_current(Vindx(id,i));
      PQd *= diagSign(id);
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

  // Now computing legendre polinomials in q and tau for the initial configuration
  Pln.cmp_single(2*amomentm(0)/cutoffq-1., pl_Q, BKdata.Nlq);          // Legendre polynomials at this Q
  Pln.cmp_single(2*(times(0)-times(1))/beta-1., pl_t, BKdata.Nlt);     // Legendre polynomials at this time

  bl::Array<double,1> times_trial(2*Norder);
  int _Nvp_ = std::max(Nvp,1); // can not assign 0 to empty file
  bl::Array<bl::TinyVector<real,3>,1> tmom_g(Ngp), tmom_v(_Nvp_);
  bl::Array<sint,1> ip_ind(Ng), iq_ind(Nv);
  bl::Array<double,1> g_trial(Ngp_tr);
  bl::Array<long double,1> v_trial(_Nvp_);
  BitArray changed_G(Ng), changed_V(Nv);

  BitArray changed_Vrtx(_Nv_);
  bl::Array<long double,1> Vrtx_trial(1);
  if (Vcmax>0){
    Vrtx_trial.resize(_Nv_);
  }
  
  int Qweight = 0;      // if Qweight==1 then we have the normalization diagram
  double Nweight = 0;   // how many times did we see measuring diagram.
  int NlowT = 0;        // how many times do we see external time < 0.05*beta
  int Nmeassure = 0;    // how many times did we measure quantities.
  int Nchanged  = 0;    // when was V0norm last changed?
  double Wchanged = 0;  // what was Nweight when we last time changed V0norm?
  int Nacc_k = 0, Nacc_q = 0, Nacc_w = 0, Nacc_t = 0;
  int Nall_k = 0, Nall_q = 0, Nall_w = 0, Nall_t = 0;
  if (mpi.rank==mpi.master) PrintInfo_(0, Qweight, amomentm(0), amomentm(1), times(0)-times(1), PQ, PQ, 0, 0, log);
  //Check(0, log, -1, Nv, PQ, lmbda, times, times_to_change, Gk, G_current, V_current, diagsG, Gindx, Vindx, mom_g, mom_v, diagSign, Vtype, bPqt, qx, tx, vindx, Vrtx_current);
  t_all.start();
  for (int itt=0; itt<p.Nitt; itt++){
    double PQ_new=0;
    int icase = tBisect(drand(), Prs); // which steps should we attempt?
    if (icase==0){ // changing momentum Q or k-point
      int iloop = static_cast<int>(Nloops*drand());
      bl::TinyVector<double,3> K_new; double Ka_new; double trial_ratio=1;
      bool accept=false;
      if (iloop==0){ // this is the loop for the external momentum Q
	Nall_q += 1;
	accept = Find_new_Q_point(K_new, Ka_new, trial_ratio, momentum(iloop), amomentm(iloop), cutoffq, drand);
      }else{         // any other momentum point
	Nall_k += 1;
	accept = Find_new_k_point(K_new, Ka_new, trial_ratio, momentum(iloop), amomentm(iloop), cutoffk, drand, drand()<p.Qring, dkF, dRk);
      }
      if (accept){
	if (!Qweight){
	  bl::TinyVector<double,3> dK = K_new - momentum(iloop); // how much momentum will be changed.
	  //t1.start(); // takes 40% of the time
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
	  changed_Vrtx = 0;
	  // Vertices, which are needed to express screened interaction with the same time difference as the unscreened part.
	  if (Vcmax>0){
	    for (int id=0; id<Ndiags; id++){ // updating momentum in all propagators
	      const std::vector<int>& lindex = Loop_index[id][iloop];
	      const std::vector<int>& ltype  = Loop_type[id][iloop];
	      for (int i=0; i<lindex.size(); i++){
		if ( abs(ltype[i])==1 ){ // this is G-propagator
		  int ic = lindex[i]; // momentum of G[ic] is changed, hence when counter-term ends at ic (ic is odd), we need to update vertex, because k_o will change in MoveVertex.
		  // To correct for the unscreened part of the interaction, we need to update time-dependent interaction
		  if ( (ic%2!=0) && Vtype(id, ic/2) !=0 ) changed_Vrtx.set( Vindx(id, ic/2), 1 ); // odd vertex, which connects interaction of the counter-term
		  int i_p = diagsG(id,ic); // alternatively, incoming  vertex k_i might change in MoveVertex. If k[ic]==k_i then diagsG(ic) is the vertex where counter term ends.
		  if ( (i_p%2!=0) && Vtype(id, i_p/2)!=0 )  changed_Vrtx.set( Vindx(id, i_p/2), 1 );
		}
	      }
	    }
	    // Vertices, which are needed to express screened interaction with the same time difference as the unscreened part.
	    for (int ii=0; ii<Nv; ii++){
	      if (changed_Vrtx[ii]){
		int id = vindx(ii)[0];
		int  i = vindx(ii)[1];
		int vtyp = Vtype(id,i);
		if (vtyp==0) log<<"ERROR : changed_Vrtx is wrong, as type==0!"<<endl;
		double t1 = times(2*i);     // t1 is old time, because for equal time case t[2i+1] was actually t[2i]
		double t2 = times(2*i+1);   // t2 is the new time, because now we want t[2i+1] to be different.
		// For now we will neglect possibility of type-II vertex. Should correct later for order higher than 5!
		int i_m = i_diagsG(id,2*i+1);
		double ti = times( et(i_m) );
		int i_p = diagsG(id,2*i+1);
		double to = times( et(i_p) );
		double ii_g = Gindx(id,i_m);
		double ki = changed_G[ii_g] ? norm(tmom_g(ip_ind(ii_g))) : norm(mom_g(ii_g));
		ii_g = Gindx(id,2*i+1);
		double ko = changed_G[ii_g] ? norm(tmom_g(ip_ind(ii_g))) : norm(mom_g(ii_g));
		Vrtx_trial(ii) = MoveVertex(t2, t1, ti, to, ki, ko, Gk);
	      }
	    }
	  }
	  //t2.start(); // takes 1.5% of the time
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
	    if (vtyp==0){
	      v_trial(iq) = Vq;
	    }else{
	      intpar pq = Interp(aQ,qx);
	      intpar pt = Interp(fabs(times(2*i)-times(2*i+1)), tx);
	      if ( fabs(pt.p)>1 ) log << "ERROR tx.i=" << pt.i << " tx.p="<< pt.p << endl;
	      int Nc = abs(vtyp);
	      double ct=0;
	      double lmbda_k = 1.0;
	      for (int k=0; k<Nc; k++){
		ct += binomial(Nc,k) * lmbda_k * bPqt[Nc-k-1](pq,pt);
		lmbda_k *= (lmbda/(8*pi));
	      }
	      long double Vn = ipower(Vq, Nc+1);
	      long double Vrtx = changed_Vrtx[ii] ? Vrtx_trial(ii) : Vrtx_current(ii);
	      v_trial(iq) = (Vn * ct) * Vrtx ;
	      v_trial(iq) += Vn * lmbda_k *(1-bubble_only);
	      if (vtyp<0){
		// we also add the single-particle counter term in addition to two-particle counter term.
		int ii_g = single_counter_index(ii);
		long double g_kq = changed_G[ii_g] ? g_trial(ip_ind(ii_g)) : G_current(ii_g);
#ifdef CNTR_VN		
		if (g_kq!=0) v_trial(iq) += Vn*lmbda_spct_cutoffk(-vtyp-1)/g_kq;
#else		
		if (g_kq!=0) v_trial(iq) += exp(-aQ*2/kF)*lmbda_spct_cutoffk(-vtyp-1)/g_kq;
#endif		
	      }
	    }
	    iq_ind(ii)=iq;
	  }
	  //t2.stop();
	  // we computed the polarization diagram
	  //t3.start();// takes 22% of the time
	  PQ_new=0;
	  BKdata.PQg_new_Initialize();
	  for (int id=0; id<Ndiags; id++){
	    long double PQd = 1.0;
	    for (int i=0; i<2*Norder; i++){
	      int ii = Gindx(id,i);
	      PQd *= ( changed_G[ii] ? g_trial(ip_ind(ii)) : G_current(ii) );
	    }
	    for (int i=1; i<Norder; i++){
	      int ii = Vindx(id,i);
	      PQd *= ( changed_V[ii] ? v_trial(iq_ind(ii)) : V_current(ii) );
	    }
	    PQd *=  diagSign(id);
	    PQ_new += PQd;
	    BKdata.PQg_new_Add(id,PQd);
	  }
	  //t3.stop();
	}else{
	  double Ka_old = amomentm(iloop);
	  amomentm(iloop) = Ka_new;
	  bl::TinyVector<double,3> K_old = momentum(iloop);
	  momentum(iloop) = K_new;
	  PQ_new = V0norm * mweight(amomentm, momentum);
	  amomentm(iloop) = Ka_old;
	  momentum(iloop) = K_old;
	}
	//if (!Qweight)
	//  Check_trial_K(3, log, itt, Nv, PQ_new, PQ, lmbda, times, times_to_change, Gk, G_current, V_current,
	//               g_trial, ip_ind, v_trial, iq_ind, tmom_g, changed_G, tmom_v, changed_V,
	//               diagsG, Gindx, Vindx, mom_g, mom_v, diagSign, Vtype, bPqt, qx, tx, vindx, Vrtx_current);
	//if (std::isnan(PQ_new)){
	//  log<<" itt="<<itt<<" PQ_new="<<PQ_new << " PQ="<<PQ << " Qweight=" << Qweight << " momentum "<< endl;
	//}
	double ratio = PQ_new*trial_ratio / PQ;
	accept = fabs(ratio) > 1-drand() && PQ_new!=0;
      }
      //if ((itt+1)%Ncout==0 && mpi.rank==mpi.master) _PrintInfo_(itt+1, Qweight, icase, amomentm(1), amomentm(Nloops-1), times(0), PQ_new, PQ, NlowT/(Nmeassure+0.0), Nweight/(Nmeassure+0.0), log);
      if ((itt+1)%Ncout==0 && mpi.rank==mpi.master) _PrintInfo_(itt+1, Qweight, icase, amomentm(1), amomentm(Nloops-1), times(0), PQ_new, PQ, dk_hist, Nweight/(Nmeassure+0.0), log);
      
      if (accept){
	if (iloop==0) Nacc_q += 1; else Nacc_k += 1;
	if (!Qweight){
	  momentum(iloop) = K_new;  // this momentum was changed
	  amomentm(iloop) = Ka_new;
	  for (int ip=0; ip<loop_Gkind(iloop).extent(0); ++ip){
	    int ii   = loop_Gkind(iloop)(ip); 
	    mom_g(ii) = tmom_g(ip);
	  }
	  for (int iq=0; iq<loop_Vqind(iloop).extent(0); ++iq){ // Now do the same for the interaction
	    int ii   = loop_Vqind(iloop)(iq);
	    mom_v(ii) = tmom_v(iq);
	  }
	  for (int ip=0; ip<loop_Gkind(iloop).extent(0); ++ip){ // all G-propagators, which are changed when momentum in certain loop is changed.
	    int ii   = loop_Gkind(iloop)(ip);  // index of G-propagators, which we are now changing
	    G_current(ii)=g_trial(ip);
	  }
	  for (int iq=0; iq<loop_Vqind(iloop).extent(0); ++iq){ // Now do the same for the interaction
	    int ii   = loop_Vqind(iloop)(iq);
	    V_current(ii) = v_trial(iq);
	  }
	  if (Vcmax>0)
	    for (int ii=0; ii<Nv; ii++)
	      if (changed_Vrtx[ii]) Vrtx_current(ii) = Vrtx_trial(ii);
	}else{
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
	  changed_Vrtx = 0;
	  if (Vcmax>0){
	    // Vertices, which are needed to express screened interaction with the same time difference as the unscreened part.
	    for (int id=0; id<Ndiags; id++){ // updating momentum in all propagators
	      const std::vector<int>& lindex = Loop_index[id][iloop];
	      const std::vector<int>& ltype  = Loop_type[id][iloop];
	      for (int i=0; i<lindex.size(); i++){
		if ( abs(ltype[i])==1 ){
		  int ic = lindex[i]; // momentum of G[ic] is changed, hence when counter-term ends at ic (ic is odd), we need to update vertex, because k_o will change in MoveVertex.
		  // To correct for the unscreened part of the interaction, we need to update time-dependent interaction
		  if ( (ic%2!=0) && Vtype(id, ic/2)!=0 ) changed_Vrtx.set( Vindx(id, ic/2), 1 );
		  int i_p = diagsG(id,ic); // alternatively, incoming  vertex k_i might change in MoveVertex. If k[ic]==k_i then diagsG(ic) is the vertex where counter term ends.
		  if ( (i_p%2!=0) && Vtype(id, i_p/2)!=0 )  changed_Vrtx.set( Vindx(id, i_p/2), 1 );
		}
	      }
	    }
	    // Vertices, which are needed to express screened interaction with the same time difference as the unscreened part.
	    for (int ii=0; ii<Nv; ii++){
	      if (changed_Vrtx[ii]){
		int id = vindx(ii)[0];
		int  i = vindx(ii)[1];
		int vtyp = Vtype(id,i);
		if (vtyp==0) log<<"ERROR changed_Vrtx is wrong, as type==0!"<<endl;
		double t1 = times(2*i);     // t1 is old time, because for equal time case t[2i+1] was actually t[2i]
		double t2 = times(2*i+1);   // t2 is the new time, because now we want t[2i+1] to be different.
		// For now we will neglect possibility of type-II vertex. Should correct later!
		int i_m = i_diagsG(id,2*i+1);
		double ti = times( et(i_m) );
		int i_p = diagsG(id,2*i+1);
		double to = times( et(i_p) );
		double ki = norm(mom_g(Gindx(id,i_m)));
		double ko = norm(mom_g(Gindx(id,2*i+1)));
		Vrtx_current(ii) = MoveVertex(t2, t1, ti, to, ki, ko, Gk);
	      }
	    }
	  }
	  //t2.start(); // takes 1.5% of the time
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
	    if (vtyp==0){
	      V_current(ii) = Vq;
	    }else{
	      intpar pq = Interp(aQ,qx);
	      intpar pt = Interp(fabs(times(2*i)-times(2*i+1)), tx);
	      if ( fabs(pt.p)>1 ) log << "ERROR tx.i=" << pt.i << " tx.p="<< pt.p << endl;
	      int Nc = abs(vtyp);
	      double ct=0;
	      double lmbda_k = 1.0;
	      for (int k=0; k<Nc; k++){
		ct += binomial(Nc,k) * lmbda_k * bPqt[Nc-k-1](pq,pt);
		lmbda_k *= (lmbda/(8*pi));
	      }
	      long double Vn = ipower(Vq, Nc+1);
	      V_current(ii) = Vn * ct * Vrtx_current(ii);
	      V_current(ii) += Vn * lmbda_k *(1-bubble_only);
	      if (vtyp<0){
		// we also add the single-particle counter term in addition to two-particle counter term.
		int ii_g = single_counter_index(ii);
		long double g_kq = G_current(ii_g);
#ifdef CNTR_VN		
		if (g_kq!=0) V_current(ii) += Vn*lmbda_spct_cutoffk(-vtyp-1)/g_kq;
#else		
		if (g_kq!=0) V_current(ii) += exp(-aQ*2/kF)*lmbda_spct_cutoffk(-vtyp-1)/g_kq;
#endif		
	      }
	    }
	  }
	  //t5.stop();
	}
	PQ = PQ_new;
	if (!Qweight) BKdata.Set_PQg_new_to_PQg();
	//t6.start(); // takes 0.5% time
	if (iloop==0) Pln.cmp_single(2*amomentm(iloop)/cutoffq-1., pl_Q, BKdata.Nlq);  // update Legendre Polynomials
	//t6.stop();
	//if (!Qweight) Check(1, log, itt, Nv, PQ, lmbda, times, times_to_change, Gk, G_current, V_current, diagsG, Gindx, Vindx, mom_g, mom_v, diagSign, Vtype, bPqt, qx, tx, vindx, Vrtx_current);
      }
    } else if (icase==1){  // changing one of the time variables
      Nall_t += 1;
      int itime = times_2_change[ static_cast<int>( times_2_change.size()*drand() ) ];
      changed_G=0;              // which propagators are being changed?
      changed_Vrtx = 0;
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
	  if (Vcmax>0){
	    int i_n = diagsG(id,ivertex);
	    std::array<int,2> ips = {i_n, i_pre_vertex}; // CORRECT std::array<int> ips size!!!
	    for(const auto& ip: ips)
	      if ( (ip%2!=0) && Vtype(id,ip/2)!=0 )  changed_Vrtx.set( Vindx(id, ip/2), 1 );
	  }
	}
      }else{
	double t_new = beta*drand();
	// unscreened interaction, hence changing two times simultaneously!
	int ivertex = itime;
	times_trial(itime) = t_new;
	if (itime%2==0){ // this is the time for static interaction
	  for (int ivertex=itime; ivertex<itime+2; ivertex++){
	    for (int id=0; id<Ndiags; id++){
	      int i_pre_vertex = i_diagsG(id,ivertex); // WARNING : MAYBE YOU DO NOT NEED TO FLAG (itime+1) vertex when we have dynamic counter term!!
	      changed_G.set(Gindx(id,ivertex),1);     // these two propagators are changed because of the new time.
	      changed_G.set(Gindx(id,i_pre_vertex),1);
	      if (Vcmax>0){
		int i_n = diagsG(id,ivertex);
		std::array<int,3> ips = {i_n, ivertex, i_pre_vertex};
		for(const auto& ip: ips)
		  if ( (ip%2!=0) && Vtype(id,ip/2)!=0 )  changed_Vrtx.set( Vindx(id, ip/2), 1 );
	      }
	    }
	  }
	}else{ // this time is for dynamic interaction only
	  for (int id=0; id<Ndiags; id++){
	    if (Vcmax>0){
	      std::array<int,1> ips = {ivertex};
	      for(const auto& ip: ips)
		if ( (ip%2!=0) && Vtype(id,ip/2)!=0 )  changed_Vrtx.set( Vindx(id, ip/2), 1 );
	    }
	  }
	}
      }
      if (! Qweight){
      	//t7.start(); // takes 11% of the time
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
	if (Vcmax>0){
	  int iq=0;
	  // Vertices, which are needed to express screened interaction with the same time difference as the unscreened part.
	  for (int ii=0; ii<Nv; ii++){ 
	    if (changed_Vrtx[ii]){
	      int id = vindx(ii)[0];
	      int  i = vindx(ii)[1];
	      int vtyp = Vtype(id,i);
	      if (vtyp==0) cout<<"ERROR changed_Vrtx is wrong, as type==0!"<<endl;
	      double t1 = times_trial(2*i);     // t1 is old time, because for equal time case t[2i+1] was actually t[2i]
	      double t2 = times_trial(2*i+1);   // t2 is the new time, because now we want t[2i+1] to be different.
	      // For now we will neglect possibility of type-II vertex. Should correct later!
	      int i_m = i_diagsG(id,2*i+1);
	      double ti = times_trial( et(i_m) );
	      int i_p = diagsG(id,2*i+1);
	      double to = times_trial( et(i_p) );
	      double ki = norm(mom_g(Gindx(id,i_m)));
	      double ko = norm(mom_g(Gindx(id,2*i+1)));
	      Vrtx_trial(ii) = MoveVertex(t2, t1, ti, to, ki, ko, Gk);
	      double aQ = norm(mom_v(ii));
	      double Vq = 8*pi/(aQ*aQ+lmbda);
	      intpar pq = Interp(aQ,qx);
	      intpar pt = Interp(fabs(t1-t2), tx);
	      if ( fabs(pt.p)>1 ) log << "ERROR tx.i=" << pt.i << " tx.p="<< pt.p << endl;
	      int Nc = abs(vtyp);
	      double ct=0;
	      double lmbda_k = 1.0;
	      for (int k=0; k<Nc; k++){
		ct += binomial(Nc,k) * lmbda_k * bPqt[Nc-k-1](pq,pt);
		lmbda_k *= (lmbda/(8*pi));
	      }
	      long double Vn = ipower(Vq, Nc+1);
	      v_trial(iq) = Vn * ct * Vrtx_trial(ii);
	      v_trial(iq) += Vn * lmbda_k *(1-bubble_only);
	      if (vtyp<0){
		// we also add the single-particle counter term in addition to two-particle counter term.
		int ii_g = single_counter_index(ii);
		long double g_kq = changed_G[ii_g] ? g_trial(ip_ind(ii_g)) : G_current(ii_g);
#ifdef CNTR_VN
		if (g_kq!=0) v_trial(iq) += Vn*lmbda_spct_cutoffk(-vtyp-1)/g_kq;
#else		
		if (g_kq!=0) v_trial(iq) += exp(-aQ*2/kF)*lmbda_spct_cutoffk(-vtyp-1)/g_kq;
#endif		
	      }
	      iq_ind(ii)=iq;
	      iq++;
	    }
	  }
	}
	//t7.stop();
	//t8.start(); // takes 8% of the time
	PQ_new=0;   // recalculating PQ, taking into account one change of time.
	BKdata.PQg_new_Initialize();
	for (int id=0; id<Ndiags; id++){
	  long double PQd = 1.0;
	  for (int i=0; i<2*Norder; i++){
	    int ii = Gindx(id,i);
	    PQd *= ( changed_G[ii] ? g_trial(ip_ind(ii)) : G_current(ii) );
	  }
	  for (int i=1; i<Norder; i++){
	    int ii = Vindx(id,i);
	    PQd *= ( changed_Vrtx[ii] ? v_trial(iq_ind(ii)) : V_current(ii) );
	  }
	  PQd *= diagSign(id);
	  PQ_new += PQd;
	  BKdata.PQg_new_Add(id,PQd);
	}
	//t8.stop();
	//if (!Qweight)
	//  Check_trial_T(3, log, itt, Nv, PQ_new, PQ, lmbda, times_trial, times_to_change, Gk, G_current, V_current,
	//		g_trial, ip_ind, v_trial, iq_ind, changed_G, changed_V,
	//		diagsG, Gindx, Vindx, mom_g, mom_v, diagSign, Vtype, bPqt, qx, tx, vindx, Vrtx_current);
	//if (std::isnan(PQ_new)){
	//  log<<" itt="<<itt<<" PQ_new="<<PQ_new << " PQ="<<PQ << " Qweight=" << Qweight << " time "<< endl;
	//}
      }else{
	PQ_new = V0norm * mweight(amomentm, momentum);
      }
      double ratio = PQ_new/PQ;
      bool accept = fabs(ratio) > 1-drand() && PQ_new!=0;
      if ((itt+1)%Ncout==0 && mpi.rank==mpi.master)
	_PrintInfo_(itt+1, Qweight, icase, amomentm(1), amomentm(Nloops-1), times(itime), PQ_new, PQ, dk_hist, Nweight/(Nmeassure+0.0),log);
      if (accept){
	//t9.start(); // takes 1.7% of time
	Nacc_t +=1 ;
	times = times_trial;  // the new times are now accepted. We change the independent time variable, or two internal times.
	if (!Qweight){
	  for (int ii=0; ii<Ng; ++ii)
	    if (changed_G[ii])
	      G_current(ii) = g_trial(ip_ind(ii)); // The green's functions in the loop have been recalculated, and are now stored.
	  if (Vcmax>0){
	    for (int ii=0; ii<Nv; ii++){
	      if (changed_Vrtx[ii]){
		Vrtx_current(ii) = Vrtx_trial(ii);
		V_current(ii) = v_trial(iq_ind(ii));
	      }
	    }
	  }
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
	  if (Vcmax>0){
	    for (int ii=0; ii<Nv; ii++){
	      if (changed_Vrtx[ii]){
		int id = vindx(ii)[0];
		int  i = vindx(ii)[1];
		int vtyp = Vtype(id,i);
		if (vtyp==0) cout<<"ERROR changed_Vrtx is wrong, as type==0!"<<endl;
		double t1 = times(2*i);     // t1 is old time, because for equal time case t[2i+1] was actually t[2i]
		double t2 = times(2*i+1);   // t2 is the new time, because now we want t[2i+1] to be different.
		// For now we will neglect possibility of type-II vertex. Should correct later!
		int i_m = i_diagsG(id,2*i+1);
		double ti = times( et(i_m) );
		int i_p = diagsG(id,2*i+1);
		double to = times( et(i_p) );
		double ki = norm(mom_g(Gindx(id,i_m)));
		double ko = norm(mom_g(Gindx(id,2*i+1)));
		Vrtx_current(ii) = MoveVertex(t2, t1, ti, to, ki, ko, Gk);
		double aQ = norm(mom_v(ii));
		double Vq = 8*pi/(aQ*aQ+lmbda);
		intpar pq = Interp(aQ,qx);
		intpar pt = Interp(fabs(t1-t2), tx);
		if ( fabs(pt.p)>1 ) log << "ERROR tx.i=" << pt.i << " tx.p="<< pt.p << endl;
		int Nc = abs(vtyp);
		double ct=0;
		double lmbda_k = 1.0;
		for (int k=0; k<Nc; k++){
		  ct += binomial(Nc,k) * lmbda_k * bPqt[Nc-k-1](pq,pt);
		  lmbda_k *= (lmbda/(8*pi));
		}
		long double Vn = ipower(Vq, Nc+1);
		V_current(ii) = Vn * ct * Vrtx_current(ii);
		V_current(ii) += Vn * lmbda_k *(1-bubble_only);
		if (vtyp<0){
		  // we also add the single-particle counter term in addition to two-particle counter term.
		  int ii_g = single_counter_index(ii);
		  long double g_kq = G_current(ii_g);
#ifdef CNTR_VN		  
		  if (g_kq!=0) V_current(ii) += Vn*lmbda_spct_cutoffk(-vtyp-1)/g_kq;
#else		  
		  if (g_kq!=0) V_current(ii) += exp(-aQ*2/kF)*lmbda_spct_cutoffk(-vtyp-1)/g_kq;
#endif		  
		}
	      }
	    }
	  }
	}
	//t9.stop();
	PQ = PQ_new;
	if (!Qweight) BKdata.Set_PQg_new_to_PQg();
	//t10.start(); // takes 0.5% of the time
	if (itime==0) Pln.cmp_single(2*(times(0)-times(1))/beta-1., pl_t, BKdata.Nlt);  // update Legendre Polynomials
	//t10.stop();
	//if (!Qweight) Check(2, log, itt, Nv, PQ, lmbda, times, times_to_change, Gk, G_current, V_current, diagsG, Gindx, Vindx, mom_g, mom_v, diagSign, Vtype, bPqt, qx, tx, vindx, Vrtx_current);
      }
      //log<<itt<<" G_current="<<G_current<<" V_current="<<V_current<<endl;
    }else{  // normalization diagram step
      Nall_w += 1;
      //t11.start(); // takes 0.3% of the time
      if (Qweight){ // trying to jump back into physical hilbert space
	PQ_new=0;
	BKdata.PQg_new_Initialize();
	for (int id=0; id<Ndiags; id++){
	  long double PQd = 1.0;
	  for (int i=0; i<2*Norder; i++) PQd *= G_current(Gindx(id,i)); // we have all propagators precomputed, hence it is easy to compute the diagrams.
	  for (int i=1; i<Norder; i++) PQd *= V_current(Vindx(id,i));
	  PQd *= diagSign(id);
	  PQ_new += PQd;
	  BKdata.PQg_new_Add(id,PQd);
	}
      }else{        // or if we are in the physical Hilbert space, we try to jump into the normalization diagram
	PQ_new = V0norm * mweight(amomentm, momentum);
      }
      //if (std::isnan(PQ_new)){
      //  log<<" itt="<<itt<<" PQ_new="<<PQ_new << " PQ="<<PQ << " Qweight=" << Qweight << " meassuring" << endl;
      //}
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
      //t11.stop();
    }
    
    // Measuring
    if (itt>=Nwarm && itt%tmeassure==0){
      t_mes.start(); // takes 1% time
      Nmeassure += 1;
      double Qa = amomentm(0);
      double t = times(0)-times(1);
      double cw = 1./rw.wt(t);
      double sp = sign(PQ) * cw;
      int iq = std::min(int(Qa/cutoffq*Nq), Nq-1);
      int it = std::min(int(t/beta*Nt), Nt-1);
      if (Qweight){
	Pnorm += Qa*Qa * cw;                 // |Q|^2 * reweighting
	Nweight+=1;
      }else{
	Pbin(it,iq) += sp;
	BKdata.Meassure(itt,tmeassure,PQ,sp,pl_Q,pl_t,momentum(0),amomentm(0),mom_g,Gindx);
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
      if ( itt>2e5/mpi.size && itt%(static_cast<int>(100000/mpi.size+1000)*tmeassure) == 0){
	double last_occurence = (Nweight-Wchanged)/(Nmeassure-Nchanged); // occurence from the last time we changed it
#ifdef _MPI
	double total_occurence=0;
	MPI_Allreduce(&last_occurence, &total_occurence, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	last_occurence = total_occurence/mpi.size;
#endif	
	if ( last_occurence > 0.15){ // decrease by two
	  V0norm /= 2.;
	  Pnorm /= 2.;
	  Nchanged = Nmeassure;
	  Wchanged = Nweight;
	  if (mpi.rank==mpi.master) log<<std::left<<(itt/1.0e6)<<"M occurence="<<std::setw(8)<<last_occurence<<" V0 reduced to "<<V0norm<<std::endl;
	}
	if ( last_occurence < 0.03){ // increase by two
	  V0norm *= 2.;
	  Pnorm *= 2.;
	  Nchanged = Nmeassure;
	  Wchanged = Nweight;
	  if (mpi.rank==mpi.master) log<<std::left<<(itt/1.0e6)<<"M occurence="<<std::setw(8)<<last_occurence<<" V0 increased to "<<V0norm<<std::endl;
	}
      }
      if (itt%(500000*tmeassure) == 0){
	if (Nmeassure-Nweight>1000){
#ifdef _MPI	  
	  MPI_Allreduce(MPI_IN_PLACE, mweight.K_hist.data(), mweight.K_hist.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
	  // We can get overflow during MPI for a long run. We should use a constant for normalization.
	  // We could here normalize K_hist (with knorm), and than when we measure, we would
	  // add instead of adding unity, we would add 1./knorm
	  dk_hist *= mweight.Normalize_K_histogram();
	  if (dk_hist < 1e-16) dk_hist = 1.0;
	  mweight.Recompute(/*K_hist,*/ mpi.rank==mpi.master);
	  // If we are in a meassuring space, we have to update now the value of the meassuring diagram as the weight has changed.
	  if (Qweight) PQ = V0norm * mweight(amomentm, momentum);
	}
      }
      t_mes.stop();
    }
  }
  t_all.stop();
  BKdata.C_Pln *= 1.0/Nmeassure;
  Pbin         *= 1.0/Nmeassure;
  Pnorm        *= 1.0/Nmeassure;
  double occurence = static_cast<double>(Nweight)/static_cast<double>(Nmeassure);

  if (GetHistogram){
    K_Ghist *= 1.0/Nmeassure;
    K_Vhist *= 1.0/Nmeassure;
  }
  
#ifdef _MPI  
  double dat[2] = {Pnorm, occurence};
  if (mpi.rank==mpi.master){
    MPI_Reduce(MPI_IN_PLACE, BKdata.C_Pln.data(), BKdata.C_Pln.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE,  Pbin.data(),  Pbin.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, dat, 2, MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, mweight.K_hist.data(), mweight.K_hist.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    if (GetHistogram){
      MPI_Reduce(MPI_IN_PLACE, K_Ghist.data(), K_Ghist.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
      MPI_Reduce(MPI_IN_PLACE, K_Vhist.data(), K_Vhist.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    }
  }else{
    MPI_Reduce(BKdata.C_Pln.data(), BKdata.C_Pln.data(), BKdata.C_Pln.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    MPI_Reduce( Pbin.data(),  Pbin.data(),  Pbin.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    MPI_Reduce(dat, dat, 2, MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    MPI_Reduce(mweight.K_hist.data(), mweight.K_hist.data(), mweight.K_hist.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    if (GetHistogram){
      MPI_Reduce(K_Ghist.data(), K_Ghist.data(), K_Ghist.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
      MPI_Reduce(K_Vhist.data(), K_Vhist.data(), K_Vhist.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    }
  }
  if (mpi.rank==mpi.master){
    BKdata.C_Pln *= 1./mpi.size;
    Pbin  *= 1./mpi.size;
    Pnorm = dat[0]/mpi.size;
    occurence = dat[1]/mpi.size;
    //
    mweight.K_hist *= 1./mpi.size;
  }
#endif

  if (mpi.rank==mpi.master){
    log<<"total acceptance rate="<<(Nacc_k+Nacc_q+Nacc_t+Nacc_w)/(p.Nitt+0.0)<<" "<<" k-acceptance="<<(Nacc_k/(Nall_k+0.0))<<" q-acceptance="<<(Nacc_q/(Nall_q+0.0))<<" t-acceptanece="<<(Nacc_t/(Nall_t+0.0))<<" w-acceptanece="<<(Nacc_w/(Nall_w+0.0))<<std::endl;
    log<<" k-trials="<<(Nall_k/(p.Nitt+0.0))<<" q-trials="<<(Nall_q/(p.Nitt+0.0))<<" t-trials="<<(Nall_t/(p.Nitt+0.0))<<" w-trials="<<(Nall_w/(p.Nitt+0.0))<<std::endl;
    //cout<<"t1="<<t1.elapsed()<<" t2="<<t2.elapsed()<<" t3="<<t3.elapsed()<<" t4="<<t4.elapsed()<<" t5="<<t5.elapsed()<<" t6="<<t6.elapsed()<<" t7="<<t7.elapsed()<<" t8="<<t8.elapsed()<<" t9="<<t9.elapsed()<<" t10="<<t10.elapsed()<<" t11="<<t11.elapsed()<<std::endl;
    log<<"total time="<<t_all.elapsed()<<" measuring time="<<t_mes.elapsed()<<std::endl;

    
    BKdata.C_Pln *= 1.0/(4*pi) * (fabs(V0norm)/Pnorm); // 4*pi because we must divide by 1/(4*pi*q**2), and 1/q**2 was already done in sampling
    Pbin  *= 1.0/(4*pi) * (fabs(V0norm)/Pnorm);
    log<<"  meassuring diagram occurence frequency="<<occurence<<" and its norm Pnorm="<<Pnorm<<std::endl;
    
    // Proper normalization of the resulting Monte Carlo data
    double norm = ipower( beta/((2*pi)*(2*pi)*(2*pi)), Norder) * (4*pi*cutoffq*cutoffq*cutoffq/3);
    BKdata.C_Pln *= norm;

    // Later we will use Nthbin/2.0*Nkbin/2.0  normalization in the Kernel to transform from binned data to function values.
    BKdata.Normalize(beta, cutoffq);
    
    double dq_binning = cutoffq/Nq;
    double dt_binning = beta/Nt;
    Pbin *= (norm/(dq_binning * dt_binning));
    
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



template<typename GK, typename Data>
void sample_static_fastD_combined(Data& BKdata, bl::Array<double,2>& Pbin, std::ostream& log,
				  double lmbda, const std::vector<double>& lmbda_spct, const GK& Gk, const params& p,
				  const bl::Array<unsigned short,2>& diagsG, const bl::Array<float,1>& diagSign,
				  const std::vector<std::vector<std::vector<int> > >& Loop_index, const std::vector<std::vector<std::vector<int> > >& Loop_type,
				  bl::Array<char,2>& Vtype, my_mpi& mpi)
{
  int bubble_only = 0;
  typedef double real;
  log.precision(12);
  bool Q0w0 = BKdata.Q0w0;
  double Q_external=BKdata.Q_external;
  if (!Q0w0){
    if ( BKdata.Nlt!=p.Nlt || BKdata.Nlq!=p.Nlq){
      log<<"ERROR : Dimensions of C_Pln is wrong : either "<<BKdata.Nlt<<" != "<<p.Nlt<<" or "<<BKdata.Nlq<<" != "<<p.Nlq<<std::endl;
      exit(1);
    }
    if (Pbin.extent(0)!=p.Nt || Pbin.extent(1)!=p.Nq) Pbin.resize(p.Nt,p.Nq); Pbin=0.0;
  }else{
    if (Pbin.extent(0)!=p.Nt) Pbin.resize(p.Nt,1); Pbin=0.0;
  }
  
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
  
  //CounterCoulombV Vqc(lmbda);

  // Set up which times need change when counter term is dynamic
  vector<int> times_2_change;
  std::set<int> times_to_change;
  int Vcmax=0;
  {
    std::set<int> counter_types;
    times_to_change.insert(0);
    int i_start = Q0w0 ? 1 : 0;   // CHANGE
    for (int i=i_start; i<Norder; i++) times_to_change.insert(2*i);
    // default vertices to change time will be : [0,2,...,2*(Norder-1)]
    for (int id=0; id<Ndiags; id++)    // Checking where can we add single-particle counter terms. 
      for (int j=1; j<Norder; j++)     // skip meassuring line
	if (Vtype(id,j)>0){
	  counter_types.insert(Vtype(id,j));
	  times_to_change.insert(2*j+1);  // since the counter-term is dynamic, we need to make vertex 2j+1 different from 2j.
	}
    if (counter_types.size()>0){
      Vcmax = *counter_types.rbegin();
    }
    times_2_change.resize(times_to_change.size());
    int j=0;
    for (auto it=times_to_change.begin(); it!=times_to_change.end(); ++it, ++j) times_2_change[j] = (*it);
    log<<"times_to_change : ";
    for (auto it=times_to_change.begin(); it!=times_to_change.end(); ++it) log << (*it) <<",  ";
    log<<std::endl;
    log<<"counter_types : ";
    for (auto it=counter_types.begin(); it!=counter_types.end(); ++it) log << (*it) <<",  ";
    log<<std::endl;
    log << "times_2_change: "<<endl;
    for (int i=0; i<times_2_change.size(); i++) log << times_2_change[i] << ",";
    log << endl;
  }
  // Reads numerically evaluated Bubble diagram.
  vector<Spline2D<double> > bPqt(Vcmax);
  bl::Array<double,1> qx, tx;  
  if (Vcmax>0) ReadCounterTermPolarization(bPqt, qx, tx, p.beta);
  bl::Array<int,2> binomial = BinomialCoeff(Vcmax);
  
  int tmeassure=p.tmeassure, Ncout=p.Ncout, Nwarm=p.Nwarm;
  double V0norm = p.V0norm, dkF = p.dkF, kF=p.kF;// V0exp= p.V0exp;
  double cutoffk = p.cutoffk, dRk = p.dRk;
  double beta = p.beta;
  double cutoffq = p.cutoffq;
  int Nt = p.Nt, Nq = p.Nq;


  // THIS PART IS DIFFERENT
  // Everything for Legendre Polynomials in time and momentum
  LegendrePl Pln;
  bl::Array<double,1> pl_Q;  // Will contain legendre Pl(2*q/qmax-1)
  bl::Array<double,1> pl_t;  // Will contain legendre Pl(2*t//beta-1)
  if (!Q0w0){
    Pln.resize(std::max(BKdata.Nlq,BKdata.Nlt));
    pl_Q.resize(BKdata.Nlq+1);  // Will contain legendre Pl(2*q/qmax-1)
    pl_t.resize(BKdata.Nlt+1);  // Will contain legendre Pl(2*t//beta-1)
  }
  BKdata.C_Pln = 0.0;            // cummulative result for legendre expansion
  //LegendrePl Pln(std::max(BKdata.Nlq,BKdata.Nlt));
  //bl::Array<double,1> pl_Q(BKdata.Nlq+1);  // Will contain legendre Pl(2*q/qmax-1)
  //bl::Array<double,1> pl_t(BKdata.Nlt+1);  // Will contain legendre Pl(2*t//beta-1)
  //BKdata.C_Pln = 0.0;            // cummulative result for legendre expansion
  // THIS PART IS DIFFERENT
  
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
  findUnique_noindx_spct(Gindx, Vindx, gindx, vindx, N0v, diagsG, Vtype, Loop_index, Loop_type, 0, true, debug, log);

  // Once more we go through single-particle counter terms and create index single_counter_index
  bl::Array<int,1> single_counter_index;
  Find_Single_Particle_Counter_G_index(single_counter_index, Ndiags, Norder, vindx.extent(0), diagsG, Vtype, Gindx, Vindx);
  
  // This is related to Baym-Kadanoff approach. For efficiency, we want to find which diagram have common values of momenta at the right hand side of the diagram.
  BKdata.FindGroups(diagsG, Loop_index, Loop_type, log);
  
  if (Loop_index.size()!=Ndiags){ log<<"Wrong size of Loop_index "<<Loop_index.size()<<" instead of "<<Ndiags<<std::endl; exit(1);}
  int Nloops = Loop_index[0].size();
  if (Nloops != Norder+1){ log<<"Expecting Nloops==Norder+1 in Loops_index, but got Nloops="<<Nloops<<" Norder="<<Norder<<std::endl; exit(1);}
  int Ng = gindx.extent(0);
  int Nv = vindx.extent(0);
  int _Nv_ = std::max(Nv,1); // can not assign 0 to empty file

  bl::Array<bl::Array<sint,1>,1> loop_Gkind, loop_Vqind;
  bl::Array<bl::Array<char,1>,1> loop_Gksgn, loop_Vqsgn;
  int Ngp, Nvp;
  
  // Find index-tables to all momentum loops, so that we do not need to loop over all diagrams, but rather only over all differet propagators corresponding to certain loop.
  // Here we also take into account that single-particle counter terms are sometimes added.
  Get_GVind(loop_Gkind, loop_Gksgn, loop_Vqind, loop_Vqsgn, Ngp, Nvp, Loop_index, Loop_type, Gindx, Vindx, single_counter_index, lmbda_spct, Vtype, true, log);
  int Ngp_tr = GetMax_NGp_trial(i_diagsG, Gindx, Ng, Ngp);

  // DEBUG
  if (mpi.rank==mpi.master){
    log<<"lmbda_spct=";
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
    log<<"loop_Vqind:"<<std::endl;
    for (int iloop=0; iloop<Nloops; ++iloop){
      for (int iq=0; iq<loop_Vqind(iloop).extent(0); ++iq){ // Now do the same for the interaction
	int ii   = loop_Vqind(iloop)(iq);
	int isgn = loop_Vqsgn(iloop)(iq);
	
	int id = vindx(ii)[0];
	int  i = vindx(ii)[1];
	int vtyp = Vtype(id,i);
	
	log<<"iloop="<<iloop<<" iq="<<iq<<" ii_v="<<ii<<" isgn="<<isgn<<" which comes from id="<< id <<" and pair =("<<2*i<<","<<2*i+1<<") of type="<<vtyp<<std::endl;
      }
    }
  }
  // END DEBUG
  
  Reweight_time rw(beta,p.lmbdat);
  int Nbin = 513;
  //int Nbin = 129;
  bl::Array<double,1> lmbda_spct_cutoffk(lmbda_spct.size());
  // Instead of using : lmbda_spct, which is q-independent, we will introduce "artificial" q-dependence, which gives identical integral.
  // When lmbda_spct is q-independent, we should normalize it by  lmbda_spct * 1/Integrate[d^3 1/(2*pi)^3].
  // The "artificial" dependence we choose is lmbda_spct * (Vq)^2, in which case the normalization constant is
  //     lmbda_spct * (Vq)^2 / Integrate[d^3q Vq^2/(2*pi)^3 ]
  //  which is
  //     lmbda_spct * (Vq)^2 / ( 2/sqrt(lmbda) *atan(ck/sqrt(lmbda)) - 2*ck/(ck^2+lmbda) )
  std::vector<double> nrm = Nrm(lmbda, cutoffk, lmbda_spct.size()+2);
  for (int i=0; i<lmbda_spct.size(); ++i){
#ifdef CNTR_VN
    // [Vq^2, Vq^3, Vq^4,..]
    lmbda_spct_cutoffk(i) = lmbda_spct[i]*nrm[i+2];
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
  {
    double Qa = Q0w0 ? Q_external : p.kF*drand(); // absolute value of external momentum
    momentum(0) = 0.0,0.0,Qa; // external momentum Q
    amomentm(0) = Qa;
    for (int ik=1; ik<Nloops; ik++){
      double th = pi*(1-drand()), phi = 2*pi*(1-drand());
      momentum(ik) = kF*sin(th)*cos(phi), kF*sin(th)*sin(phi), kF*cos(th);
      amomentm(ik) = kF;
    }
  }
  log.precision(12);
  log<<"Starting times="<<times<<" and starting momenta="<<momentum<<std::endl;

  bl::Array<double,1> G_current(Ng);
  bl::Array<long double,1> V_current(_Nv_);
  // This is momentum for all diagrams and all propagators
  bl::Array<bl::TinyVector<real,3>,1> mom_g(Ng), mom_v(Nv);
  bl::Array<long double,1> Vrtx_current(1);
  if (Vcmax>0) Vrtx_current.resize(_Nv_);
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
      for (int i=0; i<2*Norder; i++){
	//int i_final = diagsG(id,i);
	int i_previ = i_diagsG(id,i);
	bl::TinyVector<double,3> k_in = mom_G(id,i_previ);
	bl::TinyVector<double,3> k_out = mom_G(id,i);
	bl::TinyVector<double,3> q = mom_V(id,i/2)*(1.-2*(i%2));
	if (fabs(norm(k_in-k_out)-norm(q))>1e-6){
	  log<<"ERROR: diagram "<<id<<" vertex i="<<i<<" k_in="<<k_in<<" k_out="<<k_out<<" q="<<q<<std::endl;
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
    Vrtx_current=0;
    // Vertices, which are needed to express screened interaction with the same time difference as the unscreened part.
    for (int ii=0; ii<Nv; ++ii){
      int id = vindx(ii)[0];
      int  i = vindx(ii)[1];
      int vtyp = Vtype(id,i);
      if (vtyp!=0){
	double t1 = times(2*i);     // t1 is old time, because for equal time case t[2i+1] was actually t[2i]
	double t2 = times(2*i+1);   // t2 is the new time, because now we want t[2i+1] to be different.
	// For now we will neglect possibility of type-II vertex. Should correct later!
	int i_m = i_diagsG(id,2*i+1);
	double ti = times( et(i_m) );
	int i_p = diagsG(id,2*i+1);
	double to = times( et(i_p) );
	double ki = norm(mom_G(id,i_m));
	double ko = norm(mom_G(id,2*i+1));
	Vrtx_current(ii) = MoveVertex(t2, t1, ti, to, ki, ko, Gk);
	//double direct_vertex = Gk(ki, t2-ti)*Gk(ko, to-t2)/(Gk(ki, t1-ti)*Gk(ko, to-t1) );
	//log<<"t1="<<t1<<" t2="<<t2<<" i_m="<<i_m<<" ti="<<ti<<" i_p="<<i_p<<" to="<<to<<" ki="<<ki<<" ko="<<ko<<" vrtx="<<Vrtx_current(ii)<<"  direct_vrtx="<<direct_vertex<<endl;
	if ((Vtype(id,i_m/2)!=0 && (i_m%2==1)) || (Vtype(id,i_p/2)!=0 && (i_p%2==1))){
	  log<<"ERROR : We have type2, which is not yet implemented!  id="<<id<<" i="<<i<<" vtyp="<<vtyp<<endl;
	  log<<"..... i_m="<<i_m<<" i_p="<<i_p<<" 2*i="<< 2*i << endl;
	  log<<" 1 : "<< (Vtype(id,i_m/2)!=0 && (i_m != 2*i)) << " 2 : "<< (Vtype(id,i_p/2)!=0 && (i_p != 2*i)) << endl;
	}
      }
    }
    //log<<"Vrtx_current="<<Vrtx_current<<endl;
    V_current=0;  
    for (int ii=0; ii<Nv; ++ii){
      int id = vindx(ii)[0];
      int  i = vindx(ii)[1];
      double aQ = norm(mom_V(id,i));
      int vtyp = Vtype(id,i);
      double Vq = 8*pi/(aQ*aQ+lmbda);
      if (vtyp==0){
	V_current(ii) = Vq;
      }else{
	intpar pq = Interp(aQ,qx);
	intpar pt = Interp(fabs(times(2*i)-times(2*i+1)), tx);
	int Nc = abs(vtyp);
	double ct=0;
	double lmbda_k = 1.0;
	for (int k=0; k<Nc; k++){
	  ct += binomial(Nc,k) * lmbda_k * bPqt[Nc-k-1](pq,pt);
	  lmbda_k *= (lmbda/(8*pi));
	}
	long double Vn = ipower(Vq, Nc+1);
	V_current(ii) = Vn * ct * Vrtx_current(ii);
	V_current(ii) += Vn * lmbda_k *(1-bubble_only);
	if (vtyp < 0){
	  // we also add the single-particle counter term in addition to two-particle counter term.
	  int ii_g = single_counter_index(ii);
	  long double g_kq = G_current(ii_g);
#ifdef CNTR_VN
	  // vtype=-1, Nc=1 : lmbda[0] -> Vn = Vq^2
	  // vtype=-2, Nc=2 : lmbda[1] -> Vn = Vq^3
	  if (g_kq!=0) V_current(ii) += Vn*lmbda_spct_cutoffk(-vtyp-1)/g_kq;
#else	  
	  if (g_kq!=0) V_current(ii) += exp(-aQ*2/kF)*lmbda_spct_cutoffk(-vtyp-1)/g_kq;
#endif	  
	}
      }
    }
    //log<<"V_current="<<V_current<<endl;
    PQ=0;
    BKdata.PQg_Initialize();
    for (int id=0; id<Ndiags; id++){
      long double PQd = 1.0;
      for (int i=0; i<2*Norder; i++) PQd *= G_current(Gindx(id,i));
      for (int i=1; i<Norder; i++) PQd *= V_current(Vindx(id,i));
      PQd *= diagSign(id);
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
    Pln.cmp_single(2*amomentm(0)/cutoffq-1., pl_Q, BKdata.Nlq);          // Legendre polynomials at this Q
    Pln.cmp_single(2*(times(0)-times(1))/beta-1., pl_t, BKdata.Nlt);     // Legendre polynomials at this time
  }
  
  bl::Array<double,1> times_trial(2*Norder);
  int _Nvp_ = std::max(Nvp,1); // can not assign 0 to empty file
  bl::Array<bl::TinyVector<real,3>,1> tmom_g(Ngp), tmom_v(_Nvp_);
  bl::Array<sint,1> ip_ind(Ng), iq_ind(Nv);
  bl::Array<double,1> g_trial(Ngp_tr);
  bl::Array<long double,1> v_trial(_Nvp_);
  BitArray changed_G(Ng), changed_V(Nv);

  BitArray changed_Vrtx(_Nv_);
  bl::Array<long double,1> Vrtx_trial(1);
  if (Vcmax>0){
    Vrtx_trial.resize(_Nv_);
  }
  
  int Qweight = 0;      // if Qweight==1 then we have the normalization diagram
  double Nweight = 0;   // how many times did we see measuring diagram.
  int NlowT = 0;        // how many times do we see external time < 0.05*beta
  int Nmeassure = 0;    // how many times did we measure quantities.
  int Nchanged  = 0;    // when was V0norm last changed?
  double Wchanged = 0;  // what was Nweight when we last time changed V0norm?
  int Nacc_k = 0, Nacc_q = 0, Nacc_w = 0, Nacc_t = 0;
  int Nall_k = 0, Nall_q = 0, Nall_w = 0, Nall_t = 0;
  if (mpi.rank==mpi.master) PrintInfo_(0, Qweight, amomentm(0), amomentm(1), times(0)-times(1), PQ, PQ, 0, 0, log);
  //Check(0, log, -1, Nv, PQ, lmbda, times, times_to_change, Gk, G_current, V_current, diagsG, Gindx, Vindx, mom_g, mom_v, diagSign, Vtype, bPqt, qx, tx, vindx, Vrtx_current);

  t_all.start();
  for (int itt=0; itt<p.Nitt; itt++){
    double PQ_new=0;
    int icase = tBisect(drand(), Prs); // which steps should we attempt?
    if (icase==0){ // changing momentum Q or k-point
      int iloop = (!Q0w0) ? static_cast<int>(Nloops*drand()) : 1+static_cast<int>((Nloops-1)*drand());
      bl::TinyVector<double,3> K_new; double Ka_new; double trial_ratio=1;
      bool accept=false;
      if (iloop==0){ // this is the loop for the external momentum Q
	Nall_q += 1;
	accept = Find_new_Q_point(K_new, Ka_new, trial_ratio, momentum(iloop), amomentm(iloop), cutoffq, drand);
      }else{         // any other momentum point
	Nall_k += 1;
	accept = Find_new_k_point(K_new, Ka_new, trial_ratio, momentum(iloop), amomentm(iloop), cutoffk, drand, drand()<p.Qring, dkF, dRk);
      }
      if (accept){
	if (!Qweight){
	  bl::TinyVector<double,3> dK = K_new - momentum(iloop); // how much momentum will be changed.
	  //t1.start(); // takes 40% of the time
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
	  changed_Vrtx = 0;
	  // Vertices, which are needed to express screened interaction with the same time difference as the unscreened part.
	  if (Vcmax>0){
	    for (int id=0; id<Ndiags; id++){ // updating momentum in all propagators
	      const std::vector<int>& lindex = Loop_index[id][iloop];
	      const std::vector<int>& ltype  = Loop_type[id][iloop];
	      for (int i=0; i<lindex.size(); i++){
		if ( abs(ltype[i])==1 ){ // this is G-propagator
		  int ic = lindex[i]; // momentum of G[ic] is changed, hence when counter-term ends at ic (ic is odd), we need to update vertex, because k_o will change in MoveVertex.
		  // To correct for the unscreened part of the interaction, we need to update time-dependent interaction
		  if ( (ic%2!=0) && Vtype(id, ic/2) !=0 ) changed_Vrtx.set( Vindx(id, ic/2), 1 ); // odd vertex, which connects interaction of the counter-term
		  int i_p = diagsG(id,ic); // alternatively, incoming  vertex k_i might change in MoveVertex. If k[ic]==k_i then diagsG(ic) is the vertex where counter term ends.
		  if ( (i_p%2!=0) && Vtype(id, i_p/2)!=0 )  changed_Vrtx.set( Vindx(id, i_p/2), 1 );
		}
	      }
	    }
	    // Vertices, which are needed to express screened interaction with the same time difference as the unscreened part.
	    for (int ii=0; ii<Nv; ii++){
	      if (changed_Vrtx[ii]){
		int id = vindx(ii)[0];
		int  i = vindx(ii)[1];
		int vtyp = Vtype(id,i);
		if (vtyp==0) log<<"ERROR : changed_Vrtx is wrong, as type==0!"<<endl;
		double t1 = times(2*i);     // t1 is old time, because for equal time case t[2i+1] was actually t[2i]
		double t2 = times(2*i+1);   // t2 is the new time, because now we want t[2i+1] to be different.
		// For now we will neglect possibility of type-II vertex. Should correct later for order higher than 5!
		int i_m = i_diagsG(id,2*i+1);
		double ti = times( et(i_m) );
		int i_p = diagsG(id,2*i+1);
		double to = times( et(i_p) );
		double ii_g = Gindx(id,i_m);
		double ki = changed_G[ii_g] ? norm(tmom_g(ip_ind(ii_g))) : norm(mom_g(ii_g));
		ii_g = Gindx(id,2*i+1);
		double ko = changed_G[ii_g] ? norm(tmom_g(ip_ind(ii_g))) : norm(mom_g(ii_g));
		Vrtx_trial(ii) = MoveVertex(t2, t1, ti, to, ki, ko, Gk);
	      }
	    }
	  }
	  //t2.start(); // takes 1.5% of the time
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
	    if (vtyp==0){
	      v_trial(iq) = Vq;
	    }else{
	      intpar pq = Interp(aQ,qx);
	      intpar pt = Interp(fabs(times(2*i)-times(2*i+1)), tx);
	      int Nc = abs(vtyp);
	      double ct=0;
	      double lmbda_k = 1.0;
	      for (int k=0; k<Nc; k++){
		ct += binomial(Nc,k) * lmbda_k * bPqt[Nc-k-1](pq,pt);
		lmbda_k *= (lmbda/(8*pi));
	      }
	      long double Vn = ipower(Vq, Nc+1);
	      long double Vrtx = changed_Vrtx[ii] ? Vrtx_trial(ii) : Vrtx_current(ii);
	      v_trial(iq) = (Vn * ct) * Vrtx ;
	      v_trial(iq) += Vn * lmbda_k *(1-bubble_only);
	      if (vtyp<0){
		// we also add the single-particle counter term in addition to two-particle counter term.
		int ii_g = single_counter_index(ii);
		long double g_kq = changed_G[ii_g] ? g_trial(ip_ind(ii_g)) : G_current(ii_g);
#ifdef CNTR_VN		
		if (g_kq!=0) v_trial(iq) += Vn*lmbda_spct_cutoffk(-vtyp-1)/g_kq;
#else		
		if (g_kq!=0) v_trial(iq) += exp(-aQ*2/kF)*lmbda_spct_cutoffk(-vtyp-1)/g_kq;
#endif		
	      }
	    }
	    iq_ind(ii)=iq;
	  }
	  //t2.stop();
	  // we computed the polarization diagram
	  //t3.start();// takes 22% of the time
	  PQ_new=0;
	  BKdata.PQg_new_Initialize();
	  for (int id=0; id<Ndiags; id++){
	    long double PQd = 1.0;
	    for (int i=0; i<2*Norder; i++){
	      int ii = Gindx(id,i);
	      PQd *= ( changed_G[ii] ? g_trial(ip_ind(ii)) : G_current(ii) );
	    }
	    for (int i=1; i<Norder; i++){
	      int ii = Vindx(id,i);
	      PQd *= ( changed_V[ii] ? v_trial(iq_ind(ii)) : V_current(ii) );
	    }
	    PQd *=  diagSign(id);
	    PQ_new += PQd;
	    BKdata.PQg_new_Add(id,PQd);
	  }
	  //t3.stop();
	}else{
	  double Ka_old = amomentm(iloop);
	  amomentm(iloop) = Ka_new;
	  bl::TinyVector<double,3> K_old = momentum(iloop);
	  momentum(iloop) = K_new;
	  PQ_new = V0norm * mweight(amomentm, momentum);
	  amomentm(iloop) = Ka_old;
	  momentum(iloop) = K_old;
	}
	//if (!Qweight)
	//  Check_trial_K(3, log, itt, Nv, PQ_new, PQ, lmbda, times, times_to_change, Gk, G_current, V_current,
	//               g_trial, ip_ind, v_trial, iq_ind, tmom_g, changed_G, tmom_v, changed_V,
	//               diagsG, Gindx, Vindx, mom_g, mom_v, diagSign, Vtype, bPqt, qx, tx, vindx, Vrtx_current);
	//if (std::isnan(PQ_new)){
	//  log<<" itt="<<itt<<" PQ_new="<<PQ_new << " PQ="<<PQ << " Qweight=" << Qweight << " momentum "<< endl;
	//}
	double ratio = PQ_new*trial_ratio / PQ;
	accept = fabs(ratio) > 1-drand() && PQ_new!=0;
      }
      //if ((itt+1)%Ncout==0 && mpi.rank==mpi.master) _PrintInfo_(itt+1, Qweight, icase, amomentm(1), amomentm(Nloops-1), times(0), PQ_new, PQ, NlowT/(Nmeassure+0.0), Nweight/(Nmeassure+0.0), log);
      if ((itt+1)%Ncout==0 && mpi.rank==mpi.master) _PrintInfo_(itt+1, Qweight, icase, amomentm(1), amomentm(Nloops-1), times(0), PQ_new, PQ, dk_hist, Nweight/(Nmeassure+0.0), log);
      
      if (accept){
	if (iloop==0) Nacc_q += 1; else Nacc_k += 1;
	if (!Qweight){
	  momentum(iloop) = K_new;  // this momentum was changed
	  amomentm(iloop) = Ka_new;
	  for (int ip=0; ip<loop_Gkind(iloop).extent(0); ++ip){
	    int ii   = loop_Gkind(iloop)(ip); 
	    mom_g(ii) = tmom_g(ip);
	  }
	  for (int iq=0; iq<loop_Vqind(iloop).extent(0); ++iq){ // Now do the same for the interaction
	    int ii   = loop_Vqind(iloop)(iq);
	    mom_v(ii) = tmom_v(iq);
	  }
	  for (int ip=0; ip<loop_Gkind(iloop).extent(0); ++ip){ // all G-propagators, which are changed when momentum in certain loop is changed.
	    int ii   = loop_Gkind(iloop)(ip);  // index of G-propagators, which we are now changing
	    G_current(ii)=g_trial(ip);
	  }
	  for (int iq=0; iq<loop_Vqind(iloop).extent(0); ++iq){ // Now do the same for the interaction
	    int ii   = loop_Vqind(iloop)(iq);
	    V_current(ii) = v_trial(iq);
	  }
	  if (Vcmax>0)
	    for (int ii=0; ii<Nv; ii++)
	      if (changed_Vrtx[ii]) Vrtx_current(ii) = Vrtx_trial(ii);
	}else{
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
	  changed_Vrtx = 0;
	  if (Vcmax>0){
	    // Vertices, which are needed to express screened interaction with the same time difference as the unscreened part.
	    for (int id=0; id<Ndiags; id++){ // updating momentum in all propagators
	      const std::vector<int>& lindex = Loop_index[id][iloop];
	      const std::vector<int>& ltype  = Loop_type[id][iloop];
	      for (int i=0; i<lindex.size(); i++){
		if ( abs(ltype[i])==1 ){
		  int ic = lindex[i]; // momentum of G[ic] is changed, hence when counter-term ends at ic (ic is odd), we need to update vertex, because k_o will change in MoveVertex.
		  // To correct for the unscreened part of the interaction, we need to update time-dependent interaction
		  if ( (ic%2!=0) && Vtype(id, ic/2)!=0 ) changed_Vrtx.set( Vindx(id, ic/2), 1 );
		  int i_p = diagsG(id,ic); // alternatively, incoming  vertex k_i might change in MoveVertex. If k[ic]==k_i then diagsG(ic) is the vertex where counter term ends.
		  if ( (i_p%2!=0) && Vtype(id, i_p/2)!=0 )  changed_Vrtx.set( Vindx(id, i_p/2), 1 );
		}
	      }
	    }
	    // Vertices, which are needed to express screened interaction with the same time difference as the unscreened part.
	    for (int ii=0; ii<Nv; ii++){
	      if (changed_Vrtx[ii]){
		int id = vindx(ii)[0];
		int  i = vindx(ii)[1];
		int vtyp = Vtype(id,i);
		if (vtyp==0) log<<"ERROR changed_Vrtx is wrong, as type==0!"<<endl;
		double t1 = times(2*i);     // t1 is old time, because for equal time case t[2i+1] was actually t[2i]
		double t2 = times(2*i+1);   // t2 is the new time, because now we want t[2i+1] to be different.
		// For now we will neglect possibility of type-II vertex. Should correct later!
		int i_m = i_diagsG(id,2*i+1);
		double ti = times( et(i_m) );
		int i_p = diagsG(id,2*i+1);
		double to = times( et(i_p) );
		double ki = norm(mom_g(Gindx(id,i_m)));
		double ko = norm(mom_g(Gindx(id,2*i+1)));
		Vrtx_current(ii) = MoveVertex(t2, t1, ti, to, ki, ko, Gk);
	      }
	    }
	  }
	  //t2.start(); // takes 1.5% of the time
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
	    if (vtyp==0){
	      V_current(ii) = Vq;
	    }else{
	      intpar pq = Interp(aQ,qx);
	      intpar pt = Interp(fabs(times(2*i)-times(2*i+1)), tx);
	      int Nc = abs(vtyp);
	      double ct=0;
	      double lmbda_k = 1.0;
	      for (int k=0; k<Nc; k++){
		ct += binomial(Nc,k) * lmbda_k * bPqt[Nc-k-1](pq,pt);
		lmbda_k *= (lmbda/(8*pi));
	      }
	      long double Vn = ipower(Vq, Nc+1);
	      V_current(ii) = Vn * ct * Vrtx_current(ii);
	      V_current(ii) += Vn * lmbda_k *(1-bubble_only);
	      if (vtyp<0){
		// we also add the single-particle counter term in addition to two-particle counter term.
		int ii_g = single_counter_index(ii);
		long double g_kq = G_current(ii_g);
#ifdef CNTR_VN		
		if (g_kq!=0) V_current(ii) += Vn*lmbda_spct_cutoffk(-vtyp-1)/g_kq;
#else		
		if (g_kq!=0) V_current(ii) += exp(-aQ*2/kF)*lmbda_spct_cutoffk(-vtyp-1)/g_kq;
#endif		
	      }
	    }
	  }
	  //t5.stop();
	}
	PQ = PQ_new;
	if (!Qweight) BKdata.Set_PQg_new_to_PQg();
	//t6.start(); // takes 0.5% time
	if (iloop==0 && !Q0w0) Pln.cmp_single(2*amomentm(iloop)/cutoffq-1., pl_Q, BKdata.Nlq);  // update Legendre Polynomials
	//t6.stop();
	//if (!Qweight) Check(1, log, itt, Nv, PQ, lmbda, times, times_to_change, Gk, G_current, V_current, diagsG, Gindx, Vindx, mom_g, mom_v, diagSign, Vtype, bPqt, qx, tx, vindx, Vrtx_current);
      }
    } else if (icase==1){  // changing one of the time variables
      Nall_t += 1;
      int itime = times_2_change[ static_cast<int>( times_2_change.size()*drand() ) ];
      changed_G=0;              // which propagators are being changed?
      changed_Vrtx = 0;
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
	  if (Vcmax>0){
	    int i_n = diagsG(id,ivertex);
	    std::array<int,2> ips = {i_n, i_pre_vertex};
	    for(const auto& ip: ips)
	      if ( (ip%2!=0) && Vtype(id,ip/2)!=0 )  changed_Vrtx.set( Vindx(id, ip/2), 1 );
	  }
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
	      if (Vcmax>0){
		int i_n = diagsG(id,ivertex);
		std::array<int,3> ips = {i_n, ivertex, i_pre_vertex};
		for(const auto& ip: ips)
		  if ( (ip%2!=0) && Vtype(id,ip/2)!=0 )  changed_Vrtx.set( Vindx(id, ip/2), 1 );
	      }
	    }
	  }
	}else{ // this time is for dynamic interaction only
	  for (int id=0; id<Ndiags; id++){
	    if (Vcmax>0){
	      std::array<int,1> ips = {ivertex};
	      for(const auto& ip: ips)
		if ( (ip%2!=0) && Vtype(id,ip/2)!=0 )  changed_Vrtx.set( Vindx(id, ip/2), 1 );
	    }
	  }
	}
      }
      if (! Qweight){
      	//t7.start(); // takes 11% of the time
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
	if (Vcmax>0){
	  int iq=0;
	  // Vertices, which are needed to express screened interaction with the same time difference as the unscreened part.
	  for (int ii=0; ii<Nv; ii++){ 
	    if (changed_Vrtx[ii]){
	      int id = vindx(ii)[0];
	      int  i = vindx(ii)[1];
	      int vtyp = Vtype(id,i);
	      if (vtyp==0) cout<<"ERROR changed_Vrtx is wrong, as type==0!"<<endl;
	      double t1 = times_trial(2*i);     // t1 is old time, because for equal time case t[2i+1] was actually t[2i]
	      double t2 = times_trial(2*i+1);   // t2 is the new time, because now we want t[2i+1] to be different.
	      // For now we will neglect possibility of type-II vertex. Should correct later!
	      int i_m = i_diagsG(id,2*i+1);
	      double ti = times_trial( et(i_m) );
	      int i_p = diagsG(id,2*i+1);
	      double to = times_trial( et(i_p) );
	      double ki = norm(mom_g(Gindx(id,i_m)));
	      double ko = norm(mom_g(Gindx(id,2*i+1)));
	      Vrtx_trial(ii) = MoveVertex(t2, t1, ti, to, ki, ko, Gk);
	      double aQ = norm(mom_v(ii));
	      double Vq = 8*pi/(aQ*aQ+lmbda);
	      intpar pq = Interp(aQ,qx);
	      intpar pt = Interp(fabs(t1-t2), tx);
	      int Nc = abs(vtyp);
	      double ct=0;
	      double lmbda_k = 1.0;
	      for (int k=0; k<Nc; k++){
		ct += binomial(Nc,k) * lmbda_k * bPqt[Nc-k-1](pq,pt);
		lmbda_k *= (lmbda/(8*pi));
	      }
	      long double Vn = ipower(Vq, Nc+1);
	      v_trial(iq) = Vn * ct * Vrtx_trial(ii);
	      v_trial(iq) += Vn * lmbda_k *(1-bubble_only);
	      if (vtyp<0){
		// we also add the single-particle counter term in addition to two-particle counter term.
		int ii_g = single_counter_index(ii);
		long double g_kq = changed_G[ii_g] ? g_trial(ip_ind(ii_g)) : G_current(ii_g);
#ifdef CNTR_VN
		if (g_kq!=0) v_trial(iq) += Vn*lmbda_spct_cutoffk(-vtyp-1)/g_kq;
#else		
		if (g_kq!=0) v_trial(iq) += exp(-aQ*2/kF)*lmbda_spct_cutoffk(-vtyp-1)/g_kq;
#endif		
	      }
	      iq_ind(ii)=iq;
	      iq++;
	    }
	  }
	}
	//t7.stop();
	//t8.start(); // takes 8% of the time
	PQ_new=0;   // recalculating PQ, taking into account one change of time.
	BKdata.PQg_new_Initialize();
	for (int id=0; id<Ndiags; id++){
	  long double PQd = 1.0;
	  for (int i=0; i<2*Norder; i++){
	    int ii = Gindx(id,i);
	    PQd *= ( changed_G[ii] ? g_trial(ip_ind(ii)) : G_current(ii) );
	  }
	  for (int i=1; i<Norder; i++){
	    int ii = Vindx(id,i);
	    PQd *= ( changed_Vrtx[ii] ? v_trial(iq_ind(ii)) : V_current(ii) );
	  }
	  PQd *= diagSign(id);
	  PQ_new += PQd;
	  BKdata.PQg_new_Add(id,PQd);
	}
	//t8.stop();
	//if (!Qweight)
	//  Check_trial_T(3, log, itt, Nv, PQ_new, PQ, lmbda, times_trial, times_to_change, Gk, G_current, V_current,
	//		g_trial, ip_ind, v_trial, iq_ind, changed_G, changed_V,
	//		diagsG, Gindx, Vindx, mom_g, mom_v, diagSign, Vtype, bPqt, qx, tx, vindx, Vrtx_current);
	//if (std::isnan(PQ_new)){
	//  log<<" itt="<<itt<<" PQ_new="<<PQ_new << " PQ="<<PQ << " Qweight=" << Qweight << " time "<< endl;
	//}
      }else{
	PQ_new = V0norm * mweight(amomentm, momentum);
      }
      double ratio = PQ_new/PQ;
      bool accept = fabs(ratio) > 1-drand() && PQ_new!=0;
      if ((itt+1)%Ncout==0 && mpi.rank==mpi.master)
	_PrintInfo_(itt+1, Qweight, icase, amomentm(1), amomentm(Nloops-1), times(itime), PQ_new, PQ, dk_hist, Nweight/(Nmeassure+0.0),log);
      if (accept){
	//t9.start(); // takes 1.7% of time
	Nacc_t +=1 ;
	times = times_trial;  // the new times are now accepted. We change the independent time variable, or two internal times.
	if (!Qweight){
	  for (int ii=0; ii<Ng; ++ii)
	    if (changed_G[ii])
	      G_current(ii) = g_trial(ip_ind(ii)); // The green's functions in the loop have been recalculated, and are now stored.
	  if (Vcmax>0){
	    for (int ii=0; ii<Nv; ii++){
	      if (changed_Vrtx[ii]){
		Vrtx_current(ii) = Vrtx_trial(ii);
		V_current(ii) = v_trial(iq_ind(ii));
	      }
	    }
	  }
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
	  if (Vcmax>0){
	    for (int ii=0; ii<Nv; ii++){
	      if (changed_Vrtx[ii]){
		int id = vindx(ii)[0];
		int  i = vindx(ii)[1];
		int vtyp = Vtype(id,i);
		if (vtyp==0) cout<<"ERROR changed_Vrtx is wrong, as type==0!"<<endl;
		double t1 = times(2*i);     // t1 is old time, because for equal time case t[2i+1] was actually t[2i]
		double t2 = times(2*i+1);   // t2 is the new time, because now we want t[2i+1] to be different.
		// For now we will neglect possibility of type-II vertex. Should correct later!
		int i_m = i_diagsG(id,2*i+1);
		double ti = times( et(i_m) );
		int i_p = diagsG(id,2*i+1);
		double to = times( et(i_p) );
		double ki = norm(mom_g(Gindx(id,i_m)));
		double ko = norm(mom_g(Gindx(id,2*i+1)));
		Vrtx_current(ii) = MoveVertex(t2, t1, ti, to, ki, ko, Gk);
		double aQ = norm(mom_v(ii));
		double Vq = 8*pi/(aQ*aQ+lmbda);
		intpar pq = Interp(aQ,qx);
		intpar pt = Interp(fabs(t1-t2), tx);
		int Nc = abs(vtyp);
		double ct=0;
		double lmbda_k = 1.0;
		for (int k=0; k<Nc; k++){
		  ct += binomial(Nc,k) * lmbda_k * bPqt[Nc-k-1](pq,pt);
		  lmbda_k *= (lmbda/(8*pi));
		}
		long double Vn = ipower(Vq, Nc+1);
		V_current(ii) = Vn * ct * Vrtx_current(ii);
		V_current(ii) += Vn * lmbda_k *(1-bubble_only);
		if (vtyp<0){
		  // we also add the single-particle counter term in addition to two-particle counter term.
		  int ii_g = single_counter_index(ii);
		  long double g_kq = G_current(ii_g);
#ifdef CNTR_VN		  
		  if (g_kq!=0) V_current(ii) += Vn*lmbda_spct_cutoffk(-vtyp-1)/g_kq;
#else		  
		  if (g_kq!=0) V_current(ii) += exp(-aQ*2/kF)*lmbda_spct_cutoffk(-vtyp-1)/g_kq;
#endif		  
		}
	      }
	    }
	  }
	}
	//t9.stop();
	PQ = PQ_new;
	if (!Qweight) BKdata.Set_PQg_new_to_PQg();
	//t10.start(); // takes 0.5% of the time
	if (itime==0 && !Q0w0) Pln.cmp_single(2*(times(0)-times(1))/beta-1., pl_t, BKdata.Nlt);  // update Legendre Polynomials
	//t10.stop();
	//if (!Qweight) Check(2, log, itt, Nv, PQ, lmbda, times, times_to_change, Gk, G_current, V_current, diagsG, Gindx, Vindx, mom_g, mom_v, diagSign, Vtype, bPqt, qx, tx, vindx, Vrtx_current);
      }
      //log<<itt<<" G_current="<<G_current<<" V_current="<<V_current<<endl;
    }else{  // normalization diagram step
      Nall_w += 1;
      //t11.start(); // takes 0.3% of the time
      if (Qweight){ // trying to jump back into physical hilbert space
	PQ_new=0;
	BKdata.PQg_new_Initialize();
	for (int id=0; id<Ndiags; id++){
	  long double PQd = 1.0;
	  for (int i=0; i<2*Norder; i++) PQd *= G_current(Gindx(id,i)); // we have all propagators precomputed, hence it is easy to compute the diagrams.
	  for (int i=1; i<Norder; i++) PQd *= V_current(Vindx(id,i));
	  PQd *= diagSign(id);
	  PQ_new += PQd;
	  BKdata.PQg_new_Add(id,PQd);
	}
      }else{        // or if we are in the physical Hilbert space, we try to jump into the normalization diagram
	PQ_new = V0norm * mweight(amomentm, momentum);
      }
      //if (std::isnan(PQ_new)){
      //  log<<" itt="<<itt<<" PQ_new="<<PQ_new << " PQ="<<PQ << " Qweight=" << Qweight << " meassuring" << endl;
      //}
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
      //t11.stop();
    }
    
    // Measuring
    if (itt>=Nwarm && itt%tmeassure==0){
      t_mes.start(); // takes 1% time
      Nmeassure += 1;
      double Qa = amomentm(0);
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
	Pbin(it,iq) += sp;
	BKdata.Meassure(itt,tmeassure,PQ,sp,pl_Q,pl_t,momentum(0),amomentm(0),mom_g,Gindx);
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
      if ( itt>2e5/mpi.size && itt%(static_cast<int>(100000/mpi.size+1000)*tmeassure) == 0){
	double last_occurence = (Nweight-Wchanged)/(Nmeassure-Nchanged); // occurence from the last time we changed it
#ifdef _MPI
	double total_occurence=0;
	MPI_Allreduce(&last_occurence, &total_occurence, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	last_occurence = total_occurence/mpi.size;
#endif	
	if ( last_occurence > 0.15){ // decrease by two
	  V0norm /= 2.;
	  Pnorm /= 2.;
	  Nchanged = Nmeassure;
	  Wchanged = Nweight;
	  if (mpi.rank==mpi.master) log<<std::left<<(itt/1.0e6)<<"M occurence="<<std::setw(8)<<last_occurence<<" V0 reduced to "<<V0norm<<std::endl;
	}
	if ( last_occurence < 0.03){ // increase by two
	  V0norm *= 2.;
	  Pnorm *= 2.;
	  Nchanged = Nmeassure;
	  Wchanged = Nweight;
	  if (mpi.rank==mpi.master) log<<std::left<<(itt/1.0e6)<<"M occurence="<<std::setw(8)<<last_occurence<<" V0 increased to "<<V0norm<<std::endl;
	}
      }
      if (itt%(500000*tmeassure) == 0){
	if (Nmeassure-Nweight>1000){
#ifdef _MPI	  
	  MPI_Allreduce(MPI_IN_PLACE, mweight.K_hist.data(), mweight.K_hist.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
	  // We can get overflow during MPI for a long run. We should use a constant for normalization.
	  // We could here normalize K_hist (with knorm), and than when we measure, we would
	  // add instead of adding unity, we would add 1./knorm
	  dk_hist *= mweight.Normalize_K_histogram();
	  if (dk_hist < 1e-16) dk_hist = 1.0;
	  mweight.Recompute(mpi.rank==mpi.master);
	  // If we are in a meassuring space, we have to update now the value of the meassuring diagram as the weight has changed.
	  if (Qweight) PQ = V0norm * mweight(amomentm, momentum);
	}
      }
      t_mes.stop();
    }
  }
  t_all.stop();
  BKdata.C_Pln *= 1.0/Nmeassure;
  Pbin         *= 1.0/Nmeassure;
  Pnorm        *= 1.0/Nmeassure;
  double occurence = static_cast<double>(Nweight)/static_cast<double>(Nmeassure);

  if (GetHistogram){
    K_Ghist *= 1.0/Nmeassure;
    K_Vhist *= 1.0/Nmeassure;
  }
  
#ifdef _MPI  
  double dat[2] = {Pnorm, occurence};
  if (mpi.rank==mpi.master){
    MPI_Reduce(MPI_IN_PLACE, BKdata.C_Pln.data(), BKdata.C_Pln.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE,  Pbin.data(),  Pbin.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, dat, 2, MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, mweight.K_hist.data(), mweight.K_hist.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    if (GetHistogram){
      MPI_Reduce(MPI_IN_PLACE, K_Ghist.data(), K_Ghist.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
      MPI_Reduce(MPI_IN_PLACE, K_Vhist.data(), K_Vhist.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    }
  }else{
    MPI_Reduce(BKdata.C_Pln.data(), BKdata.C_Pln.data(), BKdata.C_Pln.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    MPI_Reduce( Pbin.data(),  Pbin.data(),  Pbin.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    MPI_Reduce(dat, dat, 2, MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    MPI_Reduce(mweight.K_hist.data(), mweight.K_hist.data(), mweight.K_hist.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    if (GetHistogram){
      MPI_Reduce(K_Ghist.data(), K_Ghist.data(), K_Ghist.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
      MPI_Reduce(K_Vhist.data(), K_Vhist.data(), K_Vhist.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    }
  }
  if (mpi.rank==mpi.master){
    BKdata.C_Pln *= 1./mpi.size;
    Pbin         *= 1./mpi.size;
    Pnorm        = dat[0]/mpi.size;
    occurence    = dat[1]/mpi.size;
    //
    mweight.K_hist *= 1./mpi.size;
  }
#endif

  if (mpi.rank==mpi.master){
    log<<"total acceptance rate="<<(Nacc_k+Nacc_q+Nacc_t+Nacc_w)/(p.Nitt+0.0)<<" "<<" k-acceptance="<<(Nacc_k/(Nall_k+0.0))<<" q-acceptance="<<(Nacc_q/(Nall_q+0.0))<<" t-acceptanece="<<(Nacc_t/(Nall_t+0.0))<<" w-acceptanece="<<(Nacc_w/(Nall_w+0.0))<<std::endl;
    log<<" k-trials="<<(Nall_k/(p.Nitt+0.0))<<" q-trials="<<(Nall_q/(p.Nitt+0.0))<<" t-trials="<<(Nall_t/(p.Nitt+0.0))<<" w-trials="<<(Nall_w/(p.Nitt+0.0))<<std::endl;
    //cout<<"t1="<<t1.elapsed()<<" t2="<<t2.elapsed()<<" t3="<<t3.elapsed()<<" t4="<<t4.elapsed()<<" t5="<<t5.elapsed()<<" t6="<<t6.elapsed()<<" t7="<<t7.elapsed()<<" t8="<<t8.elapsed()<<" t9="<<t9.elapsed()<<" t10="<<t10.elapsed()<<" t11="<<t11.elapsed()<<std::endl;
    log<<"total time="<<t_all.elapsed()<<" measuring time="<<t_mes.elapsed()<<std::endl;

    double dOmega = (!Q0w0) ? 1.0/(4*pi) : 1.0;
    BKdata.C_Pln *= dOmega * (fabs(V0norm)/Pnorm); // 4*pi because we must divide by 1/(4*pi*q**2), and 1/q**2 was already done in sampling
    Pbin  *= (dOmega * fabs(V0norm)/Pnorm);
    log<<"  meassuring diagram occurence frequency="<<occurence<<" and its norm Pnorm="<<Pnorm<<std::endl;
    
    // Proper normalization of the resulting Monte Carlo data
    double dOmega2 = (!Q0w0) ? (4*pi*cutoffq*cutoffq*cutoffq/3) : 1.0;
    double norm = ipower( beta/((2*pi)*(2*pi)*(2*pi)), Norder) * dOmega2;
    BKdata.C_Pln *= norm;

    // Later we will use Nthbin/2.0*Nkbin/2.0  normalization in the Kernel to transform from binned data to function values.
    BKdata.Normalize(beta, cutoffq);

    Pbin *= norm;
    // Q0w0 : Paver <= sum(Pbin)/beta;
    
    double dq_binning = cutoffq/Nq;
    double dt_binning = beta/Nt;
    if (!Q0w0)
      Pbin *= 1.0/(dq_binning * dt_binning);
    
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

template<typename GK, typename Data>
void sample_static_fastC_combined(Data& BKdata, bl::Array<double,2>& Pbin, std::ostream& log,
				  double lmbda, const std::vector<double>& lmbda_spct, const GK& Gk, const params& p,
				  const bl::Array<unsigned short,2>& diagsG,
				  const bl::Array<float,1>& diagSign,
				  const std::vector<std::vector<std::vector<int> > >& Loop_index, const std::vector<std::vector<std::vector<int> > >& Loop_type,
				  bl::Array<char,2>& Vtype, my_mpi& mpi)
{
  typedef double real;
  log.precision(12);
  bool Q0w0 = BKdata.Q0w0;
  double Q_external=BKdata.Q_external;
  if (!Q0w0){
    if ( BKdata.Nlt!=p.Nlt || BKdata.Nlq!=p.Nlq){
      log<<"ERROR : Dimensions of C_Pln is wrong : either "<<BKdata.Nlt<<" != "<<p.Nlt<<" or "<<BKdata.Nlq<<" != "<<p.Nlq<<std::endl;
      exit(1);
    }
    if (Pbin.extent(0)!=p.Nt || Pbin.extent(1)!=p.Nq) Pbin.resize(p.Nt,p.Nq); Pbin=0.0;
  }else{
    if (Pbin.extent(0)!=p.Nt) Pbin.resize(p.Nt,1); Pbin=0.0;
  }
  
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
  
  //CounterCoulombV Vqc(lmbda);

  // Set up which times need change when counter term is dynamic
  vector<int> times_2_change;
  std::set<int> times_to_change;
  int Vcmax=0;
  {
    //std::set<int> counter_types;
    times_to_change.insert(0);
    int i_start = Q0w0 ? 1 : 0;   // CHANGE
    for (int i=i_start; i<Norder; i++) times_to_change.insert(2*i);
    // default vertices to change time will be : [0,2,...,2*(Norder-1)]
    times_2_change.resize(times_to_change.size());
    int j=0;
    for (auto it=times_to_change.begin(); it!=times_to_change.end(); ++it, ++j) times_2_change[j] = (*it);
    log<<"times_to_change : ";
    for (auto it=times_to_change.begin(); it!=times_to_change.end(); ++it) log << (*it) <<",  ";
    log<<std::endl;
    //log<<"counter_types : ";
    //for (auto it=counter_types.begin(); it!=counter_types.end(); ++it) log << (*it) <<",  ";
    log<<std::endl;
    log << "times_2_change: "<<endl;
    for (int i=0; i<times_2_change.size(); i++) log << times_2_change[i] << ",";
    log << endl;
  }
  // Reads numerically evaluated Bubble diagram.
  
  int tmeassure=p.tmeassure, Ncout=p.Ncout, Nwarm=p.Nwarm;
  double V0norm = p.V0norm, dkF = p.dkF, kF=p.kF;// V0exp= p.V0exp;
  double cutoffk = p.cutoffk, dRk = p.dRk;
  double beta = p.beta;
  double cutoffq = p.cutoffq;
  int Nt = p.Nt, Nq = p.Nq;


  // Everything for Legendre Polynomials in time and momentum
  LegendrePl Pln;
  bl::Array<double,1> pl_Q;  // Will contain legendre Pl(2*q/qmax-1)
  bl::Array<double,1> pl_t;  // Will contain legendre Pl(2*t//beta-1)
  if (!Q0w0){
    Pln.resize(std::max(BKdata.Nlq,BKdata.Nlt));
    pl_Q.resize(BKdata.Nlq+1);  // Will contain legendre Pl(2*q/qmax-1)
    pl_t.resize(BKdata.Nlt+1);  // Will contain legendre Pl(2*t//beta-1)
  }
  BKdata.C_Pln = 0.0;            // cummulative result for legendre expansion
  
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
  findUnique_noindx_spct(Gindx, Vindx, gindx, vindx, N0v, diagsG, Vtype, Loop_index, Loop_type, 0, false, debug, log);

  // Once more we go through single-particle counter terms and create index single_counter_index
  bl::Array<int,1> single_counter_index;
  Find_Single_Particle_Counter_G_index(single_counter_index, Ndiags, Norder, vindx.extent(0), diagsG, Vtype, Gindx, Vindx);
  
  // This is related to Baym-Kadanoff approach. For efficiency, we want to find which diagram have common values of momenta at the right hand side of the diagram.
  BKdata.FindGroups(diagsG, Loop_index, Loop_type, log);
  
  if (Loop_index.size()!=Ndiags){ log<<"Wrong size of Loop_index "<<Loop_index.size()<<" instead of "<<Ndiags<<std::endl; exit(1);}
  int Nloops = Loop_index[0].size();
  if (Nloops != Norder+1){ log<<"Expecting Nloops==Norder+1 in Loops_index, but got Nloops="<<Nloops<<" Norder="<<Norder<<std::endl; exit(1);}
  int Ng = gindx.extent(0);
  int Nv = vindx.extent(0);
  int _Nv_ = std::max(Nv,1); // can not assign 0 to empty file

  bl::Array<bl::Array<sint,1>,1> loop_Gkind, loop_Vqind;
  bl::Array<bl::Array<char,1>,1> loop_Gksgn, loop_Vqsgn;
  int Ngp, Nvp;
  
  // Find index-tables to all momentum loops, so that we do not need to loop over all diagrams, but rather only over all differet propagators corresponding to certain loop.
  // Here we also take into account that single-particle counter terms are sometimes added.
  Get_GVind(loop_Gkind, loop_Gksgn, loop_Vqind, loop_Vqsgn, Ngp, Nvp, Loop_index, Loop_type, Gindx, Vindx, single_counter_index, lmbda_spct, Vtype, false, log);
  int Ngp_tr = GetMax_NGp_trial(i_diagsG, Gindx, Ng, Ngp);

  // DEBUG
  if (mpi.rank==mpi.master){
    log<<"lmbda_spct=";
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
    log<<"loop_Vqind:"<<std::endl;
    for (int iloop=0; iloop<Nloops; ++iloop){
      for (int iq=0; iq<loop_Vqind(iloop).extent(0); ++iq){ // Now do the same for the interaction
	int ii   = loop_Vqind(iloop)(iq);
	int isgn = loop_Vqsgn(iloop)(iq);
	
	int id = vindx(ii)[0];
	int  i = vindx(ii)[1];
	int vtyp = Vtype(id,i);
	
	log<<"iloop="<<iloop<<" iq="<<iq<<" ii_v="<<ii<<" isgn="<<isgn<<" which comes from id="<< id <<" and pair =("<<2*i<<","<<2*i+1<<") of type="<<vtyp<<std::endl;
      }
    }
  }
  // END DEBUG
  
  Reweight_time rw(beta,p.lmbdat);
  int Nbin = 513;
  //int Nbin = 129;
  bl::Array<double,1> lmbda_spct_cutoffk(lmbda_spct.size());
  // Instead of using : lmbda_spct, which is q-independent, we will introduce "artificial" q-dependence, which gives identical integral.
  // When lmbda_spct is q-independent, we should normalize it by  lmbda_spct * 1/Integrate[d^3 1/(2*pi)^3].
  // The "artificial" dependence we choose is lmbda_spct * (Vq)^2, in which case the normalization constant is
  //     lmbda_spct * (Vq)^2 / Integrate[d^3q Vq^2/(2*pi)^3 ]
  //  which is
  //     lmbda_spct * (Vq)^2 / ( 2/sqrt(lmbda) *atan(ck/sqrt(lmbda)) - 2*ck/(ck^2+lmbda) )
  std::vector<double> nrm = Nrm(lmbda, cutoffk, lmbda_spct.size()+2);
  for (int i=0; i<lmbda_spct.size(); ++i){
#ifdef CNTR_VN
    // [Vq^2, Vq^3, Vq^4,..]
    lmbda_spct_cutoffk(i) = lmbda_spct[i]*nrm[i+2];
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
  {
    double Qa = Q0w0 ? Q_external : p.kF*drand(); // absolute value of external momentum
    momentum(0) = 0.0,0.0,Qa; // external momentum Q
    amomentm(0) = Qa;
    for (int ik=1; ik<Nloops; ik++){
      double th = pi*(1-drand()), phi = 2*pi*(1-drand());
      momentum(ik) = kF*sin(th)*cos(phi), kF*sin(th)*sin(phi), kF*cos(th);
      amomentm(ik) = kF;
    }
  }
  log.precision(12);
  log<<"Starting times="<<times<<" and starting momenta="<<momentum<<std::endl;

  bl::Array<double,1> G_current(Ng);
  bl::Array<long double,1> V_current(_Nv_);
  // This is momentum for all diagrams and all propagators
  bl::Array<bl::TinyVector<real,3>,1> mom_g(Ng), mom_v(Nv);
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
      for (int i=0; i<2*Norder; i++){
	//int i_final = diagsG(id,i);
	int i_previ = i_diagsG(id,i);
	bl::TinyVector<double,3> k_in = mom_G(id,i_previ);
	bl::TinyVector<double,3> k_out = mom_G(id,i);
	bl::TinyVector<double,3> q = mom_V(id,i/2)*(1.-2*(i%2));
	if (fabs(norm(k_in-k_out)-norm(q))>1e-6){
	  log<<"ERROR: diagram "<<id<<" vertex i="<<i<<" k_in="<<k_in<<" k_out="<<k_out<<" q="<<q<<std::endl;
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
    for (int ii=0; ii<Nv; ++ii){
      int id = vindx(ii)[0];
      int  i = vindx(ii)[1];
      double aQ = norm(mom_V(id,i));
      int vtyp = Vtype(id,i);
      double Vq = 8*pi/(aQ*aQ+lmbda);
      if (vtyp==0){
	V_current(ii) = Vq;
      }else{
	int Nc = abs(vtyp);
	long double Vn = ipower(Vq, Nc+1);
	V_current(ii) = Vn * ipower(lmbda/(8*pi), Nc);
	if (vtyp < 0){
	  // we also add the single-particle counter term in addition to two-particle counter term.
	  int ii_g = single_counter_index(ii);
	  long double g_kq = G_current(ii_g);
#ifdef CNTR_VN
	  // vtype=-1, Nc=1 : lmbda[0] -> Vn = Vq^2
	  // vtype=-2, Nc=2 : lmbda[1] -> Vn = Vq^3
	  if (g_kq!=0) V_current(ii) += Vn*lmbda_spct_cutoffk(-vtyp-1)/g_kq;
#else	  
	  if (g_kq!=0) V_current(ii) += exp(-aQ*2/kF)*lmbda_spct_cutoffk(-vtyp-1)/g_kq;
#endif	  
	}
      }
    }
    PQ=0;
    BKdata.PQg_Initialize();
    for (int id=0; id<Ndiags; id++){
      long double PQd = 1.0;
      for (int i=0; i<2*Norder; i++) PQd *= G_current(Gindx(id,i));
      for (int i=1; i<Norder; i++) PQd *= V_current(Vindx(id,i));
      PQd *= diagSign(id);
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
    Pln.cmp_single(2*amomentm(0)/cutoffq-1., pl_Q, BKdata.Nlq);          // Legendre polynomials at this Q
    Pln.cmp_single(2*(times(0)-times(1))/beta-1., pl_t, BKdata.Nlt);     // Legendre polynomials at this time
  }
  
  bl::Array<double,1> times_trial(2*Norder);
  int _Nvp_ = std::max(Nvp,1); // can not assign 0 to empty file
  bl::Array<bl::TinyVector<real,3>,1> tmom_g(Ngp), tmom_v(_Nvp_);
  bl::Array<sint,1> ip_ind(Ng), iq_ind(Nv);
  bl::Array<double,1> g_trial(Ngp_tr);
  bl::Array<long double,1> v_trial(_Nvp_);
  BitArray changed_G(Ng), changed_V(Nv);
  int Qweight = 0;      // if Qweight==1 then we have the normalization diagram
  double Nweight = 0;   // how many times did we see measuring diagram.
  int NlowT = 0;        // how many times do we see external time < 0.05*beta
  int Nmeassure = 0;    // how many times did we measure quantities.
  int Nchanged  = 0;    // when was V0norm last changed?
  double Wchanged = 0;  // what was Nweight when we last time changed V0norm?
  int Nacc_k = 0, Nacc_q = 0, Nacc_w = 0, Nacc_t = 0;
  int Nall_k = 0, Nall_q = 0, Nall_w = 0, Nall_t = 0;
  if (mpi.rank==mpi.master) PrintInfo_(0, Qweight, amomentm(0), amomentm(1), times(0)-times(1), PQ, PQ, 0, 0, log);
  //Check(0, log, -1, Nv, PQ, lmbda, times, times_to_change, Gk, G_current, V_current, diagsG, Gindx, Vindx, mom_g, mom_v, diagSign, Vtype, bPqt, qx, tx, vindx, Vrtx_current);

  t_all.start();
  for (int itt=0; itt<p.Nitt; itt++){
    double PQ_new=0;
    int icase = tBisect(drand(), Prs); // which steps should we attempt?
    if (icase==0){ // changing momentum Q or k-point
      int iloop = (!Q0w0) ? static_cast<int>(Nloops*drand()) : 1+static_cast<int>((Nloops-1)*drand());
      bl::TinyVector<double,3> K_new; double Ka_new; double trial_ratio=1;
      bool accept=false;
      if (iloop==0){ // this is the loop for the external momentum Q
	Nall_q += 1;
	accept = Find_new_Q_point(K_new, Ka_new, trial_ratio, momentum(iloop), amomentm(iloop), cutoffq, drand);
      }else{         // any other momentum point
	Nall_k += 1;
	accept = Find_new_k_point(K_new, Ka_new, trial_ratio, momentum(iloop), amomentm(iloop), cutoffk, drand, drand()<p.Qring, dkF, dRk);
      }
      if (accept){
	if (!Qweight){
	  bl::TinyVector<double,3> dK = K_new - momentum(iloop); // how much momentum will be changed.
	  //t1.start(); // takes 40% of the time
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
	  //t2.start(); // takes 1.5% of the time
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
	    if (vtyp==0){
	      v_trial(iq) = Vq;
	    }else{
	      int Nc = abs(vtyp);
	      long double Vn = ipower(Vq, Nc+1);
	      //long double Vrtx = changed_Vrtx[ii] ? Vrtx_trial(ii) : Vrtx_current(ii);
	      //v_trial(iq) = (Vn * ct) * Vrtx ;
	      v_trial(iq) = Vn * ipower(lmbda/(8*pi),Nc);
	      if (vtyp<0){
		// we also add the single-particle counter term in addition to two-particle counter term.
		int ii_g = single_counter_index(ii);
		long double g_kq = changed_G[ii_g] ? g_trial(ip_ind(ii_g)) : G_current(ii_g);
#ifdef CNTR_VN		
		if (g_kq!=0) v_trial(iq) += Vn*lmbda_spct_cutoffk(-vtyp-1)/g_kq;
#else		
		if (g_kq!=0) v_trial(iq) += exp(-aQ*2/kF)*lmbda_spct_cutoffk(-vtyp-1)/g_kq;
#endif		
	      }
	    }
	    iq_ind(ii)=iq;
	  }
	  //t2.stop();
	  // we computed the polarization diagram
	  //t3.start();// takes 22% of the time
	  PQ_new=0;
	  BKdata.PQg_new_Initialize();
	  for (int id=0; id<Ndiags; id++){
	    long double PQd = 1.0;
	    for (int i=0; i<2*Norder; i++){
	      int ii = Gindx(id,i);
	      PQd *= ( changed_G[ii] ? g_trial(ip_ind(ii)) : G_current(ii) );
	    }
	    for (int i=1; i<Norder; i++){
	      int ii = Vindx(id,i);
	      PQd *= ( changed_V[ii] ? v_trial(iq_ind(ii)) : V_current(ii) );
	    }
	    PQd *=  diagSign(id);
	    PQ_new += PQd;
	    BKdata.PQg_new_Add(id,PQd);
	  }
	  //t3.stop();
	}else{
	  double Ka_old = amomentm(iloop);
	  amomentm(iloop) = Ka_new;
	  bl::TinyVector<double,3> K_old = momentum(iloop);
	  momentum(iloop) = K_new;
	  PQ_new = V0norm * mweight(amomentm, momentum);
	  amomentm(iloop) = Ka_old;
	  momentum(iloop) = K_old;
	}
	double ratio = PQ_new*trial_ratio / PQ;
	accept = fabs(ratio) > 1-drand() && PQ_new!=0;
      }
      //if ((itt+1)%Ncout==0 && mpi.rank==mpi.master) _PrintInfo_(itt+1, Qweight, icase, amomentm(1), amomentm(Nloops-1), times(0), PQ_new, PQ, NlowT/(Nmeassure+0.0), Nweight/(Nmeassure+0.0), log);
      if ((itt+1)%Ncout==0 && mpi.rank==mpi.master) _PrintInfo_(itt+1, Qweight, icase, amomentm(1), amomentm(Nloops-1), times(0), PQ_new, PQ, dk_hist, Nweight/(Nmeassure+0.0), log);
      
      if (accept){
	if (iloop==0) Nacc_q += 1; else Nacc_k += 1;
	if (!Qweight){
	  momentum(iloop) = K_new;  // this momentum was changed
	  amomentm(iloop) = Ka_new;
	  for (int ip=0; ip<loop_Gkind(iloop).extent(0); ++ip){
	    int ii   = loop_Gkind(iloop)(ip); 
	    mom_g(ii) = tmom_g(ip);
	  }
	  for (int iq=0; iq<loop_Vqind(iloop).extent(0); ++iq){ // Now do the same for the interaction
	    int ii   = loop_Vqind(iloop)(iq);
	    mom_v(ii) = tmom_v(iq);
	  }
	  for (int ip=0; ip<loop_Gkind(iloop).extent(0); ++ip){ // all G-propagators, which are changed when momentum in certain loop is changed.
	    int ii   = loop_Gkind(iloop)(ip);  // index of G-propagators, which we are now changing
	    G_current(ii)=g_trial(ip);
	  }
	  for (int iq=0; iq<loop_Vqind(iloop).extent(0); ++iq){ // Now do the same for the interaction
	    int ii   = loop_Vqind(iloop)(iq);
	    V_current(ii) = v_trial(iq);
	  }
	}else{
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
	  //t2.start(); // takes 1.5% of the time
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
	    if (vtyp==0){
	      V_current(ii) = Vq;
	    }else{
	      int Nc = abs(vtyp);
	      long double Vn = ipower(Vq, Nc+1);
	      V_current(ii) = Vn * ipower(lmbda/(8*pi),Nc);
	      if (vtyp<0){
		// we also add the single-particle counter term in addition to two-particle counter term.
		int ii_g = single_counter_index(ii);
		long double g_kq = G_current(ii_g);
#ifdef CNTR_VN		
		if (g_kq!=0) V_current(ii) += Vn*lmbda_spct_cutoffk(-vtyp-1)/g_kq;
#else		
		if (g_kq!=0) V_current(ii) += exp(-aQ*2/kF)*lmbda_spct_cutoffk(-vtyp-1)/g_kq;
#endif		
	      }
	    }
	  }
	  //t5.stop();
	}
	PQ = PQ_new;
	if (!Qweight) BKdata.Set_PQg_new_to_PQg();
	//t6.start(); // takes 0.5% time
	if (iloop==0 && !Q0w0) Pln.cmp_single(2*amomentm(iloop)/cutoffq-1., pl_Q, BKdata.Nlq);  // update Legendre Polynomials
	//t6.stop();
	//if (!Qweight) Check(1, log, itt, Nv, PQ, lmbda, times, times_to_change, Gk, G_current, V_current, diagsG, Gindx, Vindx, mom_g, mom_v, diagSign, Vtype, bPqt, qx, tx, vindx, Vrtx_current);
      }
    } else if (icase==1){  // changing one of the time variables
      Nall_t += 1;
      int itime = times_2_change[ static_cast<int>( times_2_change.size()*drand() ) ];
      changed_G=0;              // which propagators are being changed?
      //changed_Vrtx = 0;
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
      	//t7.start(); // takes 11% of the time
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
	//t7.stop();
	//t8.start(); // takes 8% of the time
	PQ_new=0;   // recalculating PQ, taking into account one change of time.
	BKdata.PQg_new_Initialize();
	for (int id=0; id<Ndiags; id++){
	  long double PQd = 1.0;
	  for (int i=0; i<2*Norder; i++){
	    int ii = Gindx(id,i);
	    PQd *= ( changed_G[ii] ? g_trial(ip_ind(ii)) : G_current(ii) );
	  }
	  for (int i=1; i<Norder; i++){
	    int ii = Vindx(id,i);
	    //PQd *= ( changed_Vrtx[ii] ? v_trial(iq_ind(ii)) : V_current(ii) );
	    PQd *= V_current(ii);
	  }
	  PQd *= diagSign(id);
	  PQ_new += PQd;
	  BKdata.PQg_new_Add(id,PQd);
	}
	//t8.stop();
      }else{
	PQ_new = V0norm * mweight(amomentm, momentum);
      }
      double ratio = PQ_new/PQ;
      bool accept = fabs(ratio) > 1-drand() && PQ_new!=0;
      if ((itt+1)%Ncout==0 && mpi.rank==mpi.master)
	_PrintInfo_(itt+1, Qweight, icase, amomentm(1), amomentm(Nloops-1), times(itime), PQ_new, PQ, dk_hist, Nweight/(Nmeassure+0.0),log);
      if (accept){
	//t9.start(); // takes 1.7% of time
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
	//t9.stop();
	PQ = PQ_new;
	if (!Qweight) BKdata.Set_PQg_new_to_PQg();
	//t10.start(); // takes 0.5% of the time
	if (itime==0 && !Q0w0) Pln.cmp_single(2*(times(0)-times(1))/beta-1., pl_t, BKdata.Nlt);  // update Legendre Polynomials
	//t10.stop();
      }
    }else{  // normalization diagram step
      Nall_w += 1;
      //t11.start(); // takes 0.3% of the time
      if (Qweight){ // trying to jump back into physical hilbert space
	PQ_new=0;
	BKdata.PQg_new_Initialize();
	for (int id=0; id<Ndiags; id++){
	  long double PQd = 1.0;
	  for (int i=0; i<2*Norder; i++) PQd *= G_current(Gindx(id,i)); // we have all propagators precomputed, hence it is easy to compute the diagrams.
	  for (int i=1; i<Norder; i++) PQd *= V_current(Vindx(id,i));
	  PQd *= diagSign(id);
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
      //t11.stop();
    }
    
    // Measuring
    if (itt>=Nwarm && itt%tmeassure==0){
      t_mes.start(); // takes 1% time
      Nmeassure += 1;
      double Qa = amomentm(0);
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
	Pbin(it,iq) += sp;
	BKdata.Meassure(itt,tmeassure,PQ,sp,pl_Q,pl_t,momentum(0),amomentm(0),mom_g,Gindx);
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
      if ( itt>2e5/mpi.size && itt%(static_cast<int>(100000/mpi.size+1000)*tmeassure) == 0){
	double last_occurence = (Nweight-Wchanged)/(Nmeassure-Nchanged); // occurence from the last time we changed it
#ifdef _MPI
	double total_occurence=0;
	MPI_Allreduce(&last_occurence, &total_occurence, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	last_occurence = total_occurence/mpi.size;
#endif	
	if ( last_occurence > 0.15){ // decrease by two
	  V0norm /= 2.;
	  Pnorm /= 2.;
	  Nchanged = Nmeassure;
	  Wchanged = Nweight;
	  if (mpi.rank==mpi.master) log<<std::left<<(itt/1.0e6)<<"M occurence="<<std::setw(8)<<last_occurence<<" V0 reduced to "<<V0norm<<std::endl;
	}
	if ( last_occurence < 0.03){ // increase by two
	  V0norm *= 2.;
	  Pnorm *= 2.;
	  Nchanged = Nmeassure;
	  Wchanged = Nweight;
	  if (mpi.rank==mpi.master) log<<std::left<<(itt/1.0e6)<<"M occurence="<<std::setw(8)<<last_occurence<<" V0 increased to "<<V0norm<<std::endl;
	}
      }
      if (itt%(500000*tmeassure) == 0){
	if (Nmeassure-Nweight>1000){
#ifdef _MPI	  
	  MPI_Allreduce(MPI_IN_PLACE, mweight.K_hist.data(), mweight.K_hist.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
	  // We can get overflow during MPI for a long run. We should use a constant for normalization.
	  // We could here normalize K_hist (with knorm), and than when we measure, we would
	  // add instead of adding unity, we would add 1./knorm
	  dk_hist *= mweight.Normalize_K_histogram();
	  if (dk_hist < 1e-16) dk_hist = 1.0;
	  mweight.Recompute(mpi.rank==mpi.master);
	  // If we are in a meassuring space, we have to update now the value of the meassuring diagram as the weight has changed.
	  if (Qweight) PQ = V0norm * mweight(amomentm, momentum);
	}
      }
      t_mes.stop();
    }
  }
  t_all.stop();
  BKdata.C_Pln *= 1.0/Nmeassure;
  Pbin         *= 1.0/Nmeassure;
  Pnorm        *= 1.0/Nmeassure;
  double occurence = static_cast<double>(Nweight)/static_cast<double>(Nmeassure);

  if (GetHistogram){
    K_Ghist *= 1.0/Nmeassure;
    K_Vhist *= 1.0/Nmeassure;
  }
  
#ifdef _MPI  
  double dat[2] = {Pnorm, occurence};
  if (mpi.rank==mpi.master){
    MPI_Reduce(MPI_IN_PLACE, BKdata.C_Pln.data(), BKdata.C_Pln.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE,  Pbin.data(),  Pbin.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, dat, 2, MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, mweight.K_hist.data(), mweight.K_hist.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    if (GetHistogram){
      MPI_Reduce(MPI_IN_PLACE, K_Ghist.data(), K_Ghist.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
      MPI_Reduce(MPI_IN_PLACE, K_Vhist.data(), K_Vhist.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    }
  }else{
    MPI_Reduce(BKdata.C_Pln.data(), BKdata.C_Pln.data(), BKdata.C_Pln.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    MPI_Reduce( Pbin.data(),  Pbin.data(),  Pbin.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    MPI_Reduce(dat, dat, 2, MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    MPI_Reduce(mweight.K_hist.data(), mweight.K_hist.data(), mweight.K_hist.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    if (GetHistogram){
      MPI_Reduce(K_Ghist.data(), K_Ghist.data(), K_Ghist.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
      MPI_Reduce(K_Vhist.data(), K_Vhist.data(), K_Vhist.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    }
  }
  if (mpi.rank==mpi.master){
    BKdata.C_Pln *= 1./mpi.size;
    Pbin         *= 1./mpi.size;
    Pnorm        = dat[0]/mpi.size;
    occurence    = dat[1]/mpi.size;
    //
    mweight.K_hist *= 1./mpi.size;
  }
#endif

  if (mpi.rank==mpi.master){
    log<<"total acceptance rate="<<(Nacc_k+Nacc_q+Nacc_t+Nacc_w)/(p.Nitt+0.0)<<" "<<" k-acceptance="<<(Nacc_k/(Nall_k+0.0))<<" q-acceptance="<<(Nacc_q/(Nall_q+0.0))<<" t-acceptanece="<<(Nacc_t/(Nall_t+0.0))<<" w-acceptanece="<<(Nacc_w/(Nall_w+0.0))<<std::endl;
    log<<" k-trials="<<(Nall_k/(p.Nitt+0.0))<<" q-trials="<<(Nall_q/(p.Nitt+0.0))<<" t-trials="<<(Nall_t/(p.Nitt+0.0))<<" w-trials="<<(Nall_w/(p.Nitt+0.0))<<std::endl;
    //cout<<"t1="<<t1.elapsed()<<" t2="<<t2.elapsed()<<" t3="<<t3.elapsed()<<" t4="<<t4.elapsed()<<" t5="<<t5.elapsed()<<" t6="<<t6.elapsed()<<" t7="<<t7.elapsed()<<" t8="<<t8.elapsed()<<" t9="<<t9.elapsed()<<" t10="<<t10.elapsed()<<" t11="<<t11.elapsed()<<std::endl;
    log<<"total time="<<t_all.elapsed()<<" measuring time="<<t_mes.elapsed()<<std::endl;

    double dOmega = (!Q0w0) ? 1.0/(4*pi) : 1.0;
    BKdata.C_Pln *= dOmega * (fabs(V0norm)/Pnorm); // 4*pi because we must divide by 1/(4*pi*q**2), and 1/q**2 was already done in sampling
    Pbin  *= (dOmega * fabs(V0norm)/Pnorm);
    log<<"  meassuring diagram occurence frequency="<<occurence<<" and its norm Pnorm="<<Pnorm<<std::endl;
    
    // Proper normalization of the resulting Monte Carlo data
    double dOmega2 = (!Q0w0) ? (4*pi*cutoffq*cutoffq*cutoffq/3) : 1.0;
    double norm = ipower( beta/((2*pi)*(2*pi)*(2*pi)), Norder) * dOmega2;
    BKdata.C_Pln *= norm;

    // Later we will use Nthbin/2.0*Nkbin/2.0  normalization in the Kernel to transform from binned data to function values.
    BKdata.Normalize(beta, cutoffq);

    Pbin *= norm;
    // Q0w0 : Paver <= sum(Pbin)/beta;
    
    double dq_binning = cutoffq/Nq;
    double dt_binning = beta/Nt;
    if (!Q0w0)
      Pbin *= 1.0/(dq_binning * dt_binning);
    
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

double Compute_V_Hugenholtz(const bl::Array<double,2>& V12, int nvh, const std::vector<int>& sign, bl::Array<double,1>& Vtree, ostream& log)
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

void CheckTrial(int itt, ostream& log, const bl::Array<unsigned short,2>& i_diagsG, const bl::Array<char,2>& Vtype,  const bl::Array<unsigned short,2>& Vindx, const bl::Array<bl::TinyVector<double,3>,1>& mom_v2,
		const bl::Array<bl::TinyVector<double,3>,1>& mom_g,  const bl::Array<unsigned short,2>& Gindx, int N0v, const BitArray& changed_V2, const BitArray& changed_G,
		const bl::Array<bl::TinyVector<double,3>,1>& tmom_v, const bl::Array<bl::TinyVector<double,3>,1>& tmom_g,
		const bl::Array<unsigned short,1>& iq_ind2, const bl::Array<unsigned short,1>& ip_ind)
{
  int Ndiags = i_diagsG.extent(0);
  int Norder = i_diagsG.extent(1)/2;
  for (int id=0; id<Ndiags; id++){
    for (int i=0; i<Norder; i++){
      int vtyp = Vtype(id,i);
      if (vtyp>=10){
	int ii = Vindx(id,i);
	bl::TinyVector<double,3> q = changed_V2[ii-N0v]  ?  tmom_v(iq_ind2(ii-N0v)) : mom_v2(ii-N0v);
	int iig1 = Gindx(id,2*i+1);
	int iig2 = Gindx(id,static_cast<int>(i_diagsG(id,2*i)));
	bl::TinyVector<double,3> k_in = changed_G[iig1] ? tmom_g(ip_ind(iig1)) : mom_g(iig1);
	bl::TinyVector<double,3> k_ou = changed_G[iig2] ? tmom_g(ip_ind(iig2)) : mom_g(iig2);
	bl::TinyVector<double,3> q2 = k_in - k_ou;
	if (fabs(norm(q2-q))>1e-6){
	  log << itt <<") ERROR checkTrial id="<< id << " i="<< i << " in computing q2 for Hugenholtz interaction q2="<< q2 << " while q2_old+dq=" << q << std::endl;
	  exit(1);
	}
      }
    }
  }
}
void CheckAccept(int itt, ostream& log, const bl::Array<unsigned short,2>& i_diagsG, const bl::Array<char,2>& Vtype,  const bl::Array<unsigned short,2>& Vindx, const bl::Array<bl::TinyVector<double,3>,1>& mom_v2,
		 const bl::Array<bl::TinyVector<double,3>,1>& mom_g,  const bl::Array<unsigned short,2>& Gindx, int N0v)
{
  int Ndiags = i_diagsG.extent(0);
  int Norder = i_diagsG.extent(1)/2;
  for (int id=0; id<Ndiags; id++){
    for (int i=0; i<Norder; i++){
      int vtyp = Vtype(id,i);
      if (vtyp>=10){
	int ii = Vindx(id,i);
	bl::TinyVector<double,3> q = mom_v2(ii-N0v);
	int iig1 = Gindx(id,2*i+1);
	int iig2 = Gindx(id,static_cast<int>(i_diagsG(id,2*i)));
	bl::TinyVector<double,3> k_in = mom_g(iig1);
	bl::TinyVector<double,3> k_ou = mom_g(iig2);
	bl::TinyVector<double,3> q2 = k_in - k_ou;
	if (fabs(norm(q2-q))>1e-6){
	  log << itt <<") ERROR checkAccept id="<< id << " i="<< i << " in computing q2 for Hugenholtz interaction q2="<< q2 << " while q2_old+dq=" << q << std::endl;
	  exit(1);
	}else{
	  log << itt << ") Works at id="<<id<<" q("<<2*i<<"->"<<2*i+1<<")="<<mom_v2(ii-N0v)<<endl;
	}
      }
    }
  }
}



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
  int Nloops = hugh_diags.extent(0);
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
	vecList Vqh1 = Gkh(i_previ) - Gkh(2*i);
	vecList dVq = Vqh1 - Vqh(i);
	if (dVq.m.size()!=0) log << "WARNING expecting Vqh0="<< Vqh(i) << " == "<< Vqh1 <<" but they are different"<<endl;
	Vqh2(id,i) = Gkh(2*i+1) - Gkh(i_previ);
      }
    }
    if (isHugenholtz){
      for (int i=0; i<Norder; i++)
	if (abs(Vtype(id,i))>=10){
	  _hugh_diags_[i].insert(id); // This is because V12(0) is changed
	  for (auto j=Vqh2(id,i).m.begin(); j!=Vqh2(id,i).m.end(); ++j) // This is because V12(1) is changed.
	    _hugh_diags_[j->first].insert(id);
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


template<typename GK, typename Data>
void sample_static_fastC_combined(Data& BKdata, bl::Array<double,2>& Pbin, std::ostream& log,
				  double lmbda, const std::vector<double>& lmbda_spct, const GK& Gk, const params& p,
				  const bl::Array<unsigned short,2>& diagsG,
				  const std::vector<std::vector<int> >& diagSign,
				  const std::vector<std::vector<std::vector<int> > >& Loop_index, const std::vector<std::vector<std::vector<int> > >& Loop_type,
				  bl::Array<char,2>& Vtype, my_mpi& mpi)
{
  typedef double real;
  log.precision(12);
  
  bool Q0w0 = BKdata.Q0w0;
  double Q_external=BKdata.Q_external;
  log << "Q_external=" << Q_external << endl;
  if (!Q0w0){
    if ( BKdata.Nlt!=p.Nlt || BKdata.Nlq!=p.Nlq){
      log<<"ERROR : Dimensions of C_Pln is wrong : either "<<BKdata.Nlt<<" != "<<p.Nlt<<" or "<<BKdata.Nlq<<" != "<<p.Nlq<<std::endl;
      exit(1);
    }
    if (Pbin.extent(0)!=p.Nt || Pbin.extent(1)!=p.Nq) Pbin.resize(p.Nt,p.Nq); Pbin=0.0;
  }else{
    if (Pbin.extent(0)!=p.Nt) Pbin.resize(p.Nt,1); Pbin=0.0;
  }
  
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
  bl::Array<double,1> pl_Q;  // Will contain legendre Pl(2*q/qmax-1)
  bl::Array<double,1> pl_t;  // Will contain legendre Pl(2*t//beta-1)
  if (!Q0w0){
    Pln.resize(std::max(BKdata.Nlq,BKdata.Nlt));
    pl_Q.resize(BKdata.Nlq+1);  // Will contain legendre Pl(2*q/qmax-1)
    pl_t.resize(BKdata.Nlt+1);  // Will contain legendre Pl(2*t//beta-1)
  }
  BKdata.C_Pln = 0.0;            // cummulative result for legendre expansion
  
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
  BKdata.FindGroups(diagsG, Loop_index, Loop_type, log);
  
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
  std::vector<double> nrm = Nrm(lmbda, cutoffk, lmbda_spct.size()+2);
  for (int i=0; i<lmbda_spct.size(); ++i){
#ifdef CNTR_VN
    // [Vq^2, Vq^3, Vq^4,..]
    lmbda_spct_cutoffk(i) = lmbda_spct[i]*nrm[i+2];
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
  {
    double Qa = Q0w0 ? Q_external : p.kF*drand(); // absolute value of external momentum
    momentum(0) = 0.0,0.0,Qa; // external momentum Q
    amomentm(0) = Qa;
    for (int ik=1; ik<Nloops; ik++){
      double th = pi*(1-drand()), phi = 2*pi*(1-drand());
      momentum(ik) = kF*sin(th)*cos(phi), kF*sin(th)*sin(phi), kF*cos(th);
      amomentm(ik) = kF;
    }
  }
  log.precision(12);
  log<<"Starting times="<<times<<" and starting momenta="<<momentum<<std::endl;

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
	long double Vn = ipower(Vq, Nc+1);
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
    Pln.cmp_single(2*amomentm(0)/cutoffq-1., pl_Q, BKdata.Nlq);          // Legendre polynomials at this Q
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
  if (mpi.rank==mpi.master) PrintInfo_(0, Qweight, amomentm(0), amomentm(1), times(0)-times(1), PQ, PQ, 0, 0, log);
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
	accept = Find_new_Q_point(K_new, Ka_new, trial_ratio, momentum(iloop), amomentm(iloop), cutoffq, drand);
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
	      long double Vn = ipower(Vq, Nc+1);
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
	    if (false){
	      int iig1 = Gindx(id,2*i+1);
	      int iig2 = Gindx(id,static_cast<int>(i_diagsG(id,2*i)));
	      bl::TinyVector<double,3> k_in = changed_G[iig1] ? tmom_g(ip_ind(iig1)) : mom_g(iig1);
	      bl::TinyVector<double,3> k_ou = changed_G[iig2] ? tmom_g(ip_ind(iig2)) : mom_g(iig2);
	      bl::TinyVector<double,3> q2 = k_in - k_ou;
	      if (fabs(norm(q2-q))>1e-6){
		log << itt <<") ERROR 1 iq="<< iq << " iloop="<< iloop << " in computing q2 for Hugenholtz interaction q2="<< q2 << " while q2_old+dq=" << q << std::endl;
		log << "    ii="<< ii <<" id="<< id << " i="<< i<< "vtyp="<<vtyp<< endl;
		log << "    isgn=" << isgn << " q2_old=" << mom_v2(ii-N0v) << " dq="<<  dK << endl;
		log << "    fermions previous momenta :"<<endl;
		for (int ig=0; ig<2*Norder; ig++){
		  int ii = Gindx(id,ig);
		  log << "k("<<ig<<"->"<<diagsG(id,ig)<<")="<< mom_g(ii) << endl;
		}
		log << "    fermions new momenta : "<< endl;
		for (int ig=0; ig<2*Norder; ig++){
		  int ii = Gindx(id,ig);
		  bl::TinyVector<double,3> k = changed_G[ii] ? tmom_g(ip_ind(ii)) : mom_g(ii);
		  log << "k("<<ig<<"->"<<diagsG(id,ig)<<")="<< k << endl;
		}
		log << "    Vq2 old momenta : " << endl;
		for (int iv=0; iv<Norder; iv++){
		  if (Vtype(id,iv)>=10){
		    int ii = Vindx(id,iv);
		    log << "q("<<2*iv<<"->"<<2*iv+1<<")="<<mom_v2(ii-N0v)<<endl;
		  }
		}
		exit(1);
	      }
	    }
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
	if (iloop==0) Nacc_q += 1; else Nacc_k += 1;
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
	      long double Vn = ipower(Vq, Nc+1);
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
	if (iloop==0 && !Q0w0) Pln.cmp_single(2*amomentm(iloop)/cutoffq-1., pl_Q, BKdata.Nlq);  // update Legendre Polynomials
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
	Pbin(it,iq) += sp;
	BKdata.Meassure(itt,tmeassure,PQ,sp,pl_Q,pl_t,momentum(0),amomentm(0),mom_g,Gindx);
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
      if ( itt>2e5/mpi.size && itt%(static_cast<int>(100000/mpi.size+1000)*tmeassure) == 0){
	double last_occurence = (Nweight-Wchanged)/(Nmeassure-Nchanged); // occurence from the last time we changed it
#ifdef _MPI
	double total_occurence=0;
	int ierr = MPI_Allreduce(&last_occurence, &total_occurence, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	if (ierr!=0) log << "MPI_Allreduce(total_occurence) returned error="<< ierr << endl; 
	last_occurence = total_occurence/mpi.size;
#endif	
	if ( last_occurence > 0.15){ // decrease by two
	  V0norm /= 2.;
	  Pnorm /= 2.;
	  Nchanged = Nmeassure;
	  Wchanged = Nweight;
	  if (mpi.rank==mpi.master) log<<std::left<<(itt/1.0e6)<<"M occurence="<<std::setw(8)<<last_occurence<<" V0 reduced to "<<V0norm<<std::endl;
	}
	if ( last_occurence < 0.03){ // increase by two
	  V0norm *= 2.;
	  Pnorm *= 2.;
	  Nchanged = Nmeassure;
	  Wchanged = Nweight;
	  if (mpi.rank==mpi.master) log<<std::left<<(itt/1.0e6)<<"M occurence="<<std::setw(8)<<last_occurence<<" V0 increased to "<<V0norm<<std::endl;
	}
      }
      if (itt%(500000*tmeassure) == 0){
	if (Nmeassure-Nweight>1000){
#ifdef _MPI	  
	  int ierr = MPI_Allreduce(MPI_IN_PLACE, mweight.K_hist.data(), mweight.K_hist.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	  if (ierr!=0) log << "MPI_Allreduce(K_hist) returned error="<< ierr << endl;
#endif
	  // We can get overflow during MPI for a long run. We should use a constant for normalization.
	  // We could here normalize K_hist (with knorm), and than when we measure, we would
	  // add instead of adding unity, we would add 1./knorm
	  dk_hist *= mweight.Normalize_K_histogram();
	  if (dk_hist < 1e-16) dk_hist = 1.0;
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
  BKdata.C_Pln *= 1.0/Nmeassure;
  Pbin         *= 1.0/Nmeassure;
  Pnorm        *= 1.0/Nmeassure;
  double occurence = static_cast<double>(Nweight)/static_cast<double>(Nmeassure);

  if (GetHistogram){
    K_Ghist *= 1.0/Nmeassure;
    K_Vhist *= 1.0/Nmeassure;
  }
  
#ifdef _MPI  
  double dat[2] = {Pnorm, occurence};
  if (mpi.rank==mpi.master){
    MPI_Reduce(MPI_IN_PLACE, BKdata.C_Pln.data(), BKdata.C_Pln.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE,  Pbin.data(),  Pbin.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, dat, 2, MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, mweight.K_hist.data(), mweight.K_hist.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    if (GetHistogram){
      MPI_Reduce(MPI_IN_PLACE, K_Ghist.data(), K_Ghist.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
      MPI_Reduce(MPI_IN_PLACE, K_Vhist.data(), K_Vhist.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    }
  }else{
    MPI_Reduce(BKdata.C_Pln.data(), BKdata.C_Pln.data(), BKdata.C_Pln.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    MPI_Reduce( Pbin.data(),  Pbin.data(),  Pbin.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    MPI_Reduce(dat, dat, 2, MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    MPI_Reduce(mweight.K_hist.data(), mweight.K_hist.data(), mweight.K_hist.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    if (GetHistogram){
      MPI_Reduce(K_Ghist.data(), K_Ghist.data(), K_Ghist.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
      MPI_Reduce(K_Vhist.data(), K_Vhist.data(), K_Vhist.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    }
  }
  if (mpi.rank==mpi.master){
    BKdata.C_Pln *= 1./mpi.size;
    Pbin         *= 1./mpi.size;
    Pnorm        = dat[0]/mpi.size;
    occurence    = dat[1]/mpi.size;
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
    BKdata.C_Pln *= dOmega * (fabs(V0norm)/Pnorm); // 4*pi because we must divide by 1/(4*pi*q**2), and 1/q**2 was already done in sampling
    Pbin  *= (dOmega * fabs(V0norm)/Pnorm);
    log<<"  meassuring diagram occurence frequency="<<occurence<<" and its norm Pnorm="<<Pnorm<<std::endl;
    
    // Proper normalization of the resulting Monte Carlo data
    double dOmega2 = (!Q0w0) ? (4*pi*cutoffq*cutoffq*cutoffq/3) : 1.0;
    double norm = ipower( beta/((2*pi)*(2*pi)*(2*pi)), Norder) * dOmega2;
    BKdata.C_Pln *= norm;

    // Later we will use Nthbin/2.0*Nkbin/2.0  normalization in the Kernel to transform from binned data to function values.
    BKdata.Normalize(beta, cutoffq);

    Pbin *= norm;
    // Q0w0 : Paver <= sum(Pbin)/beta;
    
    double dq_binning = cutoffq/Nq;
    double dt_binning = beta/Nt;
    if (!Q0w0)
      Pbin *= 1.0/(dq_binning * dt_binning);
    
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

template<typename GK>
void sample_static_Q0_fast(std::ostream& log, double Q_external, double lmbda, double lmbda_spct, bl::Array<double,1>& Pbin, const GK& Gk, const params& p,
			   const bl::Array<unsigned short,2>& diagsG, const bl::Array<float,1>& diagSign,
			   const std::vector<std::vector<std::vector<int> > >& Loop_index, const std::vector<std::vector<std::vector<int> > >& Loop_type,
			   bl::Array<char,2>& Vtype,
			   my_mpi& mpi)
{
  //#define MPI_MREAL MPI_DOUBLE
  //  typedef double real;
  typedef float real;
#define MPI_MREAL MPI_FLOAT

  if (Pbin.extent(0)!=p.Nt) Pbin.resize(p.Nt); Pbin=0.0;
  
  const int Nr=3; // three different steps in this algorithm
  double Pr_norm = (p.Pr[0]+p.Pr[1]+p.Pr[2]); // probability for all four has to be normalized
  bl::TinyVector<double,Nr> Prs;  // cumulative probability for taking a step, needed for bisection below.
  Prs[0] = p.Pr[0]/Pr_norm;
  for (int i=1; i<Nr; i++) Prs[i] = Prs[i-1] + p.Pr[i]/Pr_norm; // prepares cumulative probability from p.Pr

  int Ndiags = diagsG.extent(0);
  if (Ndiags==0) {std::cerr<<"No diagrams to simulate...."<<std::endl; return;}
  int Norder = diagsG.extent(1)/2;
  // computing the inverse of all permutations, to provide fast lookup for the previous fermionic propagator
  bl::Array<unsigned short,2> i_diagsG(Ndiags,2*Norder);
  for (int id=0; id<Ndiags; id++)
    for (int i=0; i<2*Norder; i++)
      i_diagsG(id, static_cast<int>(diagsG(id,i)) )=i;
  
  RanGSL drand(p.iseed); // GSL random number generator
  Timer t_mes, t_all, t_Pl, t_k, t_Q, t_w, t_t;
  
  CounterCoulombV Vqc(lmbda);
  
  int tmeassure=p.tmeassure, Ncout=p.Ncout, Nwarm=p.Nwarm;
  double V0norm = p.V0norm, dkF = p.dkF, kF=p.kF;// V0exp= p.V0exp;
  double cutoffq = p.cutoffq, cutoffk = p.cutoffk, dRk = p.dRk;
  double beta = p.beta;
  int Nt = p.Nt;// Nq = p.Nq;
  
  double Pnorm = 0.0;            // normalization diagram value
  
  // Here we set Vtype(id,i) => -Vtype(id,i) for those interactions which contain single-particle counter terms.
  if (lmbda_spct!=0) Where_to_Add_Single_Particle_Counter_Term(Vtype, Ndiags, Norder, diagsG);
  // Now that Vtype is properly adjusted for single-particle counter term, we can proceed to find unique propagators.
  typedef unsigned short sint;
  bl::Array<sint,2> Gindx, Vindx;
  bl::Array<bl::TinyVector<int,2>,1> gindx, vindx;
  // Finding Unique propagators for G and V
  int N0v=0;
  findUnique_noindx_spct(Gindx, Vindx, gindx, vindx, N0v, diagsG, Vtype, Loop_index, Loop_type, 0, false, false);
  bl::Array<int,1> single_counter_index;
  if (lmbda_spct!=0) Find_Single_Particle_Counter_G_index(single_counter_index, Ndiags, Norder, vindx.extent(0), diagsG, Vtype, Gindx, Vindx);
  
  
  if (Loop_index.size()!=Ndiags) log<<"Wrong size of Loop_index "<<Loop_index.size()<<" instead of "<<Ndiags<<std::endl;
  int Nloops = Loop_index[0].size();
  if (Nloops != Norder+1) log<<"Expecting Nloops==Norder+1 in Loops_index, but got Nloops="<<Nloops<<" Norder="<<Norder<<std::endl;
  int Ng = gindx.extent(0);
  int Nv = vindx.extent(0);
  int _Nv_ = std::max(Nv,1); // can not assign 0 to empty file

  bl::Array<bl::Array<sint,1>,1> loop_Gkind, loop_Vqind;
  bl::Array<bl::Array<char,1>,1> loop_Gksgn, loop_Vqsgn;
  int Ngp, Nvp;

  Get_GVind(loop_Gkind, loop_Gksgn, loop_Vqind, loop_Vqsgn, Ngp, Nvp, Loop_index, Loop_type, Gindx, Vindx);
  int Ngp_tr = GetMax_NGp_trial(i_diagsG, Gindx, Ng, Ngp);

  Reweight_time rw(p.beta,p.lmbdat);
  int Nbin = 513;
  //int Nbin = 129;
  meassureWeight mweight(p.V0exp, p.cutoffk, p.kF, Nbin, Nloops);
  //bl::Array<real,2> K_hist(Nloops,Nbin); K_hist=0;
  double dk_hist = 1.0;
  
  // This is momentum for loops
  bl::Array<bl::TinyVector<real,3>,1> momentum(Nloops); // momentum for all loops. Each diagram of the same order has the same number of loops.
  bl::Array<real,1> amomentm(Nloops);               // absolute value of momentum for each loop
  
  // Starting point is set up to be random
  // For the times for all vertices in all the graphs
  bl::Array<real,1> times(2*Norder);
  times(0) = beta*drand(); // external time
  times(1) = 0;            // beginning time
  for (int it=1; it<Norder; it++){
    double tv = beta*drand();
    times(2*it  ) = tv;
    times(2*it+1) = tv;
  }
  // Next the momenta. For a start with nonzero weight, we want to start close to kF.
  {
    double Qa = Q_external; // absolute value of external momentum
    momentum(0) = Qa,0.0,0.0; // external momentum Q
    amomentm(0) = Qa;
    for (int ik=1; ik<Nloops; ik++){
      double th = pi*(1-drand()), phi = 2*pi*(1-drand());
      momentum(ik) = kF*sin(th)*cos(phi), kF*sin(th)*sin(phi), kF*cos(th);
      amomentm(ik) = kF;
    }
  }
  
  bl::Array<real,1> G_current(Ng), V_current(_Nv_);
  // This is momentum for all diagrams and all propagators
  bl::Array<bl::TinyVector<real,3>,1> mom_g(Ng), mom_v(Nv);
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
	    mom_G(id, lindex_i) += momentum(iloop) * sign(ltype_i);
	  }else{
	    if (lindex_i>=Norder) log<<"ERROR : writting beyond boundary"<<std::endl;
	    mom_V(id,lindex_i) += momentum(iloop) * sign(ltype_i);
	  }
	}
      }
    }
    // START DEBUGGING
    for (int id=0; id<Ndiags; id++){
      for (int i=0; i<2*Norder; i++){
	//int i_final = diagsG(id,i);
	int i_previ = i_diagsG(id,i);
	bl::TinyVector<real,3> k_in = mom_G(id,i_previ);
	bl::TinyVector<real,3> k_out = mom_G(id,i);
	bl::TinyVector<real,3> q = mom_V(id,i/2)*(1-2*(i%2));
	bl::TinyVector<real,3> qp = k_in-k_out;
	if (fabs(norm(qp)-norm(q))>1e-6){
	  log<<"ERROR: diagram "<<id<<" vertex i="<<i<<" k_in="<<k_in<<" k_out="<<k_out<<" q="<<q<<std::endl;
	}
      }
    }
    // STOP DEBUGGING
    // Finally evaluating the polarizations for all diagrams
    G_current=0;
    for (int ii=0; ii<Ng; ++ii){
      int id = gindx(ii)[0];
      int  i = gindx(ii)[1];
      double aK = norm(mom_G(id,i));
      int i_final = diagsG(id,i);
      double dt = times(i_final)-times(i);
      G_current(ii) = Gk(aK, dt);
    }
    V_current=0;  
    for (int ii=0; ii<Nv; ++ii){
      int id = vindx(ii)[0];
      int  i = vindx(ii)[1];
      double aQ = norm(mom_V(id,i));
      V_current(ii) = Vqc(aQ,Vtype(id,i));
    }
    PQ=0;
    for (int id=0; id<Ndiags; id++){
      double PQd = 1.0;
      for (int i=0; i<2*Norder; i++) PQd *= G_current(Gindx(id,i));
      for (int i=1; i<Norder; i++) PQd *= V_current(Vindx(id,i));
      PQ += PQd * diagSign(id);
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
  
  bl::Array<real,1> times_trial(2*Norder);
  int _Nvp_ = std::max(Nvp,1); // can not assign 0 to empty file
  bl::Array<bl::TinyVector<real,3>,1> tmom_g(Ngp), tmom_v(_Nvp_);
  bl::Array<sint,1> ip_ind(Ng), iq_ind(Nv);
  bl::Array<real,1> g_trial(Ngp_tr), v_trial(_Nvp_);
  BitArray changed_G(Ng), changed_V(Nv);
  
  int Qweight = 0;      // if Qweight==1 then we have the normalization diagram
  double Nweight = 0;   // how many times did we see measuring diagram.
  int NlowT = 0;        // how many times do we see external time < 0.05*beta
  int Nmeassure = 0;    // how many times did we measure quantities.
  int Nchanged  = 0;    // when was V0norm last changed?
  double Wchanged = 0;  // what was Nweight when we last time changed V0norm?
  int Nacc_k = 0, Nacc_q = 0, Nacc_w = 0, Nacc_t = 0;
  int Nall_k = 0, Nall_q = 0, Nall_w = 0, Nall_t = 0;
  if (mpi.rank==mpi.master) PrintInfo_(0, Qweight, amomentm(0), amomentm(1), times(0)-times(1), PQ, PQ, 0, 0, log);

  t_all.start();
  for (int itt=0; itt<p.Nitt; itt++){
    double PQ_new=0;
    int icase = tBisect(drand(), Prs); // which steps should we attempt?
    if (icase==0){ // changing momentum Q or k-point
      int iloop = 1 + static_cast<int>((Nloops-1)*drand()); // newer change measuring (0) loop
      bl::TinyVector<real,3> K_new; real Ka_new; double trial_ratio=1;
      bool accept=false;
      Nall_k += 1;
      accept = Find_new_k_point(K_new, Ka_new, trial_ratio, momentum(iloop), amomentm(iloop), cutoffk, drand, drand()<p.Qring, dkF, dRk);
      if (accept){
	bl::TinyVector<real,3> dK = K_new - momentum(iloop); // how much momentum will be changed.
	//t1.start();
	changed_G=0;
	for (int ip=0; ip<loop_Gkind(iloop).extent(0); ++ip){ // all G-propagators, which are changed when momentum in certain loop is changed.
	  int ii   = loop_Gkind(iloop)(ip);  // index of G-propagators, which we are now changing
	  int isgn = loop_Gksgn(iloop)(ip);  // momentum can be added or subtracted on this propagator.
	  bl::TinyVector<real,3> k = mom_g(ii) + dK * isgn; // actually changing the momentum
	  tmom_g(ip) = k;                    // remember the momentum
	  changed_G.set(ii,1);                   // remember that it is changed
	  if (!Qweight){
	    double aK = norm(k);
	    int id = gindx(ii)[0];           // find first diagram with this propagator
	    int  i = gindx(ii)[1];           // and which propagator is this in the diagram
	    int i_final = diagsG(id,i);      // times in the propagators.
	    double dt = times(i_final)-times(i);
	    g_trial(ip) = Gk(aK, dt);        // remeber the value of the propagator
	    ip_ind(ii)=ip;                   // remeber where it is stored
          }
	}
	//t1.stop();
	//t2.start();
	changed_V=0;
	for (int iq=0; iq<loop_Vqind(iloop).extent(0); ++iq){ // Now do the same for the interaction
	  int ii   = loop_Vqind(iloop)(iq);
	  int isgn = loop_Vqsgn(iloop)(iq);
	  bl::TinyVector<real,3> q =  mom_v(ii) + dK * isgn;
	  tmom_v(iq) = q;
	  changed_V.set(ii,1);
	  if (!Qweight){
	    double aQ = norm(q);
	    int id = vindx(ii)[0];
	    int  i = vindx(ii)[1];
	    v_trial(iq) = Vqc(aQ, Vtype(id,i));
	    iq_ind(ii)=iq;
	  }
	}
	//t2.stop();
	if (! Qweight){
	  //t3.start();
	  PQ_new=0;
	  for (int id=0; id<Ndiags; id++){
	    double PQd = 1.0;
	    for (int i=0; i<2*Norder; i++){
	      int ii = Gindx(id,i);
	      PQd *= ( changed_G[ii] ? g_trial(ip_ind(ii)) : G_current(ii) );
	    }
	    for (int i=1; i<Norder; i++){
	      int ii = Vindx(id,i);
	      PQd *= ( changed_V[ii] ? v_trial(iq_ind(ii)) : V_current(ii) );
	    }
	    PQ_new += PQd * diagSign(id);
	  }
	  //t3.stop();
	}else{
	  real Ka_old = amomentm(iloop);
	  amomentm(iloop) = Ka_new;
	  bl::TinyVector<double,3> K_old = momentum(iloop);
	  momentum(iloop) = K_new;
	  PQ_new = V0norm * mweight(amomentm, momentum);
	  amomentm(iloop) = Ka_old;
	  momentum(iloop) = K_old;
	}
	double ratio = PQ_new*trial_ratio / PQ;
	accept = fabs(ratio) > 1-drand() && PQ_new!=0;
      }
      if ((itt+1)%Ncout==0 && mpi.rank==mpi.master)
	_PrintInfo_(itt+1, Qweight, icase, amomentm(1), amomentm(Nloops-1), times(0), PQ_new, PQ, NlowT/(Nmeassure+0.0), Nweight/(Nmeassure+0.0), log);
      if (accept){
	Nacc_k += 1;
	momentum(iloop) = K_new;  // this momentum was changed
	amomentm(iloop) = Ka_new;
	for (int ip=0; ip<loop_Gkind(iloop).extent(0); ++ip){
	  int ii   = loop_Gkind(iloop)(ip); 
	  mom_g(ii) = tmom_g(ip);
	}
	for (int iq=0; iq<loop_Vqind(iloop).extent(0); ++iq){ // Now do the same for the interaction
	  int ii   = loop_Vqind(iloop)(iq);
	  mom_v(ii) = tmom_v(iq);
	}
	if (!Qweight){
	  for (int ip=0; ip<loop_Gkind(iloop).extent(0); ++ip){ // all G-propagators, which are changed when momentum in certain loop is changed.
	    int ii   = loop_Gkind(iloop)(ip);  // index of G-propagators, which we are now changing
	    G_current(ii)=g_trial(ip);
	  }
	  for (int iq=0; iq<loop_Vqind(iloop).extent(0); ++iq){ // Now do the same for the interaction
	    int ii   = loop_Vqind(iloop)(iq);
	    V_current(ii) = v_trial(iq);
	  }
	}else{
	  // since we have the measuring diagram, we did not calculate G_trial and V_trial before, hence we need to evaluate them now.
	  //t5.start();
	  for (int ip=0; ip<loop_Gkind(iloop).extent(0); ++ip){ // all G-propagators, which are changed when momentum in certain loop is changed.
	    int ii    = loop_Gkind(iloop)(ip);  // index of G-propagators, which we are now changing
	    double aK = norm(tmom_g(ip));
	    int id = gindx(ii)[0];           // find first diagram with this propagator
	    int  i = gindx(ii)[1];           // and which propagator is this in the diagram
	    int i_final = diagsG(id,i);      // times in the propagators.
	    double dt = times(i_final)-times(i);
	    G_current(ii) = Gk(aK, dt);        // remeber the value of the propagator
	  }
	  for (int iq=0; iq<loop_Vqind(iloop).extent(0); ++iq){ // Now do the same for the interaction
	    int ii   = loop_Vqind(iloop)(iq);
	    double aQ = norm(tmom_v(iq));
	    int id = vindx(ii)[0];
	    int  i = vindx(ii)[1];
	    V_current(ii) = Vqc(aQ, Vtype(id,i));
	  }
	  //t5.stop();
	}
	PQ = PQ_new;
      }
    } else if (icase==1){  // changing one of the time variables
      Nall_t += 1;
      int itime = static_cast<int>(Norder*drand()); // which time to change? For bare interaction, there are only Norder different times.
      //t6.start();
      changed_G=0;            // which propagators are being changed?
      times_trial = times;    // times_trial will contain the trial step times.
      if (itime==0){          // this is the measuring time with vertex=0
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
	for (int ivertex=2*itime; ivertex<=2*itime+1; ++ivertex){ // 2*itime and 2*itime+1 are changed at the same time, because they are equal
	  times_trial(ivertex) = t_new;
	  for (int id=0; id<Ndiags; id++){
	    int i_pre_vertex = i_diagsG(id,ivertex);
	    changed_G.set(Gindx(id,ivertex),1);        // these two propagators are changed because of the new time.
	    changed_G.set(Gindx(id,i_pre_vertex),1);
	  }
	}
      }
      //t6.stop();
      if (! Qweight){
	//t7.start();
	int ip=0;
	for (int ii=0; ii<Ng; ++ii){
	  if (changed_G[ii]){
	    int id = gindx(ii)[0];
	    int  i = gindx(ii)[1];
	    double aK = norm(mom_g(ii));
	    int i_final = diagsG(id,i);
	    double dt = times_trial(i_final)-times_trial(i);
	    g_trial(ip) = Gk(aK, dt);
	    ip_ind(ii)=ip;
	    ip+=1;
	  }
	}
	//t7.stop();
	//t8.start();
	PQ_new=0; // recalculating PQ, taking into account one change of time.
	for (int id=0; id<Ndiags; id++){
	  double PQd = 1.0;
	  for (int i=0; i<2*Norder; i++){
	    int ii = Gindx(id,i);
	    PQd *= ( changed_G[ii] ? g_trial(ip_ind(ii)) : G_current(ii) );
	  }
	  for (int i=1; i<Norder; i++) PQd *= V_current(Vindx(id,i)); // interaction does not depend on time, hence it is not changed here.
	  PQ_new += PQd * diagSign(id);
	}
	//t8.stop();
      }else{
	PQ_new = V0norm * mweight(amomentm, momentum);
      }
      double ratio = PQ_new/PQ;
      bool accept = fabs(ratio) > 1-drand() && PQ_new!=0;
      if ((itt+1)%Ncout==0 && mpi.rank==mpi.master)
	_PrintInfo_(itt+1, Qweight, icase, amomentm(1), amomentm(Nloops-1), times(2*itime), PQ_new, PQ, NlowT/(Nmeassure+0.0), Nweight/(Nmeassure+0.0), log);
      if (accept){
	//t9.start();
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
	      G_current(ii) = Gk( norm(mom_g(ii)),  times(i_final)-times(i)); // these propagators were changed, but they were not computed before.
	    }
	  }
	}
	//t9.stop();
	PQ = PQ_new;
      }
    }else{  // normalization diagram step
      Nall_w += 1;
      //t10.start();
      if (Qweight){ // trying to jump back into physical hilbert space
	PQ_new=0;
	for (int id=0; id<Ndiags; id++){
	  double PQd = 1.0;
	  for (int i=0; i<2*Norder; i++) PQd *= G_current(Gindx(id,i)); // we have all propagators precomputed, hence it is easy to compute the diagrams.
	  for (int i=1; i<Norder; i++) PQd *= V_current(Vindx(id,i));
	  PQ_new += PQd * diagSign(id);
	}
      }else{        // or if we are in the physical Hilbert space, we try to jump into the normalization diagram
	PQ_new = V0norm * mweight(amomentm, momentum);
      }
      //t10.stop();
      double ratio = PQ_new/PQ;
      bool accept = fabs(ratio) > 1-drand() && PQ_new!=0;
      if ((itt+1)%Ncout==0 && mpi.rank==mpi.master)
	  _PrintInfo_(itt+1, Qweight, icase, amomentm(1), amomentm(Nloops-1), times(0), PQ_new, PQ, NlowT/(Nmeassure+0.0), Nweight/(Nmeassure+0.0),log);
      if (accept){
	Nacc_w += 1;
	Qweight = 1-Qweight;
	PQ = PQ_new;
      }
    }
    // Measuring
    if (itt>=Nwarm && itt%tmeassure==0){
      t_mes.start();
      Nmeassure += 1;
      double t = times(0)-times(1);
      double cw = 1./rw.wt(t);
      double sp = sign(PQ) * cw;
      int it = std::min(int(t/beta*Nt), Nt-1);
      if (Qweight){
	Pnorm += /*Qa*Qa*/ cw;                 // |Q|^2 * reweighting
	Nweight+=1;
      }else{
	Pbin(it) += sp; 
      }
      if (t < 0.05*beta){
	NlowT +=1 ;
      }

      if (!Qweight){
	mweight.Add_to_K_histogram(dk_hist, momentum, cutoffq, cutoffk);
	/*
	int iik = static_cast<int>( amomentm(0)/cutoffq * Nbin);
	if (iik>=Nbin) iik=Nbin-1;
	K_hist(0,iik) += 1.0;
	for (int ik=1; ik<Nloops; ik++){
	  real k = amomentm(ik);
	  int iik = static_cast<int>(k/cutoffk * Nbin);
	  if (iik>=Nbin) iik=Nbin-1;
	  K_hist(ik,iik) += 1./(k*k);
	}
	*/
      }

      if ( itt>0 && itt%(2000*tmeassure) == 0){
	double last_occurence = (Nweight-Wchanged)/(Nmeassure-Nchanged); // occurence from the last time we changed it
#ifdef _MPI
	double total_occurence=0;
	MPI_Allreduce(&last_occurence, &total_occurence, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	last_occurence = total_occurence/mpi.size;
#endif	
	if ( last_occurence > 0.15){ // decrease by two
	  V0norm /= 2.;
	  Pnorm /= 2.;
	  Nchanged = Nmeassure;
	  Wchanged = Nweight;
	  if (mpi.rank==mpi.master) log<<std::left<<(itt/1.0e6)<<"M occurence="<<std::setw(8)<<last_occurence<<" V0 reduced to "<<V0norm<<std::endl;
	}
	if ( last_occurence < 0.03){ // increase by two
	  V0norm *= 2.;
	  Pnorm *= 2.;
	  Nchanged = Nmeassure;
	  Wchanged = Nweight;
	  if (mpi.rank==mpi.master) log<<std::left<<(itt/1.0e6)<<"M occurence="<<std::setw(8)<<last_occurence<<" V0 increased to "<<V0norm<<std::endl;
	}
      }
      if (itt%(500000*tmeassure) == 0){
	if (Nmeassure-Nweight>1000){
#ifdef _MPI	  
	  MPI_Allreduce(MPI_IN_PLACE, mweight.K_hist.data(), mweight.K_hist.size(), MPI_MREAL, MPI_SUM, MPI_COMM_WORLD);
#endif
	  dk_hist *= mweight.Normalize_K_histogram();
	  mweight.Recompute(mpi.rank==mpi.master); // Nweight has to be finite, otherwise there is no information stored anyway.
	  // OLD_BUG : If we are in a meassuring space, we have to update now the value of the meassuring diagram as the weight has changed.
	  if (Qweight) PQ = V0norm * mweight(amomentm, momentum);
	}
      }
      t_mes.stop();
    }
  }
  t_all.stop();
  Pbin      *= 1.0/Nmeassure;
  Pnorm     *= 1.0/Nmeassure;
  double occurence = static_cast<double>(Nweight)/static_cast<double>(Nmeassure);
  
  mweight.K_hist *= 1.0/Nmeassure;
  
#ifdef _MPI  
  double dat[2] = {Pnorm, occurence};
  if (mpi.rank==mpi.master){
    MPI_Reduce(MPI_IN_PLACE,  Pbin.data(),  Pbin.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, dat, 2, MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, mweight.K_hist.data(), mweight.K_hist.size(), MPI_MREAL, MPI_SUM, mpi.master, MPI_COMM_WORLD);
  }else{
    MPI_Reduce( Pbin.data(),  Pbin.data(),  Pbin.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    MPI_Reduce(dat, dat, 2, MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    MPI_Reduce(mweight.K_hist.data(), mweight.K_hist.data(), mweight.K_hist.size(), MPI_MREAL, MPI_SUM, mpi.master, MPI_COMM_WORLD);
  }
  if (mpi.rank==mpi.master){
    Pbin  *= 1./mpi.size;
    Pnorm = dat[0]/mpi.size;
    occurence = dat[1]/mpi.size;
    //
    mweight.K_hist *= 1./mpi.size;
  }
#endif
  log<<"total acceptance rate="<<(Nacc_k+Nacc_q+Nacc_t+Nacc_w)/(p.Nitt+0.0)<<" "<<" k-acceptance="<<(Nacc_k/(Nall_k+0.0))<<" t-acceptanece="<<(Nacc_t/(Nall_t+0.0))<<" w-acceptanece="<<(Nacc_w/(Nall_w+0.0))<<std::endl;
  log<<" k-trials="<<(Nall_k/(p.Nitt+0.0))<<" q-trials="<<(Nall_q/(p.Nitt+0.0))<<" t-trials="<<(Nall_t/(p.Nitt+0.0))<<" w-trials="<<(Nall_w/(p.Nitt+0.0))<<std::endl;
  //log<<"t1="<<t1.elapsed()<<" t2="<<t2.elapsed()<<" t3="<<t3.elapsed()<<" t4="<<t4.elapsed()<<" t5="<<t5.elapsed()<<" t6="<<t6.elapsed()<<" t7="<<t7.elapsed()<<" t8="<<t8.elapsed()<<" t9="<<t9.elapsed()<<" t10="<<t10.elapsed()<<std::endl;
  log<<"total time="<<t_all.elapsed()<<" measuring time="<<t_mes.elapsed()<<std::endl;
  
  if (mpi.rank==mpi.master){
    Pbin  *= (fabs(V0norm)/Pnorm);
    log<<"  meassuring diagram occurence frequency="<<occurence<<" and its norm Pnorm="<<Pnorm<<std::endl;
    
    for (int ik=0; ik<mweight.K_hist.extent(0); ik++){
      double dsum=0;
      for (int i=0; i<Nbin; i++) dsum += mweight.K_hist(ik,i);
      mweight.K_hist(ik,bl::Range::all()) *= 1./dsum;
      std::ofstream hout((std::string("Q_hist.")+std::to_string(ik)).c_str());
      double cutoff = (ik>0)? cutoffk : cutoffq;
      for (int i=0; i<Nbin; i++){
	hout<< (i+0.5)/Nbin*cutoff/kF << " " << mweight.K_hist(ik,i)<<std::endl;
      }
    }
  }
}

template<typename GK>
std::tuple<double,double,double,double> sample_Density_C_orig(std::ostream& log, double lmbda, const std::vector<double>& lmbda_spct, const GK& Gk, const params& p,
							 const bl::Array<unsigned short,2>& diagsG, const bl::Array<float,1>& diagSign,
							 const std::vector<std::vector<std::vector<int> > >& Loop_index, const std::vector<std::vector<std::vector<int> > >& Loop_type,
							 bl::Array<char,2>& Vtype, my_mpi& mpi)
{
  typedef double real;
  log.precision(12);
  //log<<"Inside sample_Density_C"<<endl;
  
  const int Nr=3; // three different steps in this algorithm
  double Pr_norm = (p.Pr[0]+p.Pr[1]+p.Pr[2]); // probability for all four has to be normalized
  bl::TinyVector<double,Nr> Prs;  // cumulative probability for taking a step, needed for bisection below.
  Prs[0] = p.Pr[0]/Pr_norm;
  for (int i=1; i<Nr; i++) Prs[i] = Prs[i-1] + p.Pr[i]/Pr_norm; // prepares cumulative probability from p.Pr

  int Ndiags = diagsG.extent(0);
  if (Ndiags==0) {
    log<<"No diagrams to simulate...."<<std::endl;
    return std::make_tuple(0.,0.,0.,0.);
  }
  int Norder = diagsG.extent(1)/2;
  // computing the inverse of all permutations, to provide fast lookup for the previous fermionic propagator
  bl::Array<unsigned short,2> i_diagsG(Ndiags,2*Norder);
  for (int id=0; id<Ndiags; id++)
    for (int i=0; i<2*Norder; i++)
      i_diagsG(id, static_cast<int>(diagsG(id,i)) )=i;
  
  RanGSL drand(p.iseed); // GSL random number generator
  Timer t_mes, t_all, t_Pl, t_k, t_Q, t_w, t_t;
  
  CounterCoulombV Vqc(lmbda);
  
  int tmeassure=p.tmeassure, Ncout=p.Ncout, Nwarm=p.Nwarm;
  double V0norm = p.V0norm, dkF = p.dkF, kF=p.kF;// V0exp= p.V0exp;
  double cutoffk = p.cutoffk, dRk = p.dRk;
  double beta = p.beta;
  
  double Pnorm = 0.0;            // normalization diagram value

  // Here we set Vtype(id,i) => -Vtype(id,i) for those interactions which contain single-particle counter terms.
  Where_to_Add_Single_Particle_Counter_Term(Vtype, lmbda_spct, Ndiags, Norder, diagsG);
  
  // Now that Vtype is properly adjusted for single-particle counter term, we can proceed to find unique propagators.
  typedef unsigned short sint;
  bl::Array<sint,2> Gindx, Vindx;
  bl::Array<bl::TinyVector<int,2>,1> gindx, vindx;
  // Finding Unique propagators for G and V
  bool debug = true;
  int N0v=0;
  findUnique_noindx_spct(Gindx, Vindx, gindx, vindx, N0v, diagsG, Vtype, Loop_index, Loop_type, 1, false, debug, log);

  // Once more we go through single-particle counter terms and create index single_counter_index
  bl::Array<int,1> single_counter_index;
  Find_Single_Particle_Counter_G_index(single_counter_index, Ndiags, Norder, vindx.extent(0), diagsG, Vtype, Gindx, Vindx);
  
  if (Loop_index.size()!=Ndiags) {log<<"Wrong size of Loop_index "<<Loop_index.size()<<" instead of "<<Ndiags<<std::endl; exit(1);}
  int Nloops = Loop_index[0].size();
  // Here we are sampling density, which has one loop less than polarization!
  if (Nloops != Norder) {log<<"Expecting Nloops==Norder in Loops_index, but got Nloops="<<Nloops<<" Norder="<<Norder<<std::endl; exit(1);}
  int Ng = gindx.extent(0);
  int Nv = vindx.extent(0);
  int _Nv_ = std::max(Nv,1); // can not assign 0 to empty file

  bl::Array<bl::Array<sint,1>,1> loop_Gkind, loop_Vqind;
  bl::Array<bl::Array<char,1>,1> loop_Gksgn, loop_Vqsgn;
  int Ngp, Nvp;
  // Find index-tables to all momentum loops, so that we do not need to loop over all diagrams, but rather only over all differet propagators corresponding to certain loop.
  // Here we also take into account that single-particle counter terms are sometimes added.
  Get_GVind(loop_Gkind, loop_Gksgn, loop_Vqind, loop_Vqsgn, Ngp, Nvp, Loop_index, Loop_type, Gindx, Vindx, single_counter_index, lmbda_spct, Vtype, false, log);
  int Ngp_tr = GetMax_NGp_trial(i_diagsG, Gindx, Ng, Ngp);
  
  // DEBUG
  if (mpi.rank==mpi.master){
    log<<"lmbda_spct=";
    for (int i=0; i<lmbda_spct.size(); ++i) log<<lmbda_spct[i]<<",";
    log<<std::endl;
    log<<"single_counter_index="<<single_counter_index<<std::endl;
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
    log<<"loop_Vqind:"<<std::endl;
    for (int iloop=0; iloop<Nloops; ++iloop){
      for (int iq=0; iq<loop_Vqind(iloop).extent(0); ++iq){ // Now do the same for the interaction
	int ii   = loop_Vqind(iloop)(iq);
	int isgn = loop_Vqsgn(iloop)(iq);
	int id = vindx(ii)[0];
	int  i = vindx(ii)[1];
	int vtyp = Vtype(id,i);
	log<<"iloop="<<iloop<<" iq="<<iq<<" ii_v="<<ii<<" isgn="<<isgn<<" which comes from id="<< id <<" and pair =("<<2*i<<","<<2*i+1<<") of type="<<vtyp<<std::endl;
      }
    }
  }
  // END DEBUG
  
  int Nbin = 513;
  //int Nbin = 129;
  bl::Array<double,1> lmbda_spct_cutoffk(lmbda_spct.size());
  for (int i=0; i<lmbda_spct.size(); ++i)
    lmbda_spct_cutoffk(i) = lmbda_spct[i]*(2*pi*2*pi*2*pi)/(4*pi*ipower(cutoffk,3)/3.);
  
  meassureWeight mweight(p.V0exp, p.cutoffk, p.kF, Nbin, Nloops, 0);
  double dk_hist = 1.0;
  
  // This is momentum for loops
  bl::Array<bl::TinyVector<double,3>,1> momentum(Nloops); // momentum for all loops. Each diagram of the same order has the same number of loops.
  bl::Array<double,1> amomentm(Nloops);               // absolute value of momentum for each loop

  // Starting point is set up to be random
  // For the times for all vertices in all the graphs
  bl::Array<double,1> times(2*Norder);
  times(0) = -1e-15; // for density, we need t=0^-
  times(1) = 0;     // beginning time
  for (int it=1; it<Norder; it++){
    double tv = beta*drand();
    times(2*it  ) = tv;
    times(2*it+1) = tv;
  }
  // Next the momenta. For a start with nonzero weight, we want to start close to kF.
  {
    for (int ik=0; ik<Nloops; ik++){
      double th = drand()*pi, ph = drand()*(2*pi);
      double st = sin(th), ct = cos(th), cp = cos(ph), sp = sin(ph);
      momentum(ik) = kF*st*cp, kF*st*sp, kF*ct;
      amomentm(ik) = kF;
    }
  }
  log.precision(12);
  log<<"Starting times="<<times<<" and starting momenta="<<momentum<<std::endl;
  
  bl::Array<double,1> G_current(Ng);
  bl::Array<double,1> V_current(_Nv_);
  // This is momentum for all diagrams and all propagators
  bl::Array<bl::TinyVector<real,3>,1> mom_g(Ng), mom_v(Nv);
  bl::Array<double,1> PQg(Ndiags);
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
	    mom_V(id, lindex_i) += momentum(iloop) * dsign(ltype_i);
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
	if (fabs(norm(k_in-k_out-q))>1e-6){
	  log<<"ERROR : diagram "<<id<<" vertex i="<<i<<" k_in="<<k_in<<" k_out="<<k_out<<" q="<<q<<std::endl;
	}
      }
    }
    // STOP DEBUGGING
    // Finally evaluating the polarizations for all diagrams
    G_current=0;
    for (int ii=0; ii<Ng; ++ii){
      int id = gindx(ii)[0];
      int  i = gindx(ii)[1];
      double aK = norm(mom_G(id,i));
      int i_final = diagsG(id,i);
      double dt = times(i_final)-times(i);
      G_current(ii) = Gk(aK, dt);
    }
    V_current=0;  
    for (int ii=0; ii<Nv; ++ii){
      int id = vindx(ii)[0];
      int  i = vindx(ii)[1];
      double aQ = norm(mom_V(id,i));
      int vtyp = Vtype(id,i);
      double Vq = Vqc(aQ,abs(vtyp));
      if (vtyp>=0){
	V_current(ii) = Vq;
      }else{// we also add the single-particle counter term in addition to two-particle counter term.
	int ii_g = single_counter_index(ii);
	double g_kq = G_current(ii_g);
	V_current(ii) = Vq;
	if (g_kq!=0) V_current(ii) += lmbda_spct_cutoffk(-vtyp-1)/g_kq;
	{
	  int _id_ = gindx(ii_g)[0];
	  int _i_  = gindx(ii_g)[1];
	  int _i_final_ = diagsG(_id_,_i_);
	  if (fabs( times(_i_final_)-times(_i_) )>1e-6)
	    log<<"ERROR G_{k+q}(dt) should have argument dt=0, but is not : "<<times(_i_final_)-times(_i_)<<" with i_final="<<_i_final_<<" i="<<_i_<<" at id="<<_id_<<std::endl; 
	}
      }
    }
    PQ=0;
    for (int id=0; id<Ndiags; id++){
      double PQd = 1.0;
      for (int i=1; i<2*Norder; i++) PQd *= G_current(Gindx(id,i));
      for (int i=1; i<Norder; i++) PQd *= V_current(Vindx(id,i));
      PQd *= diagSign(id);
      PQ += PQd;
      PQg(id) = PQd;
    }

    // Now we create smaller arrays mom_g and mom_v, which contain the same information as mom_G and mom_V.
    mom_g=0.0; mom_v=0.0;
    for (int id=0; id<Ndiags; id++){
      for (int i=1; i<2*Norder; i++){
	int ii = Gindx(id,i);
	mom_g(ii) = mom_G(id,i);
      }
      for (int i=1; i<Norder; i++){
	int ii = Vindx(id,i);
	mom_v(ii) = mom_V(id,i);
      }
    }
  }

  bl::Array<double,1> times_trial(2*Norder);
  int _Nvp_ = std::max(Nvp,1); // can not assign 0 to empty file
  bl::Array<bl::TinyVector<real,3>,1> tmom_g(Ngp), tmom_v(_Nvp_);
  bl::Array<sint,1> ip_ind(Ng), iq_ind(Nv);
  bl::Array<double,1> g_trial(Ngp_tr);
  bl::Array<double,1> v_trial(_Nvp_);
  BitArray changed_G(Ng), changed_V(Nv);
  
  int Qweight = 0;      // if Qweight==1 then we have the normalization diagram
  double Nweight = 0;   // how many times did we see measuring diagram.
  //int NlowT = 0;        // how many times do we see external time < 0.05*beta
  int Nmeassure = 0;    // how many times did we measure quantities.
  int Nchanged  = 0;    // when was V0norm last changed?
  double Wchanged = 0;  // what was Nweight when we last time changed V0norm?
  int Nacc_k = 0, Nacc_q = 0, Nacc_w = 0, Nacc_t = 0;
  int Nall_k = 0, Nall_q = 0, Nall_w = 0, Nall_t = 0;
  if (mpi.rank==mpi.master) PrintInfo_(0, Qweight, amomentm(0), amomentm(1), times(0)-times(1), PQ, PQ, 0, 0, log);
  double Pw0=0, Ekin=0;
  bl::Array<double,1> PQg_new(Ndiags);
  
  t_all.start();
  for (int itt=0; itt<p.Nitt; itt++){
    double PQ_new=0;
    int icase = tBisect(drand(), Prs); // which steps should we attempt?
    if (Norder==1){
      // only at order 1 we have no time variable to move, hence icase==1 should not occur
      while (icase==1) icase = tBisect(drand(), Prs); 
    }
    if (icase==0){ // changing momentum Q or k-point
      int iloop = static_cast<int>(Nloops*drand()); // newer change measuring (0) loop
      bl::TinyVector<double,3> K_new; double Ka_new; double trial_ratio=1;
      bool accept=false;
      Nall_k += 1;
      accept = Find_new_k_point(K_new, Ka_new, trial_ratio, momentum(iloop), amomentm(iloop), cutoffk, drand, drand()<p.Qring, dkF, dRk);
      if (accept){
	bl::TinyVector<double,3> dK = K_new - momentum(iloop); // how much momentum will be changed.
	changed_G = 0;
	for (int ip=0; ip<loop_Gkind(iloop).extent(0); ++ip){ // all G-propagators, which are changed when momentum in certain loop is changed.
	  int ii   = loop_Gkind(iloop)(ip);  // index of G-propagators, which we are now changing
	  double isgn = loop_Gksgn(iloop)(ip);  // momentum can be added or subtracted on this propagator.
	  bl::TinyVector<double,3> k = mom_g(ii) + dK * isgn; // actually changing the momentum
	  tmom_g(ip) = k;                    // remember the momentum
	  changed_G.set(ii,1);                   // remember that it is changed
	  if (!Qweight){
	    double aK = norm(k);
	    int id = gindx(ii)[0];           // find first diagram with this propagator
	    int  i = gindx(ii)[1];           // and which propagator is this in the diagram
	    int i_final = diagsG(id,i);      // times in the propagators.
	    double dt = times(i_final)-times(i);
	    g_trial(ip) = Gk(aK, dt);        // remeber the value of the propagator
	    ip_ind(ii)=ip;                   // remeber where it is stored
          }
	}
	//t1.stop();
	changed_V = 0;
	for (int iq=0; iq<loop_Vqind(iloop).extent(0); ++iq){ // Now do the same for the interaction
	  int ii   = loop_Vqind(iloop)(iq);
	  double isgn = loop_Vqsgn(iloop)(iq);
	  bl::TinyVector<double,3> q =  mom_v(ii) + dK * isgn;
	  tmom_v(iq) = q;
	  changed_V.set(ii,1);
	  if (!Qweight){
	    double aQ = norm(q);
	    int id = vindx(ii)[0];
	    int  i = vindx(ii)[1];
	    int vtyp = Vtype(id,i);
	    double Vq = Vqc(aQ, abs(vtyp)); // This is Vq^2*lambda/(8*pi)
	    if (vtyp>=0){
	      v_trial(iq) = Vq;
	    }else{
	      // we also add the single-particle counter term in addition to two-particle counter term.
	      int ii_g = single_counter_index(ii);
	      double g_kq = ( changed_G[ii_g] ? g_trial(ip_ind(ii_g)) : G_current(ii_g) );
	      v_trial(iq) = Vq;
	      if (g_kq!=0) v_trial(iq) += lmbda_spct_cutoffk(-vtyp-1)/g_kq;
	    }
	    iq_ind(ii)=iq;
	  }
	}
	if (! Qweight){
	  PQ_new=0;
	  for (int id=0; id<Ndiags; id++){
	    double PQd = 1.0;
	    for (int i=1; i<2*Norder; i++){
	      int ii = Gindx(id,i);
	      PQd *= ( changed_G[ii] ? g_trial(ip_ind(ii)) : G_current(ii) );
	    }
	    for (int i=1; i<Norder; i++){
	      int ii = Vindx(id,i);
	      PQd *= ( changed_V[ii] ? v_trial(iq_ind(ii)) : V_current(ii) );
	    }
	    PQd *= diagSign(id);
	    PQ_new += PQd;
	    PQg_new(id) = PQd;
	  }
	}else{
	  double Ka_old = amomentm(iloop);
	  amomentm(iloop) = Ka_new;
	  bl::TinyVector<double,3> K_old = momentum(iloop);
	  momentum(iloop) = K_new;
	  PQ_new = V0norm * mweight(amomentm, momentum);
	  amomentm(iloop) = Ka_old;
	  momentum(iloop) = K_old;
	}
	double ratio = PQ_new*trial_ratio / PQ;
	accept = fabs(ratio) > 1-drand() && PQ_new!=0;
      }
      if ((itt+1)%Ncout==0 && mpi.rank==mpi.master) _PrintInfo_(itt+1, Qweight, icase, amomentm(1), amomentm(Nloops-1), times(0), PQ_new, PQ, dk_hist, Nweight/(Nmeassure+0.0), log);
      if (accept){
	Nacc_k += 1;
	momentum(iloop)  = K_new;  // this momentum was changed
	amomentm(iloop)  = Ka_new;
	for (int ip=0; ip<loop_Gkind(iloop).extent(0); ++ip){
	  int ii   = loop_Gkind(iloop)(ip); 
	  mom_g(ii) = tmom_g(ip);
	}
	for (int iq=0; iq<loop_Vqind(iloop).extent(0); ++iq){ // Now do the same for the interaction
	  int ii   = loop_Vqind(iloop)(iq);
	  mom_v(ii) = tmom_v(iq);
	}
	if (!Qweight){
	  for (int ip=0; ip<loop_Gkind(iloop).extent(0); ++ip){ // all G-propagators, which are changed when momentum in certain loop is changed.
	    int ii   = loop_Gkind(iloop)(ip);  // index of G-propagators, which we are now changing
	    G_current(ii)=g_trial(ip);
	  }
	  for (int iq=0; iq<loop_Vqind(iloop).extent(0); ++iq){ // Now do the same for the interaction
	    int ii   = loop_Vqind(iloop)(iq);
	    V_current(ii) = v_trial(iq);
	  }
	}else{
	  // since we have the measuring diagram, we did not calculate G_trial and V_trial before, hence we need to evaluate them now.
	  for (int ip=0; ip<loop_Gkind(iloop).extent(0); ++ip){ // all G-propagators, which are changed when momentum in certain loop is changed.
	    int ii    = loop_Gkind(iloop)(ip);  // index of G-propagators, which we are now changing
	    double aK = norm(tmom_g(ip));
	    int id = gindx(ii)[0];           // find first diagram with this propagator
	    int  i = gindx(ii)[1];           // and which propagator is this in the diagram
	    int i_final = diagsG(id,i);      // times in the propagators.
	    double dt = times(i_final)-times(i);
	    G_current(ii) = Gk(aK, dt);        // remeber the value of the propagator
	  }
	  for (int iq=0; iq<loop_Vqind(iloop).extent(0); ++iq){ // Now do the same for the interaction
	    int ii   = loop_Vqind(iloop)(iq);
	    double aQ = norm(tmom_v(iq));
	    int id = vindx(ii)[0];
	    int  i = vindx(ii)[1];
	    int vtyp = Vtype(id,i);         // is it counterterm (vtyp!=0) of just coulomb (vtyp==0)
	    double Vq = Vqc(aQ, abs(vtyp)); // This is Vq*(Vq*lambda/(8*pi))^n
	    if (vtyp>=0){
	      V_current(ii) = Vq;
	    }else{
	      // we also add the single-particle counter term in addition to two-particle counter term.
	      int ii_g = single_counter_index(ii);
	      double g_kq = G_current(ii_g);
	      V_current(ii) = Vq;
	      if (g_kq!=0) V_current(ii) += lmbda_spct_cutoffk(-vtyp-1)/g_kq;
	    }
	  }
	  //t_k5.stop();
	}
	PQ = PQ_new;
	PQg = PQg_new;
      }
    } else if (icase==1){  // changing one of the time variables
      Nall_t += 1;
      int itime = 1 + static_cast<int>((Norder-1)*drand()); // which time to change? For bare interaction, there are only Norder different times. but we never change external time here.
      changed_G=0;               // which propagators are being changed?
      times_trial = times;       // times_trial will contain the trial step times.
      {
	double t_new = beta*drand();
	// unscreened interaction, hence changing two times simultaneously!
	for (int ivertex=2*itime; ivertex<=2*itime+1; ++ivertex){ // 2*itime and 2*itime+1 are changed at the same time, because they are equal
	  times_trial(ivertex) = t_new;
	  for (int id=0; id<Ndiags; id++){
	    int i_pre_vertex = i_diagsG(id,ivertex);
	    if (Gindx(id,ivertex)!=std::numeric_limits<sint>().max()) changed_G.set(Gindx(id,ivertex),1);        // these two propagators are changed because of the new time.
	    if (Gindx(id,i_pre_vertex)!=std::numeric_limits<sint>().max()) changed_G.set(Gindx(id,i_pre_vertex),1);
	  }
	}
      }
      if (! Qweight){
	int ip=0;
	for (int ii=0; ii<Ng; ++ii){
	  if (changed_G[ii]){
	    int id = gindx(ii)[0];
	    int  i = gindx(ii)[1];
	    double aK = norm(mom_g(ii));
	    int i_final = diagsG(id,i);
	    double dt = times_trial(i_final)-times_trial(i);
	    g_trial(ip) = Gk(aK, dt);
	    ip_ind(ii)=ip;
	    ip+=1;
	  }
	}
	PQ_new=0; // recalculating PQ, taking into account one change of time.
	for (int id=0; id<Ndiags; id++){
	  double PQd = 1.0;
	  for (int i=1; i<2*Norder; i++){
	    int ii = Gindx(id,i);
	    PQd *= ( changed_G[ii] ? g_trial(ip_ind(ii)) : G_current(ii) );
	  }
	  for (int i=1; i<Norder; i++) PQd *= V_current(Vindx(id,i)); // interaction does not depend on time, hence it is not changed here.
	  PQd *= diagSign(id);
	  PQ_new += PQd;
	  PQg_new(id) = PQd;
	}
      }else{
	PQ_new = V0norm * mweight(amomentm, momentum);
      }
      double ratio = PQ_new/PQ;
      bool accept = fabs(ratio) > 1-drand() && PQ_new!=0;
      if ((itt+1)%Ncout==0 && mpi.rank==mpi.master)
	_PrintInfo_(itt+1, Qweight, icase, amomentm(1), amomentm(Nloops-1), times(2*itime), PQ_new, PQ, dk_hist, Nweight/(Nmeassure+0.0), log);
      if (accept){
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
	      G_current(ii) = Gk( norm(mom_g(ii)),  times(i_final)-times(i)); // these propagators were changed, but they were not computed before.
	    }
	  }
	}
	PQ = PQ_new;
	PQg = PQg_new;
      }
    }else{  // normalization diagram step
      Nall_w += 1;
      if (Qweight){ // trying to jump back into physical hilbert space
	PQ_new=0;
	for (int id=0; id<Ndiags; id++){
	  double PQd = 1.0;
	  for (int i=1; i<2*Norder; i++) PQd *= G_current(Gindx(id,i)); // we have all propagators precomputed, hence it is easy to compute the diagrams.
	  for (int i=1; i<Norder; i++) PQd *= V_current(Vindx(id,i));
	  PQd *= diagSign(id);
	  PQ_new += PQd;
	  PQg_new(id) = PQd;
	}
      }else{        // or if we are in the physical Hilbert space, we try to jump into the normalization diagram
	PQ_new = V0norm * mweight(amomentm, momentum);
      }
      double ratio = PQ_new/PQ;
      bool accept = fabs(ratio) > 1-drand() && PQ_new!=0;
      if ((itt+1)%Ncout==0 && mpi.rank==mpi.master)
	_PrintInfo_(itt+1, Qweight, icase, amomentm(1), amomentm(Nloops-1), times(0), PQ_new, PQ, dk_hist, Nweight/(Nmeassure+0.0),log);
      if (accept){
	Nacc_w += 1;
	Qweight = 1-Qweight;
	PQ = PQ_new;
	PQg = PQg_new;
      }
    }
    // Measuring
    if (itt>=Nwarm && itt%tmeassure==0){
      t_mes.start();
      Nmeassure += 1;
      double sp = sign(PQ);
      if (Qweight){
	Pnorm   += 1;
	Nweight += 1;
      }else{
	Pw0 += sp;
	for (int id=0; id<Ndiags; id++){
	  int ii = Gindx(id,1); // k_1 is external momentum!
	  double Qa = norm(mom_g(ii));
	  Ekin += sp * Qa*Qa * PQg(id)/PQ;
	}
	if (fabs(sum(PQg)/PQ-1)>1e-5) log<<"WARNING : sum(PQg) != PQ. Check PQg !"<<endl;
      }
      if (!Qweight){
	mweight.Add_to_K_histogram(dk_hist, momentum, cutoffk, cutoffk);
      }
      if ( itt>0 && itt%(2000*tmeassure) == 0){
	double last_occurence = (Nweight-Wchanged)/(Nmeassure-Nchanged); // occurence from the last time we changed it
#ifdef _MPI
	double total_occurence=0;
	MPI_Allreduce(&last_occurence, &total_occurence, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	last_occurence = total_occurence/mpi.size;
#endif
	if ( last_occurence > 0.15){ // decrease by two
	  V0norm /= 2.;
	  Pnorm /= 2.;
	  Nchanged = Nmeassure;
	  Wchanged = Nweight;
	  if (mpi.rank==mpi.master) log<<std::left<<(itt/1.0e6)<<"M occurence="<<std::setw(8)<<last_occurence<<" V0 reduced to "<<V0norm<<std::endl;
	}
	if ( last_occurence < 0.03){ // increase by two
	  V0norm *= 2.;
	  Pnorm *= 2.;
	  Nchanged = Nmeassure;
	  Wchanged = Nweight;
	  if (mpi.rank==mpi.master) log<<std::left<<(itt/1.0e6)<<"M occurence="<<std::setw(8)<<last_occurence<<" V0 increased to "<<V0norm<<std::endl;
	}
      }
      if (itt%(500000*tmeassure) == 0){
	if (Nmeassure-Nweight>1000){
#ifdef _MPI	  
	  MPI_Allreduce(MPI_IN_PLACE, mweight.K_hist.data(), mweight.K_hist.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
	  // We can get overflow during MPI for a long run. We should use a constant for normalization.
	  // We could here normalize K_hist (with knorm), and than when we measure, we would
	  // add instead of adding unity, we would add 1./knorm
	  dk_hist *= mweight.Normalize_K_histogram();
	  if (dk_hist < 1e-16) dk_hist = 1.0;
	  mweight.Recompute(mpi.rank==mpi.master);
	  if (Qweight) PQ = V0norm * mweight(amomentm, momentum);
	}
      }
      t_mes.stop();
    }
  }
  t_all.stop();
  Pw0       *= 1.0/Nmeassure;
  Ekin      *= 1.0/Nmeassure;
  Pnorm     *= 1.0/Nmeassure;
  double occurence = static_cast<double>(Nweight)/static_cast<double>(Nmeassure);
  
  double Pw02 = Pw0*Pw0;
  double Ekin2 = Ekin*Ekin;
#ifdef _MPI  
  double dat[6] = {Pnorm, occurence, Pw0, Pw02, Ekin, Ekin2};
  if (mpi.rank==mpi.master){
    MPI_Reduce(MPI_IN_PLACE, dat, 6, MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, mweight.K_hist.data(), mweight.K_hist.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
  }else{
    MPI_Reduce(dat, dat, 6, MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    MPI_Reduce(mweight.K_hist.data(), mweight.K_hist.data(), mweight.K_hist.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
  }
  if (mpi.rank==mpi.master){
    Pnorm     = dat[0]/mpi.size;
    occurence = dat[1]/mpi.size;
    Pw0       = dat[2]/mpi.size;
    Pw02      = dat[3]/mpi.size;
    Ekin      = dat[4]/mpi.size;
    Ekin2     = dat[5]/mpi.size;
    //
    mweight.K_hist *= 1./mpi.size;
  }
#endif
  double sigmaPw=0;
  double sigmaEkin=0;
  if (mpi.rank==mpi.master){
    log<<"total acceptance rate="<<(Nacc_k+Nacc_q+Nacc_t+Nacc_w)/(p.Nitt+0.0)<<" "<<" k-acceptance="<<(Nacc_k/(Nall_k+0.0))<<" t-acceptanece="<<(Nacc_t/(Nall_t+0.0))<<" w-acceptanece="<<(Nacc_w/(Nall_w+0.0))<<std::endl;
    log<<" k-trials="<<(Nall_k/(p.Nitt+0.0))<<" q-trials="<<(Nall_q/(p.Nitt+0.0))<<" t-trials="<<(Nall_t/(p.Nitt+0.0))<<" w-trials="<<(Nall_w/(p.Nitt+0.0))<<std::endl;
    log<<"total time="<<t_all.elapsed()<<" measuring time="<<t_mes.elapsed()<<std::endl;

    double fct = fabs(V0norm)/Pnorm;
    Pw0  *= fct;
    Pw02 *= fct*fct;
    sigmaPw = sqrt(fabs(Pw02 - Pw0*Pw0))/sqrt(mpi.size);

    Ekin *= fct;
    Ekin2 *= fct*fct;
    sigmaEkin = sqrt(fabs(Ekin2-Ekin*Ekin))/sqrt(mpi.size);

    // Proper normalization of the resulting Monte Carlo data
    double norm = ipower( 1/((2*pi)*(2*pi)*(2*pi)), Norder) *  ipower( beta, Norder-1); // one order less, because one G-propagator is missing and one time is missing.
    Pw0     *= norm;
    sigmaPw *= norm;
    Ekin *= norm;
    sigmaEkin *= norm;
    
    log<<"  meassuring diagram occurence frequency="<<occurence<<" and its norm Pnorm="<<Pnorm<<" density="<<Pw0<<" with error "<<sigmaPw<<std::endl;
    for (int ik=0; ik<Nloops; ik++){
      double dsum=0;
      for (int i=0; i<Nbin; i++) dsum += mweight.K_hist(ik,i);
      mweight.K_hist(ik,bl::Range::all()) *= 1./dsum;
      std::ofstream hout((std::string("Q_hist.")+std::to_string(ik)).c_str());
      double cutoff = cutoffk;
      for (int i=0; i<Nbin; i++){
	hout<< (i+0.5)/Nbin*cutoff/kF << " " << mweight.K_hist(ik,i)<<std::endl;
      }
    }
  }
  return std::make_tuple(Pw0,sigmaPw,Ekin,sigmaEkin);
}


template<typename GK>
void sample_static_fastV(bl::Array<double,4>& C_Pln, bl::Array<double,2>& Pbin, std::ostream& log,
			 double lmbda, double lmbda_spct, const GK& Gk, const params& p,
			 const bl::Array<unsigned short,2>& diagsG, const bl::Array<float,1>& diagSign,
			 const std::vector<std::vector<std::vector<int> > >& Loop_index, const std::vector<std::vector<std::vector<int> > >& Loop_type,
			 bl::Array<char,2>& Vtype, const bl::Array<double,1>& kxb,
			 my_mpi& mpi)
{
//typedef double real;
//#define MPI_WNREAL MPI_DOUBLE
  typedef float real;
#define MPI_WNREAL MPI_FLOAT

  /** Should be inside BK **/
  int Nthbin = C_Pln.extent(0);  //8
  int Nlt    = C_Pln.extent(2)-1;//24
  int Nlq    = C_Pln.extent(3)-1;//18
  /** Should be inside BK **/
  
  if ( Nthbin!= p.Nthbin || Nlt!=p.Nlt || Nlq!=p.Nlq){
    std::cerr<<"ERROR : Dimensions of C_Pln is wrong : either "<<Nthbin<<" != "<<p.Nthbin<<" or "<<Nlt<<" != "<<p.Nlt<<" or "<<Nlq<<" != "<<p.Nlq<<std::endl;
    exit(1);
  }
  
  if (Pbin.extent(0)!=p.Nt || Pbin.extent(1)!=p.Nq) Pbin.resize(p.Nt,p.Nq); Pbin=0.0;

  const int Nr=3; // three different steps in this algorithm
  double Pr_norm = (p.Pr[0]+p.Pr[1]+p.Pr[2]); // probability for all four has to be normalized
  bl::TinyVector<double,Nr> Prs;  // cumulative probability for taking a step, needed for bisection below.
  Prs[0] = p.Pr[0]/Pr_norm;
  for (int i=1; i<Nr; i++) Prs[i] = Prs[i-1] + p.Pr[i]/Pr_norm; // prepares cumulative probability from p.Pr

  int Ndiags = diagsG.extent(0);
  if (Ndiags==0) {std::cerr<<"No diagrams to simulate...."<<std::endl; return;}
  int Norder = diagsG.extent(1)/2;
  // computing the inverse of all permutations, to provide fast lookup for the previous fermionic propagator
  bl::Array<unsigned short,2> i_diagsG(Ndiags,2*Norder);
  for (int id=0; id<Ndiags; id++)
    for (int i=0; i<2*Norder; i++)
      i_diagsG(id, static_cast<int>(diagsG(id,i)) )=i;
  
  RanGSL drand(p.iseed); // GSL random number generator
  Timer t_mes, t_all, t_Pl, t_k, t_Q, t_w, t_t;
  
  CounterCoulombV Vqc(lmbda);
  
  int tmeassure=p.tmeassure, Ncout=p.Ncout, Nwarm=p.Nwarm;
  double V0norm = p.V0norm, dkF = p.dkF, kF=p.kF;// V0exp= p.V0exp;
  double cutoffk = p.cutoffk, dRk = p.dRk;
  double beta = p.beta;
  double cutoffq = p.cutoffq;
  int Nt = p.Nt, Nq = p.Nq;

  // Everything for Legendre Polynomials in time and momentum
  LegendrePl Pln(std::max(Nlq,Nlt));
  bl::Array<double,1> pl_Q(Nlq+1);  // Will contain legendre Pl(2*q/qmax-1)
  bl::Array<double,1> pl_t(Nlt+1);  // Will contain legendre Pl(2*t//beta-1)
  bl::Array<double,2> Pl_Pl_tensor_Product(Nlt+1,Nlq+1);
  int Nlqp1 = Nlq+1;
  
  C_Pln = 0.0;                   // cummulative result for legendre expansion
  double Pnorm = 0.0;            // normalization diagram value

  // Here we set Vtype(id,i) => -Vtype(id,i) for those interactions which contain single-particle counter terms.
  if (lmbda_spct!=0) Where_to_Add_Single_Particle_Counter_Term(Vtype, Ndiags, Norder, diagsG);
  // Now that Vtype is properly adjusted for single-particle counter term, we can proceed to find unique propagators.
  typedef unsigned short sint;
  bl::Array<sint,2> Gindx, Vindx;
  bl::Array<bl::TinyVector<int,2>,1> gindx, vindx;
  int N0v=0;
  findUnique_noindx_spct(Gindx, Vindx, gindx, vindx, N0v, diagsG, Vtype, Loop_index, Loop_type, 0, false, false);
  
  // Once more we go through single-particle counter terms and create index single_counter_index
  bl::Array<int,1> single_counter_index;
  if (lmbda_spct!=0) Find_Single_Particle_Counter_G_index(single_counter_index, Ndiags, Norder, vindx.extent(0), diagsG, Vtype, Gindx, Vindx);


  /** Should be inside BK **/
  // This is related to Baym-Kadanoff approach. For efficiency, we want to find which diagram have common values of momenta at the right hand side of the diagram.
  bl::Array<int,1> BKgroups;
  bl::Array<sint,1> BKindex(Ndiags);
  findBaymKadanoffGroups(BKgroups,BKindex,diagsG,Loop_index,Loop_type);
  bl::Array<double,1> PQg(BKgroups.extent(0));
  /*
  std::cout<<"BKgroups:"<<std::endl;
  for (int i=0; i<BKgroups.extent(0); i++){
    std::cout<<i<<" "<<BKgroups(i)<<std::endl;
  }
  std::cout<<"BKindx:"<<std::endl;
  for (int i=0; i<BKindex.extent(0); i++){
    std::cout<<i<<" "<<BKindex(i)<<std::endl;
  }
  */
  /** Should be inside BK **/
  
  if (Loop_index.size()!=Ndiags){ log<<"Wrong size of Loop_index "<<Loop_index.size()<<" instead of "<<Ndiags<<std::endl; exit(1);}
  int Nloops = Loop_index[0].size();
  if (Nloops != Norder+1){ log<<"Expecting Nloops==Norder+1 in Loops_index, but got Nloops="<<Nloops<<" Norder="<<Norder<<std::endl; exit(1);}
  int Ng = gindx.extent(0);
  int Nv = vindx.extent(0);
  int _Nv_ = std::max(Nv,1); // can not assign 0 to empty file

  bl::Array<bl::Array<sint,1>,1> loop_Gkind, loop_Vqind;
  bl::Array<bl::Array<char,1>,1> loop_Gksgn, loop_Vqsgn;
  int Ngp, Nvp;
  // Here we also take into account that single-particle counter terms are sometimes added.
  Get_GVind(loop_Gkind, loop_Gksgn, loop_Vqind, loop_Vqsgn, Ngp, Nvp, Loop_index, Loop_type, Gindx, Vindx, single_counter_index, lmbda_spct);
  int Ngp_tr = GetMax_NGp_trial(i_diagsG, Gindx, Ng, Ngp);



  // DEBUG
  log<<"lmbda_spct="<<lmbda_spct<<std::endl;
  for (int id=0; id<Ndiags; id++){
    for (int j=1; j<Norder; j++){
      if (Vtype(id,j)<0){
	int ii_v = Vindx(id,j);
	int ii_g = single_counter_index(ii_v);
	log<<"id="<<std::setw(3)<<id<<" diag=(";
	for (int k=0; k<Norder*2; k++) log<< diagsG(id,k) <<", ";
	log<<")  V_between=("<<2*j<<","<<2*j+1<<") with ii_v="<<ii_v<<" and ii_g="<<ii_g<<" which comes from id="<<gindx(ii_g)[0]<<" and i=("<<gindx(ii_g)[1]<<"->"<<diagsG(gindx(ii_g)[0],gindx(ii_g)[1])<< ")"<<std::endl;
      }
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

      log<<"iloop="<<iloop<<" iq="<<iq<<" ii_v="<<ii<<" isgn="<<isgn<<" which comes from id="<< id <<" and pair =("<<2*i<<","<<2*i+1<<") of type="<<vtyp<<std::endl;
    }
  }
  // END DEBUG

  
  Reweight_time rw(beta,p.lmbdat);
  int Nbin = 513;
  //int Nbin = 129;
  double lmbda_spct_cutoffk = lmbda_spct*(2*pi*2*pi*2*pi)/(4*pi*ipower(cutoffk,3)/3.);
  //meassureWeight mweight(p.V0exp, p.cutoffk, kF, beta, Nbin);
  meassureWeight mweight(p.V0exp, p.cutoffk, kF, Nbin, Nloops);
  //bl::Array<real,2> K_hist(Nloops,Nbin); K_hist=0;
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
    times(2*it+1) = tv;
  }
  // Next the momenta. For a start with nonzero weight, we want to start close to kF.
  {
    double Qa = p.kF*drand(); // absolute value of external momentum
    momentum(0) = 0.0,0.0,Qa; // external momentum Q
    amomentm(0) = Qa;
    double kf_3 = p.kF/sqrt(3.);
    for (int ik=1; ik<Nloops; ik++){
      momentum(ik) = drand()*kf_3, drand()*kf_3, drand()*kf_3;
      amomentm(ik) = norm(momentum(ik));
    }
  }
  
  bl::Array<double,1> G_current(Ng), V_current(_Nv_);
  // This is momentum for all diagrams and all propagators
  bl::Array<bl::TinyVector<real,3>,1> mom_g(Ng), mom_v(Nv);
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
      for (int i=0; i<2*Norder; i++){
	//int i_final = diagsG(id,i);
	int i_previ = i_diagsG(id,i);
	bl::TinyVector<double,3> k_in = mom_G(id,i_previ);
	bl::TinyVector<double,3> k_out = mom_G(id,i);
	bl::TinyVector<double,3> q = mom_V(id,i/2)*(1-2*(i%2));
	if (fabs(norm(k_in-k_out)-norm(q))>1e-6){
	  log<<"ERROR: diagram "<<id<<" vertex i="<<i<<" k_in="<<k_in<<" k_out="<<k_out<<" q="<<q<<std::endl;
	  log<<"   i_previ="<<i_previ<<" q_i="<< (i/2) <<" mom_V="<<mom_V(id,i/2)<<std::endl;
	}
      }
    }
    /** Should be inside BK **/
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
    /** Should be inside BK **/
    // STOP DEBUGGING
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
    // Finally evaluating the polarizations for all diagrams
    G_current=0;
    for (int ii=0; ii<Ng; ++ii){
      int id = gindx(ii)[0];
      int  i = gindx(ii)[1];
      double aK = norm(mom_G(id,i));
      int i_final = diagsG(id,i);
      double dt = times(i_final)-times(i);
      G_current(ii) = Gk(aK, dt);
    }
    V_current=0;  
    for (int ii=0; ii<Nv; ++ii){
      int id = vindx(ii)[0];
      int  i = vindx(ii)[1];
      double aQ = norm(mom_V(id,i));
      double Vq = Vqc(aQ,abs(Vtype(id,i)));
      if (Vtype(id,i)>=0){
	V_current(ii) = Vq;
      }else{// we also add the single-particle counter term in addition to two-particle counter term.
	int ii_g = single_counter_index(ii);
	double g_kq = G_current(ii_g);
	V_current(ii) = Vq;
	if (g_kq!=0) V_current(ii) += lmbda_spct_cutoffk/g_kq;
      }
    }
    PQ=0;
    /** Should be inside BK **/
    PQg=0;
    /** Should be inside BK **/
    for (int id=0; id<Ndiags; id++){
      double PQd = 1.0;
      for (int i=0; i<2*Norder; i++) PQd *= G_current(Gindx(id,i));
      for (int i=1; i<Norder; i++) PQd *= V_current(Vindx(id,i));
      PQd *= diagSign(id);
      PQ += PQd;
      /** Should be inside BK **/
      PQg(BKindex(id)) += PQd;
      /** Should be inside BK **/
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
  
  Pln.cmp_single(2*amomentm(0)/cutoffq-1., pl_Q, Nlq);          // Legendre polynomials at this Q
  Pln.cmp_single(2*(times(0)-times(1))/beta-1., pl_t, Nlt);     // Legendre polynomials at this time

  bl::Array<double,1> times_trial(2*Norder);
  int _Nvp_ = std::max(Nvp,1); // can not assign 0 to empty file
  bl::Array<bl::TinyVector<real,3>,1> tmom_g(Ngp), tmom_v(_Nvp_);
  bl::Array<sint,1> ip_ind(Ng), iq_ind(Nv);
  bl::Array<double,1> g_trial(Ngp_tr), v_trial(_Nvp_);
  BitArray changed_G(Ng), changed_V(Nv);

  /** Should be inside BK **/
  bl::Array<double,1> PQg_new(BKgroups.extent(0));
  /** Should be inside BK **/
  
  int Qweight = 0;      // if Qweight==1 then we have the normalization diagram
  double Nweight = 0;   // how many times did we see measuring diagram.
  int NlowT = 0;        // how many times do we see external time < 0.05*beta
  int Nmeassure = 0;    // how many times did we measure quantities.
  int Nchanged  = 0;    // when was V0norm last changed?
  double Wchanged = 0;  // what was Nweight when we last time changed V0norm?
  int Nacc_k = 0, Nacc_q = 0, Nacc_w = 0, Nacc_t = 0;
  int Nall_k = 0, Nall_q = 0, Nall_w = 0, Nall_t = 0;
  if (mpi.rank==mpi.master) PrintInfo_(0, Qweight, amomentm(0), amomentm(1), times(0)-times(1), PQ, PQ, 0, 0, log);
  
  t_all.start();
  for (int itt=0; itt<p.Nitt; itt++){
    double PQ_new=0;
    int icase = tBisect(drand(), Prs); // which steps should we attempt?
    if (icase==0){ // changing momentum Q or k-point
      int iloop = static_cast<int>(Nloops*drand());
      bl::TinyVector<double,3> K_new; double Ka_new; double trial_ratio=1;
      bool accept=false;
      if (iloop==0){ // this is the loop for the external momentum Q
	Nall_q += 1;

	/** Maybe Should be inside BK **/
	accept = Find_new_zQ_point(K_new, Ka_new, trial_ratio, momentum(iloop), amomentm(iloop), cutoffq, drand);
	/** Should be inside BK **/
	
      }else{         // any other momentum point
	Nall_k += 1;
	accept = Find_new_k_point(K_new, Ka_new, trial_ratio, momentum(iloop), amomentm(iloop), cutoffk, drand, drand()<p.Qring, dkF, dRk);
      }
      if (accept){
	bl::TinyVector<real,3> dK = K_new - momentum(iloop); // how much momentum will be changed.
	//t1.start(); // takes 40% of the time
	changed_G = 0;
	for (int ip=0; ip<loop_Gkind(iloop).extent(0); ++ip){ // all G-propagators, which are changed when momentum in certain loop is changed.
	  int ii   = loop_Gkind(iloop)(ip);  // index of G-propagators, which we are now changing
	  double isgn = loop_Gksgn(iloop)(ip);  // momentum can be added or subtracted on this propagator.
	  bl::TinyVector<real,3> k = mom_g(ii) + dK * isgn; // actually changing the momentum
	  tmom_g(ip) = k;                    // remember the momentum
	  changed_G.set(ii,1);                   // remember that it is changed
	  if (!Qweight){
	    double aK = norm(k);
	    int id = gindx(ii)[0];           // find first diagram with this propagator
	    int  i = gindx(ii)[1];           // and which propagator is this in the diagram
	    int i_final = diagsG(id,i);      // times in the propagators.
	    double dt = times(i_final)-times(i);
	    g_trial(ip) = Gk(aK, dt);        // remeber the value of the propagator
	    ip_ind(ii)=ip;                   // remeber where it is stored
          }
	}
	//t1.stop();
	//t2.start(); // takes 1.5% of the time
	changed_V = 0;
	for (int iq=0; iq<loop_Vqind(iloop).extent(0); ++iq){ // Now do the same for the interaction
	  int ii   = loop_Vqind(iloop)(iq);
	  double isgn = loop_Vqsgn(iloop)(iq);
	  bl::TinyVector<double,3> q =  mom_v(ii) + dK * isgn;
	  tmom_v(iq) = q;
	  changed_V.set(ii,1);
	  if (!Qweight){
	    double aQ = norm(q);
	    int id = vindx(ii)[0];
	    int  i = vindx(ii)[1];
	    int vtyp = Vtype(id,i);
	    double Vq = Vqc(aQ, abs(vtyp)); // This is Vq^2*lambda/(8*pi)
	    //if (isgn==0 && vtyp>=0){ std::cerr<<"ERROR : isgn==0 and vtype>=0  : Should not happen!"<<std::endl;} 
	    if (vtyp>=0){
	      v_trial(iq) = Vq;
	    }else{
	      // we also add the single-particle counter term in addition to two-particle counter term.
	      int ii_g = single_counter_index(ii);
	      double g_kq = changed_G[ii_g] ? g_trial(ip_ind(ii_g)) : G_current(ii_g);
	      v_trial(iq) = Vq;
	      if (g_kq!=0) v_trial(iq) += lmbda_spct_cutoffk/g_kq;
	    }
	    iq_ind(ii)=iq;
	  }
	}
	//t2.stop();
	if (! Qweight){ // we computed the polarization diagram
	  //t3.start();// takes 22% of the time
	  PQ_new=0;
	  /** Should be inside BK **/
	  PQg_new=0;
	  /** Should be inside BK **/
	  for (int id=0; id<Ndiags; id++){
	    double PQd = 1.0;
	    for (int i=0; i<2*Norder; i++){
	      int ii = Gindx(id,i);
	      PQd *= ( changed_G[ii] ? g_trial(ip_ind(ii)) : G_current(ii) );
	    }
	    for (int i=1; i<Norder; i++){
	      int ii = Vindx(id,i);
	      PQd *= ( changed_V[ii] ? v_trial(iq_ind(ii)) : V_current(ii) );
	    }
	    PQd *=  diagSign(id);
	    PQ_new += PQd;
	    /** Should be inside BK **/
	    PQg_new(BKindex(id)) += PQd;
	    /** Should be inside BK **/
	  }
	  //t3.stop();
	}else{
	  double Ka_old = amomentm(iloop);
	  amomentm(iloop) = Ka_new;
	  bl::TinyVector<double,3> K_old = momentum(iloop);
	  momentum(iloop) = K_new;
	  PQ_new = V0norm * mweight(amomentm, momentum);
	  amomentm(iloop) = Ka_old;
	  momentum(iloop) = K_old;
	}
	double ratio = PQ_new*trial_ratio / PQ;
	accept = fabs(ratio) > 1-drand() && PQ_new!=0;
      }
      if ((itt+1)%Ncout==0 && mpi.rank==mpi.master) _PrintInfo_(itt+1, Qweight, icase, amomentm(1), amomentm(Nloops-1), times(0), PQ_new, PQ, NlowT/(Nmeassure+0.0), Nweight/(Nmeassure+0.0), log);
      if (accept){
	if (iloop==0) Nacc_q += 1; else Nacc_k += 1;
	momentum(iloop) = K_new;  // this momentum was changed
	amomentm(iloop) = Ka_new;
	for (int ip=0; ip<loop_Gkind(iloop).extent(0); ++ip){
	  int ii   = loop_Gkind(iloop)(ip); 
	  mom_g(ii) = tmom_g(ip);
	}
	for (int iq=0; iq<loop_Vqind(iloop).extent(0); ++iq){ // Now do the same for the interaction
	  int ii   = loop_Vqind(iloop)(iq);
	  //int isgn = loop_Vqsgn(iloop)(iq);
	  mom_v(ii) = tmom_v(iq);
	}
	if (!Qweight){
	  for (int ip=0; ip<loop_Gkind(iloop).extent(0); ++ip){ // all G-propagators, which are changed when momentum in certain loop is changed.
	    int ii   = loop_Gkind(iloop)(ip);  // index of G-propagators, which we are now changing
	    G_current(ii)=g_trial(ip);
	  }
	  for (int iq=0; iq<loop_Vqind(iloop).extent(0); ++iq){ // Now do the same for the interaction
	    int ii   = loop_Vqind(iloop)(iq);
	    V_current(ii) = v_trial(iq);
	  }
	}else{
	  // since we have the measuring diagram, we did not calculate G_trial and V_trial before, hence we need to evaluate them now.
	  //t5.start(); // takes 3% of the time
	  for (int ip=0; ip<loop_Gkind(iloop).extent(0); ++ip){ // all G-propagators, which are changed when momentum in certain loop is changed.
	    int ii    = loop_Gkind(iloop)(ip);  // index of G-propagators, which we are now changing
	    double aK = norm(tmom_g(ip));
	    int id = gindx(ii)[0];           // find first diagram with this propagator
	    int  i = gindx(ii)[1];           // and which propagator is this in the diagram
	    int i_final = diagsG(id,i);      // times in the propagators.
	    double dt = times(i_final)-times(i);
	    G_current(ii) = Gk(aK, dt);        // remeber the value of the propagator
	  }
	  for (int iq=0; iq<loop_Vqind(iloop).extent(0); ++iq){ // Now do the same for the interaction
	    int ii   = loop_Vqind(iloop)(iq);
	    double aQ = norm(tmom_v(iq));
	    int id = vindx(ii)[0];
	    int  i = vindx(ii)[1];
	    int vtyp = Vtype(id,i);         // is it counterterm (vtyp!=0) of just coulomb (vtyp==0)
	    double Vq = Vqc(aQ, abs(vtyp)); // This is Vq*(Vq*lambda/(8*pi))^n
	    if (vtyp>=0){
	      V_current(ii) = Vq;
	    }else{
	      // we also add the single-particle counter term in addition to two-particle counter term.
	      int ii_g = single_counter_index(ii);
	      double g_kq = G_current(ii_g);
	      V_current(ii) = Vq;
	      if (g_kq!=0) V_current(ii) += lmbda_spct_cutoffk/g_kq;
	    }
	  }
	  //t5.stop();
	}
	PQ = PQ_new;

/** Should be inside BK **/	
	if (!Qweight) PQg = PQg_new;
/** Should be inside BK **/
	
	//t6.start(); // takes 0.5% time
	if (iloop==0) Pln.cmp_single(2*amomentm(iloop)/cutoffq-1., pl_Q, Nlq);  // update Legendre Polynomials
	//t6.stop();
      }
    } else if (icase==1){  // changing one of the time variables
      Nall_t += 1;
      int itime = static_cast<int>(Norder*drand()); // which time to change? For bare interaction, there are only Norder different times.
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
	for (int ivertex=2*itime; ivertex<=2*itime+1; ++ivertex){ // 2*itime and 2*itime+1 are changed at the same time, because they are equal
	  times_trial(ivertex) = t_new;
	  for (int id=0; id<Ndiags; id++){
	    int i_pre_vertex = i_diagsG(id,ivertex);
	    changed_G.set(Gindx(id,ivertex),1);     // these two propagators are changed because of the new time.
	    changed_G.set(Gindx(id,i_pre_vertex),1);
	  }
	}
      }
      if (! Qweight){
	//t7.start(); // takes 11% of the time
	int ip=0;
	for (int ii=0; ii<Ng; ++ii){
	  if (changed_G[ii]){
	    int id = gindx(ii)[0];
	    int  i = gindx(ii)[1];
	    double aK = norm(mom_g(ii));
	    int i_final = diagsG(id,i);
	    double dt = times_trial(i_final)-times_trial(i);
	    g_trial(ip) = Gk(aK, dt);
	    ip_ind(ii)=ip;
	    ip+=1;
	  }
	}
	//t7.stop();
	//t8.start(); // takes 8% of the time
	PQ_new=0;   // recalculating PQ, taking into account one change of time.
	/** Should be inside BK **/
	PQg_new=0;
	/** Should be inside BK **/
	for (int id=0; id<Ndiags; id++){
	  double PQd = 1.0;
	  for (int i=0; i<2*Norder; i++){
	    int ii = Gindx(id,i);
	    PQd *= ( changed_G[ii] ? g_trial(ip_ind(ii)) : G_current(ii) );
	  }
	  for (int i=1; i<Norder; i++) PQd *= V_current(Vindx(id,i)); // interaction does not depend on time, hence it is not changed here.
	  PQd *= diagSign(id);
	  PQ_new += PQd;
	  /** Should be inside BK **/
	  PQg_new(BKindex(id)) += PQd;
	  /** Should be inside BK **/
	}
	//t8.stop();
      }else{
	PQ_new = V0norm * mweight(amomentm, momentum);
      }
      double ratio = PQ_new/PQ;
      bool accept = fabs(ratio) > 1-drand() && PQ_new!=0;
      if ((itt+1)%Ncout==0 && mpi.rank==mpi.master)
	_PrintInfo_(itt+1, Qweight, icase, amomentm(1), amomentm(Nloops-1), times(2*itime), PQ_new, PQ, NlowT/(Nmeassure+0.0), Nweight/(Nmeassure+0.0),log);
      if (accept){
	//t9.start(); // takes 1.7% of time
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
	      G_current(ii) = Gk( norm(mom_g(ii)),  times(i_final)-times(i)); // these propagators were changed, but they were not computed before.
	    }
	  }
	}
	//t9.stop();
	PQ = PQ_new;
	/** Should be inside BK **/
	if (!Qweight) PQg = PQg_new;
	/** Should be inside BK **/
	//t10.start(); // takes 0.5% of the time
	if (itime==0) Pln.cmp_single(2*(times(0)-times(1))/beta-1., pl_t, Nlt);  // update Legendre Polynomials
	//t10.stop();
      }
    }else{  // normalization diagram step
      Nall_w += 1;
      //t11.start(); // takes 0.3% of the time
      if (Qweight){ // trying to jump back into physical hilbert space
	PQ_new=0;
	/** Should be inside BK **/
	PQg_new=0;
	/** Should be inside BK **/
	for (int id=0; id<Ndiags; id++){
	  double PQd = 1.0;
	  for (int i=0; i<2*Norder; i++) PQd *= G_current(Gindx(id,i)); // we have all propagators precomputed, hence it is easy to compute the diagrams.
	  for (int i=1; i<Norder; i++) PQd *= V_current(Vindx(id,i));
	  PQd *= diagSign(id);
	  PQ_new += PQd;
	  /** Should be inside BK **/
	  PQg_new(BKindex(id)) += PQd;
	  /** Should be inside BK **/
	}
      }else{        // or if we are in the physical Hilbert space, we try to jump into the normalization diagram
	PQ_new = V0norm * mweight(amomentm, momentum);
      }
      double ratio = PQ_new/PQ;
      bool accept = fabs(ratio) > 1-drand() && PQ_new!=0;
      if ((itt+1)%Ncout==0 && mpi.rank==mpi.master)
	  _PrintInfo_(itt+1, Qweight, icase, amomentm(1), amomentm(Nloops-1), times(0), PQ_new, PQ, NlowT/(Nmeassure+0.0), Nweight/(Nmeassure+0.0), log);
      if (accept){
	Nacc_w += 1;
	Qweight = 1-Qweight;
	PQ = PQ_new;
	/** Should be inside BK **/
	if (!Qweight) PQg = PQg_new;
	/** Should be inside BK **/
      }
      //t11.stop();
    }

    //if (itt>=Nwarm && itt%10000==0){
    //  if (!Qweight) VerifyCurrentState10(itt,beta,lmbda_spct,PQ,momentum,times,diagsG,i_diagsG,Vtype,Gindx,Vindx,Gk,diagSign,Loop_index,Loop_type,BKgroups,BKindex,Vqc,G_current,V_current);
    //}
    
    // Measuring
    if (itt>=Nwarm && itt%tmeassure==0){
      t_mes.start(); // takes 1% time
      Nmeassure += 1;
      double Qa = amomentm(0);
      double t = times(0)-times(1);
      double cw = 1./rw.wt(t);
      double sp = sign(PQ) * cw;
      int iq = std::min(int(Qa/cutoffq*Nq), Nq-1);
      int it = std::min(int(t/beta*Nt), Nt-1);

      if (Qweight){
	Pnorm += Qa*Qa * cw;                 // |Q|^2 * reweighting
	Nweight+=1;
      }else{
	Pbin(it,iq) += sp;
	/** Should be inside BK **/
	if (itt%(2000*tmeassure) == 0){ // For efficiency, do not check every time.
	  if ( fabs(sum(PQg)/PQ-1) > 1e-5 ){
	    std::cerr<<"ERROR  PQg and PQ are not equal : PQ="<< PQ << " and PQg="<< sum(PQg) << std::endl;
	  }
	}
#ifndef DGER_BLAS
	{
	  double* restrict p_Q = pl_Q.data();
	  double* restrict p_t = pl_t.data();
	  double* restrict p_p = Pl_Pl_tensor_Product.data();
	  for (int lt=0; lt<=Nlt; ++lt){
	    double t = p_t[lt]*sp;
	    for (int lq=0; lq<=Nlq; ++lq) p_p[lq] = p_Q[lq]*t;
	    p_p += Nlqp1;
	  }
	}
	for (int ig=0; ig<BKgroups.extent(0); ig++){
	  double group_weight = PQg(ig)/PQ;
	  int id_representative = BKgroups(ig);
	  bl::TinyVector<double,3> k = mom_g(Gindx(id_representative,0));
	  double nk = norm(k);
	  double cos_theta = k(2)/nk;
	  int ith = static_cast<int>(Nthbin * 0.5*(cos_theta+1));
	  if (ith>=Nthbin) ith=Nthbin-1;
	  int ik;
	  {
	    int Nkbin = kxb.extent(0)-1;
	    int klo = 0, khi=Nkbin-1;
	    ik = bisection(nk, klo, khi, Nkbin, kxb);
	  }
	  C_Pln(ith,ik,bl::Range::all(),bl::Range::all()) += Pl_Pl_tensor_Product * group_weight;
	}
#else
	int inct=1, incq=1, lda=Nlq+1, n1=Nlq+1, n2=Nlt+1;
	for (int ig=0; ig<BKgroups.extent(0); ig++){
	  double group_weight = PQg(ig)/PQ;
	  int id_representative = BKgroups(ig);
	  bl::TinyVector<double,3> k = mom_g(Gindx(id_representative,0));
	  double nk = norm(k);
	  double cos_theta = k(2)/nk;
	  int ith = static_cast<int>(Nthbin * 0.5*(cos_theta+1));
	  if (ith>=Nthbin) ith=Nthbin-1;
	  int ik;
	  {
	    int Nkbin = kxb.extent(0)-1;
	    int klo = 0, khi=Nkbin-1;
	    ik = bisection(nk, klo, khi, Nkbin, kxb);
	  }
	  double alpha = sp*group_weight;
	  double* restrict pp = &C_Pln(ith,ik,0,0);
	  dger_(&n1, &n2, &alpha, pl_Q.data(), &incq, pl_t.data(), &inct, pp, &lda);
	}
#endif
/** Should be inside BK **/	
      }
      if (t < 0.05*beta){
	NlowT +=1 ;
      }
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
      }
      if ( itt>0 && itt%(2000*tmeassure) == 0){
	double last_occurence = (Nweight-Wchanged)/(Nmeassure-Nchanged); // occurence from the last time we changed it
#ifdef _MPI
	double total_occurence=0;
	MPI_Allreduce(&last_occurence, &total_occurence, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	last_occurence = total_occurence/mpi.size;
#endif	
	if ( last_occurence > 0.15){ // decrease by two
	  V0norm /= 2.;
	  Pnorm /= 2.;
	  Nchanged = Nmeassure;
	  Wchanged = Nweight;
	  if (mpi.rank==mpi.master) log<<std::left<<(itt/1.0e6)<<"M occurence="<<std::setw(8)<<last_occurence<<" V0 reduced to "<<V0norm<<std::endl;
	}
	if ( last_occurence < 0.03){ // increase by two
	  V0norm *= 2.;
	  Pnorm *= 2.;
	  Nchanged = Nmeassure;
	  Wchanged = Nweight;
	  if (mpi.rank==mpi.master) log<<std::left<<(itt/1.0e6)<<"M occurence="<<std::setw(8)<<last_occurence<<" V0 increased to "<<V0norm<<std::endl;
	}
      }
      if (itt%(500000*tmeassure) == 0){
	if (Nmeassure-Nweight>1000){
#ifdef _MPI	  
	  MPI_Allreduce(MPI_IN_PLACE, mweight.K_hist.data(), mweight.K_hist.size(), MPI_WNREAL, MPI_SUM, MPI_COMM_WORLD);
#endif
	  dk_hist *= mweight.Normalize_K_histogram();
	  mweight.Recompute(/*K_hist,*/ mpi.rank==mpi.master);
	  // OLD_BUG : If we are in a meassuring space, we have to update now the value of the meassuring diagram as the weight has changed.
	  if (Qweight) PQ = V0norm * mweight(amomentm, momentum);
	}
      }
      t_mes.stop();
    }
  }
  t_all.stop();
  C_Pln     *= 1.0/Nmeassure;
  Pbin      *= 1.0/Nmeassure;
  Pnorm     *= 1.0/Nmeassure;
  double occurence = static_cast<double>(Nweight)/static_cast<double>(Nmeassure);

  mweight.K_hist *= 1.0/Nmeassure;

#ifdef _MPI  
  double dat[2] = {Pnorm, occurence};
  if (mpi.rank==mpi.master){
    MPI_Reduce(MPI_IN_PLACE, C_Pln.data(), C_Pln.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE,  Pbin.data(),  Pbin.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, dat, 2, MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, mweight.K_hist.data(), mweight.K_hist.size(), MPI_WNREAL, MPI_SUM, mpi.master, MPI_COMM_WORLD);
  }else{
    MPI_Reduce(C_Pln.data(), C_Pln.data(), C_Pln.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    MPI_Reduce( Pbin.data(),  Pbin.data(),  Pbin.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    MPI_Reduce(dat, dat, 2, MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    MPI_Reduce(mweight.K_hist.data(), mweight.K_hist.data(), mweight.K_hist.size(), MPI_WNREAL, MPI_SUM, mpi.master, MPI_COMM_WORLD);
  }
  if (mpi.rank==mpi.master){
    C_Pln *= 1./mpi.size;
    Pbin  *= 1./mpi.size;
    Pnorm = dat[0]/mpi.size;
    occurence = dat[1]/mpi.size;
    //
    mweight.K_hist *= 1./mpi.size;
  }
#endif

  log<<"total acceptance rate="<<(Nacc_k+Nacc_q+Nacc_t+Nacc_w)/(p.Nitt+0.0)<<" "<<" k-acceptance="<<(Nacc_k/(Nall_k+0.0))<<" q-acceptance="<<(Nacc_q/(Nall_q+0.0))<<" t-acceptanece="<<(Nacc_t/(Nall_t+0.0))<<" w-acceptanece="<<(Nacc_w/(Nall_w+0.0))<<std::endl;
  log<<" k-trials="<<(Nall_k/(p.Nitt+0.0))<<" q-trials="<<(Nall_q/(p.Nitt+0.0))<<" t-trials="<<(Nall_t/(p.Nitt+0.0))<<" w-trials="<<(Nall_w/(p.Nitt+0.0))<<std::endl;
  //cout<<"t1="<<t1.elapsed()<<" t2="<<t2.elapsed()<<" t3="<<t3.elapsed()<<" t4="<<t4.elapsed()<<" t5="<<t5.elapsed()<<" t6="<<t6.elapsed()<<" t7="<<t7.elapsed()<<" t8="<<t8.elapsed()<<" t9="<<t9.elapsed()<<" t10="<<t10.elapsed()<<" t11="<<t11.elapsed()<<std::endl;
  log<<"total time="<<t_all.elapsed()<<" measuring time="<<t_mes.elapsed()<<std::endl;
    
  if (mpi.rank==mpi.master){
    C_Pln *= 1.0/(4*pi) * (fabs(V0norm)/Pnorm); // 4*pi because we must divide by 1/(4*pi*q**2), and 1/q**2 was already done in sampling
    Pbin  *= 1.0/(4*pi) * (fabs(V0norm)/Pnorm);
    log<<"  meassuring diagram occurence frequency="<<occurence<<" and its norm Pnorm="<<Pnorm<<std::endl;

    
    // Proper normalization of the resulting Monte Carlo data
    double norm = ipower( beta/((2*pi)*(2*pi)*(2*pi)), Norder) * (4*pi*cutoffq*cutoffq*cutoffq/3);



    /** Should be different inside BK **/
    // Later we will use Nthbin/2.0*Nkbin/2.0  normalization in the Kernel to transform from binned data to function values.
    C_Pln *= (Nthbin/2.0 * norm);
    for (int lt=0; lt<=Nlt; lt++)
      for (int lq=0; lq<=Nlq; lq++)
	C_Pln(bl::Range::all(),bl::Range::all(),lt,lq) *=  ((2*lt+1.)/beta) * ((2.*lq+1.)/cutoffq);
    int Nkbin = kxb.extent(0)-1;
    for (int ik=0; ik<Nkbin; ik++)
      C_Pln(bl::Range::all(),ik,bl::Range::all(),bl::Range::all()) *= 1./(kxb(ik+1)-kxb(ik));
    
    double dq_binning = cutoffq/Nq;
    double dt_binning = beta/Nt;
    Pbin *= (norm/(dq_binning * dt_binning));
    /** Should be inside BK **/
    
    
    for (int ik=0; ik<Nloops; ik++){
      double dsum=0;
      for (int i=0; i<Nbin; i++) dsum += mweight.K_hist(ik,i);
      mweight.K_hist(ik,bl::Range::all()) *= 1./dsum;
      std::ofstream hout((std::string("Q_hist.")+std::to_string(ik)).c_str());
      double cutoff = (ik>0)? cutoffk : cutoffq;
      for (int i=0; i<Nbin; i++){
	hout<< (i+0.5)/Nbin*cutoff/kF << " " << mweight.K_hist(ik,i)<<std::endl;
      }
    }
  }
}


template<int Nlt, int Nlq, typename GK>
void sample_static_fast(bl::Array<double,2>& C_Pln, bl::Array<double,2>& Pbin, std::ostream& log,
			double lmbda, const GK& Gk, const params& p,
			const bl::Array<unsigned short,2>& diagsG, const bl::Array<float,1>& diagSign,
			const std::vector<std::vector<std::vector<int> > >& Loop_index, const std::vector<std::vector<std::vector<int> > >& Loop_type,
			const bl::Array<char,2>& Vtype, my_mpi& mpi)
{
#define MPI_NREAL MPI_DOUBLE
  typedef double real;
//#define MPI_NREAL MPI_FLOAT
//  typedef float real;
  
  const int Nr=3; // three different steps in this algorithm
  double Pr_norm = (p.Pr[0]+p.Pr[1]+p.Pr[2]); // probability for all four has to be normalized
  bl::TinyVector<double,Nr> Prs;  // cumulative probability for taking a step, needed for bisection below.
  Prs[0] = p.Pr[0]/Pr_norm;
  for (int i=1; i<Nr; i++) Prs[i] = Prs[i-1] + p.Pr[i]/Pr_norm; // prepares cumulative probability from p.Pr

  if (Pbin.extent(0)!=p.Nt || Pbin.extent(1)!=p.Nq) Pbin.resize(p.Nt,p.Nq); Pbin=0.0;

  int Ndiags = diagsG.extent(0);
  int Norder = diagsG.extent(1)/2;
  // computing the inverse of all permutations, to provide fast lookup for the previous fermionic propagator
  bl::Array<unsigned short,2> i_diagsG(Ndiags,2*Norder);
  for (int id=0; id<Ndiags; id++)
    for (int i=0; i<2*Norder; i++)
      i_diagsG(id, static_cast<int>(diagsG(id,i)) )=i;
  
  RanGSL drand(p.iseed); // GSL random number generator
  Timer t_mes, t_all, t_Pl, t_k, t_Q, t_w, t_t;
  //Timer t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11;
  
  CounterCoulombV Vqc(lmbda);
  
  int tmeassure=p.tmeassure, Ncout=p.Ncout, Nwarm=p.Nwarm;
  double V0norm = p.V0norm, dkF = p.dkF, kF=p.kF;// V0exp= p.V0exp;
  double cutoffq = p.cutoffq, cutoffk = p.cutoffk, dRk = p.dRk;
  double beta = p.beta;
  int Nt = p.Nt, Nq = p.Nq;

  // Everything for Legendre Polynomials in time and momentum
  LegendreQ<Nlq> Plq(0);
  LegendreQ<Nlt> Plt(0);
  bl::TinyVector<double,Nlq+1> pl_Q;  // Will contain legendre Pl(2*q/qmax-1)
  bl::TinyVector<double,Nlt+1> ul_t;  // Will contain legendre Pl(2*t//beta-1)

  C_Pln = 0.0;                   // cummulative result for legendre expansion
  double Pnorm = 0.0;            // normalization diagram value

  typedef unsigned short sint;
  bl::Array<sint,2> Gindx, Vindx;
  bl::Array<bl::TinyVector<int,2>,1> gindx, vindx;
  int N0v=0;
  findUnique_noindx_spct(Gindx, Vindx, gindx, vindx, N0v, diagsG, Vtype, Loop_index, Loop_type, 0, false, false);
  
  if (Loop_index.size()!=Ndiags) log<<"Wrong size of Loop_index "<<Loop_index.size()<<" instead of "<<Ndiags<<std::endl;
  int Nloops = Loop_index[0].size();
  if (Nloops != Norder+1) log<<"Expecting Nloops==Norder+1 in Loops_index, but got Nloops="<<Nloops<<" Norder="<<Norder<<std::endl;
  int Ng = gindx.extent(0);
  int Nv = vindx.extent(0);
  int _Nv_ = std::max(Nv,1); // can not assign 0 to empty file

  bl::Array<bl::Array<sint,1>,1> loop_Gkind, loop_Vqind;
  bl::Array<bl::Array<char,1>,1> loop_Gksgn, loop_Vqsgn;
  int Ngp, Nvp;
  Get_GVind(loop_Gkind, loop_Gksgn, loop_Vqind, loop_Vqsgn, Ngp, Nvp, Loop_index, Loop_type, Gindx, Vindx);
  int Ngp_tr = GetMax_NGp_trial(i_diagsG, Gindx, Ng, Ngp);

  // DEBUG
  {
    log<<"loop_Vqind:"<<std::endl;
    for (int iloop=0; iloop<Nloops; ++iloop){
      for (int iq=0; iq<loop_Vqind(iloop).extent(0); ++iq){ // Now do the same for the interaction
	int ii   = loop_Vqind(iloop)(iq);
	int isgn = loop_Vqsgn(iloop)(iq);

	int id = vindx(ii)[0];
	int  i = vindx(ii)[1];
	int vtyp = Vtype(id,i);

	log<<"iloop="<<iloop<<" iq="<<iq<<" ii_v="<<ii<<" isgn="<<isgn<<" which comes from id="<< id <<" and pair =("<<2*i<<","<<2*i+1<<") of type="<<vtyp<<std::endl;
      }
    }
  }
  
  Reweight_time rw(beta,p.lmbdat);
  int Nbin = 513;
  //int Nbin = 129;
  //meassureWeight mweight(p.V0exp, p.cutoffk, kF, beta, Nbin);
  meassureWeight mweight(p.V0exp, p.cutoffk, kF, Nbin, Nloops);
  //bl::Array<real,2> K_hist(Nloops,Nbin); K_hist=0;
  double dk_hist=1.0;
  
  // This is momentum for loops
  bl::Array<bl::TinyVector<real,3>,1> momentum(Nloops); // momentum for all loops. Each diagram of the same order has the same number of loops.
  bl::Array<real,1> amomentm(Nloops);               // absolute value of momentum for each loop
  
  // Starting point is set up to be random
  // For the times for all vertices in all the graphs
  bl::Array<real,1> times(2*Norder);
  times(0) = beta*drand(); // external time
  times(1) = 0;            // beginning time
  for (int it=1; it<Norder; it++){
    double tv = beta*drand();
    times(2*it  ) = tv;
    times(2*it+1) = tv;
  }
  // Next the momenta. For a start with nonzero weight, we want to start close to kF.
  {
    double Qa = p.kF*drand(); // absolute value of external momentum
    momentum(0) = 0.0,0.0,Qa; // external momentum Q
    amomentm(0) = Qa;
    double kf_3 = p.kF/sqrt(3.);
    for (int ik=1; ik<Nloops; ik++){
      momentum(ik) = drand()*kf_3, drand()*kf_3, drand()*kf_3;
      amomentm(ik) = norm(momentum(ik));
    }
  }

  std::cout<<"Starting times="<<times<<" and starting momenta="<<momentum<<std::endl;
  
  bl::Array<real,1> G_current(Ng), V_current(_Nv_);
  // This is momentum for all diagrams and all propagators
  bl::Array<bl::TinyVector<real,3>,1> mom_g(Ng), mom_v(Nv);
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
      for (int i=0; i<2*Norder; i++){
	//int i_final = diagsG(id,i);
	int i_previ = i_diagsG(id,i);
	bl::TinyVector<real,3> k_in = mom_G(id,i_previ);
	bl::TinyVector<real,3> k_out = mom_G(id,i);
	bl::TinyVector<real,3> q = mom_V(id,i/2)*(1.-2*(i%2));
	if (fabs(norm(k_in-k_out)-norm(q))>1e-6){
	  log<<"ERROR: diagram "<<id<<" vertex i="<<i<<" k_in="<<k_in<<" k_out="<<k_out<<" q="<<q<<std::endl;
	  log<<"   i_previ="<<i_previ<<" q_i="<< (i/2) <<" mom_V="<<mom_V(id,i/2)<<std::endl;
	}
      }
    }
    // STOP DEBUGGING
    // Finally evaluating the polarizations for all diagrams
    G_current=0;
    for (int ii=0; ii<Ng; ++ii){
      int id = gindx(ii)[0];
      int  i = gindx(ii)[1];
      double aK = norm(mom_G(id,i));
      int i_final = diagsG(id,i);
      double dt = times(i_final)-times(i);
      G_current(ii) = Gk(aK, dt);
    }
    V_current=0;  
    for (int ii=0; ii<Nv; ++ii){
      int id = vindx(ii)[0];
      int  i = vindx(ii)[1];
      double aQ = norm(mom_V(id,i));
      V_current(ii) = Vqc(aQ,Vtype(id,i));
    }
    PQ=0;
    for (int id=0; id<Ndiags; id++){
      double PQd = 1.0;
      for (int i=0; i<2*Norder; i++) PQd *= G_current(Gindx(id,i));
      for (int i=1; i<Norder; i++) PQd *= V_current(Vindx(id,i));
      PQ += PQd * diagSign(id);
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
  
  Plq.cmp_single(2*amomentm(0)/cutoffq-1., pl_Q);          // Legendre polynomials at this Q
  Plt.cmp_single(2*(times(0)-times(1))/beta-1., ul_t);     // Legendre polynomials at this time

  bl::Array<real,1> times_trial(2*Norder);
  int _Nvp_ = std::max(Nvp,1); // can not assign 0 to empty file
  bl::Array<bl::TinyVector<real,3>,1> tmom_g(Ngp), tmom_v(_Nvp_);
  bl::Array<sint,1> ip_ind(Ng), iq_ind(Nv);
  bl::Array<real,1> g_trial(Ngp_tr), v_trial(_Nvp_);
  BitArray changed_G(Ng), changed_V(Nv);
  
  int Qweight = 0;      // if Qweight==1 then we have the normalization diagram
  double Nweight = 0;   // how many times did we see measuring diagram.
  int NlowT = 0;        // how many times do we see external time < 0.05*beta
  int Nmeassure = 0;    // how many times did we measure quantities.
  int Nchanged  = 0;    // when was V0norm last changed?
  double Wchanged = 0;  // what was Nweight when we last time changed V0norm?
  int Nacc_k = 0, Nacc_q = 0, Nacc_w = 0, Nacc_t = 0;
  int Nall_k = 0, Nall_q = 0, Nall_w = 0, Nall_t = 0;
  if (mpi.rank==mpi.master) PrintInfo_(0, Qweight, amomentm(0), amomentm(1), times(0)-times(1), PQ, PQ, 0, 0, log);

  t_all.start();
  for (int itt=0; itt<p.Nitt; itt++){
    double PQ_new=0;
    int icase = tBisect(drand(), Prs); // which steps should we attempt?
    if (icase==0){ // changing momentum Q or k-point
      int iloop = static_cast<int>(Nloops*drand());
      bl::TinyVector<real,3> K_new; real Ka_new; double trial_ratio=1;
      bool accept=false;
      if (iloop==0){ // this is the loop for the external momentum Q
	Nall_q += 1;
	accept = Find_new_Q_point(K_new, Ka_new, trial_ratio, momentum(iloop), amomentm(iloop), cutoffq, drand);
      }else{         // any other momentum point
	Nall_k += 1;
	accept = Find_new_k_point(K_new, Ka_new, trial_ratio, momentum(iloop), amomentm(iloop), cutoffk, drand, drand()<p.Qring, dkF, dRk);
      }
      if (accept){
	bl::TinyVector<real,3> dK = K_new - momentum(iloop); // how much momentum will be changed.
	//t1.start(); // takes 40% of the time
	changed_G = 0;
	for (int ip=0; ip<loop_Gkind(iloop).extent(0); ++ip){ // all G-propagators, which are changed when momentum in certain loop is changed.
	  int ii   = loop_Gkind(iloop)(ip);  // index of G-propagators, which we are now changing
	  double isgn = loop_Gksgn(iloop)(ip);  // momentum can be added or subtracted on this propagator.
	  bl::TinyVector<real,3> k = mom_g(ii) + dK * isgn; // actually changing the momentum
	  tmom_g(ip) = k;                    // remember the momentum
	  changed_G.set(ii,1);                   // remember that it is changed
	  if (!Qweight){
	    double aK = norm(k);
	    int id = gindx(ii)[0];           // find first diagram with this propagator
	    int  i = gindx(ii)[1];           // and which propagator is this in the diagram
	    int i_final = diagsG(id,i);      // times in the propagators.
	    double dt = times(i_final)-times(i);
	    g_trial(ip) = Gk(aK, dt);        // remeber the value of the propagator
	    ip_ind(ii)=ip;                   // remeber where it is stored
          }
	}
	//t1.stop();
	//t2.start(); // takes 1.5% of the time
	changed_V = 0;
	for (int iq=0; iq<loop_Vqind(iloop).extent(0); ++iq){ // Now do the same for the interaction
	  int ii   = loop_Vqind(iloop)(iq);
	  double isgn = loop_Vqsgn(iloop)(iq);
	  bl::TinyVector<real,3> q =  mom_v(ii) + dK * isgn;
	  tmom_v(iq) = q;
	  changed_V.set(ii,1);
	  if (!Qweight){
	    double aQ = norm(q);
	    int id = vindx(ii)[0];
	    int  i = vindx(ii)[1];
	    v_trial(iq) = Vqc(aQ, Vtype(id,i));
	    iq_ind(ii)=iq;
	  }
	}
	//t2.stop();
	if (! Qweight){ // we computed the polarization diagram
	  //t3.start();// takes 22% of the time
	  PQ_new=0;
	  for (int id=0; id<Ndiags; id++){
	    double PQd = 1.0;
	    for (int i=0; i<2*Norder; i++){
	      int ii = Gindx(id,i);
	      PQd *= ( changed_G[ii] ? g_trial(ip_ind(ii)) : G_current(ii) );
	    }
	    for (int i=1; i<Norder; i++){
	      int ii = Vindx(id,i);
	      PQd *= ( changed_V[ii] ? v_trial(iq_ind(ii)) : V_current(ii) );
	    }
	    PQ_new += PQd * diagSign(id);
	  }
	  //t3.stop();
	}else{
	  real Ka_old = amomentm(iloop);
	  amomentm(iloop) = Ka_new;
	  bl::TinyVector<real,3> K_old = momentum(iloop);
	  momentum(iloop) = K_new;
	  PQ_new = V0norm * mweight(amomentm, momentum);
	  amomentm(iloop) = Ka_old;
	  momentum(iloop) = K_old;
	}
	double ratio = PQ_new*trial_ratio / PQ;
	accept = fabs(ratio) > 1-drand() && PQ_new!=0;
      }
      if ((itt+1)%Ncout==0 && mpi.rank==mpi.master) _PrintInfo_(itt+1, Qweight, icase, amomentm(1), amomentm(Nloops-1), times(0), PQ_new, PQ, NlowT/(Nmeassure+0.0), Nweight/(Nmeassure+0.0), log);
      
      if (accept){
	if (iloop==0) Nacc_q += 1; else Nacc_k += 1;
	momentum(iloop) = K_new;  // this momentum was changed
	amomentm(iloop) = Ka_new;
	for (int ip=0; ip<loop_Gkind(iloop).extent(0); ++ip){
	  int ii   = loop_Gkind(iloop)(ip); 
	  mom_g(ii) = tmom_g(ip);
	}
	for (int iq=0; iq<loop_Vqind(iloop).extent(0); ++iq){ // Now do the same for the interaction
	  int ii   = loop_Vqind(iloop)(iq);
	  mom_v(ii) = tmom_v(iq);
	}
	if (!Qweight){
	  for (int ip=0; ip<loop_Gkind(iloop).extent(0); ++ip){ // all G-propagators, which are changed when momentum in certain loop is changed.
	    int ii   = loop_Gkind(iloop)(ip);  // index of G-propagators, which we are now changing
	    G_current(ii)=g_trial(ip);
	  }
	  for (int iq=0; iq<loop_Vqind(iloop).extent(0); ++iq){ // Now do the same for the interaction
	    int ii   = loop_Vqind(iloop)(iq);
	    V_current(ii) = v_trial(iq);
	  }
	}else{
	  // since we have the measuring diagram, we did not calculate G_trial and V_trial before, hence we need to evaluate them now.
	  //t5.start(); // takes 3% of the time
	  for (int ip=0; ip<loop_Gkind(iloop).extent(0); ++ip){ // all G-propagators, which are changed when momentum in certain loop is changed.
	    int ii    = loop_Gkind(iloop)(ip);  // index of G-propagators, which we are now changing
	    double aK = norm(tmom_g(ip));
	    int id = gindx(ii)[0];           // find first diagram with this propagator
	    int  i = gindx(ii)[1];           // and which propagator is this in the diagram
	    int i_final = diagsG(id,i);      // times in the propagators.
	    double dt = times(i_final)-times(i);
	    G_current(ii) = Gk(aK, dt);        // remeber the value of the propagator
	  }
	  for (int iq=0; iq<loop_Vqind(iloop).extent(0); ++iq){ // Now do the same for the interaction
	    int ii   = loop_Vqind(iloop)(iq);
	    double aQ = norm(tmom_v(iq));
	    int id = vindx(ii)[0];
	    int  i = vindx(ii)[1];
	    V_current(ii) = Vqc(aQ, Vtype(id,i));
	  }
	  //t5.stop();
	}
	PQ = PQ_new;
	//t6.start(); // takes 0.5% time
	if (iloop==0) Plq.cmp_single(2*amomentm(iloop)/cutoffq-1., pl_Q);  // update Legendre Polynomials
	//t6.stop();
      }
    } else if (icase==1){  // changing one of the time variables
      Nall_t += 1;
      int itime = static_cast<int>(Norder*drand()); // which time to change? For bare interaction, there are only Norder different times.
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
	for (int ivertex=2*itime; ivertex<=2*itime+1; ++ivertex){ // 2*itime and 2*itime+1 are changed at the same time, because they are equal
	  times_trial(ivertex) = t_new;
	  for (int id=0; id<Ndiags; id++){
	    int i_pre_vertex = i_diagsG(id,ivertex);
	    changed_G.set(Gindx(id,ivertex),1);     // these two propagators are changed because of the new time.
	    changed_G.set(Gindx(id,i_pre_vertex),1);
	  }
	}
      }
      if (! Qweight){
	//t7.start(); // takes 11% of the time
	int ip=0;
	for (int ii=0; ii<Ng; ++ii){
	  if (changed_G[ii]){
	    int id = gindx(ii)[0];
	    int  i = gindx(ii)[1];
	    double aK = norm(mom_g(ii));
	    int i_final = diagsG(id,i);
	    double dt = times_trial(i_final)-times_trial(i);
	    g_trial(ip) = Gk(aK, dt);
	    ip_ind(ii)=ip;
	    ip+=1;
	  }
	}
	//t7.stop();
	//t8.start(); // takes 8% of the time
	PQ_new=0;   // recalculating PQ, taking into account one change of time.
	for (int id=0; id<Ndiags; id++){
	  double PQd = 1.0;
	  for (int i=0; i<2*Norder; i++){
	    int ii = Gindx(id,i);
	    PQd *= ( changed_G[ii] ? g_trial(ip_ind(ii)) : G_current(ii) );
	  }
	  for (int i=1; i<Norder; i++) PQd *= V_current(Vindx(id,i)); // interaction does not depend on time, hence it is not changed here.
	  PQ_new += PQd * diagSign(id);
	}
	//t8.stop();
      }else{
	PQ_new = V0norm * mweight(amomentm, momentum);
      }
      double ratio = PQ_new/PQ;
      bool accept = fabs(ratio) > 1-drand() && PQ_new!=0;
      if ((itt+1)%Ncout==0 && mpi.rank==mpi.master)
	_PrintInfo_(itt+1, Qweight, icase, amomentm(1), amomentm(Nloops-1), times(2*itime), PQ_new, PQ, NlowT/(Nmeassure+0.0), Nweight/(Nmeassure+0.0),log);
      if (accept){
	//t9.start(); // takes 1.7% of time
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
	      G_current(ii) = Gk( norm(mom_g(ii)),  times(i_final)-times(i)); // these propagators were changed, but they were not computed before.
	    }
	  }
	}
	//t9.stop();
	PQ = PQ_new;
	//t10.start(); // takes 0.5% of the time
	if (itime==0) Plt.cmp_single(2*(times(0)-times(1))/beta-1., ul_t);  // update Legendre Polynomials
	//t10.stop();
      }
    }else{  // normalization diagram step
      Nall_w += 1;
      //t11.start(); // takes 0.3% of the time
      if (Qweight){ // trying to jump back into physical hilbert space
	PQ_new=0;
	for (int id=0; id<Ndiags; id++){
	  double PQd = 1.0;
	  for (int i=0; i<2*Norder; i++) PQd *= G_current(Gindx(id,i)); // we have all propagators precomputed, hence it is easy to compute the diagrams.
	  for (int i=1; i<Norder; i++) PQd *= V_current(Vindx(id,i));
	  PQ_new += PQd * diagSign(id);
	}
      }else{        // or if we are in the physical Hilbert space, we try to jump into the normalization diagram
	PQ_new = V0norm * mweight(amomentm, momentum);
      }
      double ratio = PQ_new/PQ;
      bool accept = fabs(ratio) > 1-drand() && PQ_new!=0;
      if ((itt+1)%Ncout==0 && mpi.rank==mpi.master)
	  _PrintInfo_(itt+1, Qweight, icase, amomentm(1), amomentm(Nloops-1), times(0), PQ_new, PQ, NlowT/(Nmeassure+0.0), Nweight/(Nmeassure+0.0), log);
      if (accept){
	Nacc_w += 1;
	Qweight = 1-Qweight;
	PQ = PQ_new;
      }
      //t11.stop();
    }
    // Measuring
    if (itt>=Nwarm && itt%tmeassure==0){
      t_mes.start(); // takes 1% time
      Nmeassure += 1;
      double Qa = amomentm(0);
      double t = times(0)-times(1);
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
      if (!Qweight){
	mweight.Add_to_K_histogram(dk_hist, momentum, cutoffq, cutoffk);
	/*
	int iik = static_cast<int>( amomentm(0)/cutoffq * Nbin);
	if (iik>=Nbin) iik=Nbin-1;
	K_hist(0,iik) += 1.0;
	for (int ik=1; ik<Nloops; ik++){
	  real k = amomentm(ik);
	  int iik = static_cast<int>(k/cutoffk * Nbin);
	  if (iik>=Nbin) iik=Nbin-1;
	  K_hist(ik,iik) += 1./(k*k);
	}
	*/
      }
      if ( itt>2e5/mpi.size && itt%(static_cast<int>(100000/mpi.size+1000)*tmeassure) == 0){
	double last_occurence = (Nweight-Wchanged)/(Nmeassure-Nchanged); // occurence from the last time we changed it
#ifdef _MPI
	double total_occurence=0;
	MPI_Allreduce(&last_occurence, &total_occurence, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	last_occurence = total_occurence/mpi.size;
#endif	
	if ( last_occurence > 0.15){ // decrease by two
	  V0norm /= 2.;
	  Pnorm /= 2.;
	  Nchanged = Nmeassure;
	  Wchanged = Nweight;
	  if (mpi.rank==mpi.master) log<<std::left<<(itt/1.0e6)<<"M occurence="<<std::setw(8)<<last_occurence<<" V0 reduced to "<<V0norm<<std::endl;
	}
	if ( last_occurence < 0.03){ // increase by two
	  V0norm *= 2.;
	  Pnorm *= 2.;
	  Nchanged = Nmeassure;
	  Wchanged = Nweight;
	  if (mpi.rank==mpi.master) log<<std::left<<(itt/1.0e6)<<"M occurence="<<std::setw(8)<<last_occurence<<" V0 increased to "<<V0norm<<std::endl;
	}
      }
      if (itt%(500000*tmeassure) == 0){
	if (Nmeassure-Nweight>1000){
#ifdef _MPI	  
	  MPI_Allreduce(MPI_IN_PLACE, mweight.K_hist.data(), mweight.K_hist.size(), MPI_NREAL, MPI_SUM, MPI_COMM_WORLD);
#endif
	  dk_hist *= mweight.Normalize_K_histogram();
	  mweight.Recompute(/*K_hist,*/ mpi.rank==mpi.master);
	  // OLD_BUG : If we are in a meassuring space, we have to update now the value of the meassuring diagram as the weight has changed.
	  if (Qweight) PQ = V0norm * mweight(amomentm, momentum);
	}
      }
      t_mes.stop();
    }
  }
  t_all.stop();
  C_Pln     *= 1.0/Nmeassure;
  Pbin      *= 1.0/Nmeassure;
  Pnorm     *= 1.0/Nmeassure;
  double occurence = static_cast<double>(Nweight)/static_cast<double>(Nmeassure);

  mweight.K_hist *= 1.0/Nmeassure;

#ifdef _MPI  
  double dat[2] = {Pnorm, occurence};
  if (mpi.rank==mpi.master){
    MPI_Reduce(MPI_IN_PLACE, C_Pln.data(), C_Pln.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE,  Pbin.data(),  Pbin.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, dat, 2, MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, mweight.K_hist.data(), mweight.K_hist.size(), MPI_NREAL, MPI_SUM, mpi.master, MPI_COMM_WORLD);
  }else{
    MPI_Reduce(C_Pln.data(), C_Pln.data(), C_Pln.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    MPI_Reduce( Pbin.data(),  Pbin.data(),  Pbin.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    MPI_Reduce(dat, dat, 2, MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    MPI_Reduce(mweight.K_hist.data(), mweight.K_hist.data(), mweight.K_hist.size(), MPI_NREAL, MPI_SUM, mpi.master, MPI_COMM_WORLD);
  }
  if (mpi.rank==mpi.master){
    C_Pln *= 1./mpi.size;
    Pbin  *= 1./mpi.size;
    Pnorm = dat[0]/mpi.size;
    occurence = dat[1]/mpi.size;
    //
    mweight.K_hist *= 1./mpi.size;
  }
#endif

  log<<"total acceptance rate="<<(Nacc_k+Nacc_q+Nacc_t+Nacc_w)/(p.Nitt+0.0)<<" "<<" k-acceptance="<<(Nacc_k/(Nall_k+0.0))<<" q-acceptance="<<(Nacc_q/(Nall_q+0.0))<<" t-acceptanece="<<(Nacc_t/(Nall_t+0.0))<<" w-acceptanece="<<(Nacc_w/(Nall_w+0.0))<<std::endl;
  log<<" k-trials="<<(Nall_k/(p.Nitt+0.0))<<" q-trials="<<(Nall_q/(p.Nitt+0.0))<<" t-trials="<<(Nall_t/(p.Nitt+0.0))<<" w-trials="<<(Nall_w/(p.Nitt+0.0))<<std::endl;
  //cout<<"t1="<<t1.elapsed()<<" t2="<<t2.elapsed()<<" t3="<<t3.elapsed()<<" t4="<<t4.elapsed()<<" t5="<<t5.elapsed()<<" t6="<<t6.elapsed()<<" t7="<<t7.elapsed()<<" t8="<<t8.elapsed()<<" t9="<<t9.elapsed()<<" t10="<<t10.elapsed()<<" t11="<<t11.elapsed()<<std::endl;
  log<<"total time="<<t_all.elapsed()<<" measuring time="<<t_mes.elapsed()<<std::endl;
    
  if (mpi.rank==mpi.master){
    C_Pln *= 1.0/(4*pi) * (fabs(V0norm)/Pnorm); // 4*pi because we must divide by 1/(4*pi*q**2), and 1/q**2 was already done in sampling
    Pbin  *= 1.0/(4*pi) * (fabs(V0norm)/Pnorm);
    log<<"  meassuring diagram occurence frequency="<<occurence<<" and its norm Pnorm="<<Pnorm<<std::endl;
    
    for (int ik=0; ik<mweight.K_hist.extent(0); ik++){
      double dsum=0;
      for (int i=0; i<Nbin; i++) dsum += mweight.K_hist(ik,i);
      mweight.K_hist(ik,bl::Range::all()) *= 1./dsum;
      std::ofstream hout((std::string("Q_hist.")+std::to_string(ik)).c_str());
      double cutoff = (ik>0)? cutoffk : cutoffq;
      for (int i=0; i<Nbin; i++){
	hout<< (i+0.5)/Nbin*cutoff/kF << " " << mweight.K_hist(ik,i)<<std::endl;
      }
    }
  }
}




template<int Nlt, int Nlq, typename GK>
void sample_static(bl::Array<double,2>& C_Pln, bl::Array<double,2>& Pbin, double lmbda, const GK& Gk,
		   const params& p, const bl::Array<int,2>& diagsG, const bl::Array<double,1>& diagSign,
		   const std::vector<std::vector<std::vector<int> > >& Loop_index, const std::vector<std::vector<std::vector<int> > >& Loop_type,
		   const bl::Array<int,2>& Vtype, const bl::Array<int,1>& indx, my_mpi& mpi)
{
  const int Nr=3; // three different steps in this algorithm
  double Pr_norm = (p.Pr[0]+p.Pr[1]+p.Pr[2]); // probability for all four has to be normalized
  bl::TinyVector<double,Nr> Prs;  // cumulative probability for taking a step, needed for bisection below.
  Prs[0] = p.Pr[0]/Pr_norm;
  for (int i=1; i<Nr; i++) Prs[i] = Prs[i-1] + p.Pr[i]/Pr_norm; // prepares cumulative probability from p.Pr

  if (Pbin.extent(0)!=p.Nt || Pbin.extent(1)!=p.Nq) Pbin.resize(p.Nt,p.Nq); Pbin=0.0;

  //cout<<"Vtype="<<Vtype<<std::endl;
  //std::cout<<"indx="<<indx<<std::endl;
  
  int Ndiags = diagsG.extent(0);  int Norder = diagsG.extent(1)/2;
  // computing the inverse of all permutations, to provide fast lookup for the previous fermionic propagator
  bl::Array<int,2> i_diagsG(Ndiags,2*Norder);
  for (int id=0; id<Ndiags; id++)
    for (int i=0; i<2*Norder; i++)
      i_diagsG(id,diagsG(id,i))=i;
  
  RanGSL drand(p.iseed); // GSL random number generator
  Timer t_mes, t_all, t_Pl, t_k, t_Q, t_w, t_t;
  
  //ScreenedCoulombV Vq(lmbda);
  CounterCoulombV Vqc(lmbda);
  
  int tmeassure=p.tmeassure, Ncout=p.Ncout, Nwarm=p.Nwarm;
  double V0norm = p.V0norm, dkF = p.dkF, kF=p.kF;// V0exp= p.V0exp;
  double cutoffq = p.cutoffq, cutoffk = p.cutoffk, dRk = p.dRk;
  double beta = p.beta;
  int Nt = p.Nt, Nq = p.Nq;
  
  
  LegendreQ<Nlq> Plq(0);
  LegendreQ<Nlt> Plt(0);
  bl::TinyVector<double,Nlq+1> pl_Q;  // Will contain legendre Pl(2*q/qmax-1)
  bl::TinyVector<double,Nlt+1> ul_t;

  C_Pln = 0.0;                   // cummulative result for legendre expansion
  double Pnorm = 0.0;            // normalization diagram value
  
  if (Loop_index.size()!=Ndiags) std::cerr<<"Wrong size of Loop_index "<<Loop_index.size()<<" instead of "<<Ndiags<<std::endl;
  int Nloops = Loop_index[0].size();
  if (Nloops != Norder+1) std::cerr<<"Expecting Nloops==Norder+1 in Loops_index, but got Nloops="<<Nloops<<" Norder="<<Norder<<std::endl;

  Reweight_time rw(beta,p.lmbdat);
  int Nbin = 513;
  //int Nbin = 129;
  //meassureWeight mweight(p.V0exp, p.cutoffk, kF, beta, Nbin);
  meassureWeight mweight(p.V0exp, p.cutoffk, kF, Nbin, Nloops);
  //bl::Array<double,2> K_hist(Nloops,Nbin); K_hist=0;
  //bl::Array<double,2> T_hist((Norder-1)*2,Nbin); T_hist=0;
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
  for (int it=1; it<Norder; it++){
    double tv = beta*drand();
    times(2*it  ) = tv;
    times(2*it+1) = tv;
  }
  // Next the momenta
  {
    double Qa = p.kF*drand(); // absolute value of external momentum
    momentum(0) = Qa,0.0,0.0; // external momentum Q
    amomentm(0) = Qa;
    double kf_3 = p.kF/sqrt(3.);
    for (int ik=1; ik<Nloops; ik++){
      momentum(ik) = drand()*kf_3, drand()*kf_3, drand()*kf_3;
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
  // DEBUGGING
  for (int id=0; id<Ndiags; id++){
    for (int i=0; i<2*Norder; i++){
      //int i_final = diagsG(id,i);
      int i_previ = i_diagsG(id,i);
      bl::TinyVector<double,3> k_in = mom_G(id,i_previ);
      bl::TinyVector<double,3> k_out = mom_G(id,i);
      bl::TinyVector<double,3> q = mom_V(id,i/2)*(1.-2*(i%2));
      if (fabs(norm(k_in-k_out-q))>1e-6){
	std::cerr<<"ERROR : diagram "<<id<<" vertex i="<<i<<" k_in="<<k_in<<" k_out="<<k_out<<" q="<<q<<std::endl;
      }
    }
  }
  // DEBUGGING
  // Finally evaluating the polarizations for all diagrams
  bl::Array<double,2> G_current(Ndiags,2*Norder), V_current(Ndiags,Norder);
  G_current=0; V_current=0;
  double PQ=0;
  for (int id=0; id<Ndiags; id++){
    double PQd = 1.0;
    for (int i=0; i<2*Norder; i++){
      double aK = norm(mom_G(id,i));
      int i_final = diagsG(id,i);
      double dt = times(i_final)-times(i);
      G_current(id,i) = Gk(aK, dt);
      PQd *= G_current(id,i);
    }
    for (int i=1; i<Norder; i++){ // do not add V for the meassuring line
      double aQ = norm(mom_V(id,i));
      V_current(id,i) = Vqc(aQ,Vtype(id,i));
      PQd *= V_current(id,i);
    }
    PQ += PQd * diagSign(id);
  }

  // DEBUGGING
  for (int id=0; id<Ndiags; id++){
    for (int i=0; i<2*Norder; i++)
      if (fabs(G_current(id,i))==0){
	std::cerr<<"G_current("<<id<<","<<i<<") not set!"<<std::endl;
      }
    for (int i=1; i<Norder; i++)
      if (fabs(V_current(id,i))==0){
	std::cerr<<"V_current("<<id<<","<<i<<") not set!"<<std::endl;
      }
  }
  // DEBUGGING
  
  Plq.cmp_single(2*amomentm(0)/cutoffq-1., pl_Q); // Legendre polynomials at this Q
  Plt.cmp_single(2*(times(0)-times(1))/beta-1., ul_t);     // Legendre polynomials at this time

  bl::Array<bl::TinyVector<double,3>,2> tmom_G(Ndiags,2*Norder); // momentum for 2*Norder fermionic propagators
  bl::Array<bl::TinyVector<double,3>,2> tmom_V(Ndiags,  Norder); // momentum for Norder bosonic propagators
  bl::Array<int,2> changed_G(Ndiags,2*Norder), changed_V(Ndiags,Norder);
  bl::Array<double,2> G_trial(Ndiags,2*Norder), V_trial(Ndiags,  Norder);
  bl::Array<double,1> times_trial(2*Norder);
    
  int Qweight = 0;      // if Qweight==1 then we have the normalization diagram
  double Nweight = 0;   // how many times did we see measuring diagram.
  int NlowT = 0;        // how many times do we see external time < 0.05*beta
  int Nmeassure = 0;    // how many times did we measure quantities.
  //double aver_sign = 0; // sign
  int Nacc_k = 0, Nacc_q = 0, Nacc_w = 0, Nacc_t = 0;
  int Nall_k = 0, Nall_q = 0, Nall_w = 0, Nall_t = 0;
  if (mpi.rank==mpi.master) PrintInfo_(0, Qweight, amomentm(0), amomentm(1), times(0)-times(1), PQ, PQ, 0, 0);

  t_all.start();
  for (int itt=0; itt<p.Nitt; itt++){
    double PQ_new=0;
    int icase = tBisect(drand(), Prs); // which steps should we attempt?
    if (icase==0){ // changing momentum Q or k-point
      int iloop = static_cast<int>(Nloops*drand());
      bl::TinyVector<double,3> K_new; double Ka_new, trial_ratio=1;
      bool accept=false;
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
	changed_G = 0; changed_V = 0;
	for (int id=0; id<Ndiags; id++){ // updating momentum in all propagators
	  const std::vector<int>& lindex = Loop_index[id][iloop];
	  const std::vector<int>& ltype  = Loop_type[id][iloop];
	  for (int i=0; i<lindex.size(); i++){
	    if ( abs(ltype[i])==1 ){
	      tmom_G(id, lindex[i]) += dK * dsign(ltype[i]);
	      changed_G(id, lindex[i])=1;
	    }else{
	      tmom_V(id,lindex[i]) += dK * dsign(ltype[i]);
	      changed_V(id, lindex[i])=1;
	    }
	  }
	}
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
	    for (int i=1; i<Norder; i++){ // do not add V for the meassuring line
	      if (changed_V(id,i)){
		double aQ = norm(tmom_V(id,i));
		V_trial(id,i) = Vqc(aQ, Vtype(id,i));
		PQd *= V_trial(id,i);
	      }else{
		PQd *= V_current(id,i);
	      }
	    }
	    PQ_new += PQd * diagSign(id);
	  }
	}else{
	  double Ka_old = amomentm(iloop);
	  amomentm(iloop) = Ka_new;
	  bl::TinyVector<double,3> K_old = momentum(iloop);
	  momentum(iloop) = K_new;
	  PQ_new = V0norm * mweight(amomentm, momentum/*,times*/);
	  amomentm(iloop) = Ka_old;
	  momentum(iloop) = K_old;
	  //std::cout<<"PQ_meassure="<<PQ_new<<" k="<<norm(momentum(1))/kF<<std::endl;
	}
	double ratio = PQ_new*trial_ratio / PQ;
	accept = fabs(ratio) > 1-drand() && PQ_new!=0;
	//if (ratio<0) clog<<"SIGN PROBLEM at q!"<<std::endl;
      }
      if ((itt+1)%Ncout==0){
	if (mpi.rank==mpi.master) _PrintInfo_(itt+1, Qweight, icase, amomentm(1), amomentm(Nloops-1), times(0), PQ_new, PQ, NlowT/(Nmeassure+0.0), Nweight/(Nmeassure+0.0));
      }
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
	    for (int i=1; i<Norder; i++) // do not add V for the meassuring line
	      if (changed_V(id,i))
		V_current(id,i) = V_trial(id,i); // The interactions in the loop have been recalculated and stored.
	  }
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
	    for (int i=1; i<Norder; i++)
	      if (changed_V(id,i)){
		double aQ = norm(tmom_V(id,i));
		V_current(id,i) = Vqc(aQ, Vtype(id,i));
	      }
	  }
	}
	//if ((itt+1)%Ncout==0) PrintInfo_(itt+1, Qweight, amomentm(0), amomentm(1), times(0), PQ_new, PQ, NlowT/(Nmeassure+0.0), Nweight/(Nmeassure+0.0));
	PQ = PQ_new;
	if (iloop==0) Plq.cmp_single(2*amomentm(iloop)/cutoffq-1., pl_Q);  // update Legendre Polynomials
      }
    } else if (icase==1){  // changing one of the time variables
      Nall_t += 1;
      int itime = static_cast<int>(Norder*drand()); // which time to change? For bare interaction, there are only Norder different times.
      changed_G=0; changed_V=0;  // which propagators are being changed?
      times_trial = times;       // times_trial will contain the trial step times.
      if (itime==0){           // this is the measuring time with vertex=0
	double t_new = rw.gtau(drand());
	if (t_new<1e-15) t_new = 1e-15; // to be on the right branchcut
	times_trial(0) = t_new;
	int ivertex=0;
	for (int id=0; id<Ndiags; id++){
	  int i_pre_vertex = i_diagsG(id,ivertex);
	  changed_G(id,ivertex)=1;         // these two propagators contain vertex=0.
	  changed_G(id,i_pre_vertex)=1;
	}
      }else{
	double t_new = beta*drand();
	// unscreened interaction, hence changing two times simultaneously!
	for (int ivertex=2*itime; ivertex<=2*itime+1; ++ivertex){ // 2*itime and 2*itime+1 are changed at the same time, because they are equal
	  times_trial(ivertex) = t_new;
	  for (int id=0; id<Ndiags; id++){
	    int i_pre_vertex = i_diagsG(id,ivertex);
	    changed_G(id,ivertex)=1;        // these two propagators are changed because of the new time.
	    changed_G(id,i_pre_vertex)=1;
	  }
	}
      }
      if (! Qweight){
	PQ_new=0; // recalculating PQ, taking into account one change of time.
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
	  for (int i=1; i<Norder; i++) PQd *= V_current(id,i); // interaction does not depend on time, hence it is not changed here.
	  PQ_new += PQd * diagSign(id);
	}
      }else{
	PQ_new = V0norm * mweight(amomentm, momentum/*, times_trial*/);
	//std::cout<<"PQ_meassure="<<PQ_new<<" k="<<norm(momentum(1))/kF<<std::endl;
      }
      double ratio = PQ_new/PQ;
      bool accept = fabs(ratio) > 1-drand() && PQ_new!=0;
      if ((itt+1)%Ncout==0){
	if (mpi.rank==mpi.master) _PrintInfo_(itt+1, Qweight, icase, amomentm(1), amomentm(Nloops-1), times(2*itime), PQ_new, PQ, NlowT/(Nmeassure+0.0), Nweight/(Nmeassure+0.0));
      }
      //if (ratio<0) clog<<"SIGN PROBLEM at time with Qweight="<<Qweight<<std::endl;
      if (accept){
	Nacc_t +=1 ;
	times = times_trial;  // the new times are now accepted. We change the independent time variable, or two internal times.
	if (!Qweight){
	  for (int id=0; id<Ndiags; id++)
	    for (int i=0; i<2*Norder; i++)
	      if (changed_G(id,i))
		G_current(id,i) = G_trial(id,i);  // just saving the propagators which were changed.
	}else{
	  for (int id=0; id<Ndiags; id++)
	    for (int i=0; i<2*Norder; i++)
	      if (changed_G(id,i)){
		int i_final = diagsG(id,i);
		G_current(id,i) = Gk( norm(mom_G(id,i)),  times(i_final)-times(i)); // these propagators were changed, but they were not computed before.
	      }
	}
	//if ((itt+1)%Ncout==0) PrintInfo_(itt+1, Qweight, amomentm(0), amomentm(1), times(2*itime), PQ_new, PQ, NlowT/(Nmeassure+0.0), Nweight/(Nmeassure+0.0));
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
	  for (int i=1; i<Norder; i++) PQd *= V_current(id,i);
	  PQ_new += PQd * diagSign(id);
	}
      }else{        // or if we are in the physical Hilbert space, we try to jump into the normalization diagram
	PQ_new = V0norm * mweight(amomentm, momentum/*, times*/);
	//std::cout<<"PQ_meassure="<<PQ_new<<" k="<<norm(momentum(1))/kF<<std::endl;
      }
      double ratio = PQ_new/PQ;
      bool accept = fabs(ratio) > 1-drand() && PQ_new!=0;
      if ((itt+1)%Ncout==0){
	if (mpi.rank==mpi.master) _PrintInfo_(itt+1, Qweight, icase, amomentm(1), amomentm(Nloops-1), times(0), PQ_new, PQ, NlowT/(Nmeassure+0.0), Nweight/(Nmeassure+0.0));
      }
      if (accept){
	Nacc_w += 1;
	Qweight = 1-Qweight;
	PQ = PQ_new;
      }
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
      //aver_sign += sp;

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
	/*
	for (int it=2; it<2*Norder; it++){
	  int iit = static_cast<int>(times(it)/beta * Nbin);
	  if (iit>=Nbin) iit=Nbin-1;
	  T_hist(it-2,iit) += 1;
	}
	*/
      }
      if ( itt>5e5 && itt%(50000*tmeassure) == 0){
	double occurence = static_cast<double>(Nweight)/static_cast<double>(Nmeassure);
#ifdef _MPI
	double total_occurence=0;
	MPI_Allreduce(&occurence, &total_occurence, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	occurence = total_occurence/mpi.size;
	//std::cout<<"Now occurence at "<<mpi.rank<<" = "<<occurence<<std::endl;
#endif	
	if ( occurence > 0.25){ // decrease by two
	  V0norm /= 2.;
	  Pnorm /= 2.;
	  if (mpi.rank==mpi.master) std::cout<<"occurence="<<occurence<<" V0 reduced to "<<V0norm<<std::endl;
	}
	if ( occurence < 0.02){ // increase by two
	  V0norm *= 2.;
	  Pnorm *= 2.;
	  if (mpi.rank==mpi.master) std::cout<<"occurence="<<occurence<<" V0 increased to "<<V0norm<<std::endl;
	}
      }
      if (itt%(500000*tmeassure) == 0){
	if (Nmeassure-Nweight>1000){
#ifdef _MPI	  
	  MPI_Allreduce(MPI_IN_PLACE, mweight.K_hist.data(), mweight.K_hist.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
	  dk_hist *= mweight.Normalize_K_histogram();
	  mweight.Recompute(/*mweight.K_hist,*/ mpi.rank==mpi.master);
	  if (Qweight) PQ = V0norm * mweight(amomentm, momentum/*, times*/);	  
	}
      }
      t_mes.stop();
    }
  }
  t_all.stop();
  C_Pln     *= 1.0/Nmeassure;
  Pbin      *= 1.0/Nmeassure;
  Pnorm     *= 1.0/Nmeassure;
  double occurence = static_cast<double>(Nweight)/static_cast<double>(Nmeassure);

  mweight.K_hist *= 1.0/Nmeassure;
  //T_hist *= 1.0/Nmeassure;

#ifdef _MPI  
  double dat[2] = {Pnorm, occurence};
  if (mpi.rank==mpi.master){
    MPI_Reduce(MPI_IN_PLACE, C_Pln.data(), C_Pln.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE,  Pbin.data(),  Pbin.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, dat, 2, MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, mweight.K_hist.data(), mweight.K_hist.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    //MPI_Reduce(MPI_IN_PLACE, T_hist.data(), T_hist.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
  }else{
    MPI_Reduce(C_Pln.data(), C_Pln.data(), C_Pln.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    MPI_Reduce( Pbin.data(),  Pbin.data(),  Pbin.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    MPI_Reduce(dat, dat, 2, MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    MPI_Reduce(mweight.K_hist.data(), mweight.K_hist.data(), mweight.K_hist.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    //MPI_Reduce(T_hist.data(), T_hist.data(), T_hist.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
  }
  if (mpi.rank==mpi.master){
    C_Pln *= 1./mpi.size;
    Pbin  *= 1./mpi.size;
    Pnorm = dat[0]/mpi.size;
    occurence = dat[1]/mpi.size;
    //
    mweight.K_hist *= 1./mpi.size;
    //T_hist *= 1./mpi.size;
  }
#endif
  
  std::clog<<"total acceptance rate="<<(Nacc_k+Nacc_q+Nacc_t+Nacc_w)/(p.Nitt+0.0)<<" "<<" k-acceptance="<<(Nacc_k/(Nall_k+0.0))<<" q-acceptance="<<(Nacc_q/(Nall_q+0.0))<<" t-acceptanece="<<(Nacc_t/(Nall_t+0.0))<<" w-acceptanece="<<(Nacc_w/(Nall_w+0.0))<<std::endl;
  std::clog<<" k-trials="<<(Nall_k/(p.Nitt+0.0))<<" q-trials="<<(Nall_q/(p.Nitt+0.0))<<" t-trials="<<(Nall_t/(p.Nitt+0.0))<<" w-trials="<<(Nall_w/(p.Nitt+0.0))<<std::endl;
  std::clog<<"total time="<<t_all.elapsed()<<" measuring time="<<t_mes.elapsed()<<std::endl;
    
  if (mpi.rank==mpi.master){
    C_Pln *= 1.0/(4*pi) * (fabs(V0norm)/Pnorm); // 4*pi because we must divide by 1/(4*pi*q**2), and 1/q**2 was already done in sampling
    Pbin  *= 1.0/(4*pi) * (fabs(V0norm)/Pnorm);
  
    std::clog<<"  meassuring diagram occurence frequency="<<occurence<<" and its norm Pnorm="<<Pnorm<<std::endl;
    //log<<" legendre time="<<t_Pl.elapsed()<<" t_k="<<t_k.elapsed()<<" t_Q="<<t_Q.elapsed()<<" t_t="<<t_t.elapsed()<<" t_w="<<t_w.elapsed()<<std::endl;

    for (int ik=0; ik<mweight.K_hist.extent(0); ik++){
      double dsum=0;
      for (int i=0; i<Nbin; i++) dsum += mweight.K_hist(ik,i);
      mweight.K_hist(ik,bl::Range::all()) *= 1./dsum;
      std::ofstream hout((std::string("Q_hist.")+std::to_string(ik)).c_str());
      double cutoff = (ik>0)? cutoffk : cutoffq;
      for (int i=0; i<Nbin; i++){
	hout<< (i+0.5)/Nbin*cutoff/kF << " " << mweight.K_hist(ik,i)<<std::endl;
      }
    }
    /*
    for (int it=2; it<2*Norder; it++){
      ofstream hout((string("T_hist.")+to_string(it)).c_str());
      for (int i=0; i<Nbin; i++){
	hout<< (i+0.5)/Nbin*beta << " " << T_hist(it-2,i)<<std::endl;
      }
    }
    */
  }
}


template<int Nlt, int Nlq, typename GK>
void sample_static_fast_2(bl::Array<double,2>& C_Pln, bl::Array<double,2>& Pbin, double lmbda, const GK& Gk,
			  const params& p, const bl::Array<int,2>& diagsG, const bl::Array<double,1>& diagSign,
			  const std::vector<std::vector<std::vector<int> > >& Loop_index, const std::vector<std::vector<std::vector<int> > >& Loop_type,
			  const bl::Array<int,2>& Vtype, bl::Array<int,1>& indx)
{
  const int Nr=3; // three different steps in this algorithm
  double Pr_norm = (p.Pr[0]+p.Pr[1]+p.Pr[2]); // probability for all four has to be normalized
  bl::TinyVector<double,Nr> Prs;  // cumulative probability for taking a step, needed for bisection below.
  Prs[0] = p.Pr[0]/Pr_norm;
  for (int i=1; i<Nr; i++) Prs[i] = Prs[i-1] + p.Pr[i]/Pr_norm; // prepares cumulative probability from p.Pr


  if (Pbin.extent(0)!=p.Nt || Pbin.extent(1)!=p.Nq) Pbin.resize(p.Nt,p.Nq); Pbin=0.0;

  std::cout<<"Vtype="<<Vtype<<std::endl;
  std::cout<<"indx="<<indx<<std::endl;
  
  int Ndiags = diagsG.extent(0);
  int Norder = diagsG.extent(1)/2;
  // computing the inverse of all permutations, to provide fast lookup for the previous fermionic propagator
  bl::Array<int,2> i_diagsG(Ndiags,2*Norder);
  for (int id=0; id<Ndiags; id++)
    for (int i=0; i<2*Norder; i++)
      i_diagsG(id,diagsG(id,i))=i;
  
  RanGSL drand(p.iseed); // GSL random number generator
  Timer t_mes, t_all, t_Pl, t_k, t_Q, t_w, t_t;
  //Timer t_k1, t_k2, t_k3, t_k4, t_k5, t_k6;
  //Timer t_t1, t_t2, t_t3, t_t4, t_t5, t_t6;
  
  CounterCoulombV Vqc(lmbda);
  
  int tmeassure=p.tmeassure, Ncout=p.Ncout, Nwarm=p.Nwarm;
  double V0norm = p.V0norm, dkF = p.dkF, kF=p.kF;// V0exp= p.V0exp;
  double cutoffq = p.cutoffq, cutoffk = p.cutoffk, dRk = p.dRk;
  double beta = p.beta;
  int Nt = p.Nt, Nq = p.Nq;
  
  
  LegendreQ<Nlq> Plq(0);
  LegendreQ<Nlt> Plt(0);
  bl::TinyVector<double,Nlq+1> pl_Q;  // Will contain legendre Pl(2*q/qmax-1)
  bl::TinyVector<double,Nlt+1> ul_t;

  C_Pln = 0.0;                   // cummulative result for legendre expansion
  double Pnorm = 0.0;            // normalization diagram value

  bl::Array<int,2> Gindx, Vindx;
  bl::Array<bl::TinyVector<int,2>,1> gindx, vindx;
  findUnique(Gindx, Vindx, gindx, vindx, diagsG, Loop_index, Loop_type, Vtype, indx);
  std::cout<<"Gindx="<<Gindx<<std::endl;
  std::cout<<"Vindx="<<Vindx<<std::endl;
  std::cout<<"gindx="<<gindx<<std::endl;
  std::cout<<"vindx="<<vindx<<std::endl;
  std::vector<std::vector<int>> idiag;
  invert_indx(idiag, indx);
  int Ngrps = idiag.size();
  for (int l=0; l<Ngrps; l++){
    std::cout<<l<<" : ";
    for (int i=0; i<idiag[l].size(); i++){
      std::cout<<idiag[l][i]<<" ";
    }
    std::cout<<std::endl;
  }

  
  if (Loop_index.size()!=Ndiags) std::cerr<<"Wrong size of Loop_index "<<Loop_index.size()<<" instead of "<<Ndiags<<std::endl;
  int Nloops = Loop_index[0].size();
  if (Nloops != Norder+1) std::cerr<<"Expecting Nloops==Norder+1 in Loops_index, but got Nloops="<<Nloops<<" Norder="<<Norder<<std::endl;

  Reweight_time rw(beta,p.lmbdat);
  int Nbin = 513;
  //int Nbin = 129;
  //meassureWeight mweight(p.V0exp, p.cutoffk, kF, beta, Nbin);
  meassureWeight mweight(p.V0exp, p.cutoffk, kF, Nbin, Nloops);
  bl::Array<double,2> K_hist(Nloops,Nbin); K_hist=0;
  bl::Array<double,2> T_hist((Norder-1)*2,Nbin); T_hist=0;
  
  // This is momentum for loops
  bl::Array<bl::TinyVector<double,3>,2> momentum(Ngrps,Nloops); // momentum for all loops. Each diagram of the same order has the same number of loops.
  bl::Array<double,2> amomentm(Ngrps,Nloops);               // absolute value of momentum for each loop
  // This is momentum for all diagrams and all propagators
  bl::Array<bl::TinyVector<double,3>,2> mom_G(Ndiags,2*Norder); // momentum for 2*Norder fermionic propagators
  bl::Array<bl::TinyVector<double,3>,2> mom_V(Ndiags,  Norder); // momentum for Norder bosonic propagators
  
  // Starting point is set up to be random
  // For the times for all vertices in all the graphs
  bl::Array<double,2> times(Ngrps,2*Norder);
  times(0,0) = beta*drand(); // external time
  times(0,1) = 0;            // beginning time
  for (int it=1; it<Norder; it++){
    double tv = beta*drand();
    times(0,2*it  ) = tv;
    times(0,2*it+1) = tv;
  }
  for (int ig=1; ig<Ngrps; ++ig)
    for (int it=0; it<2*Norder; ++it)
      times(ig,it) = times(0,it);
  
  // Next the momenta
  {
    double Qa = p.kF*drand(); // absolute value of external momentum
    momentum(0,0) = Qa,0.0,0.0; // external momentum Q
    amomentm(0,0) = Qa;
    double kf_3 = p.kF/sqrt(3.);
    for (int ik=1; ik<Nloops; ik++){
      momentum(0,ik) = drand()*kf_3, drand()*kf_3, drand()*kf_3;
      amomentm(0,ik) = norm(momentum(0,ik));
    }
    for (int ig=1; ig<Ngrps; ++ig)
      for (int ik=0; ik<Nloops; ++ik){
	momentum(ig,ik) = momentum(0,ik);
	amomentm(ig,ik) = amomentm(0,ik);
      }
  }
  std::cout<<"Ngrps="<<Ngrps<<std::endl;
  std::cout<<"momentum="<<momentum<<std::endl;
  std::cout<<"amomentm="<<amomentm<<std::endl;
  
  // Momenta for all propagators in all graphs, which is needed to evaluate the diagrams.
  mom_G=0.0;
  mom_V=0.0;
  for (int id=0; id<Ndiags; id++){
    int ig = indx(id);
    for (int iloop=0; iloop<Nloops; iloop++){
      const std::vector<int>& lindex = Loop_index[id][iloop];
      const std::vector<int>& ltype  = Loop_type[id][iloop];
      for (int i=0; i<lindex.size(); i++){
        if ( abs(ltype[i])==1 ){
	  mom_G(id, lindex[i]) += momentum(ig,iloop) * dsign(ltype[i]);
	}else{
	  mom_V(id,lindex[i]) += momentum(ig,iloop) * dsign(ltype[i]);
	}
      }
    }
  }
  // DEBUGGING
  for (int id=0; id<Ndiags; id++){
    for (int i=0; i<2*Norder; i++){
      //int i_final = diagsG(id,i);
      int i_previ = i_diagsG(id,i);
      bl::TinyVector<double,3> k_in = mom_G(id,i_previ);
      bl::TinyVector<double,3> k_out = mom_G(id,i);
      bl::TinyVector<double,3> q = mom_V(id,i/2)*(1.-2*(i%2));
      if (fabs(norm(k_in-k_out-q))>1e-6){
	std::cerr<<"ERROR: diagram "<<id<<" vertex i="<<i<<" k_in="<<k_in<<" k_out="<<k_out<<" q="<<q<<std::endl;
	std::cerr<<"   i_previ="<<i_previ<<" q_i="<< (i/2) <<" mom_V="<<mom_V(id,i/2)<<std::endl;
      }
    }
  }
  // DEBUGGING
  // Finally evaluating the polarizations for all diagrams
  int Ng = gindx.extent(0);
  int Nv = vindx.extent(0);
  int _Nv_ = std::max(Nv,1); // can not assign 0 to empty file
  bl::Array<double,1> G_current(Ng), V_current(_Nv_);
  G_current=0;
  for (int ii=0; ii<Ng; ++ii){
    int id = gindx(ii)[0];
    int  i = gindx(ii)[1];
    int ig = indx(id);
    double aK = norm(mom_G(id,i));
    int i_final = diagsG(id,i);
    double dt = times(ig,i_final)-times(ig,i);
    G_current(ii) = Gk(aK, dt);
  }
  V_current=0;  
  for (int ii=0; ii<Nv; ++ii){
    int id = vindx(ii)[0];
    int  i = vindx(ii)[1];
    double aQ = norm(mom_V(id,i));
    V_current(ii) = Vqc(aQ,Vtype(id,i));
  }
  bl::Array<double,1> PQs(Ngrps);
  double PQ=0;
  for (int ig=0; ig<Ngrps; ++ig){
    PQs(ig)=0;
    for (int j=0; j<idiag[ig].size(); ++j){
      int id = idiag[ig][j];
      double PQd = 1.0;
      for (int i=0; i<2*Norder; i++) PQd *= G_current(Gindx(id,i));
      for (int i=1; i<Norder; i++) PQd *= V_current(Vindx(id,i));
      PQs(ig) += PQd * diagSign(id);
    }
    PQ += PQs(ig);
  }

  // DEBUGGING
  for (int id=0; id<Ndiags; id++){
    for (int i=0; i<2*Norder; i++){
      if (fabs(G_current(Gindx(id,i)))==0){
	std::cerr<<"G_current("<<id<<","<<i<<") not set!"<<std::endl;
      }
    }
    for (int i=1; i<Norder; i++)
      if (fabs(V_current(Vindx(id,i)))==0){
	std::cerr<<"V_current("<<id<<","<<i<<") not set!"<<std::endl;
      }
  }
  // DEBUGGING
  
  Plq.cmp_single(2*amomentm(0,0)/cutoffq-1., pl_Q); // Legendre polynomials at this Q
  Plt.cmp_single(2*(times(0,0)-times(0,1))/beta-1., ul_t);     // Legendre polynomials at this time

  bl::Array<bl::TinyVector<double,3>,2> tmom_G(Ndiags,2*Norder); // momentum for 2*Norder fermionic propagators
  bl::Array<bl::TinyVector<double,3>,2> tmom_V(Ndiags,  Norder); // momentum for Norder bosonic propagators
  bl::Array<int,1> changed_G(Ng), changed_V(Nv);
  bl::Array<double,1> G_trial(Ng), V_trial(_Nv_);
  bl::Array<double,2> times_trial(Ngrps,2*Norder);
  
  int Qweight = 0;      // if Qweight==1 then we have the normalization diagram
  double Nweight = 0;   // how many times did we see measuring diagram.
  int NlowT = 0;        // how many times do we see external time < 0.05*beta
  int Nmeassure = 0;    // how many times did we measure quantities.
  double aver_sign = 0; // sign
  int Nacc_k = 0, Nacc_q = 0, Nacc_w = 0, Nacc_t = 0;
  int Nall_k = 0, Nall_q = 0, Nall_w = 0, Nall_t = 0;
  PrintInfo_(0, Qweight, amomentm(0,0), amomentm(0,1), times(0,0)-times(0,1), PQ, PQ, 0, 0);
  bl::Array<double,1> PQs_trial(Ngrps);
  
  t_all.start();
  for (int itt=0; itt<p.Nitt; itt++){
    double PQ_new=0;
    int icase = tBisect(drand(), Prs); // which steps should we attempt?
    if (icase==0){ // changing momentum Q or k-point
      int iloop = static_cast<int>(Nloops*drand());
      int igrp = static_cast<int>(Ngrps*drand());
      bl::TinyVector<double,3> K_new; double Ka_new;
      double trial_ratio=1;
      bool accept=false;
      if (iloop==0){ // this is the loop for the external momentum Q
	Nall_q += 1;
	accept = Find_new_Q_point(K_new, Ka_new, trial_ratio, momentum(igrp,iloop), amomentm(igrp,iloop), cutoffq, drand);
      }else{         // any other momentum point
	Nall_k += 1;
	accept = Find_new_k_point(K_new, Ka_new, trial_ratio, momentum(igrp,iloop), amomentm(igrp,iloop), cutoffk, drand, drand()<p.Qring, dkF, dRk);
      }
      if (accept){
	//t_k1.start(); // Takes 22% time
	tmom_G = mom_G; tmom_V = mom_V;
	changed_G = 0; changed_V = 0;
	for (int ig=0; ig<Ngrps; ++ig){
	  if (! (iloop==0 || ig==igrp) ) continue; // when iloop=0 (external variable) we want to update all diagrams. For iloop!=0, we updaate only a subgroup (ig=igrp) of diagrams.
	  bl::TinyVector<double,3> dK = K_new - momentum(ig,iloop); // how much momentum will be changed.
	  for (int j=0; j<idiag[ig].size(); ++j){ // updating momentum in all propagators
	    int id = idiag[ig][j];
	    const std::vector<int>& lindex = Loop_index[id][iloop];
	    const std::vector<int>& ltype  = Loop_type[id][iloop];
	    for (int i=0; i<lindex.size(); i++){
	      if ( abs(ltype[i])==1 ){
		tmom_G(id, lindex[i]) += dK * dsign(ltype[i]);
		changed_G(Gindx(id, lindex[i]))=1;
	      }else{
		if (lindex[i]>0){// meassuring line should not be needed to compute diagram
		  tmom_V(id,lindex[i]) += dK * dsign(ltype[i]);
		  changed_V(Vindx(id, lindex[i]))=1;
		}
	      }
	    }
	  }
	}
	//t_k1.stop();
	if (! Qweight){ // we computed the polarization diagram
	  //t_k2.start(); // takes 29% time
	  for (int ii=0; ii<Ng; ++ii){
	    if (changed_G(ii)){
	      int id = gindx(ii)[0];
	      int  i = gindx(ii)[1];
	      int ig = indx(id);
	      double aK = norm(tmom_G(id,i));
	      int i_final = diagsG(id,i);
	      double dt = times(ig,i_final)-times(ig,i);
	      G_trial(ii) = Gk(aK, dt);
	    }
	  }
	  for (int ii=0; ii<Nv; ++ii){
	    if (changed_V(ii)){
	      int id = vindx(ii)[0];
	      int  i = vindx(ii)[1];
	      double aQ = norm(tmom_V(id,i));
	      V_trial(ii) = Vqc(aQ, Vtype(id,i));
	    }
	  }
	  //t_k2.stop();
	  //t_k3.start(); // takes 8% time
	  PQ_new=0;
	  for (int ig=0; ig<Ngrps; ++ig){
	    if (iloop==0 || ig==igrp){
	      PQs_trial(ig)=0;
	      for (int j=0; j<idiag[ig].size(); ++j){
		int id = idiag[ig][j];
		double PQd = 1.0;
		for (int i=0; i<2*Norder; i++){
		  int ii = Gindx(id,i);
		  PQd *= ( changed_G(ii) ? G_trial(ii) : G_current(ii) );
		}
		for (int i=1; i<Norder; i++){
		  int ii = Vindx(id,i);
		  PQd *= ( changed_V(ii) ? V_trial(ii) : V_current(ii) );
		}
		PQs_trial(ig) += PQd * diagSign(id);
	      }
	      PQ_new += PQs_trial(ig);
	    }else{
	      PQ_new += PQs(ig); // old values has not been changed in this step
	    }
	  }
	  //t_k3.stop();
	}else{
	  PQ_new=0;
	  for (int ig=0; ig<Ngrps; ++ig){
	    double Ka_old;
	    bl::TinyVector<double,3> K_old;
	    if (iloop==0 || ig==igrp){
	      Ka_old = amomentm(ig,iloop);
	      amomentm(ig,iloop) = Ka_new;
	      K_old = momentum(ig,iloop);
	      momentum(ig,iloop) = K_new;
	    }
	    PQ_new += idiag[ig].size() * mweight(amomentm(ig,bl::Range::all())/*,times(ig,bl::Range::all())*/, momentum(ig,bl::Range::all()));
	    if (iloop==0 || ig==igrp){
	      amomentm(ig,iloop) = Ka_old;
	      momentum(ig,iloop) = K_old;
	    }
	  }
	  PQ_new *= V0norm/Ndiags;
	}
	double ratio = PQ_new*trial_ratio / PQ;
	accept = fabs(ratio) > 1-drand() && PQ_new!=0;
      }
      if ((itt+1)%Ncout==0)
	//if (mpi.rank==mpi.master)
	  _PrintInfo_(itt+1, Qweight, icase, amomentm(igrp,1), amomentm(igrp,Nloops-1), times(igrp,0), PQ_new, PQ, NlowT/(Nmeassure+0.0), Nweight/(Nmeassure+0.0));
      if (accept){
	if (iloop==0){
	  Nacc_q += 1;
	  for (int ig=0; ig<Ngrps; ++ig){ // external momentum for all diagrams has changed
	    momentum(ig,iloop) = K_new;  // this momentum was changed
	    amomentm(ig,iloop) = Ka_new;
	  }
	}else{
	  Nacc_k += 1;
	  momentum(igrp,iloop) = K_new;  // this momentum was changed
	  amomentm(igrp,iloop) = Ka_new;
	}
	mom_G = tmom_G;           // and therefore many momenta in diagrams have changed. We could optimize this, and 
	mom_V = tmom_V;           // change only those momenta that actually change....
	if (!Qweight){
	  //t_k4.start(); // takes 1% time
	  for (int ii=0; ii<Ng; ++ii)
	    if (changed_G(ii))
	      G_current(ii) = G_trial(ii); // The green's functions in the loop have been recalculated, and are now stored.
	  for (int ii=0; ii<Nv; ++ii)
	    if (changed_V(ii))
	      V_current(ii) = V_trial(ii); // The interactions in the loop have been recalculated and stored.
	  if (iloop==0) PQs       = PQs_trial;        // entire array (all diagrams) was changed
	  else          PQs(igrp) = PQs_trial(igrp);  // only one group of diagrams has changed
	  //t_k4.stop();
	}else{
	  // since we have the measuring diagram, we did not calculate G_trial and V_trial before, hence we need to evaluate them now.
	  //t_k5.start(); // takes 1% time
	  for (int ii=0; ii<Ng; ++ii){
	    if (changed_G(ii)){
	      int id = gindx(ii)[0];
	      int  i = gindx(ii)[1];
	      int ig = indx(id);
	      double aK = norm(tmom_G(id,i));
	      int i_final = diagsG(id,i);
	      double dt = times(ig,i_final)-times(ig,i);
	      G_current(ii) = Gk(aK, dt);
	    }
	  }
	  for (int ii=0; ii<Nv; ++ii){
	    if (changed_V(ii)){
	      int id = vindx(ii)[0];
	      int  i = vindx(ii)[1];
	      double aQ = norm(tmom_V(id,i));
	      V_current(ii) = Vqc(aQ, Vtype(id,i));
	    }
	  }
	  for (int ig=0; ig<Ngrps; ++ig){
	    if (iloop==0 || ig==igrp){
	      PQs(ig)=0;
	      for (int j=0; j<idiag[ig].size(); ++j){
		int id = idiag[ig][j];
		double PQd = 1.0;
		for (int i=0; i<2*Norder; ++i) PQd *= G_current(Gindx(id,i));
		for (int i=1; i<  Norder; ++i) PQd *= V_current(Vindx(id,i));
		PQs(ig) += PQd * diagSign(id);
	      }
	    }
	  }
	  //t_k5.stop();
	}
	PQ = PQ_new;
	//t_k6.start(); // takes 0.5% time
	if (iloop==0) Plq.cmp_single(2*amomentm(igrp,iloop)/cutoffq-1., pl_Q);  // update Legendre Polynomials
	//t_k6.stop();
      }
    } else if (icase==1){  // changing one of the time variables
      Nall_t += 1;
      int itime = static_cast<int>(Norder*drand()); // which time to change? For bare interaction, there are only Norder different times.
      int igrp = static_cast<int>(Ngrps*drand());
      changed_G=0; changed_V=0;  // which propagators are being changed?
      times_trial = times;       // ???? Optimize: You do not need to copy all, except if itime==0! times_trial will contain the trial step times.
      if (itime==0){           // this is the measuring time with vertex=0
	// t_t1.start(); // takes 1% time
	double t_new = rw.gtau(drand());
	if (t_new<1e-15) t_new = 1e-15; // to be on the right branchcut
	for (int ig=0; ig<Ngrps; ++ig) times_trial(ig,0) = t_new;
	int ivertex=0;
	for (int id=0; id<Ndiags; id++){
	  int i_pre_vertex = i_diagsG(id,ivertex);
	  changed_G(Gindx(id,ivertex))=1;         // these two propagators contain vertex=0.
	  changed_G(Gindx(id,i_pre_vertex))=1;
	}
	//t_t1.stop();
      }else{
	//t_t1.start(); // takes 1% time
	double t_new = beta*drand();
	// unscreened interaction, hence changing two times simultaneously!
	for (int ivertex=2*itime; ivertex<=2*itime+1; ++ivertex){ // 2*itime and 2*itime+1 are changed at the same time, because they are equal
	  times_trial(igrp,ivertex) = t_new;
	  for (int j=0; j<idiag[igrp].size(); j++){
	    int id = idiag[igrp][j];
	    int i_pre_vertex = i_diagsG(id,ivertex);
	    changed_G(Gindx(id,ivertex))=1;        // these two propagators are changed because of the new time.
	    changed_G(Gindx(id,i_pre_vertex))=1;
	  }
	}
	//t_t1.stop();
      }
      if (! Qweight){
	//t_t2.start(); // takes 10% time
	for (int ii=0; ii<Ng; ++ii){
	  if (changed_G(ii)){
	    int id = gindx(ii)[0];
	    int  i = gindx(ii)[1];
	    int ig = indx(id);
	    double aK = norm(mom_G(id,i));
	    int i_final = diagsG(id,i);
	    double dt = times_trial(ig,i_final)-times_trial(ig,i);
	    G_trial(ii) = Gk(aK, dt);
	  }
	}
	//t_t2.stop();
	//t_t3.start(); // takes 3% time
	PQ_new=0; // recalculating PQ, taking into account one change of time.
	for (int ig=0; ig<Ngrps; ++ig){
	  if (itime==0 || ig==igrp){
	    PQs_trial(ig)=0;
	    for (int j=0; j<idiag[ig].size(); ++j){
	      int id = idiag[ig][j];
	      double PQd = 1.0;
	      for (int i=0; i<2*Norder; i++){
		int ii = Gindx(id,i);
		PQd *= ( changed_G(ii) ? G_trial(ii) : G_current(ii) );
	      }
	      for (int i=1; i<Norder; i++) PQd *= V_current(Vindx(id,i)); // interaction does not depend on time, hence it is not changed here.
	      PQs_trial(ig) += PQd * diagSign(id);
	    }
	    PQ_new += PQs_trial(ig);
	  }else{
	    PQ_new += PQs(ig);
	  }
	}
	//t_t3.stop();
      }else{
	PQ_new=0;
	for (int ig=0; ig<Ngrps; ++ig)
	  PQ_new += idiag[ig].size() * mweight(amomentm(ig,bl::Range::all()), momentum(ig,bl::Range::all())/*, times_trial(ig,bl::Range::all())*/);
	PQ_new *= V0norm/Ndiags;
      }
      double ratio = PQ_new/PQ;
      bool accept = fabs(ratio) > 1-drand() && PQ_new!=0;
      if ((itt+1)%Ncout==0)
	//if (mpi.rank==mpi.master)
	  _PrintInfo_(itt+1, Qweight, icase, amomentm(igrp,1), amomentm(igrp,Nloops-1), times(igrp,2*itime), PQ_new, PQ, NlowT/(Nmeassure+0.0), Nweight/(Nmeassure+0.0));
      if (accept){
	//t_t4.start(); // takes 1% time
	Nacc_t +=1;
	times = times_trial;  // the new times are now accepted. We change the independent time variable, or two internal times.
	if (!Qweight){
	  for (int ii=0; ii<Ng; ++ii)
	    if (changed_G(ii))
	      G_current(ii) = G_trial(ii); // The green's functions in the loop have been recalculated, and are now stored.
	  if (itime==0) PQs       = PQs_trial;       // entire array (all diagrams) was changed
	  else          PQs(igrp) = PQs_trial(igrp); // only one group of diagrams has changed
	}else{
	  // since we have the measuring diagram, we did not calculate G_trial and V_trial before, hence we need to evaluate them now.
	  for (int ii=0; ii<Ng; ++ii){
	    if (changed_G(ii)){
	      int id = gindx(ii)[0];
	      int  i = gindx(ii)[1];
	      int ig = indx(id);
	      int i_final = diagsG(id,i);
	      G_current(ii) = Gk( norm(mom_G(id,i)),  times(ig,i_final)-times(ig,i)); // these propagators were changed, but they were not computed before.
	    }
	  }
	  for (int ig=0; ig<Ngrps; ++ig){
	    if (itime==0 || ig==igrp){
	      PQs(ig)=0;
	      for (int j=0; j<idiag[ig].size(); ++j){
		int id = idiag[ig][j];
		double PQd = 1.0;
		for (int i=0; i<2*Norder; i++) PQd *= G_current(Gindx(id,i));
		for (int i=1; i<  Norder; i++) PQd *= V_current(Vindx(id,i)); // interaction does not depend on time, hence it is not changed here.
		PQs(ig) += PQd * diagSign(id);
	      }
	    }
	  }
	}
	//t_t4.stop();
	PQ = PQ_new;
	//t_t5.start(); // takes 0.4% time
	if (itime==0) Plt.cmp_single(2*(times(0,0)-times(0,1))/beta-1., ul_t);  // update Legendre Polynomials
	//t_t5.stop();
      }
    }else{  // normalization diagram step
      Nall_w += 1;
      //t_t6.start();// takes 0.4% time
      if (Qweight){ // trying to jump back into physical hilbert space
	PQ_new=0;
	for (int ig=0; ig<Ngrps; ++ig) PQ_new += PQs(ig);
      }else{        // or if we are in the physical Hilbert space, we try to jump into the normalization diagram
	PQ_new = 0;
	for (int ig=0; ig<Ngrps; ++ig)
	  PQ_new += idiag[ig].size() * mweight(amomentm(ig,bl::Range::all()),momentum(ig,bl::Range::all())/*times(ig,bl::Range::all())*/);
	PQ_new *= V0norm/Ndiags;
      }
      double ratio = PQ_new/PQ;
      bool accept = fabs(ratio) > 1-drand() && PQ_new!=0;
      if ((itt+1)%Ncout==0)
	//if (mpi.rank==mpi.master)
	  _PrintInfo_(itt+1, Qweight, icase, amomentm(0,1), amomentm(0,Nloops-1), times(0,0), PQ_new, PQ, NlowT/(Nmeassure+0.0), Nweight/(Nmeassure+0.0));
      if (accept){
	Nacc_w += 1;
	Qweight = 1-Qweight;
	PQ = PQ_new;
      }
      //t_t6.stop();
    }
    // Measuring
    if (itt>=Nwarm && itt%tmeassure==0){
      t_mes.start(); // takes 2% time
      Nmeassure += 1;
      double Qa = amomentm(0,0);
      double t = times(0,0)-times(0,1);
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
	// THIS NEEDS NEW CODING. SHOULD BE REDONE WITH THE NEW MWEIGHT!
	int iik = static_cast<int>( amomentm(0,0)/cutoffq * Nbin); // all groups of diagrams should have the same external momenta
	if (iik>=Nbin) iik=Nbin-1;
	K_hist(0,iik) += 1.0;
	for (int ig=0; ig<Ngrps; ++ig){
	  for (int ik=1; ik<Nloops; ik++){
	    double k = amomentm(ig,ik);
	    int iik = static_cast<int>(k/cutoffk * Nbin);
	    if (iik>=Nbin) iik=Nbin-1;
	    K_hist(ik,iik) += 1./(k*k);
	  }
	}
	for (int ig=0; ig<Ngrps; ++ig){
	  for (int it=2; it<2*Norder; it++){
	    int iit = static_cast<int>(times(ig,it)/beta * Nbin);
	    if (iit>=Nbin) iit=Nbin-1;
	    T_hist(it-2,iit) += 1;
	  }
	}
      }
      if ( itt>5e5 && itt%(10000*tmeassure) == 0){
	double occurence = static_cast<double>(Nweight)/static_cast<double>(Nmeassure);
	if ( occurence > 0.25){ // decrease by two
	  V0norm /= 1.1;
	  Pnorm /= 1.1;
	  //if (mpi.rank==mpi.master) std::cout<<"occurence="<<occurence<<" V0 reduced to "<<V0norm<<std::endl;
	}
	if ( occurence < 0.02){ // increase by two
	  V0norm *= 1.1;
	  Pnorm *= 1.1;
	  //if (mpi.rank==mpi.master) std::cout<<"occurence="<<occurence<<" V0 increased to "<<V0norm<<std::endl;
	}
      }
      if (itt%(500000*tmeassure) == 0){
	if (Nmeassure-Nweight>1000){
	  mweight.Recompute(/*K_hist, T_hist,  mpi.rank==mpi.master*/ true);
	  if (Qweight){
	    PQ=0;
	    for (int ig=0; ig<Ngrps; ++ig)
	      PQ += idiag[ig].size() * mweight(amomentm(ig,bl::Range::all()), momentum(ig,bl::Range::all())/*times_trial(ig,bl::Range::all())*/);
	    PQ *= V0norm/Ndiags;
	  }
	}
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
  
  C_Pln *= 1.0/(4*pi); // 4*pi because we must divide by 1/(4*pi*q**2), and 1/q**2 was already done in sampling
  C_Pln *= (fabs(V0norm)/Pnorm);
  Pbin  *= 1.0/(4*pi);
  Pbin  *= (fabs(V0norm)/Pnorm);
  std::clog<<"total acceptance rate="<<(Nacc_k+Nacc_q+Nacc_t+Nacc_w)/(p.Nitt+0.0)<<" "<<" k-acceptance="<<(Nacc_k/(Nall_k+0.0))<<" q-acceptance="<<(Nacc_q/(Nall_q+0.0))<<" t-acceptanece="<<(Nacc_t/(Nall_t+0.0))<<" w-acceptanece="<<(Nacc_w/(Nall_w+0.0))<<std::endl;
  std::clog<<" k-trials="<<(Nall_k/(p.Nitt+0.0))<<" q-trials="<<(Nall_q/(p.Nitt+0.0))<<" t-trials="<<(Nall_t/(p.Nitt+0.0))<<" w-trials="<<(Nall_w/(p.Nitt+0.0))<<std::endl;
  std::clog<<"  meassuring diagram occurence frequency="<<Nweight<<" and its norm Pnorm="<<Pnorm<<std::endl;
  std::clog<<"total time="<<t_all.elapsed()<<" measuring time="<<t_mes.elapsed()<<std::endl;
  //std::clog<<" legendre time="<<t_Pl.elapsed()<<" t_k="<<t_k.elapsed()<<" t_Q="<<t_Q.elapsed()<<" t_t="<<t_t.elapsed()<<" t_w="<<t_w.elapsed()<<std::endl;
  //std::cout<<"t_k1="<<t_k1.elapsed()<<" t_k2="<<t_k2.elapsed()<<" t_k3="<<t_k3.elapsed()<<" t_k4="<<t_k4.elapsed()<<" t_k5="<<t_k5.elapsed()<<" t_k6="<<t_k6.elapsed()<<std::endl;
  //std::cout<<"t_t1="<<t_t1.elapsed()<<" t_t2="<<t_t2.elapsed()<<" t_t3="<<t_t3.elapsed()<<" t_t4="<<t_t4.elapsed()<<" t_t5="<<t_t5.elapsed()<<" t_t6="<<t_t6.elapsed()<<std::endl;
  
  K_hist *= 1.0/Nmeassure;
  T_hist *= 1.0/Nmeassure;
  for (int ik=0; ik<Nloops; ik++){
    double dsum=0;
    for (int i=0; i<Nbin; i++) dsum += K_hist(ik,i);
    K_hist(ik,bl::Range::all()) *= 1./dsum;
    std::ofstream hout((std::string("Q_hist.")+std::to_string(ik)).c_str());
    double cutoff = (ik>0)? cutoffk : cutoffq;
    for (int i=0; i<Nbin; i++){
      hout<< (i+0.5)/Nbin*cutoff/kF << " " << K_hist(ik,i)<<std::endl;
    }
  }
  for (int it=2; it<2*Norder; it++){
    std::ofstream hout((std::string("T_hist.")+std::to_string(it)).c_str());
    for (int i=0; i<Nbin; i++){
      hout<< (i+0.5)/Nbin*beta << " " << T_hist(it-2,i)<<std::endl;
    }
  }
}

template<typename GK>
void sample_static_Q0(double Q_external, double lmbda, bl::Array<double,1>& Pbin, const GK& Gk,
		      const params& p, const bl::Array<int,2>& diagsG, const bl::Array<double,1>& diagSign,
		      const std::vector<std::vector<std::vector<int> > >& Loop_index, const std::vector<std::vector<std::vector<int> > >& Loop_type,
		      const bl::Array<int,2>& Vtype, const bl::Array<int,1>& indx, my_mpi& mpi)
{
  const int Nr=3; // three different steps in this algorithm
  double Pr_norm = (p.Pr[0]+p.Pr[1]+p.Pr[2]); // probability for all four has to be normalized
  bl::TinyVector<double,Nr> Prs;  // cumulative probability for taking a step, needed for bisection below.
  Prs[0] = p.Pr[0]/Pr_norm;
  for (int i=1; i<Nr; i++) Prs[i] = Prs[i-1] + p.Pr[i]/Pr_norm; // prepares cumulative probability from p.Pr

  int Ndiags = diagsG.extent(0);
  int Norder = diagsG.extent(1)/2;
  // computing the inverse of all permutations, to provide fast lookup for the previous fermionic propagator
  bl::Array<int,2> i_diagsG(Ndiags,2*Norder);
  for (int id=0; id<Ndiags; id++)
    for (int i=0; i<2*Norder; i++)
      i_diagsG(id,diagsG(id,i))=i;
  
  RanGSL drand(p.iseed); // GSL random number generator
  Timer t_mes, t_all, t_Pl, t_k, t_Q, t_w, t_t;
  
  //ScreenedCoulombV Vq(lmbda);
  CounterCoulombV Vqc(lmbda);
  
  int tmeassure=p.tmeassure, Ncout=p.Ncout, Nwarm=p.Nwarm;
  double V0norm = p.V0norm, dkF = p.dkF, kF=p.kF;// V0exp= p.V0exp;
  double cutoffq = p.cutoffq, cutoffk = p.cutoffk, dRk = p.dRk;
  double beta = p.beta;
  int Nt = p.Nt;// Nq = p.Nq;
  
  if (Pbin.extent(0)!=p.Nt) Pbin.resize(p.Nt); Pbin=0.0;
  
  double Pnorm = 0.0;            // normalization diagram value
  
  if (Loop_index.size()!=Ndiags) std::cerr<<"Wrong size of Loop_index "<<Loop_index.size()<<" instead of "<<Ndiags<<std::endl;
  int Nloops = Loop_index[0].size();
  if (Nloops != Norder+1) std::cerr<<"Expecting Nloops==Norder+1 in Loops_index, but got Nloops="<<Nloops<<" Norder="<<Norder<<std::endl;

  
  Reweight_time rw(p.beta,p.lmbdat);
  int Nbin = 513;
  //int Nbin = 129;
  //meassureWeight mweight(p.V0exp, p.cutoffk, p.kF, p.beta, Nbin);
  meassureWeight mweight(p.V0exp, p.cutoffk, kF, Nbin, Nloops);
  //bl::Array<double,2> K_hist(Nloops,Nbin); K_hist=0;
  bl::Array<double,2> T_hist((Norder-1)*2,Nbin); T_hist=0;
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
  for (int it=1; it<Norder; it++){
    double tv = beta*drand();
    times(2*it  ) = tv;
    times(2*it+1) = tv;
  }
  // Next the momenta
  {
    double Qa = Q_external; // absolute value of external momentum
    momentum(0) = Qa,0.0,0.0; // external momentum Q
    amomentm(0) = Qa;
    double kf_3 = p.kF/sqrt(3.);
    for (int ik=1; ik<Nloops; ik++){
      momentum(ik) = drand()*kf_3, drand()*kf_3, drand()*kf_3;
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
  // Finally evaluating the polarizations for all diagrams
  bl::Array<double,2> G_current(Ndiags,2*Norder), V_current(Ndiags,Norder);
  G_current=0; V_current=0;
  double PQ=0;
  for (int id=0; id<Ndiags; id++){
    double PQd = 1.0;
    for (int i=0; i<2*Norder; i++){
      double aK = norm(mom_G(id,i));
      int i_final = diagsG(id,i);
      double dt = times(i_final)-times(i);
      G_current(id,i) = Gk(aK, dt);
      PQd *= G_current(id,i);
    }
    for (int i=1; i<Norder; i++){ // do not add V for the meassuring line
      double aQ = norm(mom_V(id,i));
      V_current(id,i) = Vqc(aQ, Vtype(id,i));
      PQd *= V_current(id,i);
    }
    PQ += PQd * diagSign(id);
  }

  bl::Array<bl::TinyVector<double,3>,2> tmom_G(Ndiags,2*Norder); // momentum for 2*Norder fermionic propagators
  bl::Array<bl::TinyVector<double,3>,2> tmom_V(Ndiags,  Norder); // momentum for Norder bosonic propagators
  bl::Array<int,2> changed_G(Ndiags,2*Norder), changed_V(Ndiags,Norder);
  bl::Array<double,2> G_trial(Ndiags,2*Norder), V_trial(Ndiags,  Norder);
  bl::Array<double,1> times_trial(2*Norder);
    
  int Qweight = 0;      // if Qweight==1 then we have the normalization diagram
  double Nweight = 0;   // how many times did we see measuring diagram.
  int NlowT = 0;        // how many times do we see external time < 0.05*beta
  int Nmeassure = 0;    // how many times did we measure quantities.
  int Nacc_k = 0, Nacc_q = 0, Nacc_w = 0, Nacc_t = 0;
  int Nall_k = 0, Nall_q = 0, Nall_w = 0, Nall_t = 0;
  if (mpi.rank==mpi.master) PrintInfo_(0, Qweight, amomentm(0), amomentm(1), times(0)-times(1), PQ, PQ, 0, 0);

  t_all.start();
  for (int itt=0; itt<p.Nitt; itt++){
    double PQ_new=0;
    int icase = tBisect(drand(), Prs); // which steps should we attempt?
    double Ka_new;
    if (icase==0){ // changing momentum Q or k-point
      int iloop = 1 + static_cast<int>((Nloops-1)*drand()); // newer change measuring (0) loop
      bl::TinyVector<double,3> K_new; double trial_ratio=1;
      bool accept=false;
      Nall_k += 1;
      accept = Find_new_k_point(K_new, Ka_new, trial_ratio, momentum(iloop), amomentm(iloop), cutoffk, drand, drand()<p.Qring, dkF, dRk);
      if (accept){
	bl::TinyVector<double,3> dK = K_new - momentum(iloop); // how much momentum will be changed.
	tmom_G = mom_G; tmom_V = mom_V;
	changed_G = 0; changed_V = 0;
	for (int id=0; id<Ndiags; id++){ // updating momentum in all propagators
	  const std::vector<int>& lindex = Loop_index[id][iloop];
	  const std::vector<int>& ltype  = Loop_type[id][iloop];
	  for (int i=0; i<lindex.size(); i++){
	    if ( abs(ltype[i])==1 ){
	      tmom_G(id, lindex[i]) += dK * dsign(ltype[i]);
	      changed_G(id, lindex[i])=1;
	    }else{
	      tmom_V(id,lindex[i]) += dK * dsign(ltype[i]);
	      changed_V(id, lindex[i])=1;
	    }
	  }
	}
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
	    for (int i=1; i<Norder; i++){ // do not add V for the meassuring line
	      if (changed_V(id,i)){
		double aQ = norm(tmom_V(id,i));
		V_trial(id,i) = Vqc(aQ, Vtype(id,i));
		PQd *= V_trial(id,i);
	      }else{
		PQd *= V_current(id,i);
	      }
	    }
	    PQ_new += PQd * diagSign(id);
	  }
	}else{
	  double Ka_old = amomentm(iloop);
	  amomentm(iloop) = Ka_new;
	  bl::TinyVector<double,3> K_old = momentum(iloop);
	  momentum(iloop) = K_new;
	  PQ_new = V0norm * mweight(amomentm,momentum/*times*/);
	  amomentm(iloop) = Ka_old;
	  momentum(iloop) = K_old;
	  //std::cout<<"PQ_meassure="<<PQ_new<<" k="<<norm(momentum(1))/kF<<std::endl;
	}
	double ratio = PQ_new*trial_ratio / PQ;
	accept = fabs(ratio) > 1-drand() && PQ_new!=0;
      }
      if ((itt+1)%Ncout==0)
	if (mpi.rank==mpi.master)
	  _PrintInfo_(itt+1, Qweight, icase, amomentm(1), amomentm(Nloops-1), times(0), PQ_new, PQ, NlowT/(Nmeassure+0.0), Nweight/(Nmeassure+0.0));
      if (accept){
	Nacc_k += 1;
	momentum(iloop)  = K_new;  // this momentum was changed
	amomentm(iloop) = Ka_new;
	mom_G = tmom_G;           // and therefore many momenta in diagrams have changed. We could optimize this, and 
	mom_V = tmom_V;           // change only those momenta that actually change....
	if (!Qweight){
	  for (int id=0; id<Ndiags; id++){
	    for (int i=0; i<2*Norder; i++)
	      if (changed_G(id,i))
		G_current(id,i) = G_trial(id,i); // The green's functions in the loop have been recalculated, and are now stored.
	    for (int i=1; i<Norder; i++) // do not add V for the meassuring line
	      if (changed_V(id,i))
		V_current(id,i) = V_trial(id,i); // The interactions in the loop have been recalculated and stored.
	  }
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
	    for (int i=1; i<Norder; i++)
	      if (changed_V(id,i)){
		double aQ = norm(tmom_V(id,i));
		V_current(id,i) = Vqc(aQ, Vtype(id,i));
	      }
	  }
	}
	PQ = PQ_new;
	//if (iloop==0) Plq.cmp_single(2*amomentm(iloop)/cutoffq-1., pl_Q);  // update Legendre Polynomials
      }
    } else if (icase==1){  // changing one of the time variables
      Nall_t += 1;
      int itime = static_cast<int>(Norder*drand()); // which time to change? For bare interaction, there are only Norder different times.
      changed_G=0; changed_V=0;  // which propagators are being changed?
      times_trial = times;       // times_trial will contain the trial step times.
      if (itime==0){           // this is the measuring time with vertex=0
	double t_new = rw.gtau(drand());
	if (t_new<1e-15) t_new = 1e-15; // to be on the right branchcut
	times_trial(0) = t_new;
	int ivertex=0;
	for (int id=0; id<Ndiags; id++){
	  int i_pre_vertex = i_diagsG(id,ivertex);
	  changed_G(id,ivertex)=1;         // these two propagators contain vertex=0.
	  changed_G(id,i_pre_vertex)=1;
	}
      }else{
	double t_new = beta*drand();
	// unscreened interaction, hence changing two times simultaneously!
	for (int ivertex=2*itime; ivertex<=2*itime+1; ++ivertex){ // 2*itime and 2*itime+1 are changed at the same time, because they are equal
	  times_trial(ivertex) = t_new;
	  for (int id=0; id<Ndiags; id++){
	    int i_pre_vertex = i_diagsG(id,ivertex);
	    changed_G(id,ivertex)=1;        // these two propagators are changed because of the new time.
	    changed_G(id,i_pre_vertex)=1;
	  }
	}
      }
      if (! Qweight){
	PQ_new=0; // recalculating PQ, taking into account one change of time.
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
	  for (int i=1; i<Norder; i++) PQd *= V_current(id,i); // interaction does not depend on time, hence it is not changed here.
	  PQ_new += PQd * diagSign(id);
	}
      }else{
	PQ_new = V0norm * mweight(amomentm, momentum /*times_trial*/);
	//std::cout<<"PQ_meassure="<<PQ_new<<" k="<<norm(momentum(1))/kF<<std::endl;
      }
      double ratio = PQ_new/PQ;
      bool accept = fabs(ratio) > 1-drand() && PQ_new!=0;
      if ((itt+1)%Ncout==0)
	if (mpi.rank==mpi.master)
	  _PrintInfo_(itt+1, Qweight, icase, amomentm(1), amomentm(Nloops-1), times(2*itime), PQ_new, PQ, NlowT/(Nmeassure+0.0), Nweight/(Nmeassure+0.0));
      if (accept){
	Nacc_t +=1 ;
	times = times_trial;  // the new times are now accepted. We change the independent time variable, or two internal times.
	if (!Qweight){
	  for (int id=0; id<Ndiags; id++)
	    for (int i=0; i<2*Norder; i++)
	      if (changed_G(id,i))
		G_current(id,i) = G_trial(id,i);  // just saving the propagators which were changed.
	}else{
	  for (int id=0; id<Ndiags; id++)
	    for (int i=0; i<2*Norder; i++)
	      if (changed_G(id,i)){
		int i_final = diagsG(id,i);
		G_current(id,i) = Gk( norm(mom_G(id,i)),  times(i_final)-times(i)); // these propagators were changed, but they were not computed before.
	      }
	}
	PQ = PQ_new;
	//if (itime==0) Plt.cmp_single(2*(times(0)-times(1))/beta-1., ul_t);  // update Legendre Polynomials
      }
    }else{  // normalization diagram step
      Nall_w += 1;
      if (Qweight){ // trying to jump back into physical hilbert space
	PQ_new=0;
	for (int id=0; id<Ndiags; id++){
	  double PQd = 1.0;
	  for (int i=0; i<2*Norder; i++) PQd *= G_current(id,i); // we have all propagators precomputed, hence it is easy to compute the diagrams.
	  for (int i=1; i<Norder; i++) PQd *= V_current(id,i);
	  PQ_new += PQd * diagSign(id);
	}
      }else{        // or if we are in the physical Hilbert space, we try to jump into the normalization diagram
	PQ_new = V0norm * mweight(amomentm, momentum/*, times*/);
	//std::cout<<"PQ_meassure="<<PQ_new<<" k="<<norm(momentum(1))/kF<<std::endl;
      }
      double ratio = PQ_new/PQ;
      bool accept = fabs(ratio) > 1-drand() && PQ_new!=0;
      if ((itt+1)%Ncout==0)
	if (mpi.rank==mpi.master)
	  _PrintInfo_(itt+1, Qweight, icase, amomentm(1), amomentm(Nloops-1), times(0), PQ_new, PQ, NlowT/(Nmeassure+0.0), Nweight/(Nmeassure+0.0));
      if (accept){
	Nacc_w += 1;
	Qweight = 1-Qweight;
	PQ = PQ_new;
      }
    }
    if (PQ==0){
      std::cout<<itt<<"  PQ_new="<<PQ_new<<" which is strange icase="<<icase<<" Qweight="<<Qweight<<" PQ="<<PQ<<" ... Recomputing..."<<std::endl;
      if (icase==0) std::cout << " Ka_new=" << Ka_new/cutoffk << std::endl;
      /*
      {
	VerifyCurrentState(itt, beta, Qweight, PQ, diagsG, i_diagsG, diagSign, Gk, Vqc, Loop_index, Loop_type, momentum, times, mom_G, mom_V, G_current, V_current);
	
	double PQ_tmp=0;
	for (int id=0; id<Ndiags; id++){
	  double PQd = 1.0;
	  std::cout<<"---- diag="<<id<<std::endl;
	  for (int i=0; i<2*Norder; i++){
	    PQd *= G_current(id,i); // we have all propagators precomputed, hence it is easy to compute the diagrams.
	    std::cout<<" G("<<i<<")="<<G_current(id,i)<<" PQd="<<PQd<<std::endl;
	  }
	  for (int i=1; i<Norder; i++){
	    PQd *= V_current(id,i);
	    std::cout<<" V("<<i<<")="<<V_current(id,i)<<" PQd="<<PQd<<std::endl;
	  }
	  PQ_new += PQd * diagSign(id);
	  std::cout<<" sign*PQd="<<PQd * diagSign(id)<<std::endl;
	}
      }
      */
    }
    // Measuring
    if (itt>=Nwarm && itt%tmeassure==0){
      t_mes.start();
      Nmeassure += 1;
      //double Qa = amomentm(0);
      double t = times(0)-times(1);
      double cw = 1./rw.wt(t);
      double sp = sign(PQ) * cw;
      //int iq = min(int(Qa/cutoffq*Nq), Nq-1);
      int it = std::min(int(t/beta*Nt), Nt-1);
      if (Qweight){
	Pnorm += /*Qa*Qa*/ cw;                 // |Q|^2 * reweighting
	Nweight+=1;
      }else{
	Pbin(it) += sp; 
      }
      if (t < 0.05*beta){
	NlowT +=1 ;
      }

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

      if ( itt>5e5 && itt%(50000*tmeassure) == 0){
	double occurence = static_cast<double>(Nweight)/static_cast<double>(Nmeassure);
#ifdef _MPI
	double total_occurence=0;
	MPI_Allreduce(&occurence, &total_occurence, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	occurence = total_occurence/mpi.size;
#endif	
	if ( occurence > 0.25){ // decrease by two
	  V0norm /= 2.;
	  Pnorm /= 2.;
	  if (mpi.rank==mpi.master) std::cout<<"occurence="<<occurence<<" V0 reduced to "<<V0norm<<std::endl;
	}
	if ( occurence < 0.02){ // increase by two
	  V0norm *= 2.;
	  Pnorm *= 2.;
	  if (mpi.rank==mpi.master) std::cout<<"occurence="<<occurence<<" V0 increased to "<<V0norm<<std::endl;
	}
      }

      if (itt%(500000*tmeassure) == 0){
	if (Nmeassure-Nweight>1000){
#ifdef _MPI	  
	  MPI_Allreduce(MPI_IN_PLACE, mweight.K_hist.data(), mweight.K_hist.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
	  dk_hist *= mweight.Normalize_K_histogram();
	  mweight.Recompute(/*K_hist, T_hist,*/ mpi.rank==mpi.master); // Nweight has to be finite, otherwise there is no information stored anyway.
	  if (Qweight) PQ = V0norm * mweight(amomentm, momentum/*times*/);
	}
      }
      t_mes.stop();
    }
  }
  t_all.stop();
  Pbin      *= 1.0/Nmeassure;
  Pnorm     *= 1.0/Nmeassure;
  double occurence = static_cast<double>(Nweight)/static_cast<double>(Nmeassure);
  
  //mweight.K_hist *= 1.0/Nmeassure;
  T_hist *= 1.0/Nmeassure;

#ifdef _MPI  
  double dat[2] = {Pnorm, occurence};
  if (mpi.rank==mpi.master){
    MPI_Reduce(MPI_IN_PLACE,  Pbin.data(),  Pbin.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, dat, 2, MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, mweight.K_hist.data(), mweight.K_hist.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, T_hist.data(), T_hist.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
  }else{
    MPI_Reduce( Pbin.data(),  Pbin.data(),  Pbin.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    MPI_Reduce(dat, dat, 2, MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    MPI_Reduce(mweight.K_hist.data(), mweight.K_hist.data(), mweight.K_hist.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    MPI_Reduce(T_hist.data(), T_hist.data(), T_hist.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
  }
  if (mpi.rank==mpi.master){
    Pbin  *= 1./mpi.size;
    Pnorm = dat[0]/mpi.size;
    occurence = dat[1]/mpi.size;
    //
    mweight.K_hist *= 1./mpi.size;
    T_hist *= 1./mpi.size;
  }
#endif
  std::clog<<"total acceptance rate="<<(Nacc_k+Nacc_q+Nacc_t+Nacc_w)/(p.Nitt+0.0)<<" "<<" k-acceptance="<<(Nacc_k/(Nall_k+0.0))<<" t-acceptanece="<<(Nacc_t/(Nall_t+0.0))<<" w-acceptanece="<<(Nacc_w/(Nall_w+0.0))<<std::endl;
  std::clog<<" k-trials="<<(Nall_k/(p.Nitt+0.0))<<" q-trials="<<(Nall_q/(p.Nitt+0.0))<<" t-trials="<<(Nall_t/(p.Nitt+0.0))<<" w-trials="<<(Nall_w/(p.Nitt+0.0))<<std::endl;
  std::clog<<"total time="<<t_all.elapsed()<<" measuring time="<<t_mes.elapsed()<<std::endl;
  
  if (mpi.rank==mpi.master){
    Pbin  *= (fabs(V0norm)/Pnorm);
    std::clog<<"  meassuring diagram occurence frequency="<<occurence<<" and its norm Pnorm="<<Pnorm<<std::endl;
    
    for (int ik=0; ik<Nloops; ik++){
      double dsum=0;
      for (int i=0; i<Nbin; i++) dsum += mweight.K_hist(ik,i);
      mweight.K_hist(ik,bl::Range::all()) *= 1./dsum;
      std::ofstream hout((std::string("Q_hist.")+std::to_string(ik)).c_str());
      double cutoff = (ik>0)? cutoffk : cutoffq;
      for (int i=0; i<Nbin; i++){
	hout<< (i+0.5)/Nbin*cutoff/kF << " " << mweight.K_hist(ik,i)<<std::endl;
      }
    }
    for (int it=2; it<2*Norder; it++){
      std::ofstream hout((std::string("T_hist.")+std::to_string(it)).c_str());
      for (int i=0; i<Nbin; i++){
	hout<< (i+0.5)/Nbin*beta << " " << T_hist(it-2,i)<<std::endl;
      }
    }
  }
}



template<typename GK>
double sample_static_Q0_t0_fast(double Q_external, double t_external, double lmbda, const GK& Gk,
				 const params& p, const bl::Array<int,2>& diagsG, const bl::Array<double,1>& diagSign,
				 const std::vector<std::vector<std::vector<int> > >& Loop_index, const std::vector<std::vector<std::vector<int> > >& Loop_type,
				 const bl::Array<int,2>& Vtype, const bl::Array<int,1>& indx, my_mpi& mpi)
{
  const int Nr=3; // three different steps in this algorithm
  double Pr_norm = (p.Pr[0]+p.Pr[1]+p.Pr[2]); // probability for all four has to be normalized
  bl::TinyVector<double,Nr> Prs;  // cumulative probability for taking a step, needed for bisection below.
  Prs[0] = p.Pr[0]/Pr_norm;
  for (int i=1; i<Nr; i++) Prs[i] = Prs[i-1] + p.Pr[i]/Pr_norm; // prepares cumulative probability from p.Pr

  int Ndiags = diagsG.extent(0);
  int Norder = diagsG.extent(1)/2;
  // computing the inverse of all permutations, to provide fast lookup for the previous fermionic propagator
  bl::Array<int,2> i_diagsG(Ndiags,2*Norder);
  for (int id=0; id<Ndiags; id++)
    for (int i=0; i<2*Norder; i++)
      i_diagsG(id,diagsG(id,i))=i;
  
  RanGSL drand(p.iseed); // GSL random number generator
  Timer t_mes, t_all, t_Pl, t_k, t_Q, t_w, t_t;
  
  CounterCoulombV Vqc(lmbda);
  
  int tmeassure=p.tmeassure, Ncout=p.Ncout, Nwarm=p.Nwarm;
  double V0norm = p.V0norm, dkF = p.dkF, kF=p.kF;// V0exp= p.V0exp;
  double cutoffq = p.cutoffq, cutoffk = p.cutoffk, dRk = p.dRk;
  double beta = p.beta;
  int Nt = p.Nt;// Nq = p.Nq;
  
  double Pnorm = 0.0;            // normalization diagram value
  
  bl::Array<int,2> Gindx, Vindx;
  bl::Array<bl::TinyVector<int,2>,1> gindx, vindx;
  int N0v=0;
  findUnique_noindx_spct(Gindx, Vindx, gindx, vindx, N0v, diagsG, Vtype, Loop_index, Loop_type);

  if (Loop_index.size()!=Ndiags) std::cerr<<"Wrong size of Loop_index "<<Loop_index.size()<<" instead of "<<Ndiags<<std::endl;
  int Nloops = Loop_index[0].size();
  if (Nloops != Norder+1) std::cerr<<"Expecting Nloops==Norder+1 in Loops_index, but got Nloops="<<Nloops<<" Norder="<<Norder<<std::endl;

  Reweight_time rw(p.beta,p.lmbdat);
  int Nbin = 513;
  //int Nbin = 129;
  //meassureWeight mweight(p.V0exp, p.cutoffk, p.kF, p.beta, Nbin);
  meassureWeight mweight(p.V0exp, p.cutoffk, kF, Nbin, Nloops);
  //bl::Array<double,2> K_hist(Nloops,Nbin); K_hist=0;
  bl::Array<double,2> T_hist((Norder-1)*2,Nbin); T_hist=0;
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
  times(0) = t_external+1e-15; // external time (for correct ordering, a small time)
  times(1) = 0;            // beginning time
  for (int it=1; it<Norder; it++){
    double tv = beta*drand();
    times(2*it  ) = tv;
    times(2*it+1) = tv;
  }
  // Next the momenta
  {
    double Qa = Q_external; // absolute value of external momentum
    momentum(0) = Qa,0.0,0.0; // external momentum Q
    amomentm(0) = Qa;
    double kf_3 = p.kF/sqrt(3.);
    for (int ik=1; ik<Nloops; ik++){
      momentum(ik) = drand()*kf_3, drand()*kf_3, drand()*kf_3;
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
  // DEBUGGING
  for (int id=0; id<Ndiags; id++){
    for (int i=0; i<2*Norder; i++){
      //int i_final = diagsG(id,i);
      int i_previ = i_diagsG(id,i);
      bl::TinyVector<double,3> k_in = mom_G(id,i_previ);
      bl::TinyVector<double,3> k_out = mom_G(id,i);
      bl::TinyVector<double,3> q = mom_V(id,i/2)*(1.-2*(i%2));
      if (fabs(norm(k_in-k_out)-norm(q))>1e-6){
	std::cerr<<"ERROR: diagram "<<id<<" vertex i="<<i<<" k_in="<<k_in<<" k_out="<<k_out<<" q="<<q<<std::endl;
      }
    }
  }
  // DEBUGGING
  // Finally evaluating the polarizations for all diagrams
  int Ng = gindx.extent(0);
  int Nv = vindx.extent(0);
  int _Nv_ = std::max(Nv,1); // can not assign 0 to empty file
  //bl::Array<double,2> G_current(Ndiags,2*Norder), V_current(Ndiags,Norder);
  bl::Array<double,1> G_current(Ng), V_current(_Nv_);
  G_current=0;
  for (int ii=0; ii<Ng; ++ii){
    int id = gindx(ii)[0];
    int  i = gindx(ii)[1];
    double aK = norm(mom_G(id,i));
    int i_final = diagsG(id,i);
    double dt = times(i_final)-times(i);
    G_current(ii) = Gk(aK, dt);
  }
  V_current=0;  
  for (int ii=0; ii<Nv; ++ii){
    int id = vindx(ii)[0];
    int  i = vindx(ii)[1];
    double aQ = norm(mom_V(id,i));
    V_current(ii) = Vqc(aQ,Vtype(id,i));
  }
  double PQ=0;
  for (int id=0; id<Ndiags; id++){
    double PQd = 1.0;
    for (int i=0; i<2*Norder; i++) PQd *= G_current(Gindx(id,i));
    for (int i=1; i<Norder; i++) PQd *= V_current(Vindx(id,i));
    PQ += PQd * diagSign(id);
  }

  bl::Array<bl::TinyVector<double,3>,2> tmom_G(Ndiags,2*Norder); // momentum for 2*Norder fermionic propagators
  bl::Array<bl::TinyVector<double,3>,2> tmom_V(Ndiags,  Norder); // momentum for Norder bosonic propagators
  bl::Array<int,1> changed_G(Ng), changed_V(Nv);
  bl::Array<double,1> G_trial(Ng), V_trial(_Nv_);
  bl::Array<double,1> times_trial(2*Norder);
    
  int Qweight = 0;      // if Qweight==1 then we have the normalization diagram
  double Nweight = 0;   // how many times did we see measuring diagram.
  int NlowT = 0;        // how many times do we see external time < 0.05*beta
  int Nmeassure = 0;    // how many times did we measure quantities.
  int Nacc_k = 0, Nacc_q = 0, Nacc_w = 0, Nacc_t = 0;
  int Nall_k = 0, Nall_q = 0, Nall_w = 0, Nall_t = 0;
  if (mpi.rank==mpi.master) PrintInfo_(0, Qweight, amomentm(0), amomentm(1), times(0)-times(1), PQ, PQ, 0, 0);
  double Pt0=0;

  t_all.start();
  for (int itt=0; itt<p.Nitt; itt++){
    double PQ_new=0;
    int icase = tBisect(drand(), Prs); // which steps should we attempt?
    if (Norder==1){
      // only at order 1 we have no time variable to move, hence icase==1 should not occur
      while (icase==1) icase = tBisect(drand(), Prs); 
    }
    double Ka_new;

    if (icase==0){ // changing momentum Q or k-point
      int iloop = 1 + static_cast<int>((Nloops-1)*drand()); // newer change measuring (0) loop
      bl::TinyVector<double,3> K_new; double trial_ratio=1;
      bool accept=false;
      Nall_k += 1;
      accept = Find_new_k_point(K_new, Ka_new, trial_ratio, momentum(iloop), amomentm(iloop), cutoffk, drand, drand()<p.Qring, dkF, dRk);
      if (accept){
	bl::TinyVector<double,3> dK = K_new - momentum(iloop); // how much momentum will be changed.
	tmom_G = mom_G; tmom_V = mom_V;
	changed_G = 0; changed_V = 0;
	for (int id=0; id<Ndiags; id++){ // updating momentum in all propagators
	  const std::vector<int>& lindex = Loop_index[id][iloop];
	  const std::vector<int>& ltype  = Loop_type[id][iloop];
	  for (int i=0; i<lindex.size(); i++){
	    if ( abs(ltype[i])==1 ){
	      tmom_G(id, lindex[i]) += dK * dsign(ltype[i]);
	      changed_G(Gindx(id, lindex[i]))=1;
	    }else{
	      if (lindex[i]>0){
		tmom_V(id,lindex[i]) += dK * dsign(ltype[i]);
		changed_V(Vindx(id, lindex[i]))=1;
	      }
	    }
	  }
	}
	if (! Qweight){ // we computed the polarization diagram
	  for (int ii=0; ii<Ng; ++ii){
	    if (changed_G(ii)){
	      int id = gindx(ii)[0];
	      int  i = gindx(ii)[1];
	      double aK = norm(tmom_G(id,i));
	      int i_final = diagsG(id,i);
	      double dt = times(i_final)-times(i);
	      G_trial(ii) = Gk(aK, dt);
	    }
	  }
	  for (int ii=0; ii<Nv; ++ii){
	    if (changed_V(ii)){
	      int id = vindx(ii)[0];
	      int  i = vindx(ii)[1];
	      double aQ = norm(tmom_V(id,i));
	      V_trial(ii) = Vqc(aQ, Vtype(id,i));
	    }
	  }
	  PQ_new=0;
	  for (int id=0; id<Ndiags; id++){
	    double PQd = 1.0;
	    for (int i=0; i<2*Norder; i++){
	      int ii = Gindx(id,i);
	      PQd *= ( changed_G(ii) ? G_trial(ii) : G_current(ii) );
	    }
	    for (int i=1; i<Norder; i++){
	      int ii = Vindx(id,i);
	      PQd *= ( changed_V(ii) ? V_trial(ii) : V_current(ii) );
	    }
	    PQ_new += PQd * diagSign(id);
	  }
	}else{
	  double Ka_old = amomentm(iloop);
	  amomentm(iloop) = Ka_new;
	  bl::TinyVector<double,3> K_old = momentum(iloop);
	  momentum(iloop) = K_new;
	  PQ_new = V0norm * mweight(amomentm, momentum/*,times*/);
	  amomentm(iloop) = Ka_old;
	  momentum(iloop) = K_old;
	  //std::cout<<"PQ_meassure="<<PQ_new<<" k="<<norm(momentum(1))/kF<<std::endl;
	}
	double ratio = PQ_new*trial_ratio / PQ;
	accept = fabs(ratio) > 1-drand() && PQ_new!=0;
      }
      if ((itt+1)%Ncout==0)
	if (mpi.rank==mpi.master)
	  _PrintInfo_(itt+1, Qweight, icase, amomentm(1), amomentm(Nloops-1), times(0), PQ_new, PQ, NlowT/(Nmeassure+0.0), Nweight/(Nmeassure+0.0));
      if (accept){
	Nacc_k += 1;
	momentum(iloop)  = K_new;  // this momentum was changed
	amomentm(iloop) = Ka_new;
	mom_G = tmom_G;           // and therefore many momenta in diagrams have changed. We could optimize this, and 
	mom_V = tmom_V;           // change only those momenta that actually change....
	if (!Qweight){
	  for (int ii=0; ii<Ng; ++ii)
	    if (changed_G(ii))
	      G_current(ii) = G_trial(ii); // The green's functions in the loop have been recalculated, and are now stored.
	  for (int ii=0; ii<Nv; ++ii)
	    if (changed_V(ii))
	      V_current(ii) = V_trial(ii); // The interactions in the loop have been recalculated and stored.
	}else{
	  // since we have the measuring diagram, we did not calculate G_trial and V_trial before, hence we need to evaluate them now.
	  for (int ii=0; ii<Ng; ++ii){
	    if (changed_G(ii)){
	      int id = gindx(ii)[0];
	      int  i = gindx(ii)[1];
	      double aK = norm(tmom_G(id,i));
	      int i_final = diagsG(id,i);
	      double dt = times(i_final)-times(i);
	      G_current(ii) = Gk(aK, dt);
	    }
	  }
	  for (int ii=0; ii<Nv; ++ii){
	    if (changed_V(ii)){
	      int id = vindx(ii)[0];
	      int  i = vindx(ii)[1];
	      double aQ = norm(tmom_V(id,i));
	      V_current(ii) = Vqc(aQ, Vtype(id,i));
	    }
	  }
	  //t_k5.stop();
	}
	PQ = PQ_new;
	//if (iloop==0) Plq.cmp_single(2*amomentm(iloop)/cutoffq-1., pl_Q);  // update Legendre Polynomials
      }
    } else if (icase==1){  // changing one of the time variables
      Nall_t += 1;
      int itime = 1 + static_cast<int>((Norder-1)*drand()); // which time to change? For bare interaction, there are only Norder different times. but we never change external time here.
      changed_G=0; changed_V=0;  // which propagators are being changed?
      times_trial = times;       // times_trial will contain the trial step times.

      if (itime==0){           // this is the measuring time with vertex=0
	double t_new = rw.gtau(drand());
	if (t_new<1e-15) t_new = 1e-15; // to be on the right branchcut
	times_trial(0) = t_new;
	int ivertex=0;
	for (int id=0; id<Ndiags; id++){
	  int i_pre_vertex = i_diagsG(id,ivertex);
	  changed_G(Gindx(id,ivertex))=1;         // these two propagators contain vertex=0.
	  changed_G(Gindx(id,i_pre_vertex))=1;
	}
      }else{
	double t_new = beta*drand();
	// unscreened interaction, hence changing two times simultaneously!
	for (int ivertex=2*itime; ivertex<=2*itime+1; ++ivertex){ // 2*itime and 2*itime+1 are changed at the same time, because they are equal
	  times_trial(ivertex) = t_new;
	  for (int id=0; id<Ndiags; id++){
	    int i_pre_vertex = i_diagsG(id,ivertex);
	    changed_G(Gindx(id,ivertex))=1;        // these two propagators are changed because of the new time.
	    changed_G(Gindx(id,i_pre_vertex))=1;
	  }
	}
      }
      
      if (! Qweight){
	for (int ii=0; ii<Ng; ++ii){
	  if (changed_G(ii)){
	    int id = gindx(ii)[0];
	    int  i = gindx(ii)[1];
	    double aK = norm(mom_G(id,i));
	    int i_final = diagsG(id,i);
	    double dt = times_trial(i_final)-times_trial(i);
	    G_trial(ii) = Gk(aK, dt);
	  }
	}
	PQ_new=0; // recalculating PQ, taking into account one change of time.
	for (int id=0; id<Ndiags; id++){
	  double PQd = 1.0;
	  for (int i=0; i<2*Norder; i++){
	    int ii = Gindx(id,i);
	    PQd *= ( changed_G(ii) ? G_trial(ii) : G_current(ii) );
	  }
	  for (int i=1; i<Norder; i++) PQd *= V_current(Vindx(id,i)); // interaction does not depend on time, hence it is not changed here.
	  PQ_new += PQd * diagSign(id);
	}
      }else{
	PQ_new = V0norm * mweight(amomentm, momentum/*, times_trial*/);
	//std::cout<<"PQ_meassure="<<PQ_new<<" k="<<norm(momentum(1))/kF<<std::endl;
      }
      double ratio = PQ_new/PQ;
      bool accept = fabs(ratio) > 1-drand() && PQ_new!=0;
      if ((itt+1)%Ncout==0)
	if (mpi.rank==mpi.master)
	  _PrintInfo_(itt+1, Qweight, icase, amomentm(1), amomentm(Nloops-1), times(2*itime), PQ_new, PQ, NlowT/(Nmeassure+0.0), Nweight/(Nmeassure+0.0));
      if (accept){
	Nacc_t +=1 ;
	times = times_trial;  // the new times are now accepted. We change the independent time variable, or two internal times.
	if (!Qweight){
	  for (int ii=0; ii<Ng; ++ii)
	    if (changed_G(ii))
	      G_current(ii) = G_trial(ii); // The green's functions in the loop have been recalculated, and are now stored.
	}else{
	  // since we have the measuring diagram, we did not calculate G_trial and V_trial before, hence we need to evaluate them now.
	  for (int ii=0; ii<Ng; ++ii){
	    if (changed_G(ii)){
	      int id = gindx(ii)[0];
	      int  i = gindx(ii)[1];
	      int i_final = diagsG(id,i);
	      G_current(ii) = Gk( norm(mom_G(id,i)),  times(i_final)-times(i)); // these propagators were changed, but they were not computed before.
	    }
	  }
	}
	PQ = PQ_new;
	//if (itime==0) Plt.cmp_single(2*(times(0)-times(1))/beta-1., ul_t);  // update Legendre Polynomials
      }
    }else{  // normalization diagram step
      Nall_w += 1;
      if (Qweight){ // trying to jump back into physical hilbert space
	PQ_new=0;
	for (int id=0; id<Ndiags; id++){
	  double PQd = 1.0;
	  for (int i=0; i<2*Norder; i++) PQd *= G_current(Gindx(id,i)); // we have all propagators precomputed, hence it is easy to compute the diagrams.
	  for (int i=1; i<Norder; i++) PQd *= V_current(Vindx(id,i));
	  PQ_new += PQd * diagSign(id);
	}
      }else{        // or if we are in the physical Hilbert space, we try to jump into the normalization diagram
	PQ_new = V0norm * mweight(amomentm, momentum/*, times*/);
	//std::cout<<"PQ_meassure="<<PQ_new<<" k="<<norm(momentum(1))/kF<<std::endl;
      }
      double ratio = PQ_new/PQ;
      bool accept = fabs(ratio) > 1-drand() && PQ_new!=0;
      if ((itt+1)%Ncout==0)
	if (mpi.rank==mpi.master)
	  _PrintInfo_(itt+1, Qweight, icase, amomentm(1), amomentm(Nloops-1), times(0), PQ_new, PQ, NlowT/(Nmeassure+0.0), Nweight/(Nmeassure+0.0));
      if (accept){
	Nacc_w += 1;
	Qweight = 1-Qweight;
	PQ = PQ_new;
      }
    }
    if (PQ==0){
      std::cout<<itt<<"  PQ_new="<<PQ_new<<" which is strange icase="<<icase<<" Qweight="<<Qweight<<" PQ="<<PQ<<" ... Recomputing..."<<std::endl;
      if (icase==0) std::cout << " Ka_new=" << Ka_new/cutoffk << std::endl;
      /*
      {
	VerifyCurrentState(itt, beta, Qweight, PQ, diagsG, i_diagsG, diagSign, Gk, Vqc, Loop_index, Loop_type, momentum, times, mom_G, mom_V, G_current, V_current);
	
	double PQ_tmp=0;
	for (int id=0; id<Ndiags; id++){
	  double PQd = 1.0;
	  std::cout<<"---- diag="<<id<<std::endl;
	  for (int i=0; i<2*Norder; i++){
	    PQd *= G_current(id,i); // we have all propagators precomputed, hence it is easy to compute the diagrams.
	    std::cout<<" G("<<i<<")="<<G_current(id,i)<<" PQd="<<PQd<<std::endl;
	  }
	  for (int i=1; i<Norder; i++){
	    PQd *= V_current(id,i);
	    std::cout<<" V("<<i<<")="<<V_current(id,i)<<" PQd="<<PQd<<std::endl;
	  }
	  PQ_new += PQd * diagSign(id);
	  std::cout<<" sign*PQd="<<PQd * diagSign(id)<<std::endl;
	}
      }
      */
    }
    // Measuring
    if (itt>=Nwarm && itt%tmeassure==0){
      t_mes.start();
      Nmeassure += 1;
      //double Qa = amomentm(0);
      //double t = times(0)-times(1);
      //double cw = 1./rw.wt(t);
      double cw = 1.0;
      double sp = sign(PQ) * cw;
      //int iq = min(int(Qa/cutoffq*Nq), Nq-1);
      //int it = min(int(t/beta*Nt), Nt-1);
      if (Qweight){
	Pnorm += /*Qa*Qa*/ cw;                 // |Q|^2 * reweighting
	Nweight+=1;
      }else{
	Pt0 += sp; 
      }

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

      if ( itt>5e5 && itt%(50000*tmeassure) == 0){
	double occurence = static_cast<double>(Nweight)/static_cast<double>(Nmeassure);
#ifdef _MPI
	//std::cout<<"  occurence on rank "<<mpi.rank<<" is "<<occurence<<std::endl;
	double total_occurence=0;
	MPI_Allreduce(&occurence, &total_occurence, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	occurence = total_occurence/mpi.size;
	//std::cout<<"  so that average occurence on rank "<<mpi.rank<<" is "<<occurence<<std::endl;
#endif	
	if ( occurence > 0.25){ // decrease by two
	  V0norm /= 2.;
	  Pnorm /= 2.;
	  if (mpi.rank==mpi.master) std::cout<<"occurence="<<occurence<<" V0 reduced to "<<V0norm<<std::endl;
	}
	if ( occurence < 0.02){ // increase by two
	  V0norm *= 2.;
	  Pnorm *= 2.;
	  if (mpi.rank==mpi.master) std::cout<<"occurence="<<occurence<<" V0 increased to "<<V0norm<<std::endl;
	}
      }

      if (itt%(500000*tmeassure) == 0){
	if (Nmeassure-Nweight>1000){
#ifdef _MPI	  
	  MPI_Allreduce(MPI_IN_PLACE, mweight.K_hist.data(), mweight.K_hist.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
	  dk_hist *= mweight.Normalize_K_histogram();
	  mweight.Recompute(/*K_hist, T_hist, */mpi.rank==mpi.master); // Nweight has to be finite, otherwise there is no information stored anyway.
	  if (Qweight) PQ = V0norm * mweight(amomentm, momentum/*, times*/);
	}
      }
      t_mes.stop();
    }
  }
  t_all.stop();
  Pt0      *= 1.0/Nmeassure;
  Pnorm     *= 1.0/Nmeassure;
  double occurence = static_cast<double>(Nweight)/static_cast<double>(Nmeassure);
  
  //mweight.K_hist *= 1.0/Nmeassure;
  T_hist *= 1.0/Nmeassure;

#ifdef _MPI  
  double dat[3] = {Pnorm, occurence, Pt0};
  if (mpi.rank==mpi.master){
    MPI_Reduce(MPI_IN_PLACE, dat, 3, MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, mweight.K_hist.data(), mweight.K_hist.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, T_hist.data(), T_hist.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
  }else{
    MPI_Reduce(dat, dat, 3, MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    MPI_Reduce(mweight.K_hist.data(), mweight.K_hist.data(), mweight.K_hist.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    MPI_Reduce(T_hist.data(), T_hist.data(), T_hist.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
  }
  if (mpi.rank==mpi.master){
    Pnorm     = dat[0]/mpi.size;
    occurence = dat[1]/mpi.size;
    Pt0       = dat[2]/mpi.size;
    //
    mweight.K_hist *= 1./mpi.size;
    T_hist *= 1./mpi.size;
  }
#endif
  std::clog<<"total acceptance rate="<<(Nacc_k+Nacc_q+Nacc_t+Nacc_w)/(p.Nitt+0.0)<<" "<<" k-acceptance="<<(Nacc_k/(Nall_k+0.0))<<" t-acceptanece="<<(Nacc_t/(Nall_t+0.0))<<" w-acceptanece="<<(Nacc_w/(Nall_w+0.0))<<std::endl;
  std::clog<<" k-trials="<<(Nall_k/(p.Nitt+0.0))<<" q-trials="<<(Nall_q/(p.Nitt+0.0))<<" t-trials="<<(Nall_t/(p.Nitt+0.0))<<" w-trials="<<(Nall_w/(p.Nitt+0.0))<<std::endl;
  std::clog<<"total time="<<t_all.elapsed()<<" measuring time="<<t_mes.elapsed()<<std::endl;
  
  if (mpi.rank==mpi.master){
    Pt0  *= (fabs(V0norm)/Pnorm);
    std::clog<<"  meassuring diagram occurence frequency="<<occurence<<" and its norm Pnorm="<<Pnorm<<std::endl;
    
    for (int ik=0; ik<Nloops; ik++){
      double dsum=0;
      for (int i=0; i<Nbin; i++) dsum += mweight.K_hist(ik,i);
      mweight.K_hist(ik,bl::Range::all()) *= 1./dsum;
      std::ofstream hout((std::string("Q_hist.")+std::to_string(ik)).c_str());
      double cutoff = (ik>0)? cutoffk : cutoffq;
      for (int i=0; i<Nbin; i++){
	hout<< (i+0.5)/Nbin*cutoff/kF << " " << mweight.K_hist(ik,i)<<std::endl;
      }
    }
    for (int it=2; it<2*Norder; it++){
      std::ofstream hout((std::string("T_hist.")+std::to_string(it)).c_str());
      for (int i=0; i<Nbin; i++){
	hout<< (i+0.5)/Nbin*beta << " " << T_hist(it-2,i)<<std::endl;
      }
    }
  }
  return Pt0;
}



void PrintLogInformation(ostream& log, int itt, int icase, double PQ_new, double iseed, int Qweight,
			 const bl::Array<double,1>& G_current, 
			 const bl::Array<long double,1>& V_current, 
			 const BitArray& changed_G, const BitArray& changed_V,
			 int Ndiags, int Norder, int Nv, 
			 const bl::Array<unsigned short,2>& Gindx, const bl::Array<unsigned short,2>& Vindx,
			 const bl::Array<double,1>& g_trial,
			 const bl::Array<unsigned short,1>& ip_ind,
			 const bl::Array<long double,1>& v_trial,
			 const bl::Array<unsigned short,1>& iq_ind,
			 const BitArray& changed_Vrtx,
			 const bl::Array<long double,1>& Vrtx_current,
			 const bl::Array<long double,1>& Vrtx_trial)
{
	  
  log << "ERROR : PQ_new="<<PQ_new<<" starting with iseed="<<iseed<<" at itt="<<itt<<" icase="<<icase<<" Qweight="<< Qweight << endl;
  log << "G_current="<<G_current << " V_current="<< V_current << endl;
  log << "changed_G=" << changed_G << endl;
  log << "changed_V=" << changed_V << endl;
  log << " ...contributions" << endl;
  for (int id=0; id<Ndiags; id++){
    log << "id="<< id << " gg= ";
    for (int i=1; i<2*Norder; i++){
      int ii = Gindx(id,i);
      double gg =  changed_G[ii] ? g_trial(ip_ind(ii)) : G_current(ii);
      log << gg << " * ";
    }
    log << " vv= ";
    for (int i=1; i<Norder; i++){
      int ii = Vindx(id,i);
      long double vv = changed_V[ii] ? v_trial(iq_ind(ii)) : V_current(ii);
      log << vv << " * ";
    }
    log << endl;
  }
  log<< "changed_Vrtx=" << changed_Vrtx << endl;
  for (int ii=0; ii<Nv; ++ii){
    long double Vrtx = changed_Vrtx[ii] ? Vrtx_trial(ii) : Vrtx_current(ii);
    log << "Vrtx[="<<ii<<"]= "<< Vrtx << " which is changed "<< changed_Vrtx[ii] << endl;
  }
}



template<typename GK>
std::tuple<double,double,double,double> sample_Density_D(std::ostream& log, double lmbda, const std::vector<double>& lmbda_spct, const GK& Gk, const params& p,
							 const bl::Array<unsigned short,2>& diagsG, const bl::Array<float,1>& diagSign,
							 const std::vector<std::vector<std::vector<int> > >& Loop_index, const std::vector<std::vector<std::vector<int> > >& Loop_type,
							 bl::Array<char,2>& Vtype, my_mpi& mpi)
{
  int bubble_only=0;
  typedef double real;
  log.precision(12);
  //log<<"Inside sample_Density_D"<<endl;
  
  const int Nr=3; // three different steps in this algorithm
  double Pr_norm = (p.Pr[0]+p.Pr[1]+p.Pr[2]); // probability for all four has to be normalized
  bl::TinyVector<double,Nr> Prs;  // cumulative probability for taking a step, needed for bisection below.
  Prs[0] = p.Pr[0]/Pr_norm;
  for (int i=1; i<Nr; i++) Prs[i] = Prs[i-1] + p.Pr[i]/Pr_norm; // prepares cumulative probability from p.Pr
  
  int Ndiags = diagsG.extent(0);
  if (Ndiags==0) {
    log<<"No diagrams to simulate...."<<std::endl;
    return std::make_tuple(0.,0.,0.,0.);
  }
  int Norder = diagsG.extent(1)/2;
  // computing the inverse of all permutations, to provide fast lookup for the previous fermionic propagator
  bl::Array<unsigned short,2> i_diagsG(Ndiags,2*Norder);
  for (int id=0; id<Ndiags; id++)
    for (int i=0; i<2*Norder; i++)
      i_diagsG(id, static_cast<int>(diagsG(id,i)) )=i;
  
  RanGSL drand(p.iseed); // GSL random number generator
  Timer t_mes, t_all, t_Pl, t_k, t_Q, t_w, t_t;
  
  //CounterCoulombV Vqc(lmbda);

  // Set up which times need change when counter term is dynamic
  vector<int> times_2_change;
  std::set<int> times_to_change;
  int Vcmax=0;
  {
    std::set<int> counter_types;
    for (int i=1; i<Norder; i++) times_to_change.insert(2*i);
    // default vertices to change time will be : [0,2,...,2*(Norder-1)]
    for (int id=0; id<Ndiags; id++)    // Checking where can we add single-particle counter terms. 
      for (int j=1; j<Norder; j++)     // skip meassuring line
	if (Vtype(id,j)>0){
	  counter_types.insert(Vtype(id,j));
	  times_to_change.insert(2*j+1);  // since the counter-term is dynamic, we need to make vertex 2j+1 different from 2j.
	}
    if (counter_types.size()>0){
      Vcmax = *counter_types.rbegin();
    }
    times_2_change.resize(times_to_change.size());
    int j=0;
    for (auto it=times_to_change.begin(); it!=times_to_change.end(); ++it, ++j) times_2_change[j] = (*it);
    log<<"times_to_change : ";
    for (auto it=times_to_change.begin(); it!=times_to_change.end(); ++it) log << (*it) <<",  ";
    log<<std::endl;
    log<<"counter_types : ";
    for (auto it=counter_types.begin(); it!=counter_types.end(); ++it) log << (*it) <<",  ";
    log<<std::endl;
    log << "times_2_change: "<<endl;
    for (int i=0; i<times_2_change.size(); i++) log << times_2_change[i] << ",";
    log << endl;
  }
  // Reads numerically evaluated Bubble diagram.
  vector<Spline2D<double> > bPqt(Vcmax);
  bl::Array<double,1> qx, tx;  
  if (Vcmax>0) ReadCounterTermPolarization(bPqt, qx, tx, p.beta);
  bl::Array<int,2> binomial = BinomialCoeff(Vcmax);
  
  int tmeassure=p.tmeassure, Ncout=p.Ncout, Nwarm=p.Nwarm;
  double V0norm = p.V0norm, dkF = p.dkF, kF=p.kF;// V0exp= p.V0exp;
  double cutoffk = p.cutoffk, dRk = p.dRk;
  double beta = p.beta;

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
  findUnique_noindx_spct(Gindx, Vindx, gindx, vindx, N0v, diagsG, Vtype, Loop_index, Loop_type, 1, true, debug, log);

  // Once more we go through single-particle counter terms and create index single_counter_index
  bl::Array<int,1> single_counter_index;
  Find_Single_Particle_Counter_G_index(single_counter_index, Ndiags, Norder, vindx.extent(0), diagsG, Vtype, Gindx, Vindx);
  
  if (Loop_index.size()!=Ndiags){ log<<"Wrong size of Loop_index "<<Loop_index.size()<<" instead of "<<Ndiags<<std::endl; exit(1);}
  int Nloops = Loop_index[0].size();
  // Here we are sampling density, which has one loop less than polarization!
  if (Nloops != Norder){ log<<"Expecting Nloops==Norder in Loops_index, but got Nloops="<<Nloops<<" Norder="<<Norder<<std::endl; exit(1);}
  int Ng = gindx.extent(0);
  int Nv = vindx.extent(0);
  int _Nv_ = std::max(Nv,1); // can not assign 0 to empty file

  bl::Array<bl::Array<sint,1>,1> loop_Gkind, loop_Vqind;
  bl::Array<bl::Array<char,1>,1> loop_Gksgn, loop_Vqsgn;
  int Ngp, Nvp;
  // Find index-tables to all momentum loops, so that we do not need to loop over all diagrams, but rather only over all differet propagators corresponding to certain loop.
  // Here we also take into account that single-particle counter terms are sometimes added.
  Get_GVind(loop_Gkind, loop_Gksgn, loop_Vqind, loop_Vqsgn, Ngp, Nvp, Loop_index, Loop_type, Gindx, Vindx, single_counter_index, lmbda_spct, Vtype, true, log);
  int Ngp_tr = GetMax_NGp_trial(i_diagsG, Gindx, Ng, Ngp);
  
  // DEBUG
  if (mpi.rank==mpi.master){
    log<<"lmbda_spct=";
    for (int i=0; i<lmbda_spct.size(); ++i) log<<lmbda_spct[i]<<",";
    log<<std::endl;
    log<<"single_counter_index="<<single_counter_index<<std::endl;
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
    log<<"loop_Vqind:"<<std::endl;
    for (int iloop=0; iloop<Nloops; ++iloop){
      for (int iq=0; iq<loop_Vqind(iloop).extent(0); ++iq){ // Now do the same for the interaction
	int ii   = loop_Vqind(iloop)(iq);
	int isgn = loop_Vqsgn(iloop)(iq);
	int id = vindx(ii)[0];
	int  i = vindx(ii)[1];
	int vtyp = Vtype(id,i);
	log<<"iloop="<<iloop<<" iq="<<iq<<" ii_v="<<ii<<" isgn="<<isgn<<" which comes from id="<< id <<" and pair =("<<2*i<<","<<2*i+1<<") of type="<<vtyp<<std::endl;
      }
    }
  }
  // END DEBUG
  
  int Nbin = 513;
  //  int Nbin = 129;
  bl::Array<double,1> lmbda_spct_cutoffk(lmbda_spct.size());
  // Instead of using : lmbda_spct, which is q-independent, we will introduce "artificial" q-dependence, which gives identical integral.
  // When lmbda_spct is q-independent, we should normalize it by  lmbda_spct * 1/Integrate[d^3 1/(2*pi)^3].
  // The "artificial" dependence we choose is lmbda_spct * (Vq)^2, in which case the normalization constant is
  //     lmbda_spct * (Vq)^2 / Integrate[d^3q Vq^2/(2*pi)^3 ]
  //  which is
  //     lmbda_spct * (Vq)^2 / ( 2/sqrt(lmbda) *atan(ck/sqrt(lmbda)) - 2*ck/(ck^2+lmbda) )
  std::vector<double> nrm = Nrm(lmbda, cutoffk, lmbda_spct.size()+2);
  for (int i=0; i<lmbda_spct.size(); ++i){
#ifdef CNTR_VN
    // [Vq^2, Vq^3, Vq^4,..]
    lmbda_spct_cutoffk(i) = lmbda_spct[i]*nrm[i+2];
#else    
    // exp(-2*q/kF)
    lmbda_spct_cutoffk(i) = lmbda_spct[i] * pi*pi*8/(kF*kF*kF*(1. - exp(-2*cutoffk/kF)*ipower(cutoffk/kF,2)*(1. + ipower(1.+kF/cutoffk,2) )));
#endif    
  }
  meassureWeight mweight(p.V0exp, p.cutoffk, p.kF, Nbin, Nloops, 0);
  double dk_hist = 1.0;
  
  // This is momentum for loops
  bl::Array<bl::TinyVector<double,3>,1> momentum(Nloops); // momentum for all loops. Each diagram of the same order has the same number of loops.
  bl::Array<double,1> amomentm(Nloops);               // absolute value of momentum for each loop
  
  // Starting point is set up to be random
  // For the times for all vertices in all the graphs
  bl::Array<double,1> times(2*Norder);
  times(0) = -1e-15; // external time
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
  {
    for (int ik=0; ik<Nloops; ik++){
      double th = drand()*pi, ph = drand()*(2*pi);
      double st = sin(th), ct = cos(th), cp = cos(ph), sp = sin(ph);
      momentum(ik) = kF*st*cp, kF*st*sp, kF*ct;
      amomentm(ik) = kF;
    }
  }
  log.precision(12);
  log<<"Starting times="<<times<<" and starting momenta="<<momentum<<std::endl;

  bl::Array<double,1> G_current(Ng);
  bl::Array<long double,1> V_current(_Nv_);
  // This is momentum for all diagrams and all propagators
  bl::Array<bl::TinyVector<real,3>,1> mom_g(Ng), mom_v(Nv);
  bl::Array<long double,1> Vrtx_current(1);
  if (Vcmax>0) Vrtx_current.resize(_Nv_);
  bl::Array<double,1> PQg(Ndiags);
  double PQ=0;
  {

    log<<"before calculating momenta"<<endl;
    log.flush();

    
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
      for (int i=0; i<2*Norder; i++){
	//int i_final = diagsG(id,i);
	int i_previ = i_diagsG(id,i);
	bl::TinyVector<double,3> k_in = mom_G(id,i_previ);
	bl::TinyVector<double,3> k_out = mom_G(id,i);
	bl::TinyVector<double,3> q = mom_V(id,i/2)*(1.-2*(i%2));
	if (fabs(norm(k_in-k_out-q))>1e-6){
	  log<<"ERROR: diagram "<<id<<" vertex i="<<i<<" k_in="<<k_in<<" k_out="<<k_out<<" q="<<q<<std::endl;
	  log<<"   i_previ="<<i_previ<<" q_i="<< (i/2) <<" mom_V="<<mom_V(id,i/2)<<std::endl;
	}
      }
    }
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
    Vrtx_current=0;
    // Vertices, which are needed to express screened interaction with the same time difference as the unscreened part.
    for (int ii=0; ii<Nv; ++ii){
      int id = vindx(ii)[0];
      int  i = vindx(ii)[1];
      int vtyp = Vtype(id,i);
      if (vtyp!=0){
	double t1 = times(2*i);     // t1 is old time, because for equal time case t[2i+1] was actually t[2i]
	double t2 = times(2*i+1);   // t2 is the new time, because now we want t[2i+1] to be different.
	// For now we will neglect possibility of type-II vertex. Should correct later!
	int i_m = i_diagsG(id,2*i+1);
	double ti = times( et(i_m) );
	int i_p = diagsG(id,2*i+1);
	double to = times( et(i_p) );
	double ki = norm(mom_G(id,i_m));
	double ko = norm(mom_G(id,2*i+1));
	Vrtx_current(ii) = MoveVertex(t2, t1, ti, to, ki, ko, Gk);
	//double direct_vertex = Gk(ki, t2-ti)*Gk(ko, to-t2)/(Gk(ki, t1-ti)*Gk(ko, to-t1) );
	//log<<"t1="<<t1<<" t2="<<t2<<" i_m="<<i_m<<" ti="<<ti<<" i_p="<<i_p<<" to="<<to<<" ki="<<ki<<" ko="<<ko<<" vrtx="<<Vrtx_current(ii)<<"  direct_vrtx="<<direct_vertex<<endl;
	if ((Vtype(id,i_m/2)!=0 && (i_m%2==1)) || (Vtype(id,i_p/2)!=0 && (i_p%2==1))){
	  log<<"ERROR : We have type2, which is not yet implemented!  id="<<id<<" i="<<i<<" vtyp="<<vtyp<<endl;
	  log<<"..... i_m="<<i_m<<" i_p="<<i_p<<" 2*i="<< 2*i << endl;
	  log<<" 1 : "<< (Vtype(id,i_m/2)!=0 && (i_m != 2*i)) << " 2 : "<< (Vtype(id,i_p/2)!=0 && (i_p != 2*i)) << endl;
	}
      }
    }
    //log<<"Vrtx_current="<<Vrtx_current<<endl;
    V_current=0;  
    for (int ii=0; ii<Nv; ++ii){
      int id = vindx(ii)[0];
      int  i = vindx(ii)[1];
      double aQ = norm(mom_V(id,i));
      int vtyp = Vtype(id,i);
      double Vq = 8*pi/(aQ*aQ+lmbda);
      if (vtyp==0){
	V_current(ii) = Vq;
      }else{
	intpar pq = Interp(aQ,qx);
	intpar pt = Interp(fabs(times(2*i)-times(2*i+1)), tx);
	int Nc = abs(vtyp);
	double ct=0;
	double lmbda_k = 1.0;
	for (int k=0; k<Nc; k++){
	  ct += binomial(Nc,k) * lmbda_k * bPqt[Nc-k-1](pq,pt);
	  lmbda_k *= (lmbda/(8*pi));
	}
	long double Vn = ipower(Vq, Nc+1);
	V_current(ii) = Vn * ct * Vrtx_current(ii);
	V_current(ii) += Vn * lmbda_k *(1-bubble_only);
	if (vtyp < 0){
	  // we also add the single-particle counter term in addition to two-particle counter term.
	  int ii_g = single_counter_index(ii);
	  long double g_kq = G_current(ii_g);
	  //if (g_kq!=0) V_current(ii) += lmbda_spct_cutoffk(-vtyp-1)/g_kq;
#ifdef CNTR_VN
	  if (g_kq!=0) V_current(ii) += Vn*lmbda_spct_cutoffk(-vtyp-1)/g_kq;
#else	  
	  if (g_kq!=0) V_current(ii) += exp(-aQ*2/kF)*lmbda_spct_cutoffk(-vtyp-1)/g_kq;
#endif	  
	  
	}
      }
    }
    //log<<"V_current="<<V_current<<endl;
    PQ=0;
    for (int id=0; id<Ndiags; id++){
      long double PQd = 1.0;
      for (int i=1; i<2*Norder; i++) PQd *= G_current(Gindx(id,i)); // CAREFULL Density does use G(0)!
      for (int i=1; i<Norder; i++) PQd *= V_current(Vindx(id,i));
      PQd *= diagSign(id);
      PQ += PQd;
      PQg(id) = PQd;
    }
    // Now we create smaller arrays mom_g and mom_v, which contain the same information as mom_G and mom_V.
    mom_g=0.0; mom_v=0.0;
    for (int id=0; id<Ndiags; id++){
      for (int i=1; i<2*Norder; i++){
	int ii = Gindx(id,i);
	mom_g(ii) = mom_G(id,i);
      }
      for (int i=1; i<Norder; i++){
	int ii = Vindx(id,i);
	mom_v(ii) = mom_V(id,i);
      }
    }
  }
  
  bl::Array<double,1> times_trial(2*Norder);
  int _Nvp_ = std::max(Nvp,1); // can not assign 0 to empty file
  bl::Array<bl::TinyVector<real,3>,1> tmom_g(Ngp), tmom_v(_Nvp_);
  bl::Array<sint,1> ip_ind(Ng), iq_ind(Nv);
  bl::Array<double,1> g_trial(Ngp_tr);
  bl::Array<long double,1> v_trial(_Nvp_);
  BitArray changed_G(Ng), changed_V(Nv);

  BitArray changed_Vrtx(_Nv_);
  bl::Array<long double,1> Vrtx_trial(1);
  if (Vcmax>0){
    Vrtx_trial.resize(_Nv_);
  }
  
  int Qweight = 0;      // if Qweight==1 then we have the normalization diagram
  double Nweight = 0;   // how many times did we see measuring diagram.
  //int NlowT = 0;        // how many times do we see external time < 0.05*beta
  int Nmeassure = 0;    // how many times did we measure quantities.
  int Nchanged  = 0;    // when was V0norm last changed?
  double Wchanged = 0;  // what was Nweight when we last time changed V0norm?
  int Nacc_k = 0, Nacc_q = 0, Nacc_w = 0, Nacc_t = 0;
  int Nall_k = 0, Nall_q = 0, Nall_w = 0, Nall_t = 0;
  if (mpi.rank==mpi.master) PrintInfo_(0, Qweight, amomentm(0), amomentm(1), times(0)-times(1), PQ, PQ, 0, 0, log);
  double Pw0=0, Ekin=0;
  bl::Array<double,1> PQg_new(Ndiags);
  
  //Check(0, log, -1, Nv, PQ, lmbda, times, times_to_change, Gk, G_current, V_current, diagsG, Gindx, Vindx, mom_g, mom_v, diagSign, Vtype, bPqt, qx, tx, vindx, Vrtx_current);
  t_all.start();
  for (int itt=0; itt<p.Nitt; itt++){
    double PQ_new=0;
    int icase = tBisect(drand(), Prs); // which steps should we attempt?
    if (Norder==1){
      // only at order 1 we have no time variable to move, hence icase==1 should not occur
      while (icase==1) icase = tBisect(drand(), Prs); 
    }

    if (icase==0){ // changing momentum Q or k-point
      int iloop = static_cast<int>(Nloops*drand());
      bl::TinyVector<double,3> K_new; double Ka_new; double trial_ratio=1;
      bool accept=false;
      Nall_k += 1;
      accept = Find_new_k_point(K_new, Ka_new, trial_ratio, momentum(iloop), amomentm(iloop), cutoffk, drand, drand()<p.Qring, dkF, dRk);
      if (accept){
	if (!Qweight){
	  bl::TinyVector<double,3> dK = K_new - momentum(iloop); // how much momentum will be changed.
	  //t1.start(); // takes 40% of the time
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
	  changed_Vrtx = 0;
	  // Vertices, which are needed to express screened interaction with the same time difference as the unscreened part.
	  if (Vcmax>0){
	    for (int id=0; id<Ndiags; id++){ // updating momentum in all propagators
	      const std::vector<int>& lindex = Loop_index[id][iloop];
	      const std::vector<int>& ltype  = Loop_type[id][iloop];
	      for (int i=0; i<lindex.size(); i++){
		if ( abs(ltype[i])==1 ){ // this is G-propagator
		  int ic = lindex[i]; // momentum of G[ic] is changed, hence when counter-term ends at ic (ic is odd), we need to update vertex, because k_o will change in MoveVertex.
		  // To correct for the unscreened part of the interaction, we need to update time-dependent interaction
		  if ( (ic%2!=0) && Vtype(id, ic/2) !=0 ) changed_Vrtx.set( Vindx(id, ic/2), 1 ); // odd vertex, which connects interaction of the counter-term
		  int i_p = diagsG(id,ic); // alternatively, incoming  vertex k_i might change in MoveVertex. If k[ic]==k_i then diagsG(ic) is the vertex where counter term ends.
		  if ( (i_p%2!=0) && Vtype(id, i_p/2)!=0 )  changed_Vrtx.set( Vindx(id, i_p/2), 1 );
		}
	      }
	    }
	    // Vertices, which are needed to express screened interaction with the same time difference as the unscreened part.
	    for (int ii=0; ii<Nv; ii++){
	      if (changed_Vrtx[ii]){
		int id = vindx(ii)[0];
		int  i = vindx(ii)[1];
		int vtyp = Vtype(id,i);
		if (vtyp==0) log<<"ERROR : changed_Vrtx is wrong, as type==0!"<<endl;
		double t1 = times(2*i);     // t1 is old time, because for equal time case t[2i+1] was actually t[2i]
		double t2 = times(2*i+1);   // t2 is the new time, because now we want t[2i+1] to be different.
		// For now we will neglect possibility of type-II vertex. Should correct later for order higher than 5!
		int i_m = i_diagsG(id,2*i+1);
		double ti = times( et(i_m) );
		int i_p = diagsG(id,2*i+1);
		double to = times( et(i_p) );
		double ii_g = Gindx(id,i_m);
		double ki = changed_G[ii_g] ? norm(tmom_g(ip_ind(ii_g))) : norm(mom_g(ii_g));
		ii_g = Gindx(id,2*i+1);
		double ko = changed_G[ii_g] ? norm(tmom_g(ip_ind(ii_g))) : norm(mom_g(ii_g));
		Vrtx_trial(ii) = MoveVertex(t2, t1, ti, to, ki, ko, Gk);
	      }
	    }
	  }
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
	    if (vtyp==0){
	      v_trial(iq) = Vq;
	    }else{
	      intpar pq = Interp(aQ,qx);
	      intpar pt = Interp(fabs(times(2*i)-times(2*i+1)), tx);
	      int Nc = abs(vtyp);
	      double ct=0;
	      double lmbda_k = 1.0;
	      for (int k=0; k<Nc; k++){
		ct += binomial(Nc,k) * lmbda_k * bPqt[Nc-k-1](pq,pt);
		lmbda_k *= (lmbda/(8*pi));
	      }
	      long double Vn = ipower(Vq, Nc+1);
	      long double Vrtx = changed_Vrtx[ii] ? Vrtx_trial(ii) : Vrtx_current(ii);
	      v_trial(iq) = static_cast<long double>(Vn * ct) * Vrtx ;
	      v_trial(iq) += Vn * lmbda_k *(1-bubble_only);
	      if (vtyp<0){
		// we also add the single-particle counter term in addition to two-particle counter term.
		int ii_g = single_counter_index(ii);
		long double g_kq = changed_G[ii_g] ? g_trial(ip_ind(ii_g)) : G_current(ii_g);
		//if (g_kq!=0) v_trial(iq) += lmbda_spct_cutoffk(-vtyp-1)/g_kq;
#ifdef CNTR_VN		
		if (g_kq!=0) v_trial(iq) += Vn*lmbda_spct_cutoffk(-vtyp-1)/g_kq;
#else		
		if (g_kq!=0) v_trial(iq) += exp(-aQ*2/kF)*lmbda_spct_cutoffk(-vtyp-1)/g_kq;
#endif		
	      }
	    }
	    iq_ind(ii)=iq;
	  }
	  // we computed the polarization diagram
	  PQ_new=0;
	  for (int id=0; id<Ndiags; id++){
	    long double PQd = 1.0;
	    for (int i=1; i<2*Norder; i++){ // CAREFUL : density does not contain G(0)
	      int ii = Gindx(id,i);
	      PQd *= ( changed_G[ii] ? g_trial(ip_ind(ii)) : G_current(ii) );
	    }
	    for (int i=1; i<Norder; i++){
	      int ii = Vindx(id,i);
	      PQd *= ( changed_V[ii] ? v_trial(iq_ind(ii)) : V_current(ii) );
	    }
	    PQd *=  diagSign(id);
	    PQ_new += PQd;
	    PQg_new(id) = PQd;
	  }
	}else{
	  double Ka_old = amomentm(iloop);
	  amomentm(iloop) = Ka_new;
	  bl::TinyVector<double,3> K_old = momentum(iloop);
	  momentum(iloop) = K_new;
	  PQ_new = V0norm * mweight(amomentm, momentum);
	  amomentm(iloop) = Ka_old;
	  momentum(iloop) = K_old;
	}
	double ratio = PQ_new*trial_ratio / PQ;
	accept = fabs(ratio) > 1-drand() && PQ_new!=0;
	if ( std::isinf(PQ_new) or std::isnan(PQ_new) ){
	  accept = 0;
	  PrintLogInformation(log, itt, icase, PQ_new, p.iseed, Qweight, G_current, V_current, changed_G, changed_V, Ndiags, Norder, Nv, Gindx, Vindx, g_trial, ip_ind, v_trial, iq_ind, changed_Vrtx, Vrtx_current, Vrtx_trial);
	}
      }
      if ((itt+1)%Ncout==0 && mpi.rank==mpi.master) _PrintInfo_(itt+1, Qweight, icase, amomentm(1), amomentm(Nloops-1), times(0), PQ_new, PQ, dk_hist, Nweight/(Nmeassure+0.0), log);
      if (accept){
	Nacc_k += 1;
	if (!Qweight){
	  momentum(iloop) = K_new;  // this momentum was changed
	  amomentm(iloop) = Ka_new;
	  for (int ip=0; ip<loop_Gkind(iloop).extent(0); ++ip){
	    int ii   = loop_Gkind(iloop)(ip); 
	    mom_g(ii) = tmom_g(ip);
	  }
	  for (int iq=0; iq<loop_Vqind(iloop).extent(0); ++iq){ // Now do the same for the interaction
	    int ii   = loop_Vqind(iloop)(iq);
	    mom_v(ii) = tmom_v(iq);
	  }
	  for (int ip=0; ip<loop_Gkind(iloop).extent(0); ++ip){ // all G-propagators, which are changed when momentum in certain loop is changed.
	    int ii   = loop_Gkind(iloop)(ip);  // index of G-propagators, which we are now changing
	    G_current(ii)=g_trial(ip);
	  }
	  for (int iq=0; iq<loop_Vqind(iloop).extent(0); ++iq){ // Now do the same for the interaction
	    int ii   = loop_Vqind(iloop)(iq);
	    V_current(ii) = v_trial(iq);
	  }
	  if (Vcmax>0)
	    for (int ii=0; ii<Nv; ii++)
	      if (changed_Vrtx[ii]) Vrtx_current(ii) = Vrtx_trial(ii);
	}else{
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
	  changed_Vrtx = 0;
	  if (Vcmax>0){
	    // Vertices, which are needed to express screened interaction with the same time difference as the unscreened part.
	    for (int id=0; id<Ndiags; id++){ // updating momentum in all propagators
	      const std::vector<int>& lindex = Loop_index[id][iloop];
	      const std::vector<int>& ltype  = Loop_type[id][iloop];
	      for (int i=0; i<lindex.size(); i++){
		if ( abs(ltype[i])==1 ){
		  int ic = lindex[i]; // momentum of G[ic] is changed, hence when counter-term ends at ic (ic is odd), we need to update vertex, because k_o will change in MoveVertex.
		  // To correct for the unscreened part of the interaction, we need to update time-dependent interaction
		  if ( (ic%2!=0) && Vtype(id, ic/2)!=0 ) changed_Vrtx.set( Vindx(id, ic/2), 1 );
		  int i_p = diagsG(id,ic); // alternatively, incoming  vertex k_i might change in MoveVertex. If k[ic]==k_i then diagsG(ic) is the vertex where counter term ends.
		  if ( (i_p%2!=0) && Vtype(id, i_p/2)!=0 )  changed_Vrtx.set( Vindx(id, i_p/2), 1 );
		}
	      }
	    }
	    // Vertices, which are needed to express screened interaction with the same time difference as the unscreened part.
	    for (int ii=0; ii<Nv; ii++){
	      if (changed_Vrtx[ii]){
		int id = vindx(ii)[0];
		int  i = vindx(ii)[1];
		int vtyp = Vtype(id,i);
		if (vtyp==0) log<<"ERROR changed_Vrtx is wrong, as type==0!"<<endl;
		double t1 = times(2*i);     // t1 is old time, because for equal time case t[2i+1] was actually t[2i]
		double t2 = times(2*i+1);   // t2 is the new time, because now we want t[2i+1] to be different.
		// For now we will neglect possibility of type-II vertex. Should correct later!
		int i_m = i_diagsG(id,2*i+1);
		double ti = times( et(i_m) );
		int i_p = diagsG(id,2*i+1);
		double to = times( et(i_p) );
		double ki = norm(mom_g(Gindx(id,i_m)));
		double ko = norm(mom_g(Gindx(id,2*i+1)));
		Vrtx_current(ii) = MoveVertex(t2, t1, ti, to, ki, ko, Gk);
	      }
	    }
	  }
	  //t2.start(); // takes 1.5% of the time
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
	    if (vtyp==0){
	      V_current(ii) = Vq;
	    }else{
	      intpar pq = Interp(aQ,qx);
	      intpar pt = Interp(fabs(times(2*i)-times(2*i+1)), tx);
	      int Nc = abs(vtyp);
	      double ct=0;
	      double lmbda_k = 1.0;
	      for (int k=0; k<Nc; k++){
		ct += binomial(Nc,k) * lmbda_k * bPqt[Nc-k-1](pq,pt);
		lmbda_k *= (lmbda/(8*pi));
	      }
	      long double Vn = ipower(Vq, Nc+1);
	      V_current(ii) = Vn * ct * Vrtx_current(ii);
	      V_current(ii) += Vn * lmbda_k *(1-bubble_only);
	      if (vtyp<0){
		// we also add the single-particle counter term in addition to two-particle counter term.
		int ii_g = single_counter_index(ii);
		long double g_kq = G_current(ii_g);
		//if (g_kq!=0) V_current(ii) += lmbda_spct_cutoffk(-vtyp-1)/g_kq;
#ifdef CNTR_VN		
		if (g_kq!=0) V_current(ii) += Vn*lmbda_spct_cutoffk(-vtyp-1)/g_kq;
#else		
		if (g_kq!=0) V_current(ii) += exp(-aQ*2/kF)*lmbda_spct_cutoffk(-vtyp-1)/g_kq;
#endif		
	      }
	    }
	  }
	  //t5.stop();
	}
	PQ = PQ_new;
	PQg = PQg_new;
      }
    } else if (icase==1){  // changing one of the time variables
      Nall_t += 1;
      int itime = times_2_change[ static_cast<int>( times_2_change.size()*drand() ) ];
      changed_G=0;              // which propagators are being changed?
      changed_Vrtx = 0;
      times_trial = times;      // times_trial will contain the trial step times.
      if (itime==0){            // this is the measuring time with vertex=0
	log<<"ERROR : itime==0 should not occur in dDensity!"<<endl;
      }else{
	double t_new = beta*drand();
	// unscreened interaction, hence changing two times simultaneously!
	int ivertex = itime;
	times_trial(itime) = t_new;
	if (itime%2==0){ // this is the time for static interaction
	  for (int ivertex=itime; ivertex<itime+2; ivertex++){
	    for (int id=0; id<Ndiags; id++){
	      int i_pre_vertex = i_diagsG(id,ivertex);
	      if (Gindx(id,ivertex)!=std::numeric_limits<sint>().max()) changed_G.set(Gindx(id,ivertex),1);     // these two propagators are changed because of the new time.
	      if (Gindx(id,i_pre_vertex)!=std::numeric_limits<sint>().max()) changed_G.set(Gindx(id,i_pre_vertex),1);
	      if (Vcmax>0){
		int i_n = diagsG(id,ivertex);
		std::array<int,3> ips = {i_n, ivertex, i_pre_vertex};
		for(const auto& ip: ips)
		  if ( (ip%2!=0) && Vtype(id,ip/2)!=0 )  changed_Vrtx.set( Vindx(id, ip/2), 1 );
	      }
	    }
	  }
	}else{ // this time is for dynamic interaction only
	  if (Vcmax>0){
	    for (int id=0; id<Ndiags; id++){
	      std::array<int,1> ips = {ivertex};
	      for(const auto& ip: ips)
		if ( (ip%2!=0) && Vtype(id,ip/2)!=0 )  changed_Vrtx.set( Vindx(id, ip/2), 1 );
	    }
	  }
	}
      }
      if (! Qweight){
      	//t7.start(); // takes 11% of the time
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
	if (Vcmax>0){
	  int iq=0;
	  // Vertices, which are needed to express screened interaction with the same time difference as the unscreened part.
	  for (int ii=0; ii<Nv; ii++){ 
	    if (changed_Vrtx[ii]){
	      int id = vindx(ii)[0];
	      int  i = vindx(ii)[1];
	      int vtyp = Vtype(id,i);
	      if (vtyp==0) cout<<"ERROR changed_Vrtx is wrong, as type==0!"<<endl;
	      double t1 = times_trial(2*i);     // t1 is old time, because for equal time case t[2i+1] was actually t[2i]
	      double t2 = times_trial(2*i+1);   // t2 is the new time, because now we want t[2i+1] to be different.
	      // For now we will neglect possibility of type-II vertex. Should correct later!
	      int i_m = i_diagsG(id,2*i+1);
	      double ti = times_trial( et(i_m) );
	      int i_p = diagsG(id,2*i+1);
	      double to = times_trial( et(i_p) );
	      double ki = norm(mom_g(Gindx(id,i_m)));
	      double ko = norm(mom_g(Gindx(id,2*i+1)));
	      Vrtx_trial(ii) = MoveVertex(t2, t1, ti, to, ki, ko, Gk);
	      double aQ = norm(mom_v(ii));
	      double Vq = 8*pi/(aQ*aQ+lmbda);
	      intpar pq = Interp(aQ,qx);
	      intpar pt = Interp(fabs(t1-t2), tx);
	      int Nc = abs(vtyp);
	      double ct=0;
	      double lmbda_k = 1.0;
	      for (int k=0; k<Nc; k++){
		ct += binomial(Nc,k) * lmbda_k * bPqt[Nc-k-1](pq,pt);
		lmbda_k *= (lmbda/(8*pi));
	      }
	      long double Vn = ipower(Vq, Nc+1);
	      v_trial(iq) = Vn * ct * Vrtx_trial(ii);
	      v_trial(iq) += Vn * lmbda_k *(1-bubble_only);
	      if (vtyp<0){
		// we also add the single-particle counter term in addition to two-particle counter term.
		int ii_g = single_counter_index(ii);
		long double g_kq = changed_G[ii_g] ? g_trial(ip_ind(ii_g)) : G_current(ii_g);
		//if (g_kq!=0) v_trial(iq) += lmbda_spct_cutoffk(-vtyp-1)/g_kq;
#ifdef CNTR_VN
		if (g_kq!=0) v_trial(iq) += Vn*lmbda_spct_cutoffk(-vtyp-1)/g_kq;
#else		
		if (g_kq!=0) v_trial(iq) += exp(-aQ*2/kF)*lmbda_spct_cutoffk(-vtyp-1)/g_kq;
#endif		
	      }
	      iq_ind(ii)=iq;
	      iq++;
	    }
	  }
	}
	//t7.stop();
	//t8.start(); // takes 8% of the time
	PQ_new=0;   // recalculating PQ, taking into account one change of time.
	for (int id=0; id<Ndiags; id++){
	  long double PQd = 1.0;
	  for (int i=1; i<2*Norder; i++){ // CAREFULL: start with i=1 for density
	    int ii = Gindx(id,i);
	    PQd *= ( changed_G[ii] ? g_trial(ip_ind(ii)) : G_current(ii) );
	  }
	  for (int i=1; i<Norder; i++){
	    int ii = Vindx(id,i);
	    PQd *= ( changed_Vrtx[ii] ? v_trial(iq_ind(ii)) : V_current(ii) );
	  }
	  PQd *= diagSign(id);
	  PQ_new += PQd;
	  PQg_new(id) = PQd;
	}
      }else{
	PQ_new = V0norm * mweight(amomentm, momentum);
      }
      double ratio = PQ_new/PQ;
      bool accept = fabs(ratio) > 1-drand() && PQ_new!=0;
      if ( std::isinf(PQ_new) or std::isnan(PQ_new) ){
	accept = 0;
	PrintLogInformation(log, itt, icase, PQ_new, p.iseed, Qweight, G_current, V_current, changed_G, changed_V, Ndiags, Norder, Nv, Gindx, Vindx, g_trial, ip_ind, v_trial, iq_ind, changed_Vrtx, Vrtx_current, Vrtx_trial);
      }
      if ((itt+1)%Ncout==0 && mpi.rank==mpi.master)
	_PrintInfo_(itt+1, Qweight, icase, amomentm(1), amomentm(Nloops-1), times(itime), PQ_new, PQ, dk_hist, Nweight/(Nmeassure+0.0),log);
      if (accept){
	//t9.start(); // takes 1.7% of time
	Nacc_t +=1 ;
	times = times_trial;  // the new times are now accepted. We change the independent time variable, or two internal times.
	if (!Qweight){
	  for (int ii=0; ii<Ng; ++ii)
	    if (changed_G[ii])
	      G_current(ii) = g_trial(ip_ind(ii)); // The green's functions in the loop have been recalculated, and are now stored.
	  if (Vcmax>0){
	    for (int ii=0; ii<Nv; ii++){
	      if (changed_Vrtx[ii]){
		Vrtx_current(ii) = Vrtx_trial(ii);
		V_current(ii) = v_trial(iq_ind(ii));
	      }
	    }
	  }
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
	  if (Vcmax>0){
	    for (int ii=0; ii<Nv; ii++){
	      if (changed_Vrtx[ii]){
		int id = vindx(ii)[0];
		int  i = vindx(ii)[1];
		int vtyp = Vtype(id,i);
		if (vtyp==0) cout<<"ERROR changed_Vrtx is wrong, as type==0!"<<endl;
		double t1 = times(2*i);     // t1 is old time, because for equal time case t[2i+1] was actually t[2i]
		double t2 = times(2*i+1);   // t2 is the new time, because now we want t[2i+1] to be different.
		// For now we will neglect possibility of type-II vertex. Should correct later!
		int i_m = i_diagsG(id,2*i+1);
		double ti = times( et(i_m) );
		int i_p = diagsG(id,2*i+1);
		double to = times( et(i_p) );
		double ki = norm(mom_g(Gindx(id,i_m)));
		double ko = norm(mom_g(Gindx(id,2*i+1)));
		Vrtx_current(ii) = MoveVertex(t2, t1, ti, to, ki, ko, Gk);
		double aQ = norm(mom_v(ii));
		double Vq = 8*pi/(aQ*aQ+lmbda);
		intpar pq = Interp(aQ,qx);
		intpar pt = Interp(fabs(t1-t2), tx);
		int Nc = abs(vtyp);
		double ct=0;
		double lmbda_k = 1.0;
		for (int k=0; k<Nc; k++){
		  ct += binomial(Nc,k) * lmbda_k * bPqt[Nc-k-1](pq,pt);
		  lmbda_k *= (lmbda/(8*pi));
		}
		long double Vn = ipower(Vq, Nc+1);
		V_current(ii) = Vn * ct * Vrtx_current(ii);
		V_current(ii) += Vn * lmbda_k *(1-bubble_only);
		if (vtyp<0){
		  // we also add the single-particle counter term in addition to two-particle counter term.
		  int ii_g = single_counter_index(ii);
		  long double g_kq = G_current(ii_g);
		  //if (g_kq!=0) V_current(ii) += lmbda_spct_cutoffk(-vtyp-1)/g_kq;
#ifdef CNTR_VN		  
		  if (g_kq!=0) V_current(ii) += Vn*lmbda_spct_cutoffk(-vtyp-1)/g_kq;
#else		  
		  if (g_kq!=0) V_current(ii) += exp(-aQ*2/kF)*lmbda_spct_cutoffk(-vtyp-1)/g_kq;
#endif		  
		  
		}
	      }
	    }
	  }
	}
	PQ = PQ_new;
	PQg = PQg_new;
      }
      //log<<itt<<" G_current="<<G_current<<" V_current="<<V_current<<endl;
    }else{  // normalization diagram step
      Nall_w += 1;
      if (Qweight){ // trying to jump back into physical hilbert space
	PQ_new=0;
	//BKdata.PQg_new_Initialize();
	for (int id=0; id<Ndiags; id++){
	  long double PQd = 1.0;
	  for (int i=1; i<2*Norder; i++) PQd *= G_current(Gindx(id,i)); // we have all propagators precomputed, hence it is easy to compute the diagrams.
	  for (int i=1; i<Norder; i++) PQd *= V_current(Vindx(id,i));
	  PQd *= diagSign(id);
	  PQ_new += PQd;
	  PQg_new(id) = PQd;
	}
      }else{        // or if we are in the physical Hilbert space, we try to jump into the normalization diagram
	PQ_new = V0norm * mweight(amomentm, momentum);
      }
      double ratio = PQ_new/PQ;
      bool accept = fabs(ratio) > 1-drand() && PQ_new!=0;
      if ( std::isinf(PQ_new) or std::isnan(PQ_new) ){
	accept = 0;
	PrintLogInformation(log, itt, icase, PQ_new, p.iseed, Qweight, G_current, V_current, changed_G, changed_V, Ndiags, Norder, Nv, Gindx, Vindx, g_trial, ip_ind, v_trial, iq_ind, changed_Vrtx, Vrtx_current, Vrtx_trial);
      }
      if ((itt+1)%Ncout==0 && mpi.rank==mpi.master)
	  _PrintInfo_(itt+1, Qweight, icase, amomentm(1), amomentm(Nloops-1), times(0), PQ_new, PQ, dk_hist, Nweight/(Nmeassure+0.0), log);
      if (accept){
	Nacc_w += 1;
	Qweight = 1-Qweight;
	PQ = PQ_new;
	PQg = PQg_new;
      }
    }
    
    // Measuring
    if (itt>=Nwarm && itt%tmeassure==0){
      t_mes.start(); // takes 1% time
      Nmeassure += 1;
      double sp = sign(PQ);
      if (Qweight){
	Pnorm += 1;
	Nweight+=1;
      }else{
	Pw0  += sp;
	for (int id=0; id<Ndiags; id++){
	  int ii = Gindx(id,1); // k_1 is external momentum!
	  double Qa = norm(mom_g(ii));
	  Ekin += sp * Qa*Qa * PQg(id)/PQ;
	}
	if (fabs(sum(PQg)/PQ-1)>1e-5) log<<"WARNING : sum(PQg) != PQ. Check PQg !"<<endl;
      }
      if (!Qweight){
	mweight.Add_to_K_histogram(dk_hist, momentum, cutoffk, cutoffk);
      }
      //if ( itt>2e5/mpi.size && itt%(static_cast<int>(100000/mpi.size+1000)*tmeassure) == 0){
      if ( itt>0 && itt%(2000*tmeassure) == 0){
	double last_occurence = (Nweight-Wchanged)/(Nmeassure-Nchanged); // occurence from the last time we changed it
#ifdef _MPI
	double total_occurence=0;
	MPI_Allreduce(&last_occurence, &total_occurence, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	last_occurence = total_occurence/mpi.size;
#endif	
	if ( last_occurence > 0.15){ // decrease by two
	  V0norm /= 2.;
	  Pnorm /= 2.;
	  Nchanged = Nmeassure;
	  Wchanged = Nweight;
	  if (mpi.rank==mpi.master) log<<std::left<<(itt/1.0e6)<<"M occurence="<<std::setw(8)<<last_occurence<<" V0 reduced to "<<V0norm<<std::endl;
	}
	if ( last_occurence < 0.03){ // increase by two
	  V0norm *= 2.;
	  Pnorm *= 2.;
	  Nchanged = Nmeassure;
	  Wchanged = Nweight;
	  if (mpi.rank==mpi.master) log<<std::left<<(itt/1.0e6)<<"M occurence="<<std::setw(8)<<last_occurence<<" V0 increased to "<<V0norm<<std::endl;
	}
      }
      if (itt%(500000*tmeassure) == 0){
	if (Nmeassure-Nweight>1000){
#ifdef _MPI	  
	  MPI_Allreduce(MPI_IN_PLACE, mweight.K_hist.data(), mweight.K_hist.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
	  // We can get overflow during MPI for a long run. We should use a constant for normalization.
	  // We could here normalize K_hist (with knorm), and than when we measure, we would
	  // add instead of adding unity, we would add 1./knorm
	  dk_hist *= mweight.Normalize_K_histogram();
	  if (dk_hist < 1e-16) dk_hist = 1.0;
	  mweight.Recompute(mpi.rank==mpi.master);
	  // If we are in a meassuring space, we have to update now the value of the meassuring diagram as the weight has changed.
	  if (Qweight) PQ = V0norm * mweight(amomentm, momentum);
	}
      }
      t_mes.stop();
    }
  }
  t_all.stop();
  Pw0          *= 1.0/Nmeassure;
  Ekin         *= 1.0/Nmeassure;
  Pnorm        *= 1.0/Nmeassure;
  double occurence = static_cast<double>(Nweight)/static_cast<double>(Nmeassure);

  double Pw02 = Pw0*Pw0;
  double Ekin2 = Ekin*Ekin;
#ifdef _MPI  
  double dat[6] = {Pnorm, occurence, Pw0, Pw02, Ekin, Ekin2};
  if (mpi.rank==mpi.master){
    //MPI_Reduce(MPI_IN_PLACE, BKdata.C_Pln.data(), BKdata.C_Pln.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    //MPI_Reduce(MPI_IN_PLACE,  Pbin.data(),  Pbin.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, dat, 6, MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, mweight.K_hist.data(), mweight.K_hist.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
  }else{
    //MPI_Reduce(BKdata.C_Pln.data(), BKdata.C_Pln.data(), BKdata.C_Pln.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    //MPI_Reduce( Pbin.data(),  Pbin.data(),  Pbin.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    MPI_Reduce(dat, dat, 6, MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    MPI_Reduce(mweight.K_hist.data(), mweight.K_hist.data(), mweight.K_hist.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
  }
  if (mpi.rank==mpi.master){
    //BKdata.C_Pln *= 1./mpi.size;
    //Pbin  *= 1./mpi.size;
    Pnorm = dat[0]/mpi.size;
    occurence = dat[1]/mpi.size;
    Pw0   = dat[2]/mpi.size;
    Pw02  = dat[3]/mpi.size;
    Ekin  = dat[4]/mpi.size;
    Ekin2 = dat[5]/mpi.size;
    //
    mweight.K_hist *= 1./mpi.size;
  }
#endif
  double sigmaPw=0;
  double sigmaEkin=0;
  if (mpi.rank==mpi.master){
    log<<"total acceptance rate="<<(Nacc_k+Nacc_q+Nacc_t+Nacc_w)/(p.Nitt+0.0)<<" "<<" k-acceptance="<<(Nacc_k/(Nall_k+0.0))<<" t-acceptanece="<<(Nacc_t/(Nall_t+0.0))<<" w-acceptanece="<<(Nacc_w/(Nall_w+0.0))<<std::endl;
    log<<" k-trials="<<(Nall_k/(p.Nitt+0.0))<<" q-trials="<<(Nall_q/(p.Nitt+0.0))<<" t-trials="<<(Nall_t/(p.Nitt+0.0))<<" w-trials="<<(Nall_w/(p.Nitt+0.0))<<std::endl;
    //cout<<"t1="<<t1.elapsed()<<" t2="<<t2.elapsed()<<" t3="<<t3.elapsed()<<" t4="<<t4.elapsed()<<" t5="<<t5.elapsed()<<" t6="<<t6.elapsed()<<" t7="<<t7.elapsed()<<" t8="<<t8.elapsed()<<" t9="<<t9.elapsed()<<" t10="<<t10.elapsed()<<" t11="<<t11.elapsed()<<std::endl;
    log<<"total time="<<t_all.elapsed()<<" measuring time="<<t_mes.elapsed()<<std::endl;

    double fct = fabs(V0norm)/Pnorm;
    Pw0  *= fct;
    Pw02 *= fct*fct;
    sigmaPw = sqrt(fabs(Pw02 - Pw0*Pw0))/sqrt(mpi.size);
    
    Ekin *= fct;
    Ekin2 *= fct*fct;
    sigmaEkin = sqrt(fabs(Ekin2-Ekin*Ekin))/sqrt(mpi.size);
    
    // Proper normalization of the resulting Monte Carlo data
    double norm = ipower( 1/((2*pi)*(2*pi)*(2*pi)), Norder) *  ipower( beta, Norder-1); // one order less, because one G-propagator is missing and one time is missing.
    Pw0 *= norm;
    sigmaPw *= norm;
    Ekin *= norm;
    sigmaEkin *= norm;
    
    log<<"  meassuring diagram occurence frequency="<<occurence<<" and its norm Pnorm="<<Pnorm<<std::endl;
    for (int ik=0; ik<mweight.K_hist.extent(0); ik++){
      double dsum=0;
      for (int i=0; i<Nbin; i++) dsum += mweight.K_hist(ik,i);
      mweight.K_hist(ik,bl::Range::all()) *= 1./dsum;
      std::ofstream hout((std::string("Q_hist.")+std::to_string(ik)).c_str());
      double cutoff = cutoffk;
      for (int i=0; i<Nbin; i++){
	hout<< (i+0.5)/Nbin*cutoff/kF << " " << mweight.K_hist(ik,i)<<std::endl;
      }
    }
  }
  return std::make_tuple(Pw0,sigmaPw,Ekin,sigmaEkin);
}

template<typename GK>
std::tuple<double,double,double,double> sample_Density_C(std::ostream& log, double lmbda, const std::vector<double>& lmbda_spct, const GK& Gk, const params& p,
							 const bl::Array<unsigned short,2>& diagsG, const bl::Array<float,1>& diagSign,
							 const std::vector<std::vector<std::vector<int> > >& Loop_index, const std::vector<std::vector<std::vector<int> > >& Loop_type,
							 bl::Array<char,2>& Vtype, my_mpi& mpi)
{
  typedef double real;
  log.precision(12);
  
  const int Nr=3; // three different steps in this algorithm
  double Pr_norm = (p.Pr[0]+p.Pr[1]+p.Pr[2]); // probability for all four has to be normalized
  bl::TinyVector<double,Nr> Prs;  // cumulative probability for taking a step, needed for bisection below.
  Prs[0] = p.Pr[0]/Pr_norm;
  for (int i=1; i<Nr; i++) Prs[i] = Prs[i-1] + p.Pr[i]/Pr_norm; // prepares cumulative probability from p.Pr
  
  int Ndiags = diagsG.extent(0);
  if (Ndiags==0) {
    log<<"No diagrams to simulate...."<<std::endl;
    return std::make_tuple(0.,0.,0.,0.);
  }
  int Norder = diagsG.extent(1)/2;
  // computing the inverse of all permutations, to provide fast lookup for the previous fermionic propagator
  bl::Array<unsigned short,2> i_diagsG(Ndiags,2*Norder);
  for (int id=0; id<Ndiags; id++)
    for (int i=0; i<2*Norder; i++)
      i_diagsG(id, static_cast<int>(diagsG(id,i)) )=i;
  
  RanGSL drand(p.iseed); // GSL random number generator
  Timer t_mes, t_all, t_Pl, t_k, t_Q, t_w, t_t;
  
  // Set up which times need change when counter term is dynamic
  vector<int> times_2_change;
  std::set<int> times_to_change;
  {
    for (int i=1; i<Norder; i++) times_to_change.insert(2*i);
    // default vertices to change time will be : [0,2,...,2*(Norder-1)]
    times_2_change.resize(times_to_change.size());
    int j=0;
    for (auto it=times_to_change.begin(); it!=times_to_change.end(); ++it, ++j) times_2_change[j] = (*it);
    log<<"times_to_change : ";
    for (auto it=times_to_change.begin(); it!=times_to_change.end(); ++it) log << (*it) <<",  ";
    log<<std::endl;
    log << "times_2_change: "<<endl;
    for (int i=0; i<times_2_change.size(); i++) log << times_2_change[i] << ",";
    log << endl;
  }
  // Reads numerically evaluated Bubble diagram.
  
  int tmeassure=p.tmeassure, Ncout=p.Ncout, Nwarm=p.Nwarm;
  double V0norm = p.V0norm, dkF = p.dkF, kF=p.kF;// V0exp= p.V0exp;
  double cutoffk = p.cutoffk, dRk = p.dRk;
  double beta = p.beta;

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
  findUnique_noindx_spct(Gindx, Vindx, gindx, vindx, N0v, diagsG, Vtype, Loop_index, Loop_type, 1, false, debug, log);

  // Once more we go through single-particle counter terms and create index single_counter_index
  bl::Array<int,1> single_counter_index;
  Find_Single_Particle_Counter_G_index(single_counter_index, Ndiags, Norder, vindx.extent(0), diagsG, Vtype, Gindx, Vindx);
  
  if (Loop_index.size()!=Ndiags){ log<<"Wrong size of Loop_index "<<Loop_index.size()<<" instead of "<<Ndiags<<std::endl; exit(1);}
  int Nloops = Loop_index[0].size();
  // Here we are sampling density, which has one loop less than polarization!
  if (Nloops != Norder){ log<<"Expecting Nloops==Norder in Loops_index, but got Nloops="<<Nloops<<" Norder="<<Norder<<std::endl; exit(1);}
  int Ng = gindx.extent(0);
  int Nv = vindx.extent(0);
  int _Nv_ = std::max(Nv,1); // can not assign 0 to empty file

  bl::Array<bl::Array<sint,1>,1> loop_Gkind, loop_Vqind;
  bl::Array<bl::Array<char,1>,1> loop_Gksgn, loop_Vqsgn;
  int Ngp, Nvp;
  // Find index-tables to all momentum loops, so that we do not need to loop over all diagrams, but rather only over all differet propagators corresponding to certain loop.
  // Here we also take into account that single-particle counter terms are sometimes added.
  Get_GVind(loop_Gkind, loop_Gksgn, loop_Vqind, loop_Vqsgn, Ngp, Nvp, Loop_index, Loop_type, Gindx, Vindx, single_counter_index, lmbda_spct, Vtype, false, log);
  int Ngp_tr = GetMax_NGp_trial(i_diagsG, Gindx, Ng, Ngp);
  
  // DEBUG
  if (mpi.rank==mpi.master){
    log<<"lmbda_spct=";
    for (int i=0; i<lmbda_spct.size(); ++i) log<<lmbda_spct[i]<<",";
    log<<std::endl;
    log<<"single_counter_index="<<single_counter_index<<std::endl;
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
    log<<"loop_Vqind:"<<std::endl;
    for (int iloop=0; iloop<Nloops; ++iloop){
      for (int iq=0; iq<loop_Vqind(iloop).extent(0); ++iq){ // Now do the same for the interaction
	int ii   = loop_Vqind(iloop)(iq);
	int isgn = loop_Vqsgn(iloop)(iq);
	int id = vindx(ii)[0];
	int  i = vindx(ii)[1];
	int vtyp = Vtype(id,i);
	log<<"iloop="<<iloop<<" iq="<<iq<<" ii_v="<<ii<<" isgn="<<isgn<<" which comes from id="<< id <<" and pair =("<<2*i<<","<<2*i+1<<") of type="<<vtyp<<std::endl;
      }
    }
  }
  // END DEBUG
  
  int Nbin = 513;
  //int Nbin = 129;
  bl::Array<double,1> lmbda_spct_cutoffk(lmbda_spct.size());
  // Instead of using : lmbda_spct, which is q-independent, we will introduce "artificial" q-dependence, which gives identical integral.
  // When lmbda_spct is q-independent, we should normalize it by  lmbda_spct * 1/Integrate[d^3 1/(2*pi)^3].
  // The "artificial" dependence we choose is lmbda_spct * (Vq)^2, in which case the normalization constant is
  //     lmbda_spct * (Vq)^2 / Integrate[d^3q Vq^2/(2*pi)^3 ]
  //  which is
  //     lmbda_spct * (Vq)^2 / ( 2/sqrt(lmbda) *atan(ck/sqrt(lmbda)) - 2*ck/(ck^2+lmbda) )
  std::vector<double> nrm = Nrm(lmbda, cutoffk, lmbda_spct.size()+2);
  for (int i=0; i<lmbda_spct.size(); ++i){
#ifdef CNTR_VN
    // [Vq^2, Vq^3, Vq^4,..]
    lmbda_spct_cutoffk(i) = lmbda_spct[i]*nrm[i+2];
#else    
    // exp(-2*q/kF)
    lmbda_spct_cutoffk(i) = lmbda_spct[i] * pi*pi*8/(kF*kF*kF*(1. - exp(-2*cutoffk/kF)*ipower(cutoffk/kF,2)*(1. + ipower(1.+kF/cutoffk,2) )));
#endif    
  }
  meassureWeight mweight(p.V0exp, p.cutoffk, p.kF, Nbin, Nloops, 0);
  double dk_hist = 1.0;
  
  // This is momentum for loops
  bl::Array<bl::TinyVector<double,3>,1> momentum(Nloops); // momentum for all loops. Each diagram of the same order has the same number of loops.
  bl::Array<double,1> amomentm(Nloops);               // absolute value of momentum for each loop
  
  // Starting point is set up to be random
  // For the times for all vertices in all the graphs
  bl::Array<double,1> times(2*Norder);
  times(0) = -1e-15; // external time
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
  {
    for (int ik=0; ik<Nloops; ik++){
      double th = drand()*pi, ph = drand()*(2*pi);
      double st = sin(th), ct = cos(th), cp = cos(ph), sp = sin(ph);
      momentum(ik) = kF*st*cp, kF*st*sp, kF*ct;
      amomentm(ik) = kF;
    }
  }
  log.precision(12);
  log<<"Starting times="<<times<<" and starting momenta="<<momentum<<std::endl;

  bl::Array<double,1> G_current(Ng);
  bl::Array<long double,1> V_current(_Nv_);
  // This is momentum for all diagrams and all propagators
  bl::Array<bl::TinyVector<real,3>,1> mom_g(Ng), mom_v(Nv);
  bl::Array<double,1> PQg(Ndiags);
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
      for (int i=0; i<2*Norder; i++){
	//int i_final = diagsG(id,i);
	int i_previ = i_diagsG(id,i);
	bl::TinyVector<double,3> k_in = mom_G(id,i_previ);
	bl::TinyVector<double,3> k_out = mom_G(id,i);
	bl::TinyVector<double,3> q = mom_V(id,i/2)*(1.-2*(i%2));
	if (fabs(norm(k_in-k_out-q))>1e-6){
	  log<<"ERROR: diagram "<<id<<" vertex i="<<i<<" k_in="<<k_in<<" k_out="<<k_out<<" q="<<q<<std::endl;
	  log<<"   i_previ="<<i_previ<<" q_i="<< (i/2) <<" mom_V="<<mom_V(id,i/2)<<std::endl;
	}
      }
    }
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
    for (int ii=0; ii<Nv; ++ii){
      int id = vindx(ii)[0];
      int  i = vindx(ii)[1];
      double aQ = norm(mom_V(id,i));
      int vtyp = Vtype(id,i);
      double Vq = 8*pi/(aQ*aQ+lmbda);
      if (vtyp==0){
	V_current(ii) = Vq;
      }else{
	int Nc = abs(vtyp);
	long double Vn = ipower(Vq, Nc+1);
	V_current(ii) = Vn * ipower(lmbda/(8*pi), Nc);
	if (vtyp < 0){
	  // we also add the single-particle counter term in addition to two-particle counter term.
	  int ii_g = single_counter_index(ii);
	  long double g_kq = G_current(ii_g);
#ifdef CNTR_VN
	  if (g_kq!=0) V_current(ii) += Vn*lmbda_spct_cutoffk(-vtyp-1)/g_kq;
#else	  
	  if (g_kq!=0) V_current(ii) += exp(-aQ*2/kF)*lmbda_spct_cutoffk(-vtyp-1)/g_kq;
#endif	  
	  
	}
      }
    }
    //log<<"V_current="<<V_current<<endl;
    PQ=0;
    for (int id=0; id<Ndiags; id++){
      long double PQd = 1.0;
      for (int i=1; i<2*Norder; i++) PQd *= G_current(Gindx(id,i)); // CAREFULL Density does use G(0)!
      for (int i=1; i<Norder; i++) PQd *= V_current(Vindx(id,i));
      PQd *= diagSign(id);
      PQ += PQd;
      PQg(id) = PQd;
    }
    // Now we create smaller arrays mom_g and mom_v, which contain the same information as mom_G and mom_V.
    mom_g=0.0; mom_v=0.0;
    for (int id=0; id<Ndiags; id++){
      for (int i=1; i<2*Norder; i++){
	int ii = Gindx(id,i);
	mom_g(ii) = mom_G(id,i);
      }
      for (int i=1; i<Norder; i++){
	int ii = Vindx(id,i);
	mom_v(ii) = mom_V(id,i);
      }
    }
  }
  
  bl::Array<double,1> times_trial(2*Norder);
  int _Nvp_ = std::max(Nvp,1); // can not assign 0 to empty file
  bl::Array<bl::TinyVector<real,3>,1> tmom_g(Ngp), tmom_v(_Nvp_);
  bl::Array<sint,1> ip_ind(Ng), iq_ind(Nv);
  bl::Array<double,1> g_trial(Ngp_tr);
  bl::Array<long double,1> v_trial(_Nvp_);
  BitArray changed_G(Ng), changed_V(Nv);

  int Qweight = 0;      // if Qweight==1 then we have the normalization diagram
  double Nweight = 0;   // how many times did we see measuring diagram.
  //int NlowT = 0;        // how many times do we see external time < 0.05*beta
  int Nmeassure = 0;    // how many times did we measure quantities.
  int Nchanged  = 0;    // when was V0norm last changed?
  double Wchanged = 0;  // what was Nweight when we last time changed V0norm?
  int Nacc_k = 0, Nacc_q = 0, Nacc_w = 0, Nacc_t = 0;
  int Nall_k = 0, Nall_q = 0, Nall_w = 0, Nall_t = 0;
  if (mpi.rank==mpi.master) PrintInfo_(0, Qweight, amomentm(0), amomentm(1), times(0)-times(1), PQ, PQ, 0, 0, log);
  double Pw0=0, Ekin=0;
  bl::Array<double,1> PQg_new(Ndiags);
  
  //Check(0, log, -1, Nv, PQ, lmbda, times, times_to_change, Gk, G_current, V_current, diagsG, Gindx, Vindx, mom_g, mom_v, diagSign, Vtype, bPqt, qx, tx, vindx, Vrtx_current);
  t_all.start();
  for (int itt=0; itt<p.Nitt; itt++){
    double PQ_new=0;
    int icase = tBisect(drand(), Prs); // which steps should we attempt?
    if (Norder==1){
      // only at order 1 we have no time variable to move, hence icase==1 should not occur
      while (icase==1) icase = tBisect(drand(), Prs); 
    }
    if (icase==0){ // changing momentum Q or k-point
      int iloop = static_cast<int>(Nloops*drand());
      bl::TinyVector<double,3> K_new; double Ka_new; double trial_ratio=1;
      bool accept=false;
      Nall_k += 1;
      accept = Find_new_k_point(K_new, Ka_new, trial_ratio, momentum(iloop), amomentm(iloop), cutoffk, drand, drand()<p.Qring, dkF, dRk);
      if (accept){
	if (!Qweight){
	  bl::TinyVector<double,3> dK = K_new - momentum(iloop); // how much momentum will be changed.
	  //t1.start(); // takes 40% of the time
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
	    if (vtyp==0){
	      v_trial(iq) = Vq;
	    }else{
	      int Nc = abs(vtyp);
	      long double Vn = ipower(Vq, Nc+1);
	      v_trial(iq) = Vn * ipower(lmbda/(8*pi),Nc);
	      if (vtyp<0){
		// we also add the single-particle counter term in addition to two-particle counter term.
		int ii_g = single_counter_index(ii);
		long double g_kq = changed_G[ii_g] ? g_trial(ip_ind(ii_g)) : G_current(ii_g);
#ifdef CNTR_VN		
		if (g_kq!=0) v_trial(iq) += Vn*lmbda_spct_cutoffk(-vtyp-1)/g_kq;
#else		
		if (g_kq!=0) v_trial(iq) += exp(-aQ*2/kF)*lmbda_spct_cutoffk(-vtyp-1)/g_kq;
#endif		
	      }
	    }
	    iq_ind(ii)=iq;
	  }
	  // we computed the polarization diagram
	  PQ_new=0;
	  for (int id=0; id<Ndiags; id++){
	    long double PQd = 1.0;
	    for (int i=1; i<2*Norder; i++){ // CAREFUL : density does not contain G(0)
	      int ii = Gindx(id,i);
	      PQd *= ( changed_G[ii] ? g_trial(ip_ind(ii)) : G_current(ii) );
	    }
	    for (int i=1; i<Norder; i++){
	      int ii = Vindx(id,i);
	      PQd *= ( changed_V[ii] ? v_trial(iq_ind(ii)) : V_current(ii) );
	    }
	    PQd *=  diagSign(id);
	    PQ_new += PQd;
	    PQg_new(id) = PQd;
	  }
	}else{
	  double Ka_old = amomentm(iloop);
	  amomentm(iloop) = Ka_new;
	  bl::TinyVector<double,3> K_old = momentum(iloop);
	  momentum(iloop) = K_new;
	  PQ_new = V0norm * mweight(amomentm, momentum);
	  amomentm(iloop) = Ka_old;
	  momentum(iloop) = K_old;
	}
	double ratio = PQ_new*trial_ratio / PQ;
	accept = fabs(ratio) > 1-drand() && PQ_new!=0;
	if ( std::isinf(PQ_new) or std::isnan(PQ_new) ){
	  accept = 0;
	  //PrintLogInformation(log, itt, icase, PQ_new, p.iseed, Qweight, G_current, V_current, changed_G, changed_V, Ndiags, Norder, Nv, Gindx, Vindx, g_trial, ip_ind, v_trial, iq_ind, changed_Vrtx, Vrtx_current, Vrtx_trial);
	}
      }
      if ((itt+1)%Ncout==0 && mpi.rank==mpi.master) _PrintInfo_(itt+1, Qweight, icase, amomentm(1), amomentm(Nloops-1), times(0), PQ_new, PQ, dk_hist, Nweight/(Nmeassure+0.0), log);
      if (accept){
	Nacc_k += 1;
	if (!Qweight){
	  momentum(iloop) = K_new;  // this momentum was changed
	  amomentm(iloop) = Ka_new;
	  for (int ip=0; ip<loop_Gkind(iloop).extent(0); ++ip){
	    int ii   = loop_Gkind(iloop)(ip); 
	    mom_g(ii) = tmom_g(ip);
	  }
	  for (int iq=0; iq<loop_Vqind(iloop).extent(0); ++iq){ // Now do the same for the interaction
	    int ii   = loop_Vqind(iloop)(iq);
	    mom_v(ii) = tmom_v(iq);
	  }
	  for (int ip=0; ip<loop_Gkind(iloop).extent(0); ++ip){ // all G-propagators, which are changed when momentum in certain loop is changed.
	    int ii   = loop_Gkind(iloop)(ip);  // index of G-propagators, which we are now changing
	    G_current(ii)=g_trial(ip);
	  }
	  for (int iq=0; iq<loop_Vqind(iloop).extent(0); ++iq){ // Now do the same for the interaction
	    int ii   = loop_Vqind(iloop)(iq);
	    V_current(ii) = v_trial(iq);
	  }
	}else{
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
	  //t2.start(); // takes 1.5% of the time
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
	    if (vtyp==0){
	      V_current(ii) = Vq;
	    }else{
	      int Nc = abs(vtyp);
	      long double Vn = ipower(Vq, Nc+1);
	      V_current(ii) = Vn * ipower(lmbda/(8*pi),Nc);
	      if (vtyp<0){
		// we also add the single-particle counter term in addition to two-particle counter term.
		int ii_g = single_counter_index(ii);
		long double g_kq = G_current(ii_g);
#ifdef CNTR_VN		
		if (g_kq!=0) V_current(ii) += Vn*lmbda_spct_cutoffk(-vtyp-1)/g_kq;
#else		
		if (g_kq!=0) V_current(ii) += exp(-aQ*2/kF)*lmbda_spct_cutoffk(-vtyp-1)/g_kq;
#endif		
	      }
	    }
	  }
	  //t5.stop();
	}
	PQ = PQ_new;
	PQg = PQg_new;
      }
    } else if (icase==1){  // changing one of the time variables
      Nall_t += 1;
      int itime = times_2_change[ static_cast<int>( times_2_change.size()*drand() ) ];
      changed_G=0;              // which propagators are being changed?
      //changed_Vrtx = 0;
      times_trial = times;      // times_trial will contain the trial step times.
      if (itime==0){            // this is the measuring time with vertex=0
	log<<"ERROR : itime==0 should not occur in dDensity!"<<endl;
      }else{
	double t_new = beta*drand();
	// unscreened interaction, hence changing two times simultaneously!
	int ivertex = itime;
	times_trial(itime) = t_new;
	if (itime%2==0){ // this is the time for static interaction
	  for (int ivertex=itime; ivertex<itime+2; ivertex++){
	    for (int id=0; id<Ndiags; id++){
	      int i_pre_vertex = i_diagsG(id,ivertex);
	      if (Gindx(id,ivertex)!=std::numeric_limits<sint>().max()) changed_G.set(Gindx(id,ivertex),1);     // these two propagators are changed because of the new time.
	      if (Gindx(id,i_pre_vertex)!=std::numeric_limits<sint>().max()) changed_G.set(Gindx(id,i_pre_vertex),1);
	    }
	  }
	}
      }
      if (! Qweight){
      	//t7.start(); // takes 11% of the time
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
	//t7.stop();
	//t8.start(); // takes 8% of the time
	PQ_new=0;   // recalculating PQ, taking into account one change of time.
	for (int id=0; id<Ndiags; id++){
	  long double PQd = 1.0;
	  for (int i=1; i<2*Norder; i++){ // CAREFULL: start with i=1 for density
	    int ii = Gindx(id,i);
	    PQd *= ( changed_G[ii] ? g_trial(ip_ind(ii)) : G_current(ii) );
	  }
	  for (int i=1; i<Norder; i++){
	    int ii = Vindx(id,i);
	    //PQd *= ( changed_Vrtx[ii] ? v_trial(iq_ind(ii)) : V_current(ii) );
	    PQd *= V_current(ii);
	  }
	  PQd *= diagSign(id);
	  PQ_new += PQd;
	  PQg_new(id) = PQd;
	}
      }else{
	PQ_new = V0norm * mweight(amomentm, momentum);
      }
      double ratio = PQ_new/PQ;
      bool accept = fabs(ratio) > 1-drand() && PQ_new!=0;
      if ( std::isinf(PQ_new) or std::isnan(PQ_new) ){
	accept = 0;
	//PrintLogInformation(log, itt, icase, PQ_new, p.iseed, Qweight, G_current, V_current, changed_G, changed_V, Ndiags, Norder, Nv, Gindx, Vindx, g_trial, ip_ind, v_trial, iq_ind, changed_Vrtx, Vrtx_current, Vrtx_trial);
      }
      if ((itt+1)%Ncout==0 && mpi.rank==mpi.master)
	_PrintInfo_(itt+1, Qweight, icase, amomentm(1), amomentm(Nloops-1), times(itime), PQ_new, PQ, dk_hist, Nweight/(Nmeassure+0.0),log);
      if (accept){
	//t9.start(); // takes 1.7% of time
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
	PQ = PQ_new;
	PQg = PQg_new;
      }
      //log<<itt<<" G_current="<<G_current<<" V_current="<<V_current<<endl;
    }else{  // normalization diagram step
      Nall_w += 1;
      if (Qweight){ // trying to jump back into physical hilbert space
	PQ_new=0;
	//BKdata.PQg_new_Initialize();
	for (int id=0; id<Ndiags; id++){
	  long double PQd = 1.0;
	  for (int i=1; i<2*Norder; i++) PQd *= G_current(Gindx(id,i)); // we have all propagators precomputed, hence it is easy to compute the diagrams.
	  for (int i=1; i<Norder; i++) PQd *= V_current(Vindx(id,i));
	  PQd *= diagSign(id);
	  PQ_new += PQd;
	  PQg_new(id) = PQd;
	}
      }else{        // or if we are in the physical Hilbert space, we try to jump into the normalization diagram
	PQ_new = V0norm * mweight(amomentm, momentum);
      }
      double ratio = PQ_new/PQ;
      bool accept = fabs(ratio) > 1-drand() && PQ_new!=0;
      if ( std::isinf(PQ_new) or std::isnan(PQ_new) ){
	accept = 0;
	//PrintLogInformation(log, itt, icase, PQ_new, p.iseed, Qweight, G_current, V_current, changed_G, changed_V, Ndiags, Norder, Nv, Gindx, Vindx, g_trial, ip_ind, v_trial, iq_ind, changed_Vrtx, Vrtx_current, Vrtx_trial);
      }
      if ((itt+1)%Ncout==0 && mpi.rank==mpi.master)
	  _PrintInfo_(itt+1, Qweight, icase, amomentm(1), amomentm(Nloops-1), times(0), PQ_new, PQ, dk_hist, Nweight/(Nmeassure+0.0), log);
      if (accept){
	Nacc_w += 1;
	Qweight = 1-Qweight;
	PQ = PQ_new;
	PQg = PQg_new;
      }
    }
    
    // Measuring
    if (itt>=Nwarm && itt%tmeassure==0){
      t_mes.start(); // takes 1% time
      Nmeassure += 1;
      double sp = sign(PQ);
      if (Qweight){
	Pnorm += 1;
	Nweight+=1;
      }else{
	Pw0  += sp;
	for (int id=0; id<Ndiags; id++){
	  int ii = Gindx(id,1); // k_1 is external momentum!
	  double Qa = norm(mom_g(ii));
	  Ekin += sp * Qa*Qa * PQg(id)/PQ;
	}
	if (fabs(sum(PQg)/PQ-1)>1e-5) log<<"WARNING : sum(PQg) != PQ. Check PQg !"<<endl;
      }
      if (!Qweight){
	mweight.Add_to_K_histogram(dk_hist, momentum, cutoffk, cutoffk);
      }
      if ( itt>2e5/mpi.size && itt%(static_cast<int>(100000/mpi.size+1000)*tmeassure) == 0){
	//if ( itt>0 && itt%(2000*tmeassure) == 0){
	double last_occurence = (Nweight-Wchanged)/(Nmeassure-Nchanged); // occurence from the last time we changed it
#ifdef _MPI
	double total_occurence=0;
	MPI_Allreduce(&last_occurence, &total_occurence, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	last_occurence = total_occurence/mpi.size;
#endif	
	if ( last_occurence > 0.15){ // decrease by two
	  V0norm /= 2.;
	  Pnorm /= 2.;
	  Nchanged = Nmeassure;
	  Wchanged = Nweight;
	  if (mpi.rank==mpi.master) log<<std::left<<(itt/1.0e6)<<"M occurence="<<std::setw(8)<<last_occurence<<" V0 reduced to "<<V0norm<<std::endl;
	}
	if ( last_occurence < 0.03){ // increase by two
	  V0norm *= 2.;
	  Pnorm *= 2.;
	  Nchanged = Nmeassure;
	  Wchanged = Nweight;
	  if (mpi.rank==mpi.master) log<<std::left<<(itt/1.0e6)<<"M occurence="<<std::setw(8)<<last_occurence<<" V0 increased to "<<V0norm<<std::endl;
	}
      }
      if (itt%(500000*tmeassure) == 0){
	if (Nmeassure-Nweight>1000){
#ifdef _MPI	  
	  MPI_Allreduce(MPI_IN_PLACE, mweight.K_hist.data(), mweight.K_hist.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
	  // We can get overflow during MPI for a long run. We should use a constant for normalization.
	  // We could here normalize K_hist (with knorm), and than when we measure, we would
	  // add instead of adding unity, we would add 1./knorm
	  dk_hist *= mweight.Normalize_K_histogram();
	  if (dk_hist < 1e-16) dk_hist = 1.0;
	  mweight.Recompute(mpi.rank==mpi.master);
	  // If we are in a meassuring space, we have to update now the value of the meassuring diagram as the weight has changed.
	  if (Qweight) PQ = V0norm * mweight(amomentm, momentum);
	}
      }
      t_mes.stop();
    }
  }
  t_all.stop();
  Pw0          *= 1.0/Nmeassure;
  Ekin         *= 1.0/Nmeassure;
  Pnorm        *= 1.0/Nmeassure;
  double occurence = static_cast<double>(Nweight)/static_cast<double>(Nmeassure);

  double Pw02 = Pw0*Pw0;
  double Ekin2 = Ekin*Ekin;
#ifdef _MPI  
  double dat[6] = {Pnorm, occurence, Pw0, Pw02, Ekin, Ekin2};
  if (mpi.rank==mpi.master){
    MPI_Reduce(MPI_IN_PLACE, dat, 6, MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, mweight.K_hist.data(), mweight.K_hist.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
  }else{
    MPI_Reduce(dat, dat, 6, MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    MPI_Reduce(mweight.K_hist.data(), mweight.K_hist.data(), mweight.K_hist.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
  }
  if (mpi.rank==mpi.master){
    Pnorm = dat[0]/mpi.size;
    occurence = dat[1]/mpi.size;
    Pw0   = dat[2]/mpi.size;
    Pw02  = dat[3]/mpi.size;
    Ekin  = dat[4]/mpi.size;
    Ekin2 = dat[5]/mpi.size;
    mweight.K_hist *= 1./mpi.size;
  }
#endif
  double sigmaPw=0;
  double sigmaEkin=0;
  if (mpi.rank==mpi.master){
    log<<"total acceptance rate="<<(Nacc_k+Nacc_q+Nacc_t+Nacc_w)/(p.Nitt+0.0)<<" "<<" k-acceptance="<<(Nacc_k/(Nall_k+0.0))<<" t-acceptanece="<<(Nacc_t/(Nall_t+0.0))<<" w-acceptanece="<<(Nacc_w/(Nall_w+0.0))<<std::endl;
    log<<" k-trials="<<(Nall_k/(p.Nitt+0.0))<<" q-trials="<<(Nall_q/(p.Nitt+0.0))<<" t-trials="<<(Nall_t/(p.Nitt+0.0))<<" w-trials="<<(Nall_w/(p.Nitt+0.0))<<std::endl;
    //cout<<"t1="<<t1.elapsed()<<" t2="<<t2.elapsed()<<" t3="<<t3.elapsed()<<" t4="<<t4.elapsed()<<" t5="<<t5.elapsed()<<" t6="<<t6.elapsed()<<" t7="<<t7.elapsed()<<" t8="<<t8.elapsed()<<" t9="<<t9.elapsed()<<" t10="<<t10.elapsed()<<" t11="<<t11.elapsed()<<std::endl;
    log<<"total time="<<t_all.elapsed()<<" measuring time="<<t_mes.elapsed()<<std::endl;

    double fct = fabs(V0norm)/Pnorm;
    Pw0  *= fct;
    Pw02 *= fct*fct;
    sigmaPw = sqrt(fabs(Pw02 - Pw0*Pw0))/sqrt(mpi.size);
    
    Ekin *= fct;
    Ekin2 *= fct*fct;
    sigmaEkin = sqrt(fabs(Ekin2-Ekin*Ekin))/sqrt(mpi.size);
    
    // Proper normalization of the resulting Monte Carlo data
    double norm = ipower( 1/((2*pi)*(2*pi)*(2*pi)), Norder) *  ipower( beta, Norder-1); // one order less, because one G-propagator is missing and one time is missing.
    Pw0 *= norm;
    sigmaPw *= norm;
    Ekin *= norm;
    sigmaEkin *= norm;
    
    log<<"  meassuring diagram occurence frequency="<<occurence<<" and its norm Pnorm="<<Pnorm<<std::endl;
    for (int ik=0; ik<mweight.K_hist.extent(0); ik++){
      double dsum=0;
      for (int i=0; i<Nbin; i++) dsum += mweight.K_hist(ik,i);
      mweight.K_hist(ik,bl::Range::all()) *= 1./dsum;
      std::ofstream hout((std::string("Q_hist.")+std::to_string(ik)).c_str());
      double cutoff = cutoffk;
      for (int i=0; i<Nbin; i++){
	hout<< (i+0.5)/Nbin*cutoff/kF << " " << mweight.K_hist(ik,i)<<std::endl;
      }
    }
  }
  return std::make_tuple(Pw0,sigmaPw,Ekin,sigmaEkin);
}

template<typename GK>
std::tuple<double,double,double,double> sample_Density_C(std::ostream& log, double lmbda, const std::vector<double>& lmbda_spct, const GK& Gk, const params& p,
							 const bl::Array<unsigned short,2>& diagsG,
							 const std::vector<std::vector<int> >& diagSign,
							 const std::vector<std::vector<std::vector<int> > >& Loop_index, const std::vector<std::vector<std::vector<int> > >& Loop_type,
							 bl::Array<char,2>& Vtype, my_mpi& mpi)
{
  typedef double real;
  log.precision(12);
  
  const int Nr=3; // three different steps in this algorithm
  double Pr_norm = (p.Pr[0]+p.Pr[1]+p.Pr[2]); // probability for all four has to be normalized
  bl::TinyVector<double,Nr> Prs;  // cumulative probability for taking a step, needed for bisection below.
  Prs[0] = p.Pr[0]/Pr_norm;
  for (int i=1; i<Nr; i++) Prs[i] = Prs[i-1] + p.Pr[i]/Pr_norm; // prepares cumulative probability from p.Pr
  
  int Ndiags = diagsG.extent(0);
  if (Ndiags==0) {
    log<<"No diagrams to simulate...."<<std::endl;
    return std::make_tuple(0.,0.,0.,0.);
  }
  int Norder = diagsG.extent(1)/2;
  // computing the inverse of all permutations, to provide fast lookup for the previous fermionic propagator
  bl::Array<unsigned short,2> i_diagsG(Ndiags,2*Norder);
  for (int id=0; id<Ndiags; id++)
    for (int i=0; i<2*Norder; i++)
      i_diagsG(id, static_cast<int>(diagsG(id,i)) )=i;
  
  RanGSL drand(p.iseed); // GSL random number generator
  Timer t_mes, t_all, t_Pl, t_k, t_Q, t_w, t_t;
  
  // Set up which times need change when counter term is dynamic
  vector<int> times_2_change;
  std::set<int> times_to_change;
  {
    for (int i=1; i<Norder; i++) times_to_change.insert(2*i);
    // default vertices to change time will be : [2,...,2*(Norder-1)]
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
  findUnique_noindx_spct(Gindx, Vindx, gindx, vindx, N0v, diagsG, Vtype, Loop_index, Loop_type, 1, false, mpi.rank==mpi.master, log);

  // Once more we go through single-particle counter terms and create index single_counter_index
  bl::Array<int,1> single_counter_index;
  Find_Single_Particle_Counter_G_index(single_counter_index, Ndiags, Norder, vindx.extent(0), diagsG, Vtype, Gindx, Vindx);
  
  if (Loop_index.size()!=Ndiags){ log<<"Wrong size of Loop_index "<<Loop_index.size()<<" instead of "<<Ndiags<<std::endl; exit(1);}
  int Nloops = Loop_index[0].size();
  // Here we are sampling density, which has one loop less than polarization!
  if (Nloops != Norder){ log<<"Expecting Nloops==Norder in Loops_index, but got Nloops="<<Nloops<<" Norder="<<Norder<<std::endl; exit(1);}
  int Ng = gindx.extent(0);
  int Nv = vindx.extent(0);
  int _Nv_ = std::max(Nv,1); // can not assign 0 to empty file

  bl::Array<bl::Array<sint,1>,1> loop_Gkind, loop_Vqind, loop_Vqind2;
  bl::Array<bl::Array<char,1>,1> loop_Gksgn, loop_Vqsgn, loop_Vqsgn2;
  int Ngp, Nvp, Nvp2;
  bl::Array<bl::Array<sint,1>,1> hugh_diags(Nloops);
  bl::Array<sint,1> hh_indx(Ndiags);
  
  int nhid=0;
  // Find index-tables to all momentum loops, so that we do not need to loop over all diagrams, but rather only over all differet propagators corresponding to certain loop.
  // Here we also take into account that single-particle counter terms are sometimes added.
  Get_GVind_Hugenholtz(hh_indx,nhid,hugh_diags,loop_Gkind,loop_Gksgn,loop_Vqind,loop_Vqsgn,loop_Vqind2,loop_Vqsgn2,Ngp,Nvp,Nvp2,diagsG,i_diagsG,Loop_index,Loop_type,Gindx,Vindx,single_counter_index,lmbda_spct,Vtype,mpi.rank==mpi.master,false,log);
  //Get_GVind(loop_Gkind, loop_Gksgn, loop_Vqind, loop_Vqsgn, loop_Vqind2, loop_Vqsgn2, Ngp, Nvp, Nvp2, Vqh2, Loop_index, Loop_type, Gindx, Vindx, single_counter_index, lmbda_spct, Vtype, false, log);
  int Ngp_tr = GetMax_NGp_trial(i_diagsG, Gindx, Ng, Ngp);
  
  if (mpi.rank==mpi.master){
    log << "hugh_diags="<<endl;
    for (int iloop=0; iloop<Nloops; iloop++){
      log << "iloop="<< iloop << " ("<< setw(3)<< hugh_diags(iloop).extent(0) << "): ";
      for (int it=0; it<hugh_diags(iloop).extent(0); ++it) log << hugh_diags(iloop)(it) << ", "; 
      log << endl;
    }
    // log << "hh_indx="<<endl;
    // for (int id=0; id<Ndiags; id++){
    //   log << setw(2) << id << "  " << setw(2) << hh_indx(id) << endl;
    //}
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
  
  //int Nbin = 129;
  int Nbin = 513;
  bl::Array<double,1> lmbda_spct_cutoffk(lmbda_spct.size());
  // Instead of using : lmbda_spct, which is q-independent, we will introduce "artificial" q-dependence, which gives identical integral.
  // When lmbda_spct is q-independent, we should normalize it by  lmbda_spct * 1/Integrate[d^3 1/(2*pi)^3].
  // The "artificial" dependence we choose is lmbda_spct * (Vq)^2, in which case the normalization constant is
  //     lmbda_spct * (Vq)^2 / Integrate[d^3q Vq^2/(2*pi)^3 ]
  //  which is
  //     lmbda_spct * (Vq)^2 / ( 2/sqrt(lmbda) *atan(ck/sqrt(lmbda)) - 2*ck/(ck^2+lmbda) )
  std::vector<double> nrm = Nrm(lmbda, cutoffk, lmbda_spct.size()+2);
  for (int i=0; i<lmbda_spct.size(); ++i){
#ifdef CNTR_VN
    // [Vq^2, Vq^3, Vq^4,..]
    lmbda_spct_cutoffk(i) = lmbda_spct[i]*nrm[i+2];
#else    
    // exp(-2*q/kF)
    lmbda_spct_cutoffk(i) = lmbda_spct[i] * pi*pi*8/(kF*kF*kF*(1. - exp(-2*cutoffk/kF)*ipower(cutoffk/kF,2)*(1. + ipower(1.+kF/cutoffk,2) )));
#endif    
  }
  meassureWeight mweight(p.V0exp, p.cutoffk, p.kF, Nbin, Nloops, 0);
  double dk_hist = 1.0;
  
  // This is momentum for loops
  bl::Array<bl::TinyVector<double,3>,1> momentum(Nloops); // momentum for all loops. Each diagram of the same order has the same number of loops.
  bl::Array<double,1> amomentm(Nloops);               // absolute value of momentum for each loop
  
  // Starting point is set up to be random
  // For the times for all vertices in all the graphs
  bl::Array<double,1> times(2*Norder);
  times(0) = -1e-15; // external time
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
  {
    for (int ik=0; ik<Nloops; ik++){
      double th = drand()*pi, ph = 2*pi*drand();
      double st = sin(th), ct = cos(th), cp = cos(ph), sp = sin(ph);
      momentum(ik) = kF*st*cp, kF*st*sp, kF*ct;
      amomentm(ik) = kF;
    }
  }
  log.precision(12);
  log<<"Starting times="<<times<<" and starting momenta="<<momentum<<std::endl;

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
  
  bl::Array<double,1> PQg(Ndiags);
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
	long double Vn = ipower(Vq, Nc+1);
	if (g_kq!=0) V_current(ii) += Vn*lmbda_spct_cutoffk(-vtyp-1)/g_kq;
#else	  
	if (g_kq!=0) V_current(ii) += exp(-aQ*2/kF)*lmbda_spct_cutoffk(-vtyp-1)/g_kq;
#endif	  
      }
    }
    //log<<"V_current="<< V_current <<endl<<"V_current2="<< V_current2 << endl;
    PQ=0;
    for (int id=0; id<Ndiags; id++){
      long double PQd = 1.0;
      for (int i=1; i<2*Norder; i++) PQd *= G_current(Gindx(id,i)); // CAREFULL Density does not use G(0)!
      int nvh=0;
      for (int i=1; i<Norder; i++){
	int ii=Vindx(id,i);
	if (Vtype(id,i)>=10){ // collect all Hugenholtz interactions in this diagram
	  V12(nvh,0) = V_current(ii);
	  V12(nvh,1) = V_current2(ii-N0v);
	  nvh++;
	}else{
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
      PQg(id) = PQd;
    }
    // Now we create smaller arrays mom_g and mom_v, which contain the same information as mom_G and mom_V.
    mom_g=0.0; mom_v=0.0;
    for (int id=0; id<Ndiags; id++){
      for (int i=1; i<2*Norder; i++){
	int ii = Gindx(id,i);
	mom_g(ii) = mom_G(id,i);
      }
      for (int i=1; i<Norder; i++){
	int ii = Vindx(id,i);
	mom_v(ii) = mom_V(id,i);
      }
    }
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
  //int NlowT = 0;        // how many times do we see external time < 0.05*beta
  int Nmeassure = 0;    // how many times did we measure quantities.
  int Nchanged  = 0;    // when was V0norm last changed?
  double Wchanged = 0;  // what was Nweight when we last time changed V0norm?
  int Nacc_k = 0, Nacc_q = 0, Nacc_w = 0, Nacc_t = 0;
  int Nall_k = 0, Nall_q = 0, Nall_w = 0, Nall_t = 0;
  if (mpi.rank==mpi.master) PrintInfo_(0, Qweight, amomentm(0), amomentm(1), times(0)-times(1), PQ, PQ, 0, 0, log);

  double Pw0=0, Ekin=0;
  bl::Array<double,1> PQg_new(Ndiags);
  
  //Check(0, log, -1, Nv, PQ, lmbda, times, times_to_change, Gk, G_current, V_current, diagsG, Gindx, Vindx, mom_g, mom_v, diagSign, Vtype, bPqt, qx, tx, vindx, Vrtx_current);
  t_all.start();
  for (int itt=0; itt<p.Nitt; itt++){
    double PQ_new=0;
    int icase = tBisect(drand(), Prs); // which steps should we attempt?
    if (Norder==1){
      // only at order 1 we have no time variable to move, hence icase==1 should not occur
      while (icase==1) icase = tBisect(drand(), Prs); 
    }
    if (icase==0){ // changing momentum Q or k-point
      int iloop = static_cast<int>(Nloops*drand());
      bl::TinyVector<double,3> K_new; double Ka_new; double trial_ratio=1;
      bool accept=false;
      Nall_k += 1;
      accept = Find_new_k_point(K_new, Ka_new, trial_ratio, momentum(iloop), amomentm(iloop), cutoffk, drand, drand()<p.Qring, dkF, dRk);
      if (accept){
	if (!Qweight){
	  bl::TinyVector<double,3> dK = K_new - momentum(iloop); // how much momentum will be changed.
	  //t1.start(); // takes 40% of the time
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
	    if (vtyp < 0){ // we also add the single-particle counter term in addition to two-particle counter term.
	      int ii_g = single_counter_index(ii);
	      long double g_kq = changed_G[ii_g] ? g_trial(ip_ind(ii_g)) : G_current(ii_g);
#ifdef CNTR_VN		
	      long double Vn = ipower(Vq, Nc+1);
	      if (g_kq!=0) v_trial(iq) += Vn*lmbda_spct_cutoffk(-vtyp-1)/g_kq;
#else		
	      if (g_kq!=0) v_trial(iq) += exp(-aQ*2/kF)*lmbda_spct_cutoffk(-vtyp-1)/g_kq;
#endif		
	    }
	    iq_ind(ii)=iq;
	  }
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
	  // Hugenholtz diagrams are evaluated
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
	  // we computed the polarization diagram
	  PQ_new=0;
	  for (int id=0; id<Ndiags; id++){
	    long double PQd = 1.0;
	    for (int i=1; i<2*Norder; i++){ // CAREFUL : density does not contain G(0)
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
	    PQ_new += PQd;
	    PQg_new(id) = PQd;
	  }
	}else{
	  double Ka_old = amomentm(iloop);
	  amomentm(iloop) = Ka_new;
	  bl::TinyVector<double,3> K_old = momentum(iloop);
	  momentum(iloop) = K_new;
	  PQ_new = V0norm * mweight(amomentm, momentum);
	  amomentm(iloop) = Ka_old;
	  momentum(iloop) = K_old;
	}
	double ratio = PQ_new*trial_ratio / PQ;
	accept = fabs(ratio) > 1-drand() && PQ_new!=0;
      }
      if ((itt+1)%Ncout==0 && mpi.rank==mpi.master) _PrintInfo_(itt+1, Qweight, icase, amomentm(1), amomentm(Nloops-1), times(0), PQ_new, PQ, dk_hist, Nweight/(Nmeassure+0.0), log);
      if (accept){
	Nacc_k += 1;
	if (!Qweight){
	  momentum(iloop) = K_new;  // this momentum was changed
	  amomentm(iloop) = Ka_new;
	  for (int ip=0; ip<loop_Gkind(iloop).extent(0); ++ip){
	    int ii   = loop_Gkind(iloop)(ip); 
	    mom_g(ii) = tmom_g(ip);
	    G_current(ii)=g_trial(ip);
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
	}else{
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
	  //t2.start(); // takes 1.5% of the time
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
	      long double Vn = ipower(Vq, Nc+1);
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
	}
	PQ = PQ_new;
	PQg = PQg_new;
      }
    } else if (icase==1){  // changing one of the time variables
      Nall_t += 1;
      int itime = times_2_change[ static_cast<int>( times_2_change.size()*drand() ) ];
      changed_G=0;              // which propagators are being changed?
      //changed_Vrtx = 0;
      times_trial = times;      // times_trial will contain the trial step times.
      if (itime==0){            // this is the measuring time with vertex=0
	log<<"ERROR : itime==0 should not occur in dDensity!"<<endl;
      }else{
	double t_new = beta*drand();
	// unscreened interaction, hence changing two times simultaneously!
	int ivertex = itime;
	times_trial(itime) = t_new;
	if (itime%2==0){ // this is the time for static interaction
	  for (int ivertex=itime; ivertex<itime+2; ivertex++){
	    for (int id=0; id<Ndiags; id++){
	      int i_pre_vertex = i_diagsG(id,ivertex);
	      if (Gindx(id,ivertex)!=std::numeric_limits<sint>().max()) changed_G.set(Gindx(id,ivertex),1);     // these two propagators are changed because of the new time.
	      if (Gindx(id,i_pre_vertex)!=std::numeric_limits<sint>().max()) changed_G.set(Gindx(id,i_pre_vertex),1);
	    }
	  }
	}
      }
      if (! Qweight){
      	//t7.start(); // takes 11% of the time
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
	//t7.stop();
	//t8.start(); // takes 8% of the time
	PQ_new=0;   // recalculating PQ, taking into account one change of time.
	for (int id=0; id<Ndiags; id++){
	  long double PQd = 1.0;
	  for (int i=1; i<2*Norder; i++){ // CAREFULL: start with i=1 for density
	    int ii = Gindx(id,i);
	    PQd *= ( changed_G[ii] ? g_trial(ip_ind(ii)) : G_current(ii) );
	  }
	  bool isHugenholtz=false;
	  for (int i=1; i<Norder; i++)
	    if (Vtype(id,i) < 10)// non-Hugenholtz
	      PQd *= V_current(Vindx(id,i));
	    else
	      isHugenholtz=true;
	  if (isHugenholtz)
	    PQd *= V_Hugh(hh_indx(id));
	  else
	    PQd *=  diagSign[id][0];
	  PQ_new += PQd;
	  PQg_new(id) = PQd;
	}
      }else{
	PQ_new = V0norm * mweight(amomentm, momentum);
      }
      double ratio = PQ_new/PQ;
      bool accept = fabs(ratio) > 1-drand() && PQ_new!=0;
      if ((itt+1)%Ncout==0 && mpi.rank==mpi.master)
	_PrintInfo_(itt+1, Qweight, icase, amomentm(1), amomentm(Nloops-1), times(itime), PQ_new, PQ, dk_hist, Nweight/(Nmeassure+0.0),log);
      if (accept){
	//t9.start(); // takes 1.7% of time
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
	PQ = PQ_new;
	PQg = PQg_new;
      }
      //log<<itt<<" G_current="<<G_current<<" V_current="<<V_current<<endl;
    }else{  // normalization diagram step
      Nall_w += 1;
      if (Qweight){ // trying to jump back into physical hilbert space
	PQ_new=0;
	//BKdata.PQg_new_Initialize();
	for (int id=0; id<Ndiags; id++){
	  long double PQd = 1.0;
	  for (int i=1; i<2*Norder; i++) PQd *= G_current(Gindx(id,i)); // we have all propagators precomputed, hence it is easy to compute the diagrams.
	  bool isHugenholtz=false;
	  for (int i=1; i<Norder; i++)
	    if (Vtype(id,i) < 10) // non-Hugenholtz
	      PQd *= V_current(Vindx(id,i));
	    else
	      isHugenholtz=true;
	  if (isHugenholtz)
	    PQd *= V_Hugh(hh_indx(id));
	  else
	    PQd *=  diagSign[id][0];
	  PQ_new += PQd;
	  PQg_new(id) = PQd;
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
	PQg = PQg_new;
      }
    }
    
    // Measuring
    if (itt>=Nwarm && itt%tmeassure==0){
      t_mes.start(); // takes 1% time
      Nmeassure += 1;
      double sp = sign(PQ);
      if (Qweight){
	Pnorm += 1;
	Nweight+=1;
      }else{
	Pw0  += sp;
	for (int id=0; id<Ndiags; id++){
	  int ii = Gindx(id,1); // k_1 is external momentum!
	  double Qa = norm(mom_g(ii));
	  Ekin += sp * Qa*Qa * PQg(id)/PQ;
	}
	if (fabs(sum(PQg)/PQ-1)>1e-5) log<<"WARNING : sum(PQg) != PQ. Check PQg !"<<endl;
      }
      if (!Qweight){
	mweight.Add_to_K_histogram(dk_hist, momentum, cutoffk, cutoffk);
      }
      if ( itt>2e5/mpi.size && itt%(static_cast<int>(100000/mpi.size+1000)*tmeassure) == 0){
	//if ( itt>0 && itt%(2000*tmeassure) == 0){
	double last_occurence = (Nweight-Wchanged)/(Nmeassure-Nchanged); // occurence from the last time we changed it
#ifdef _MPI
	double total_occurence=0;
	MPI_Allreduce(&last_occurence, &total_occurence, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	last_occurence = total_occurence/mpi.size;
#endif	
	if ( last_occurence > 0.15){ // decrease by two
	  V0norm /= 2.;
	  Pnorm /= 2.;
	  Nchanged = Nmeassure;
	  Wchanged = Nweight;
	  if (mpi.rank==mpi.master) log<<std::left<<(itt/1.0e6)<<"M occurence="<<std::setw(8)<<last_occurence<<" V0 reduced to "<<V0norm<<std::endl;
	}
	if ( last_occurence < 0.03){ // increase by two
	  V0norm *= 2.;
	  Pnorm *= 2.;
	  Nchanged = Nmeassure;
	  Wchanged = Nweight;
	  if (mpi.rank==mpi.master) log<<std::left<<(itt/1.0e6)<<"M occurence="<<std::setw(8)<<last_occurence<<" V0 increased to "<<V0norm<<std::endl;
	}
      }
      if (itt%(500000*tmeassure) == 0){
	if (Nmeassure-Nweight>1000){
#ifdef _MPI	  
	  MPI_Allreduce(MPI_IN_PLACE, mweight.K_hist.data(), mweight.K_hist.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
	  // We can get overflow during MPI for a long run. We should use a constant for normalization.
	  // We could here normalize K_hist (with knorm), and than when we measure, we would
	  // add instead of adding unity, we would add 1./knorm
	  dk_hist *= mweight.Normalize_K_histogram();
	  if (dk_hist < 1e-16) dk_hist = 1.0;
	  mweight.Recompute(false);//(mpi.rank==mpi.master);
	  // If we are in a meassuring space, we have to update now the value of the meassuring diagram as the weight has changed.
	  if (Qweight) PQ = V0norm * mweight(amomentm, momentum);
	}
      }
      t_mes.stop();
    }
  }
  t_all.stop();
  Pw0          *= 1.0/Nmeassure;
  Ekin         *= 1.0/Nmeassure;
  Pnorm        *= 1.0/Nmeassure;
  double occurence = static_cast<double>(Nweight)/static_cast<double>(Nmeassure);

  double Pw02 = Pw0*Pw0;
  double Ekin2 = Ekin*Ekin;
#ifdef _MPI  
  double dat[6] = {Pnorm, occurence, Pw0, Pw02, Ekin, Ekin2};
  if (mpi.rank==mpi.master){
    MPI_Reduce(MPI_IN_PLACE, dat, 6, MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, mweight.K_hist.data(), mweight.K_hist.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
  }else{
    MPI_Reduce(dat, dat, 6, MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    MPI_Reduce(mweight.K_hist.data(), mweight.K_hist.data(), mweight.K_hist.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
  }
  if (mpi.rank==mpi.master){
    Pnorm = dat[0]/mpi.size;
    occurence = dat[1]/mpi.size;
    Pw0   = dat[2]/mpi.size;
    Pw02  = dat[3]/mpi.size;
    Ekin  = dat[4]/mpi.size;
    Ekin2 = dat[5]/mpi.size;
    mweight.K_hist *= 1./mpi.size;
  }
#endif
  double sigmaPw=0;
  double sigmaEkin=0;
  if (mpi.rank==mpi.master){
    log<<"total acceptance rate="<<(Nacc_k+Nacc_q+Nacc_t+Nacc_w)/(p.Nitt+0.0)<<" "<<" k-acceptance="<<(Nacc_k/(Nall_k+0.0))<<" t-acceptanece="<<(Nacc_t/(Nall_t+0.0))<<" w-acceptanece="<<(Nacc_w/(Nall_w+0.0))<<std::endl;
    log<<" k-trials="<<(Nall_k/(p.Nitt+0.0))<<" q-trials="<<(Nall_q/(p.Nitt+0.0))<<" t-trials="<<(Nall_t/(p.Nitt+0.0))<<" w-trials="<<(Nall_w/(p.Nitt+0.0))<<std::endl;
    //cout<<"t1="<<t1.elapsed()<<" t2="<<t2.elapsed()<<" t3="<<t3.elapsed()<<" t4="<<t4.elapsed()<<" t5="<<t5.elapsed()<<" t6="<<t6.elapsed()<<" t7="<<t7.elapsed()<<" t8="<<t8.elapsed()<<" t9="<<t9.elapsed()<<" t10="<<t10.elapsed()<<" t11="<<t11.elapsed()<<std::endl;
    log<<"total time="<<t_all.elapsed()<<" measuring time="<<t_mes.elapsed()<<std::endl;

    double fct = fabs(V0norm)/Pnorm;
    Pw0  *= fct;
    Pw02 *= fct*fct;
    sigmaPw = sqrt(fabs(Pw02 - Pw0*Pw0))/sqrt(mpi.size);
    
    Ekin *= fct;
    Ekin2 *= fct*fct;
    sigmaEkin = sqrt(fabs(Ekin2-Ekin*Ekin))/sqrt(mpi.size);
    
    // Proper normalization of the resulting Monte Carlo data
    double norm = ipower( 1/((2*pi)*(2*pi)*(2*pi)), Norder) *  ipower( beta, Norder-1); // one order less, because one G-propagator is missing and one time is missing.
    Pw0 *= norm;
    sigmaPw *= norm;
    Ekin *= norm;
    sigmaEkin *= norm;
    
    log<<"  meassuring diagram occurence frequency="<<occurence<<" and its norm Pnorm="<<Pnorm<<std::endl;
    for (int ik=0; ik<mweight.K_hist.extent(0); ik++){
      double dsum=0;
      for (int i=0; i<Nbin; i++) dsum += mweight.K_hist(ik,i);
      mweight.K_hist(ik,bl::Range::all()) *= 1./dsum;
      std::ofstream hout((std::string("Q_hist.")+std::to_string(ik)).c_str());
      double cutoff = cutoffk;
      for (int i=0; i<Nbin; i++){
	hout<< (i+0.5)/Nbin*cutoff/kF << " " << mweight.K_hist(ik,i)<<std::endl;
      }
    }
  }
  return std::make_tuple(Pw0,sigmaPw,Ekin,sigmaEkin);
}

template<typename GK, typename Data>
void sample_static_DiscreteC_combined(Data& BKdata, bl::Array<double,2>& Pbin, std::ostream& log, const bl::Array<double,1>& qx,
				      double lmbda, const std::vector<double>& lmbda_spct, const GK& Gk, const params& p,
				      const bl::Array<unsigned short,2>& diagsG,
				      const std::vector<std::vector<int> >& diagSign,
				      const std::vector<std::vector<std::vector<int> > >& Loop_index, const std::vector<std::vector<std::vector<int> > >& Loop_type,
				      bl::Array<char,2>& Vtype, my_mpi& mpi)
{
  typedef double real;
  log.precision(12);
  
  bool Q0w0 = BKdata.Q0w0;
  double Q_external = qx(0);
  if (Q0w0){
    log << "Q_external=" << Q_external << endl;
    if (Pbin.extent(0)!=p.Nt) Pbin.resize(p.Nt,1); Pbin=0.0;
  } else {
    if ( BKdata.Nlt!=p.Nlt || BKdata.Nq!=qx.extent(0)){
      log<<"ERROR : Dimensions of C_Pln is wrong : either "<<BKdata.Nlt<<" != "<<p.Nlt<<" or "<<BKdata.Nq<<" != "<<qx.extent(0)<<std::endl;
      exit(1);
    }
    if (Pbin.extent(0)!=p.Nt || Pbin.extent(1)!=qx.extent(0)) Pbin.resize(p.Nt,qx.extent(0)); Pbin=0.0;
  }
  
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
  BKdata.C_Pln = 0.0;            // cummulative result for legendre expansion
  
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
  BKdata.FindGroups(diagsG, Loop_index, Loop_type, log);
  
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
  std::vector<double> nrm = Nrm(lmbda, cutoffk, lmbda_spct.size()+2);
  for (int i=0; i<lmbda_spct.size(); ++i){
#ifdef CNTR_VN
    // [Vq^2, Vq^3, Vq^4,..]
    lmbda_spct_cutoffk(i) = lmbda_spct[i]*nrm[i+2];
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
  log<<"Starting times="<<times<<" and starting momenta="<<momentum<<std::endl;

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
	long double Vn = ipower(Vq, Nc+1);
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
  if (mpi.rank==mpi.master) PrintInfo_(0, Qweight, amomentm(0), amomentm(1), times(0)-times(1), PQ, PQ, 0, 0, log);
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
	      long double Vn = ipower(Vq, Nc+1);
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
	}else Nacc_k += 1;
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
	      long double Vn = ipower(Vq, Nc+1);
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
      if ( itt>2e5/mpi.size && itt%(static_cast<int>(100000/mpi.size+1000)*tmeassure) == 0){
	double last_occurence = (Nweight-Wchanged)/(Nmeassure-Nchanged); // occurence from the last time we changed it
#ifdef _MPI
	double total_occurence=0;
	int ierr = MPI_Allreduce(&last_occurence, &total_occurence, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	if (ierr!=0) log << "MPI_Allreduce(total_occurence) returned error="<< ierr << endl; 
	last_occurence = total_occurence/mpi.size;
#endif	
	if ( last_occurence > 0.15){ // decrease by two
	  V0norm /= 2.;
	  Pnorm /= 2.;
	  Nchanged = Nmeassure;
	  Wchanged = Nweight;
	  if (mpi.rank==mpi.master) log<<std::left<<(itt/1.0e6)<<"M occurence="<<std::setw(8)<<last_occurence<<" V0 reduced to "<<V0norm<<std::endl;
	}
	if ( last_occurence < 0.03){ // increase by two
	  V0norm *= 2.;
	  Pnorm *= 2.;
	  Nchanged = Nmeassure;
	  Wchanged = Nweight;
	  if (mpi.rank==mpi.master) log<<std::left<<(itt/1.0e6)<<"M occurence="<<std::setw(8)<<last_occurence<<" V0 increased to "<<V0norm<<std::endl;
	}
      }
      if (itt%(500000*tmeassure) == 0){
	if (Nmeassure-Nweight>1000){
#ifdef _MPI	  
	  int ierr = MPI_Allreduce(MPI_IN_PLACE, mweight.K_hist.data(), mweight.K_hist.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	  if (ierr!=0) log << "MPI_Allreduce(K_hist) returned error="<< ierr << endl;
#endif
	  // We can get overflow during MPI for a long run. We should use a constant for normalization.
	  // We could here normalize K_hist (with knorm), and than when we measure, we would
	  // add instead of adding unity, we would add 1./knorm
	  dk_hist *= mweight.Normalize_K_histogram();
	  if (dk_hist < 1e-16) dk_hist = 1.0;
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
  BKdata.C_Pln *= 1.0/Nmeassure;
  Pbin         *= 1.0/Nmeassure;
  Pnorm        *= 1.0/Nmeassure;
  double occurence = static_cast<double>(Nweight)/static_cast<double>(Nmeassure);

  if (GetHistogram){
    K_Ghist *= 1.0/Nmeassure;
    K_Vhist *= 1.0/Nmeassure;
  }
  
#ifdef _MPI  
  double dat[2] = {Pnorm, occurence};
  if (mpi.rank==mpi.master){
    MPI_Reduce(MPI_IN_PLACE, BKdata.C_Pln.data(), BKdata.C_Pln.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE,  Pbin.data(),  Pbin.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, dat, 2, MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, mweight.K_hist.data(), mweight.K_hist.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    if (GetHistogram){
      MPI_Reduce(MPI_IN_PLACE, K_Ghist.data(), K_Ghist.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
      MPI_Reduce(MPI_IN_PLACE, K_Vhist.data(), K_Vhist.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    }
  }else{
    MPI_Reduce(BKdata.C_Pln.data(), BKdata.C_Pln.data(), BKdata.C_Pln.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    MPI_Reduce( Pbin.data(),  Pbin.data(),  Pbin.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    MPI_Reduce(dat, dat, 2, MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    MPI_Reduce(mweight.K_hist.data(), mweight.K_hist.data(), mweight.K_hist.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    if (GetHistogram){
      MPI_Reduce(K_Ghist.data(), K_Ghist.data(), K_Ghist.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
      MPI_Reduce(K_Vhist.data(), K_Vhist.data(), K_Vhist.size(), MPI_DOUBLE, MPI_SUM, mpi.master, MPI_COMM_WORLD);
    }
  }
  if (mpi.rank==mpi.master){
    BKdata.C_Pln *= 1./mpi.size;
    Pbin         *= 1./mpi.size;
    Pnorm        = dat[0]/mpi.size;
    occurence    = dat[1]/mpi.size;
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
    BKdata.C_Pln *= dOmega * (fabs(V0norm)/Pnorm); // 4*pi because we must divide by 1/(4*pi*q**2), and 1/q**2 was already done in sampling
    Pbin         *= dOmega * (fabs(V0norm)/Pnorm);
    log<<"  meassuring diagram occurence frequency="<<occurence<<" and its norm Pnorm="<<Pnorm<<std::endl;
    
    // Proper normalization of the resulting Monte Carlo data
    //double dOmega2 = (!Q0w0) ? (4*pi*cutoffq*cutoffq*cutoffq/3) : 1.0;
    double dOmega2 = (!Q0w0) ? 4*pi*sum(qx*qx) : 1.0;
      
    double norm = ipower( beta/((2*pi)*(2*pi)*(2*pi)), Norder) * dOmega2;
    BKdata.C_Pln *= norm;
    Pbin         *= norm;
    // Later we will use Nthbin/2.0*Nkbin/2.0  normalization in the Kernel to transform from binned data to function values.
    BKdata.Normalize(beta);
    
    //double dq_binning = cutoffq/Nq;
    double dt_binning = beta/Nt;
    if (!Q0w0)
      Pbin *= 1.0/(dt_binning);
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
