// @Copyright 2018 Kristjan Haule 
#ifndef BAYMK
#define BAYMK

#define DGER_BLAS
#ifdef DGER_BLAS
extern "C" {
  void dger_(const int* n1, const int* n2, const double* alpha, const double* x, const int* incx, const double* y, const int* incy, double* A, const int* lda);  
}
#endif

class BaymKadanoff_Parent{
public:
  bl::Array<double,1>& kxb;
  bl::Array<int,1> BKgroups;
  bl::Array<unsigned short,1> BKindex;
  bl::Array<double,1> PQg, PQg_new;
public:
  BaymKadanoff_Parent(bl::Array<double,1>& _kxb_) : kxb(_kxb_)/*, Nlt(0), Nlq(0)*/ {}
  void PQg_Initialize()                { PQg = 0; }
  void PQg_Add(int id, double PQd)     { PQg(BKindex(id)) += PQd; }
  void PQg_new_Initialize()            { PQg_new = 0; }
  void PQg_new_Add(int id, double PQd) { PQg_new(BKindex(id)) += PQd; }
  void Set_PQg_new_to_PQg()            { PQg = PQg_new; }
  
  void FindGroups(const bl::Array<unsigned short,2>& diagsG, const std::vector<std::vector<std::vector<int> > >& Loop_index, const std::vector<std::vector<std::vector<int> > >& Loop_type, ostream& log){
    // For efficiency, we want to find which diagram have common values of momenta at the right hand side of the diagram.
    int Ndiags = diagsG.extent(0);
    int Nloops = Loop_index[0].size();
    BKindex.resize(Ndiags);
    // Feynman graphs ends like in the figure below:
    //   k+q
    //  --->-
    //  |     \    q
    //  |     /   vertex=0
    //  ---<-
    //    k G[0]==G_k
    // We want to find what are possibilities for G[0]. Ideally there would be just one, but since we want to have
    // diagrams which cancel by sign, we have several possibilities, which are here enumerated.
    std::deque<int> bkgroups;
    std::deque< std::deque<int> > how_many;
    for (int id=0; id<Ndiags; id++){
      std::deque<int> k_momenta;
      for (int iloop=0; iloop<Nloops; ++iloop){
	const std::vector<int>& lindex = Loop_index[id][iloop];
	const std::vector<int>& ltype  = Loop_type[id][iloop];
	for (int i=0; i<lindex.size(); i++){
	  // this shows you which momenta contribute to the G[0] propagator, and what is the loops orientation.
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
    PQg.resize(BKgroups.extent(0));
    PQg_new.resize(BKgroups.extent(0));
  }
  template <typename real>
  void TestGroups(const bl::Array<bl::TinyVector<real,3>,2>& mom_G){
    bl::Array<double,1> k_nrm(BKgroups.extent(0)), k_cth(BKgroups.extent(0));
    for (int ig=0; ig<BKgroups.extent(0); ig++){
      int id_representative = BKgroups(ig);
      bl::TinyVector<double,3> k = mom_G(id_representative,0);
      k_nrm(ig) = norm(k);
      k_cth(ig) = k(2)/k_nrm(ig);
    }
    int Ndiags = mom_G.extent(0);
    for (int id=0; id<Ndiags; id++){
      bl::TinyVector<double,3> k = mom_G(id,0);
      double nk = norm(k);
      double cos_theta = k(2)/nk;
      int ig = BKindex(id);
      if (fabs(k_nrm(ig)-nk)>1e-6){std::cerr<<"ERROR : 1) It seems BKgroups or BKindex was not properly computed!"<<std::endl; exit(1);}
      if (fabs(k_cth(ig)-cos_theta)>1e-6){ std::cerr<<"ERROR : 2) It seems BKgroups or BKindex was not properly computed! "<<std::endl; exit(1);}
    }
  }
};

class BaymKadanoff_Q0W0_Data : public BaymKadanoff_Parent{
public:
  const bool Q0w0 = true;  
  double Q_external;
  bl::Array<double,2>& C_Pln; // C_Pln(Nthbin,kxb.extent(0)-1)
  int Nthbin, Nlt, Nlq;
  
  BaymKadanoff_Q0W0_Data(bl::Array<double,2>& _C_Pln_, bl::Array<double,1>& _kxb_, double _Q_external_=0.0) :
    C_Pln(_C_Pln_), Q_external(_Q_external_), Nlt(0), Nlq(0), BaymKadanoff_Parent(_kxb_)
  {
    Nthbin = C_Pln.extent(0);
    C_Pln = 0.0;
  }
  
  template <typename real>
  void Meassure(int itt, int tmeassure, double PQ, double sp, bl::Array<double,1>& , bl::Array<double,1>& ,
		const bl::TinyVector<double,3> , double , const bl::Array<bl::TinyVector<real,3>,1>& mom_g,
		const bl::Array<unsigned short,2>& Gindx)
  {
    if (itt%(2000*tmeassure) == 0){ // For efficiency, do not check every time.
      if ( fabs(sum(PQg)/PQ-1) > 1e-4 ){
	std::cerr<<"ERROR  PQg and PQ are not equal : PQ="<< PQ << " and PQg="<< sum(PQg) << std::endl;
      }
    }
    for (int ig=0; ig<BKgroups.extent(0); ig++){
      double group_weight = PQg(ig)/PQ;
      int id_representative = BKgroups(ig);
      bl::TinyVector<double,3> k = mom_g(Gindx(id_representative,0));
      double ak = norm(k);
      double cos_theta=0;
      if (ak>0) cos_theta = k[2]/ak;
      int ith = static_cast<int>(Nthbin * 0.5*(cos_theta+1));
      if (ith>=Nthbin) ith=Nthbin-1;
      int ik;
      {
	int Nkbin = kxb.extent(0)-1;
	int klo = 0, khi=Nkbin-1;
	ik = bisection(ak, klo, khi, Nkbin, kxb);
      }
      C_Pln(ith,ik) += sp * group_weight;
    }
  }
  void Normalize(double beta, double cutoffq){
    C_Pln(bl::Range::all(),bl::Range::all()) *=  1./beta;
    int Nkbin = kxb.extent(0)-1;
    double dtheta = 2.0/Nthbin;
    // Since ik and itheta are binned, we need to divide by 1/dtheta * 1/dk
    for (int ik=0; ik<Nkbin; ik++)
      C_Pln(bl::Range::all(),ik) *= 1./(dtheta*(kxb(ik+1)-kxb(ik)));
  }
};

class BaymKadanoffData : public BaymKadanoff_Parent{
public:
  const bool Q0w0 = false;
  double Q_external;
  bl::Array<double,4>& C_Pln; // C_Pln(Nthbin,kxb.extent(0)-1,Nlt+1,Nlq+1)
  int Nthbin, Nlt, Nlq;
  bl::Array<double,2> Pl_Pl_tensor_Product;
  int Nlqp1;
public:
  BaymKadanoffData(bl::Array<double,4>& _C_Pln_, bl::Array<double,1>& _kxb_) :
    C_Pln(_C_Pln_), Q_external(0.0), BaymKadanoff_Parent(_kxb_){
    Nthbin = C_Pln.extent(0);  
    Nlt    = C_Pln.extent(2)-1;
    Nlq    = C_Pln.extent(3)-1;
    C_Pln = 0.0;
    Nlqp1 = Nlq+1;
    Pl_Pl_tensor_Product.resize(Nlt+1,Nlq+1);
  }
  template <typename real>
  void Meassure(int itt, int tmeassure, double PQ, double sp, bl::Array<double,1>& pl_Q, bl::Array<double,1>& pl_t,
		const bl::TinyVector<double,3> Q, double aQ, const bl::Array<bl::TinyVector<real,3>,1>& mom_g,
		const bl::Array<unsigned short,2>& Gindx)
  {
    //bl::Array<double,1> pl_Q(Nlq+1);  // Will contain legendre Pl(2*q/qmax-1)
    //bl::Array<double,1> pl_t(Nlt+1);  // Will contain legendre Pl(2*t//beta-1)
    
    if (itt%(2000*tmeassure) == 0){ // For efficiency, do not check every time.
      if ( fabs(sum(PQg)/PQ-1) > 1e-4 ){
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
      double ak = norm(k);
      double cos_theta=0;
      if (ak>0 && aQ>0) cos_theta = dot(k,Q)/(ak*aQ);
      int ith = static_cast<int>(Nthbin * 0.5*(cos_theta+1));
      if (ith>=Nthbin) ith=Nthbin-1;
      int ik;
      {
	int Nkbin = kxb.extent(0)-1;
	int klo = 0, khi=Nkbin-1;
	ik = bisection(ak, klo, khi, Nkbin, kxb);
      }
      C_Pln(ith,ik,bl::Range::all(),bl::Range::all()) += Pl_Pl_tensor_Product * group_weight;
    }
#else
    int inct=1, incq=1, lda=Nlq+1, n1=Nlq+1, n2=Nlt+1;
    for (int ig=0; ig<BKgroups.extent(0); ig++){
      double group_weight = PQg(ig)/PQ;
      int id_representative = BKgroups(ig);
      bl::TinyVector<double,3> k = mom_g(Gindx(id_representative,0));
      double ak = norm(k);
      double cos_theta=0;
      if (ak>0 && aQ>0) cos_theta = dot(k,Q)/(ak*aQ);
      int ith = static_cast<int>(Nthbin * 0.5*(cos_theta+1));
      if (ith>=Nthbin) ith=Nthbin-1;
      int ik;
      {
	int Nkbin = kxb.extent(0)-1;
	int klo = 0, khi=Nkbin-1;
	ik = bisection(ak, klo, khi, Nkbin, kxb);
      }
      double alpha = sp*group_weight;
      double* restrict pp = &C_Pln(ith,ik,0,0);
      dger_(&n1, &n2, &alpha, pl_Q.data(), &incq, pl_t.data(), &inct, pp, &lda);
    }
#endif
  }
  void Normalize(double beta, double cutoffq){

    for (int lt=0; lt<=Nlt; lt++)
      for (int lq=0; lq<=Nlq; lq++)
	C_Pln(bl::Range::all(),bl::Range::all(),lt,lq) *=  ((2*lt+1.)/beta) * ((2.*lq+1.)/cutoffq);

    int Nkbin = kxb.extent(0)-1;
    double dtheta = 2.0/Nthbin;
    // Since ik and itheta are binned, we need to divide by 1/dtheta * 1/dk
    for (int ik=0; ik<Nkbin; ik++)
      C_Pln(bl::Range::all(),ik,bl::Range::all(),bl::Range::all()) *= 1./(dtheta*(kxb(ik+1)-kxb(ik)));
  }
};

class BaymKadanoff_Discrete : public BaymKadanoff_Parent{
public:
  bool Q0w0;
  bl::Array<double,4>& C_Pln; // C_Pln(Nthbin, kxb.extent(0)-1, qx.extent(0), Nlt+1);
  int Nthbin, Nlt, Nq;
public:
  BaymKadanoff_Discrete(bl::Array<double,4>& _C_Pln_, bl::Array<double,1>& _kxb_) :
    C_Pln(_C_Pln_), BaymKadanoff_Parent(_kxb_)
  {
    if (C_Pln.extent(2)==1 && C_Pln.extent(3)==1) Q0w0=true;
    else Q0w0=false;
    Nthbin = C_Pln.extent(0);  
    Nq    = C_Pln.extent(2);    // 1 or more
    Nlt    = C_Pln.extent(3)-1; // 0 or more
    C_Pln = 0.0;
  }
  template <typename real>
  void Meassure(int itt, int tmeassure, double PQ, double sp, int iiQ, bl::Array<double,1>& pl_t,
		const bl::TinyVector<double,3> Q, double aQ, const bl::Array<bl::TinyVector<real,3>,1>& mom_g,
		const bl::Array<unsigned short,2>& Gindx)
  {
    //bl::Array<double,1> pl_t(Nlt+1);  // Will contain legendre Pl(2*t//beta-1)
    if (itt%(2000*tmeassure) == 0){ // For efficiency, do not check every time.
      if ( fabs(sum(PQg)/PQ-1) > 1e-4 ){
	std::cerr<<"ERROR  PQg and PQ are not equal : PQ="<< PQ << " and PQg="<< sum(PQg) << std::endl;
      }
    }
    for (int ig=0; ig<BKgroups.extent(0); ig++){
      double group_weight = PQg(ig)/PQ;
      int id_representative = BKgroups(ig);
      bl::TinyVector<double,3> k = mom_g(Gindx(id_representative,0));
      double ak = norm(k);
      double cos_theta=0;
      if (ak>0 && aQ>0) cos_theta = dot(k,Q)/(ak*aQ);
      int ith = static_cast<int>(Nthbin * 0.5*(cos_theta+1));
      if (ith>=Nthbin) ith=Nthbin-1;
      int ik;
      {
	int Nkbin = kxb.extent(0)-1;
	int klo = 0, khi=Nkbin-1;
	ik = bisection(ak, klo, khi, Nkbin, kxb);
      }
      double wgh = sp * group_weight;
      C_Pln(ith,ik,iiQ,bl::Range::all()) += pl_t(bl::Range::all()) * wgh;
    }
  }
  void Normalize(double beta)
  {
    for (int lt=0; lt<=Nlt; lt++)
      C_Pln(bl::Range::all(),bl::Range::all(),bl::Range::all(),lt) *=  (2*lt+1.)/beta;

    int Nkbin = kxb.extent(0)-1;
    double dtheta = 2.0/Nthbin;
    // Since ik and itheta are binned, we need to divide by 1/dtheta * 1/dk
    for (int ik=0; ik<Nkbin; ik++)
      C_Pln(bl::Range::all(),ik,bl::Range::all(),bl::Range::all()) *= 1./(dtheta*(kxb(ik+1)-kxb(ik)));
  }
};


class StandardParent{
public:
  void PQg_Initialize()                { }
  void PQg_Add(int id, double PQd)     { }
  void PQg_new_Initialize()            { }
  void PQg_new_Add(int id, double PQd) { }
  void Set_PQg_new_to_PQg()            { }
  void FindGroups(const bl::Array<unsigned short,2>& diagsG, const std::vector<std::vector<std::vector<int> > >& Loop_index, const std::vector<std::vector<std::vector<int> > >& Loop_type, ostream& log){}
  template <typename real>
  void TestGroups(const bl::Array<bl::TinyVector<real,3>,2>& mom_G){}
};

class StandardData : public StandardParent{
public:
  const bool Q0w0 = false;
  double Q_external;
  bl::Array<double,2>& C_Pln;
  int Nlt, Nlq;
  int Nlqp1;
public:
  StandardData(bl::Array<double,2>& _C_Pln_) : C_Pln(_C_Pln_), Q_external(0.0){
    Nlt    = C_Pln.extent(0)-1;//24
    Nlq    = C_Pln.extent(1)-1;//18
    Nlqp1 = Nlq+1;
  }
  template <typename real>
  void Meassure(int itt, int tmeassure, double PQ, double sp, bl::Array<double,1>& pl_Q, bl::Array<double,1>& pl_t,
		const bl::TinyVector<double,3> Q, double aQ, const bl::Array<bl::TinyVector<real,3>,1>& mom_g,
		const bl::Array<unsigned short,2>& Gindx)
  {
#ifndef DGER_BLAS
    double* restrict p_Q = pl_Q.data();
    double* restrict p_t = pl_t.data();
    double* restrict p_p = C_Pln.data();
    for (int lt=0; lt<=Nlt; lt++){
      double t = p_t[lt]*sp;
      for (int lq=0; lq<=Nlq; ++lq) p_p[lq] += p_Q[lq]*t;
      p_p += Nlqp1;
    }
#else
    int inct=1, incq=1, lda=Nlq+1, n1=Nlq+1, n2=Nlt+1;
    dger_(&n1, &n2, &sp, pl_Q.data(), &incq, pl_t.data(), &inct, C_Pln.data(), &lda);
#endif
  }
  void Normalize(double beta, double cutoffq){
    for (int lt=0; lt<=Nlt; lt++)
      for (int lq=0; lq<=Nlq; lq++)
	C_Pln(lt,lq) *=  ((2*lt+1.)/beta) * ((2.*lq+1.)/cutoffq);
  };
};

class Standard_Q0W0_Data : public StandardParent{
public:
  const bool Q0w0 = true;
  double Q_external;
  bl::Array<double,2>& C_Pln;
  int Nlt, Nlq;
public:
  Standard_Q0W0_Data(bl::Array<double,2>& _C_Pln_, double _Q_external_=0.0) :
    C_Pln(_C_Pln_), Nlt(0), Nlq(0), Q_external(_Q_external_) { }
  template <typename real>
  void Meassure(int itt, int tmeassure, double PQ, double sp, bl::Array<double,1>& , bl::Array<double,1>& ,
		const bl::TinyVector<double,3> , double , const bl::Array<bl::TinyVector<real,3>,1>& mom_g,
		const bl::Array<unsigned short,2>& Gindx)
  { C_Pln(0,0) += sp; }
  void Normalize(double beta, double cutoffq){C_Pln *=  1./(beta);};
};

class Standard_Discrete : public StandardParent{
public:
  bool Q0w0; // if C_Pln is of size (1,1) than Q0W0 should be true
  bl::Array<double,2>& C_Pln; // C_Pln(qx.extent(0), Nlt+1);
  int Nlt, Nq;
public:
  Standard_Discrete(bl::Array<double,2>& _C_Pln_) : C_Pln(_C_Pln_)
  {
    if ( C_Pln.extent(0)==1 && C_Pln.extent(1)==1 ) Q0w0 = true;
    else Q0w0 = false;
    Nq = C_Pln.extent(0);
    Nlt = C_Pln.extent(1)-1;
  }
  template <typename real>
  void Meassure(int itt, int tmeassure, double PQ, double sp, int iiQ, bl::Array<double,1>& pl_t,
		const bl::TinyVector<double,3> Q, double aQ, const bl::Array<bl::TinyVector<real,3>,1>& mom_g,
		const bl::Array<unsigned short,2>& Gindx)
  { C_Pln(iiQ,bl::Range::all()) += pl_t(bl::Range::all())*sp; }
  
  void Normalize(double beta){
    for (int lt=0; lt<C_Pln.extent(1); lt++)
      C_Pln(bl::Range::all(),lt) *=  ((2*lt+1.)/beta);
  };
};


class BaymKadanoff_Symmetric_Parent{
public:
  bl::Array<double,1>& kxb;
  bl::Array<int,1> BKgroups;
  bl::Array<unsigned short,1> BKindex;
  bl::Array<double,1> PQg, PQg_new;
  bl::Array<unsigned short,2> i_diagsG;
public:
  BaymKadanoff_Symmetric_Parent(bl::Array<double,1>& _kxb_) : kxb(_kxb_){}
  void PQg_Initialize()                { PQg = 0; }
  void PQg_Add(int id, double PQd)     { PQg(BKindex(id)) += PQd; }
  void PQg_new_Initialize()            { PQg_new = 0; }
  void PQg_new_Add(int id, double PQd) { PQg_new(BKindex(id)) += PQd; }
  void Set_PQg_new_to_PQg()            { PQg = PQg_new; }
  
  void FindGroups(const bl::Array<unsigned short,2>& diagsG, const std::vector<std::vector<std::vector<int> > >& Loop_index, const std::vector<std::vector<std::vector<int> > >& Loop_type, ostream& log){
    // For efficiency, we want to find which diagram have common values of momenta at the right hand side of the diagram.
    int Ndiags = diagsG.extent(0);
    int Nloops = Loop_index[0].size();
    int Norder = diagsG.extent(1)/2;
    // computing the inverse of all permutations, to provide fast lookup for the previous fermionic propagator
    i_diagsG.resize(Ndiags,2*Norder);
    for (int id=0; id<Ndiags; id++)
      for (int i=0; i<2*Norder; i++)
	i_diagsG(id, static_cast<int>(diagsG(id,i)) )=i;
    
    BKindex.resize(Ndiags);
    // Feynman graphs ends like in the figure below:
    //         k'+q       k+q
    //             -->----->-
    // vrtex=1  /    |  |     \    q
    //          \    |  |     /   vertex=0
    //            --<------<-
    //          k'         k G[0]==G_k
    //          G^i[1]=G_k'
    // We want to find what are possibilities for G[0]. Ideally there would be just one, but since we want to have
    // diagrams which cancel by sign, we have several possibilities, which are here enumerated.
    std::deque<int> bkgroups;
    std::deque< std::deque<int> > how_many;
    for (int id=0; id<Ndiags; id++){
      std::deque<int> k_momenta_right;
      std::deque<int> k_momenta_left;
      int inv_1  = i_diagsG(id, 1); // the propagators which ends at vertex=1, i.e., k' in the figure
      for (int iloop=0; iloop<Nloops; ++iloop){
	const std::vector<int>& lindex = Loop_index[id][iloop];
	const std::vector<int>& ltype  = Loop_type[id][iloop];
	for (int i=0; i<lindex.size(); i++){
	  int ltype_i = ltype[i];
	  int lindex_i = lindex[i];
	  // this shows you which momenta contribute to the G[0] propagator, and what is the loops orientation.
	  // Note that external momentum (iloop==0) can have only a single orientation (i.e.,negative) and therefore 0 can serve the purpose
	  if ( abs(ltype_i)==1  && lindex_i==0 )
	    k_momenta_right.push_back( sign(ltype_i)*iloop );
	  if ( abs(ltype_i)==1 && lindex_i==inv_1 )
	    k_momenta_left.push_back( sign(ltype_i)*iloop );
	}
      }
      // now merging left and right momentum lists
      std::deque<int> k_momenta;
      for (auto it=k_momenta_right.begin(); it!=k_momenta_right.end(); ++it)
	k_momenta.push_back( *it );
      for (auto it=k_momenta_left.begin(); it!=k_momenta_left.end(); ++it)
	k_momenta.push_back( *it );
      
      auto where = std::find(how_many.begin(), how_many.end(), k_momenta );
      int iwhere = how_many.size();
      if (where == how_many.end()){
	how_many.push_back(k_momenta);
	bkgroups.push_back(id);
      }else{
	iwhere = distance(how_many.begin(),where);
      }
      BKindex(id) = iwhere;

      log << "id="<< id << " right_momenta=";
      for (auto it=k_momenta_right.begin(); it!=k_momenta_right.end(); ++it) log << (*it) <<",";
      log << "  left_momenta=";
      for (auto it=k_momenta_left.begin(); it!=k_momenta_left.end(); ++it) log << (*it) <<","; 
      log << "   BKindex=" << iwhere << " : ";
      for (int i=0; i<k_momenta.size(); i++) log<< k_momenta[i]<<",";
      log << std::endl;
    }
    BKgroups.resize(bkgroups.size());
    for (int i=0; i<bkgroups.size(); i++) BKgroups(i) = bkgroups[i];
    PQg.resize(BKgroups.extent(0));
    PQg_new.resize(BKgroups.extent(0));
  }
  template <typename real>
  void TestGroups(const bl::Array<bl::TinyVector<real,3>,2>& mom_G){
    bl::Array<double,1> k_nrm(BKgroups.extent(0)), k_cth(BKgroups.extent(0));
    bl::Array<double,1> kp_nrm(BKgroups.extent(0)), kp_cth(BKgroups.extent(0));
    for (int ig=0; ig<BKgroups.extent(0); ig++){
      int id_representative = BKgroups(ig);
      bl::TinyVector<double,3> k = mom_G(id_representative,0);
      int inv_1  = i_diagsG(id_representative, 1);
      bl::TinyVector<double,3> kp = mom_G(id_representative,inv_1);
      k_nrm(ig) = norm(k);
      kp_nrm(ig) = norm(kp);
      k_cth(ig) = k(2)/k_nrm(ig);
      kp_cth(ig) = kp(2)/kp_nrm(ig);
    }
    int Ndiags = mom_G.extent(0);
    for (int id=0; id<Ndiags; id++){
      bl::TinyVector<double,3> k = mom_G(id,0);
      int inv_1  = i_diagsG(id, 1);
      bl::TinyVector<double,3> kp = mom_G(id,inv_1);
      double nk = norm(k);
      double cos_theta = k(2)/nk;
      double nkp = norm(kp);
      double cos_thetap = kp(2)/nkp;
      int ig = BKindex(id);
      if (fabs(k_nrm(ig)-nk)>1e-6){std::cerr<<"ERROR : 1) It seems BKgroups or BKindex was not properly computed!"<<std::endl; exit(1);}
      if (fabs(k_cth(ig)-cos_theta)>1e-6){ std::cerr<<"ERROR : 2) It seems BKgroups or BKindex was not properly computed! "<<std::endl; exit(1);}
      if (fabs(kp_nrm(ig)-nkp)>1e-6){std::cerr<<"ERROR : 3) It seems BKgroups or BKindex was not properly computed!"<<std::endl; exit(1);}
      if (fabs(kp_cth(ig)-cos_thetap)>1e-6){ std::cerr<<"ERROR : 4) It seems BKgroups or BKindex was not properly computed! "<<std::endl; exit(1);}
    }
  }
};

class BaymKadanoff_Symmetric_Q0W0_Data : public BaymKadanoff_Symmetric_Parent{
public:
  const bool Q0w0 = true;
  double Q_external;
  bl::Array<double,4>& C_Pln;
  int Nthbin, Nlt, Nlq;
public:
  BaymKadanoff_Symmetric_Q0W0_Data(bl::Array<double,4>& _C_Pln_, bl::Array<double,1>& _kxb_, double _Q_external_=0.0) :
    C_Pln(_C_Pln_), Nlt(0), Nlq(0), Q_external(_Q_external_), BaymKadanoff_Symmetric_Parent(_kxb_)
  {
    Nthbin = C_Pln.extent(0);
    C_Pln = 0.0;
  }
  template <typename real>
  void Meassure(int itt, int tmeassure, double PQ, double sp, bl::Array<double,1>& , bl::Array<double,1>& ,
		const bl::TinyVector<double,3> , double , const bl::Array<bl::TinyVector<real,3>,1>& mom_g,
		const bl::Array<unsigned short,2>& Gindx)
  {
    if (itt%(2000*tmeassure) == 0){ // For efficiency, do not check every time.
      if ( fabs(sum(PQg)/PQ-1) > 1e-4 ){
	std::cerr<<"ERROR  PQg and PQ are not equal : PQ="<< PQ << " and PQg="<< sum(PQg) << std::endl;
      }
    }
    for (int ig=0; ig<BKgroups.extent(0); ig++){
      double group_weight = PQg(ig)/PQ;
      int id_representative = BKgroups(ig);
      bl::TinyVector<double,3> k  = mom_g(Gindx(id_representative,0));
      int inv_1  = i_diagsG(id_representative, 1);
      bl::TinyVector<double,3> kp = mom_g(Gindx(id_representative,inv_1));
      double ak = norm(k);
      double akp = norm(kp);
      double cos_theta=0;
      if (ak>0) cos_theta = k[2]/ak;
      double cos_thetap=0;
      if (akp>0) cos_thetap = kp[2]/akp;
      int ith = static_cast<int>(Nthbin * 0.5*(cos_theta+1));
      if (ith>=Nthbin) ith=Nthbin-1;
      int ithp = static_cast<int>(Nthbin * 0.5*(cos_thetap+1));
      if (ithp>=Nthbin) ithp=Nthbin-1;
      int ik, ikp;
      {
	int Nkbin = kxb.extent(0)-1;
	int klo = 0, khi=Nkbin-1;
	ik = bisection(ak, klo, khi, Nkbin, kxb);
	klo = 0; khi=Nkbin-1;
	ikp = bisection(akp, klo, khi, Nkbin, kxb);
      }
      C_Pln(ith,ik,ithp,ikp) += sp * group_weight;
    }
  }
  void Normalize(double beta, double cutoffq){
    C_Pln(bl::Range::all(),bl::Range::all(),bl::Range::all(),bl::Range::all()) *=  1./beta;
    int Nkbin = kxb.extent(0)-1;
    double dtheta = 2.0/Nthbin;
    // Since ik and itheta are binned, we need to divide by 1/dtheta * 1/dk
    for (int ik=0; ik<Nkbin; ik++)
      for (int ikp=0; ikp<Nkbin; ikp++)
	C_Pln(bl::Range::all(),ik,bl::Range::all(),ikp) *= 1./(dtheta*dtheta*(kxb(ik+1)-kxb(ik))*(kxb(ikp+1)-kxb(ikp)));
  }
};

class BaymKadanoff_Symmetric_Discrete : public BaymKadanoff_Symmetric_Parent{
public:
  bool Q0w0;
  bl::Array<double,6>& C_Pln; // C_Pln(Nthbin, kxb.extent(0)-1, Nthbin, kxb.extent(0)-1, qx.extent(0), Nlt+1);
  int Nthbin, Nlt, Nq;
public:
  BaymKadanoff_Symmetric_Discrete(bl::Array<double,6>& _C_Pln_, bl::Array<double,1>& _kxb_) :
    C_Pln(_C_Pln_), BaymKadanoff_Symmetric_Parent(_kxb_)
  {
    if (C_Pln.extent(4)==1 && C_Pln.extent(5)==1) Q0w0=true;
    else Q0w0=false;
    Nthbin = C_Pln.extent(0);
    Nq     = C_Pln.extent(4);    // 1 or more
    Nlt    = C_Pln.extent(5)-1; // 0 or more
    C_Pln = 0.0;
  }
  template <typename real>
  void Meassure(int itt, int tmeassure, double PQ, double sp, int iiQ, bl::Array<double,1>& pl_t,
		const bl::TinyVector<double,3> Q, double aQ, const bl::Array<bl::TinyVector<real,3>,1>& mom_g,
		const bl::Array<unsigned short,2>& Gindx)
  {
    if (itt%(2000*tmeassure) == 0){ // For efficiency, do not check every time.
      if ( fabs(sum(PQg)/PQ-1) > 1e-4 ){
	std::cerr<<"ERROR  PQg and PQ are not equal : PQ="<< PQ << " and PQg="<< sum(PQg) << std::endl;
      }
    }
    for (int ig=0; ig<BKgroups.extent(0); ig++){
      double group_weight = PQg(ig)/PQ;
      int id_representative = BKgroups(ig);
      bl::TinyVector<double,3> k  = mom_g(Gindx(id_representative,0));
      int inv_1  = i_diagsG(id_representative, 1);
      bl::TinyVector<double,3> kp = mom_g(Gindx(id_representative,inv_1));
      double ak = norm(k);
      double akp = norm(kp);
      double cos_theta=0;
      if (ak>0) cos_theta = k[2]/ak;
      double cos_thetap=0;
      if (akp>0) cos_thetap = kp[2]/akp;
      int ith = static_cast<int>(Nthbin * 0.5*(cos_theta+1));
      if (ith>=Nthbin) ith=Nthbin-1;
      int ithp = static_cast<int>(Nthbin * 0.5*(cos_thetap+1));
      if (ithp>=Nthbin) ithp=Nthbin-1;
      int ik, ikp;
      {//     determines where is |k|=ak in the discrete kxb mesh
       //also determines where is |k'|=akp in the same kxb mesh
	int Nkbin = kxb.extent(0)-1;
	int klo = 0, khi=Nkbin-1;
	ik = bisection(ak, klo, khi, Nkbin, kxb);
	klo = 0; khi=Nkbin-1;
	ikp = bisection(akp, klo, khi, Nkbin, kxb);
      }
      double wgh = sp * group_weight;
      C_Pln(ith,ik,ithp,ikp,iiQ,bl::Range::all()) += pl_t(bl::Range::all()) * wgh;
    }
  }
  void Normalize(double beta){
    C_Pln(bl::Range::all(),bl::Range::all(),bl::Range::all(),bl::Range::all(),bl::Range::all(),bl::Range::all()) *=  1./beta;
    int Nkbin = kxb.extent(0)-1;
    double dtheta = 2.0/Nthbin;
    // Since ik and itheta are binned, we need to divide by 1/dtheta * 1/dk
    for (int ik=0; ik<Nkbin; ik++)
      for (int ikp=0; ikp<Nkbin; ikp++)
	C_Pln(bl::Range::all(),ik,bl::Range::all(),ikp,bl::Range::all(),bl::Range::all()) *= 1./(dtheta*dtheta*(kxb(ik+1)-kxb(ik))*(kxb(ikp+1)-kxb(ikp)));
  }
};


class Cmp_momenta{
  std::deque< std::deque<int> >& how_many;
public:
  Cmp_momenta(std::deque< std::deque<int> >& _how_many_) : how_many(_how_many_){}
  bool operator()(int ig1, int ig2){
    if ( how_many[ig1][0] >=100 || how_many[ig2][0] >=100 )
      return how_many[ig2][0] < how_many[ig1][0]; // only sort those whose value is 100 or more
    else
      return false; // when value is less than 100, keep them unsorted
  }
};


class BaymKadanoff_Combined{
public:
  bl::Array<double,1>& kxb;
  bl::Array<int,1> BKgroups, BK_right, BK_left;
  std::deque<int> BKgroupr, BKgroupl;
  //
  bl::Array<unsigned short,1> BKindex;
  bl::Array<double,1> PQg, PQg_new;
  bl::Array<unsigned short,2> i_diagsG;
  //
  bl::Array<int,1> ith_left; // ith_left(BKgroupl.size());
  bl::Array<int,1> ith_right;// ith_right(BKgroupr.size());
  bl::Array<int,1> ik_right; // ik_right( BKgroupr.size());
  bl::Array<int,1> ik_left;  // ik_left ( BKgroupl.size());
  //
  bool Q0w0;
  bl::Array<double,6>& C_Pln2; // C_Pln2(Nthbin, kxb.extent(0)-1, Nthbin, kxb.extent(0)-1, qx.extent(0), Nlt+1);
  bl::Array<double,4>& C_Pln1; // C_Pln1(Nthbin, kxb.extent(0)-1, qx.extent(0), Nlt+1)
  bl::Array<double,2>& C_Pln0; // C_Pln0(qx.extent(0), Nlt+1)
  bl::Array<char,2>& Vtype;
  int Nthbin, Nlt, Nq, Nlq;
  int N_BKA0, N_BKA1;
  BaymKadanoff_Combined(bl::Array<double,6>& _C_Pln2_, bl::Array<double,4>& _C_Pln1_, bl::Array<double,2>& _C_Pln0_, 
			bl::Array<double,1>& _kxb_, bl::Array<char,2>& _Vtype_) : C_Pln2(_C_Pln2_), C_Pln1(_C_Pln1_), C_Pln0(_C_Pln0_),
										  kxb(_kxb_), Vtype(_Vtype_)
  {
    Nlq = 0;
    if (C_Pln2.extent(4)==1){
      Q0w0=true;
      Nlt = 0;
      Nthbin = 1;
    } else{
      Q0w0=false;
      Nlt    = C_Pln2.extent(5)-1; // 0 or more
      Nthbin = C_Pln2.extent(0);
    }
    Nq     = C_Pln2.extent(4);    // 1 or more

    if (! (C_Pln2.extent(0)==C_Pln2.extent(2) && C_Pln2.extent(2)==C_Pln1.extent(0)) ){
      cerr << "ERROR : C_Pln2 has wrong dimension, Nthbin points " << C_Pln2.extent(0)<<", "<< C_Pln2.extent(2) << ", " << C_Pln1.extent(0) << endl;
    }
    if (! (C_Pln2.extent(1)==C_Pln2.extent(3) && C_Pln2.extent(3)==C_Pln1.extent(1)) ){
      cerr << "ERROR : C_Pln2 has wrong dimension, Nk points " << C_Pln2.extent(1)<<", "<< C_Pln2.extent(3) << ", " << C_Pln1.extent(1) << endl;
    }
    if (! (C_Pln2.extent(4)==C_Pln1.extent(2) && C_Pln1.extent(2)==C_Pln0.extent(0)) ){
      cerr << "ERROR : C_Pln2 has wrong dimensions, Nq points " << C_Pln2.extent(4)<<", "<< C_Pln1.extent(2) << ", "<< C_Pln0.extent(0) << endl;
    }
    if (! (C_Pln2.extent(5)==C_Pln1.extent(3) && C_Pln1.extent(3)==C_Pln0.extent(1)) ){
      cerr << "ERROR : C_Pln2 has wrong dimensions, Nlt points " << C_Pln2.extent(5)<<", "<< C_Pln1.extent(3) << ", "<< C_Pln0.extent(1) << endl;
    }
    C_Pln2 = 0.0;
    C_Pln1 = 0.0;
    C_Pln0 = 0.0;
  }
															
  void ResetMeasure()                  {C_Pln2=0; C_Pln1=0; C_Pln0=0;}
  void PQg_Initialize()                { PQg = 0; }
  void PQg_Add(int id, double PQd)     { PQg(BKindex(id)) += PQd; }
  void PQg_new_Initialize()            { PQg_new = 0; }
  void PQg_new_Add(int id, double PQd) { PQg_new(BKindex(id)) += PQd; }
  void Set_PQg_new_to_PQg()            { PQg = PQg_new; }
  
  void FindGroups(const bl::Array<unsigned short,2>& diagsG, const std::vector<std::vector<std::vector<int> > >& Loop_index, const std::vector<std::vector<std::vector<int> > >& Loop_type,
		  ostream& log, bool debug, double Q_external){
    //bool debug=true;
    //log<<"Inside BaymKadanoff_Combined.FindGroups" << endl;
    // For efficiency, we want to find which diagram have common values of momenta at the right hand side of the diagram.
    int Ndiags = diagsG.extent(0);
    int Nloops = Loop_index[0].size();
    int Norder = diagsG.extent(1)/2;
    // computing the inverse of all permutations, to provide fast lookup for the previous fermionic propagator
    i_diagsG.resize(Ndiags,2*Norder);
    for (int id=0; id<Ndiags; id++)
      for (int i=0; i<2*Norder; i++)
	i_diagsG(id, static_cast<int>(diagsG(id,i)) )=i;
    
    BKindex.resize(Ndiags);
    // Feynman graphs ends like in the figure below:
    //         k'+q       k+q
    //             -->----->-
    // vrtex=1  /    |  |     \    q
    //          \    |  |     /   vertex=0
    //            --<------<-
    //          k'         k G[0]==G_k
    //          G^i[1]=G_k'
    // We want to find what are possibilities for G[0]. Ideally there would be just one, but since we want to have
    // diagrams which cancel by sign, we have several possibilities, which are here enumerated.
    std::deque<int> bkgroups;
    std::deque< std::deque<int> > how_many;

    std::vector<std::deque<int> > k_momenta_right(Ndiags);
    std::vector<std::deque<int> > k_momenta_left(Ndiags);
    for (int id=0; id<Ndiags; id++){
      int post_0 = diagsG(id,0);
      int inv_0  = i_diagsG(id,0);
      int post_1 = diagsG(id,1);
      int inv_1  = i_diagsG(id, 1);// the propagators which ends at vertex=1, i.e., k' in the figure

      bool ladder_left = false;
      bool ladder_right = false;
      if (partner(inv_0)==post_0 && Vtype(id,post_0/2)==0 ) //  it has ladder as the first interaction line on the right
	ladder_right=true;                           // but should not be counter-term interaction
      if (partner(inv_1)==post_1 && Vtype(id,post_1/2)==0 ) // it has ladder as the first interaction on the left
	ladder_left=true;                            // but should not be counter-term interaction

      // Finding all momenta that appear on the left and on the right
      //std::deque<int> k_momenta_right;
      //std::deque<int> k_momenta_left;
      int iloop_start=0;
      if (Q0w0 && (fabs(Q_external)<1e-10)) iloop_start=1; // for Q==0 the external loop does not change anything
      for (int iloop=iloop_start; iloop<Nloops; ++iloop){
	const std::vector<int>& lindex = Loop_index[id][iloop];
	const std::vector<int>& ltype  = Loop_type[id][iloop];
	for (int i=0; i<lindex.size(); i++){
	  int ltype_i = ltype[i];
	  int lindex_i = lindex[i];
	  // this shows you which momenta contribute to the G[0] propagator, and what is the loops orientation.
	  // Note that external momentum (iloop==0) can have only a single orientation (i.e.,negative) and therefore 0 can serve the purpose
	  if ( abs(ltype_i)==1  && lindex_i==0 )
	    k_momenta_right[id].push_back( sign(ltype_i)*iloop );
	  if ( abs(ltype_i)==1 && lindex_i==inv_1 )
	    k_momenta_left[id].push_back( sign(ltype_i)*iloop );
	}
      }

      // now merging left and right momentum lists
      std::deque<int> k_momenta;
      
      if (ladder_right){ // only relevant in BKA==0 (but not in BKA==1 or BKA==2) because it has ladder on the right
	k_momenta.push_back( 1000 ); // just add some large number that exceeds any loop. All diagrams of this type make a single group
      }else if (ladder_left){ // relevant in BKA==1 but not BKA==2, because it does not have ladder on the right but only on the left
	k_momenta.push_back( 100 ); // this will shows that it is not BKA==2 type
	// all right momenta are relevant, and each 
	for (auto it=k_momenta_right[id].begin(); it!=k_momenta_right[id].end(); ++it)
	  k_momenta.push_back( *it );
      } else{ // no ladder on the left or right => hence relevant for BKA==2
	for (auto it=k_momenta_right[id].begin(); it!=k_momenta_right[id].end(); ++it)
	  k_momenta.push_back( *it );
	for (auto it=k_momenta_left[id].begin(); it!=k_momenta_left[id].end(); ++it)
	  k_momenta.push_back( *it );
      }
      
      // checking if such combination of momenta has already appeared before?
      auto where = std::find( how_many.begin(), how_many.end(), k_momenta );
      int iwhere = how_many.size();
      if (where == how_many.end()){ // has not appeared before, hence is a new type diagram
	how_many.push_back(k_momenta); // remember this set of momenta
	bkgroups.push_back(id);        // first diagram of this type
      }else{
	iwhere = distance(how_many.begin(),where); // this is diagram with the same momenta, its position in how_many.
      }
      BKindex(id) = iwhere;
      /*
      if (debug){
	log << "id="<< setw(3)<< id <<" (";
	for (int j=0; j<2*Norder; j++) log << diagsG(id,j)<<",";
	log<<") BKindex="<< BKindex(id) <<" ";
	log<<" ladd_right="<<ladder_right<<" ladd_left="<<ladder_left;
	log<< " right_momenta=";
	for (auto it=k_momenta_right[id].begin(); it!=k_momenta_right[id].end(); ++it) log << (*it) <<",";
	log << "  left_momenta=";
	for (auto it=k_momenta_left[id].begin(); it!=k_momenta_left[id].end(); ++it) log << (*it) <<","; 
	log << "   BKindex=" << iwhere << " : ";
	for (int i=0; i<k_momenta.size(); i++) log<< k_momenta[i]<<",";
	log << std::endl;
      }
      */
    }
    
    PQg.resize(bkgroups.size());
    PQg_new.resize(bkgroups.size());

    {// Now we sort BKindex, so that BKA==0 diagrams appear first (those with ladder at the right hand), followed by BKA==1 terms (ladder on the left side), followed by all other diagrams.
      vector<int> grp_index(bkgroups.size());
      for (int i=0; i<grp_index.size(); i++) grp_index[i]=i;  // setting up an index array
      Cmp_momenta cmp_momenta(how_many);// we will sort according to how_many[ig][0] in descending order
      std::stable_sort(grp_index.begin(), grp_index.end(), cmp_momenta); // actual sorting done
      vector<int> grp_index_1(bkgroups.size()); // will create an inverse index
      for (int i=0; i<grp_index.size(); i++) grp_index_1[grp_index[i]]=i;

      N_BKA0=0;
      if (how_many[grp_index[N_BKA0]][0]==1000) N_BKA0=1; // We have some diagrams with ladder on the right-hand side
      // BKA==0 needs diagrams in group 0
      N_BKA1=0; // BKA==1 needs diagrams from 1... N_BKA1
      // BKA==2 needs diagrams from N_BKA1 to bkgroups.size()
      for (; N_BKA1<bkgroups.size(); N_BKA1++)
	if (how_many[grp_index[N_BKA1]][0]<100) break;
      
      if (debug) log << "N_BKA0=" << N_BKA0 << " N_BKA1="<< N_BKA1 << endl; 
      /*
      if (debug){
	log<<" grp_index= ";
	for (int i=0; i<grp_index.size(); i++) log << grp_index[i] << " , ";
	log<<endl;
	log<<" grp_index_1= ";
	for (int i=0; i<grp_index.size(); i++) log << grp_index_1[i] << " , ";
	log<<endl;
      }
      */
      bl::Array<unsigned short,1> BKindex_new(Ndiags);// with the help of an inverse index, we can set each diagram to the new sorted index.
      for (int id=0; id<Ndiags; id++) BKindex_new(id) = grp_index_1[BKindex(id)]; // setting new BKindex
      for (int id=0; id<Ndiags; id++) BKindex(id) = BKindex_new(id); // and now rewriting the old BKindex
    
      BKgroups.resize(bkgroups.size()); // correcting BKgroups
      for (int ig=0; ig<BKgroups.size(); ig++){
	for (int id=0; id<Ndiags; id++)
	  if (BKindex(id)==ig) {
	    BKgroups(ig) = id;
	    break;
	  }
      }
    }
  

    BK_right.resize(BKgroups.size());
    BK_left.resize(BKgroups.size());
    BK_right = -1; BK_left = -1;
    //  BKindexr and  BKindexl not essential, can be removed
    bl::Array<int,1> BKindexr(Ndiags), BKindexl(Ndiags);
    BKindexr = -1; BKindexl = -1;
    std::deque< std::deque<int> > right_side;
    std::deque< std::deque<int> > left_side;

    for (int id=0; id<Ndiags; id++){
      int ig = BKindex(id);
      if (ig>=N_BKA0){ // need momenta on the right
	// checking if such combination of momenta has already appeared before?
	auto where = std::find( right_side.begin(), right_side.end(), k_momenta_right[id] );
	int iwhere = right_side.size();
	if (where == right_side.end()){ // has not appeared before, hence is a new type diagram
	  right_side.push_back(k_momenta_right[id]); // remember this set of momenta
	  BKgroupr.push_back( id );
	}else{
	  iwhere = distance(right_side.begin(),where); // this is diagram with the same momenta, its position in how_many.
	}
	BK_right(ig) = iwhere;
	BKindexr(id) = iwhere;
      }
      if (ig>=N_BKA1){ // need momenta on the left as well
	// checking if such combination of momenta has already appeared before?
	auto where = std::find( left_side.begin(), left_side.end(), k_momenta_left[id] );
	int iwhere = left_side.size();
	if (where == left_side.end()){ // has not appeared before, hence is a new type diagram
	  left_side.push_back(k_momenta_left[id]); // remember this set of momenta
	  BKgroupl.push_back( id );
	}else{
	  iwhere = distance(left_side.begin(),where); // this is diagram with the same momenta, its position in how_many.
	}
	BK_left(ig) = iwhere;
	BKindexl(id) = iwhere;
      }
    }

    ith_left.resize(BKgroupl.size());
    ith_right.resize(BKgroupr.size());
    ith_left=0;
    ith_right=0;
    ik_right.resize(BKgroupr.size());
    ik_left.resize(BKgroupl.size());
    
    if (debug){
      log<<"BKgroups for attaching the vertex:"<<endl;
      log<<"  #"<<" "<<"rep"<< ","<<" r"<<","<<"l "<<"  (typ;#nm) : "<<endl;
      for (int ig=0; ig<BKgroups.size(); ig++){
	log<< setw(3)<<ig<<" " << setw(3)<< BKgroups(ig)<< ","<< setw(2) << BK_right(ig)<<","<< setw(2) << BK_left(ig)<<" ";
      	int n=0;
	for (int id=0; id<Ndiags; id++) if (BKindex(id)==ig) n+=1;
	int BKA_type = 2;
	if (ig==0) BKA_type=0;
	else if (ig<N_BKA1) BKA_type=1;
	//log << " ("<<setw(3)<<how_many[grp_index[ig]][0]<<";"<<n<<") : ";
	log << " ("<<setw(3)<< BKA_type <<";#"<<setw(2)<<n<<") : ";
	for (int id=0; id<Ndiags; id++){
	  if (BKindex(id)==ig){
	    log<<" "<<id<<"=(";
	    for (int j=0; j<2*Norder; j++) log << diagsG(id,j)<<",";
	    log<<") ";
	  }
	}
	log<<endl;
      }
      log<<"BKgroups for the right side attachement"<<endl;
      for (int igr=0; igr<BKgroupr.size(); igr++){
	int n=0;
	for (int id=0; id<Ndiags; id++) if (BKindexr(id)==igr) n+=1;
	log << setw(3) << igr <<" " << setw(3) << BKgroupr[igr] << "   (#"<<setw(3)<<n<<") ";
	for (int id=0; id<Ndiags; id++){
	  if (BKindexr(id)==igr){
	    log<<" "<<id<<"=(";
	    for (int j=0; j<2*Norder; j++) log << diagsG(id,j)<<",";
	    log<<") ";
	  }
	}
	log<<endl;
      }
      log<<"BKgroups for the left side attachement"<<endl;
      for (int igl=0; igl<BKgroupl.size(); igl++){
	int n=0;
	for (int id=0; id<Ndiags; id++) if (BKindexl(id)==igl) n+=1;
	log << setw(3) << igl << " " << setw(3) << BKgroupl[igl] << "   (#"<<setw(3)<<n<<") ";
	for (int id=0; id<Ndiags; id++){
	  if (BKindexl(id)==igl){
	    log<<" "<<id<<"=(";
	    for (int j=0; j<2*Norder; j++) log << diagsG(id,j)<<",";
	    log<<") ";
	  }
	}
	log<<endl;
      }
    }
  }
  
  template <typename real>
  void TestGroups(const bl::Array<bl::TinyVector<real,3>,2>& mom_G){
    bl::Array<double,1> k_nrm(BKgroups.extent(0)), k_cth(BKgroups.extent(0));
    bl::Array<double,1> kp_nrm(BKgroups.extent(0)), kp_cth(BKgroups.extent(0));
    for (int ig=0; ig<BKgroups.extent(0); ig++){
      int id_representative = BKgroups(ig);
      bl::TinyVector<double,3> k = mom_G(id_representative,0);
      int inv_1  = i_diagsG(id_representative, 1);
      bl::TinyVector<double,3> kp = mom_G(id_representative,inv_1);
      k_nrm(ig) = norm(k);
      kp_nrm(ig) = norm(kp);
      k_cth(ig) = k(2)/k_nrm(ig);
      kp_cth(ig) = kp(2)/kp_nrm(ig);
    }
    int Ndiags = mom_G.extent(0);
    for (int id=0; id<Ndiags; id++){
      int ig = BKindex(id);
      if (ig>=N_BKA1){ // only for those they should obey the property
	bl::TinyVector<double,3> k = mom_G(id,0);
	int inv_1  = i_diagsG(id, 1);
	bl::TinyVector<double,3> kp = mom_G(id,inv_1);
	double nk = norm(k);
	double cos_theta = k(2)/nk;
	double nkp = norm(kp);
	double cos_thetap = kp(2)/nkp;
	if (fabs(k_nrm(ig)-nk)>1e-6){std::cerr<<"ERROR : 1) It seems BKgroups or BKindex was not properly computed!"<<std::endl; exit(1);}
	if (fabs(k_cth(ig)-cos_theta)>1e-6){ std::cerr<<"ERROR : 2) It seems BKgroups or BKindex was not properly computed! "<<std::endl; exit(1);}
	if (fabs(kp_nrm(ig)-nkp)>1e-6){
	  std::cerr<<"ERROR : 3) It seems BKgroups or BKindex was not properly computed!"<<std::endl;
	  std::cerr<<"id="<<id<<" BKindex="<<ig<<" nkp="<<nkp <<" kp_nrm="<<kp_nrm(ig) << endl;
	  exit(1);
	}
	if (fabs(kp_cth(ig)-cos_thetap)>1e-6){ std::cerr<<"ERROR : 4) It seems BKgroups or BKindex was not properly computed! "<<std::endl; exit(1);}
      }
    }
    // Now checking only the right-hand side
    for (int ig=0; ig<BKgroupr.size(); ig++){
      int id_representative = BKgroupr[ig];
      bl::TinyVector<double,3> k = mom_G(id_representative,0);
      k_nrm(ig) = norm(k);
      k_cth(ig) = k(2)/k_nrm(ig);
    }
    for (int id=0; id<Ndiags; id++){
      int ig = BK_right(BKindex(id));
      if (ig>=N_BKA0){ // the first group are those with ladders on the right-hand side, and will be removed in BKA>0
	bl::TinyVector<double,3> k = mom_G(id,0);
	double nk = norm(k);
	double cos_theta = k(2)/nk;
	if (fabs(k_nrm(ig)-nk)>1e-6){std::cerr<<"ERROR : 1) It seems BKgroupr or BK_right was not properly computed!"<<std::endl; exit(1);}
	if (fabs(k_cth(ig)-cos_theta)>1e-6){ std::cerr<<"ERROR : 2) It seems BKgroupr or BK_right was not properly computed! "<<std::endl; exit(1);}
      }
    }
    // Now checking only the left-hand side
    for (int ig=0; ig<BKgroupl.size(); ig++){
      int id_representative = BKgroupl[ig];
      int inv_1  = i_diagsG(id_representative, 1);
      bl::TinyVector<double,3> kp = mom_G(id_representative,inv_1);
      kp_nrm(ig) = norm(kp);
      kp_cth(ig) = kp(2)/kp_nrm(ig);
    }
    for (int id=0; id<Ndiags; id++){
      int ig = BK_left(BKindex(id));
      if (ig>=N_BKA1){
	int inv_1  = i_diagsG(id, 1);
	bl::TinyVector<double,3> kp = mom_G(id,inv_1);
	double nkp = norm(kp);
	double cos_thetap = kp(2)/nkp;
	if (fabs(kp_nrm(ig)-nkp)>1e-6){std::cerr<<"ERROR : 3) It seems BKgroupl or BK_left was not properly computed!"<<std::endl; exit(1);}
	if (fabs(kp_cth(ig)-cos_thetap)>1e-6){ std::cerr<<"ERROR : 4) It seems BKgroupl or BK_left was not properly computed! "<<std::endl; exit(1);}
      }
    }
  }

  template <typename real>
  void Meassure(int itt, int tmeassure, double PQ, double sp, int iiQ, bl::Array<double,1>& pl_t,
		const bl::TinyVector<double,3> Q, double aQ, const bl::Array<bl::TinyVector<real,3>,1>& mom_g,
		const bl::Array<unsigned short,2>& Gindx)
  {
    if (itt%(2000*tmeassure) == 0){ // For efficiency, do not check every time.
      if ( fabs(sum(PQg)/PQ-1) > 1e-4 ){
	std::cerr<<"ERROR  PQg and PQ are not equal : PQ="<< PQ << " and PQg="<< sum(PQg) << std::endl;
      }
    }
    if (N_BKA0>0) {// for BKA==0
      double wgh = sp * PQg(0)/PQ;
      C_Pln0(iiQ,bl::Range::all()) += pl_t(bl::Range::all()) * wgh;
    }
    for (int igr=0; igr<BKgroupr.size(); igr++){
      int id_representative = BKgroupr[igr];
      bl::TinyVector<double,3> k  = mom_g(Gindx(id_representative,0));
      double ak = norm(k);
      if (!Q0w0){
	double cos_theta=0;
	if (ak>0) cos_theta = k[2]/ak;
	int ith = static_cast<int>(Nthbin * 0.5*(cos_theta+1));
	if (ith>=Nthbin) ith=Nthbin-1;
	ith_right(igr) = ith;
      }
      {//     determines where is |k|=ak in the discrete kxb mesh
	int Nkbin = kxb.extent(0)-1;
	int klo = 0, khi=Nkbin-1;
	ik_right(igr) = bisection(ak, klo, khi, Nkbin, kxb);
      }
    }
    for (int igl=0; igl<BKgroupl.size(); igl++){
      int id_representative = BKgroupl[igl];
      int inv_1  = i_diagsG(id_representative, 1);
      bl::TinyVector<double,3> kp = mom_g(Gindx(id_representative,inv_1));
      double akp = norm(kp);
      if (!Q0w0){
	double cos_thetap=0;
	if (akp>0) cos_thetap = kp[2]/akp;
	int ithp = static_cast<int>(Nthbin * 0.5*(cos_thetap+1));
	if (ithp>=Nthbin) ithp=Nthbin-1;
	ith_left(igl) = ithp;
      }
      {//     determines where is |k|=ak in the discrete kxb mesh
	int Nkbin = kxb.extent(0)-1;
	int klo = 0, khi=Nkbin-1;
	ik_left(igl) = bisection(akp, klo, khi, Nkbin, kxb);
      }
    }
    
    for (int ig=N_BKA0; ig<N_BKA1; ig++){ // for BKA==1
      double group_weight = PQg(ig)/PQ;
      /*
      int id_representative = BKgroups(ig);
      bl::TinyVector<double,3> k  = mom_g(Gindx(id_representative,0));
      double ak = norm(k);
      double cos_theta=0;
      if (ak>0) cos_theta = k[2]/ak;
      int ith = static_cast<int>(Nthbin * 0.5*(cos_theta+1));
      if (ith>=Nthbin) ith=Nthbin-1;
      int ik;
      {//     determines where is |k|=ak in the discrete kxb mesh
	int Nkbin = kxb.extent(0)-1;
	int klo = 0, khi=Nkbin-1;
	ik = bisection(ak, klo, khi, Nkbin, kxb);
      }
      int igr = BK_right(ig);
      int ikn = ik_right(igr);
      int ithn = ith_right(igr);
      if (ik != ikn){
	cerr << "ERROR : ikn="<<ikn <<" and ik="<< ik << endl;
      }
      if (ithn != ith){
	cerr << "ERROR : ithn="<<ithn<<" and ith="<< ith << endl;
      }
      */
      int igr = BK_right(ig);
      int ik = ik_right(igr);
      int ith = ith_right(igr);
      double wgh = sp * group_weight;
      C_Pln1(ith,ik,iiQ,bl::Range::all()) += pl_t(bl::Range::all()) * wgh;
    }
    for (int ig=N_BKA1; ig<BKgroups.extent(0); ig++){ // for BKA==2
      double group_weight = PQg(ig)/PQ;
      /*
      int id_representative = BKgroups(ig);
      bl::TinyVector<double,3> k  = mom_g(Gindx(id_representative,0));
      int inv_1  = i_diagsG(id_representative, 1);
      bl::TinyVector<double,3> kp = mom_g(Gindx(id_representative,inv_1));
      double ak = norm(k);
      double akp = norm(kp);
      double cos_theta=0;
      if (ak>0) cos_theta = k[2]/ak;
      double cos_thetap=0;
      if (akp>0) cos_thetap = kp[2]/akp;
      int ith = static_cast<int>(Nthbin * 0.5*(cos_theta+1));
      if (ith>=Nthbin) ith=Nthbin-1;
      int ithp = static_cast<int>(Nthbin * 0.5*(cos_thetap+1));
      if (ithp>=Nthbin) ithp=Nthbin-1;
      int ik, ikp;
      {//     determines where is |k|=ak in the discrete kxb mesh
       //also determines where is |k'|=akp in the same kxb mesh
	int Nkbin = kxb.extent(0)-1;
	int klo = 0, khi=Nkbin-1;
	ik = bisection(ak, klo, khi, Nkbin, kxb);
	klo = 0; khi=Nkbin-1;
	ikp = bisection(akp, klo, khi, Nkbin, kxb);
      }
      int igr = BK_right(ig);
      int igl = BK_left(ig);
      int ikn   = ik_right (igr);
      int ithn  = ith_right(igr);
      int ikpn  = ik_left  (igl);
      int ithpn = ith_left (igl);
      if (ikn != ik){
	cerr << "ERROR ikn="<< ikn << " and ik=" << ik << endl;
      }
      if (ikpn != ikp){
	cerr << "ERROR ikpn=" << ikpn << " and ikp="<< ik << endl;
      }
      */
      int igr = BK_right(ig);
      int igl = BK_left(ig);
      int ik   = ik_right (igr);
      int ith  = ith_right(igr);
      int ikp  = ik_left  (igl);
      int ithp = ith_left (igl);
      double wgh = sp * group_weight;
      C_Pln2(ith,ik,ithp,ikp,iiQ,bl::Range::all()) += pl_t(bl::Range::all()) * wgh;
    }
  }
  void Normalize(double beta){
    C_Pln2(bl::Range::all(),bl::Range::all(),bl::Range::all(),bl::Range::all(),bl::Range::all(),bl::Range::all()) *=  1./beta;
    C_Pln1(bl::Range::all(),bl::Range::all(),bl::Range::all(),bl::Range::all()) *=  1./beta;
    C_Pln0(bl::Range::all(),bl::Range::all()) *=  1./beta;
    /*
    int Nkbin = kxb.extent(0)-1;
    double dtheta = 2.0/Nthbin;
    // Since ik and itheta are binned, we need to divide by 1/dtheta * 1/dk
    for (int ik=0; ik<Nkbin; ik++)
      for (int ikp=0; ikp<Nkbin; ikp++)
	C_Pln2(bl::Range::all(),ik,bl::Range::all(),ikp,bl::Range::all(),bl::Range::all()) *= 1./(dtheta*dtheta*(kxb(ik+1)-kxb(ik))*(kxb(ikp+1)-kxb(ikp)));
    for (int ik=0; ik<Nkbin; ik++)
      C_Pln1(bl::Range::all(),ik,bl::Range::all(),bl::Range::all()) *= 1./(dtheta*(kxb(ik+1)-kxb(ik)));
    */
  }
  void BeforeReduce(){};
  void AfterReduce(int){};
};


void dot_product(bl::Array<std::complex<double>,1>& res, const bl::Array<std::complex<double>,2>& Ax, const bl::Array<double,1>& by)
{
  for (int i=0; i<Ax.extent(0); i++){
    std::complex<double> r=0;
    for (int j=0; j<Ax.extent(1); j++){
      r += Ax(i,j)* by(j);
    }
    res(i) = r;
  }
}

class BKdataAttachVertex{
public:
  bl::Array<double,1>& kxb;
  bl::Array<int,1> BKgroups, BK_right, BK_left;
  std::deque<int> BKgroupr, BKgroupl;
  //
  bl::Array<unsigned short,1> BKindex;
  bl::Array<double,1> PQg, PQg_new;
  bl::Array<unsigned short,2> i_diagsG;
  //
  bl::Array<int,1> ith_left; // ith_left(BKgroupl.size());
  bl::Array<int,1> ith_right;// ith_right(BKgroupr.size());
  bl::Array<int,1> ik_right; // ik_right( BKgroupr.size());
  bl::Array<int,1> ik_left;  // ik_left ( BKgroupl.size());
  //
  bool Q0w0;
  //
  bl::Array<double,2>& C_Pln2; // C_Pln2(qx.extent(0), Nw+1);
  bl::Array<double,2>& C_Pln1; // C_Pln1(qx.extent(0), Nw+1);
  bl::Array<double,2>& C_Pln0; // C_Pln0(qx.extent(0), Nw+1);
  bl::Array<std::complex<double>,4>& Vertex;
  bl::Array<std::complex<double>,2>& Ker;
  bl::Array<char,2>& Vtype;
  bl::Array<std::complex<double>,1> pl_w;
  int Nthbin, Nlt, Nq, Nw, Nk;
  int N_BKA0, N_BKA1;
  BKdataAttachVertex(bl::Array<double,2>& _C_Pln2_, bl::Array<double,2>& _C_Pln1_, bl::Array<double,2>& _C_Pln0_,
		     bl::Array<double,1>& _kxb_, bl::Array<char,2>& _Vtype_,
		     bl::Array<std::complex<double>,4>& _Vertex_,
		     bl::Array<std::complex<double>,2>& _Ker_iOm_lt_) :
    C_Pln2(_C_Pln2_), C_Pln1(_C_Pln1_), C_Pln0(_C_Pln0_), kxb(_kxb_), Vtype(_Vtype_), Vertex(_Vertex_), Ker(_Ker_iOm_lt_), Q0w0(false)
  {
    Nq = C_Pln2.extent(0);
    //Nw = C_Pln2.extent(1);
    Nw = Vertex.extent(3);
    // Vertex = zeros( ( Nthbin, len(kxb)-1, len(qx), Nw ), dtype=complex)
    Nthbin = Vertex.extent(0);
    Nk = Vertex.extent(1);
    if (Nq != Vertex.extent(2))
      cerr << "ERROR : Vertex.extent(2) is wrong "<< Vertex.extent(2) <<" instead of Nq="<< Nq << endl;
    if (Nw+1 != C_Pln2.extent(1))
      cerr << "ERROR : Vertex.extent(3) is wrong " << Vertex.extent(2) <<" instead of Nw="<< Nw << endl;
    // Ker = zeros( (Nw,Nlt+1) )
    Nlt = Ker.extent(1)-1;
    if (Nw != Ker.extent(0))
      cerr << "ERROR : Ker.extent(0) is wrong " << Ker.extent(0) << " instead of Nw="<< Nw << endl;

    if (! (C_Pln2.extent(0)==C_Pln1.extent(0) && C_Pln2.extent(0)==C_Pln0.extent(0)) )
      cerr << "ERROR : C_Pln2 has wrong dimension, Nq points " << C_Pln2.extent(0)<<", "<< C_Pln1.extent(0) << ", " << C_Pln0.extent(0) << endl;
    if (! (C_Pln2.extent(1)==C_Pln1.extent(1) && C_Pln2.extent(1)==C_Pln0.extent(1)) )
      cerr << "ERROR : C_Pln2 has wrong dimension, Nw points " << C_Pln2.extent(1)<<", "<< C_Pln1.extent(1) << ", " << C_Pln0.extent(1) << endl;

    if (Nw==1) Nlt=0; // If we are interested in Om=0, we need only the first legendre Polynomial, hence Nlt can be set to zero.
    
    pl_w.resize(Nw);
    C_Pln2 = 0.0;
    C_Pln1 = 0.0;
    C_Pln0 = 0.0;
  }
															
  void ResetMeasure()                  {C_Pln2=0; C_Pln1=0; C_Pln0=0;}
  void PQg_Initialize()                { PQg = 0; }
  void PQg_Add(int id, double PQd)     { PQg(BKindex(id)) += PQd; }
  void PQg_new_Initialize()            { PQg_new = 0; }
  void PQg_new_Add(int id, double PQd) { PQg_new(BKindex(id)) += PQd; }
  void Set_PQg_new_to_PQg()            { PQg = PQg_new; }
  
  void FindGroups(const bl::Array<unsigned short,2>& diagsG, const std::vector<std::vector<std::vector<int> > >& Loop_index, const std::vector<std::vector<std::vector<int> > >& Loop_type,
		  ostream& log, bool debug, double Q_external){
    //bool debug=true;
    //log<<"Inside BaymKadanoff_Combined.FindGroups" << endl;
    // For efficiency, we want to find which diagram have common values of momenta at the right hand side of the diagram.
    int Ndiags = diagsG.extent(0);
    int Nloops = Loop_index[0].size();
    int Norder = diagsG.extent(1)/2;
    // computing the inverse of all permutations, to provide fast lookup for the previous fermionic propagator
    i_diagsG.resize(Ndiags,2*Norder);
    for (int id=0; id<Ndiags; id++)
      for (int i=0; i<2*Norder; i++)
	i_diagsG(id, static_cast<int>(diagsG(id,i)) )=i;
    
    BKindex.resize(Ndiags);
    // Feynman graphs ends like in the figure below:
    //         k'+q       k+q
    //             -->----->-
    // vrtex=1  /    |  |     \    q
    //          \    |  |     /   vertex=0
    //            --<------<-
    //          k'         k G[0]==G_k
    //          G^i[1]=G_k'
    // We want to find what are possibilities for G[0]. Ideally there would be just one, but since we want to have
    // diagrams which cancel by sign, we have several possibilities, which are here enumerated.
    std::deque<int> bkgroups;
    std::deque< std::deque<int> > how_many;

    std::vector<std::deque<int> > k_momenta_right(Ndiags);
    std::vector<std::deque<int> > k_momenta_left(Ndiags);
    for (int id=0; id<Ndiags; id++){
      int post_0 = diagsG(id,0);
      int inv_0  = i_diagsG(id,0);
      int post_1 = diagsG(id,1);
      int inv_1  = i_diagsG(id, 1);// the propagators which ends at vertex=1, i.e., k' in the figure

      bool ladder_left = false;
      bool ladder_right = false;
      if (partner(inv_0)==post_0 && Vtype(id,post_0/2)==0 ) //  it has ladder as the first interaction line on the right
	ladder_right=true;                           // but should not be counter-term interaction
      if (partner(inv_1)==post_1 && Vtype(id,post_1/2)==0 ) // it has ladder as the first interaction on the left
	ladder_left=true;                            // but should not be counter-term interaction

      // Finding all momenta that appear on the left and on the right
      //std::deque<int> k_momenta_right;
      //std::deque<int> k_momenta_left;
      int iloop_start=0;
      if (Q0w0 && (fabs(Q_external)<1e-10)) iloop_start=1; // for Q==0 the external loop does not change anything
      for (int iloop=iloop_start; iloop<Nloops; ++iloop){
	const std::vector<int>& lindex = Loop_index[id][iloop];
	const std::vector<int>& ltype  = Loop_type[id][iloop];
	for (int i=0; i<lindex.size(); i++){
	  int ltype_i = ltype[i];
	  int lindex_i = lindex[i];
	  // this shows you which momenta contribute to the G[0] propagator, and what is the loops orientation.
	  // Note that external momentum (iloop==0) can have only a single orientation (i.e.,negative) and therefore 0 can serve the purpose
	  if ( abs(ltype_i)==1  && lindex_i==0 )
	    k_momenta_right[id].push_back( sign(ltype_i)*iloop );
	  if ( abs(ltype_i)==1 && lindex_i==inv_1 )
	    k_momenta_left[id].push_back( sign(ltype_i)*iloop );
	}
      }

      // now merging left and right momentum lists
      std::deque<int> k_momenta;
      
      if (ladder_right){ // only relevant in BKA==0 (but not in BKA==1 or BKA==2) because it has ladder on the right
	k_momenta.push_back( 1000 ); // just add some large number that exceeds any loop. All diagrams of this type make a single group
      }else if (ladder_left){ // relevant in BKA==1 but not BKA==2, because it does not have ladder on the right but only on the left
	k_momenta.push_back( 100 ); // this will shows that it is not BKA==2 type
	// all right momenta are relevant, and each 
	for (auto it=k_momenta_right[id].begin(); it!=k_momenta_right[id].end(); ++it)
	  k_momenta.push_back( *it );
      } else{ // no ladder on the left or right => hence relevant for BKA==2
	for (auto it=k_momenta_right[id].begin(); it!=k_momenta_right[id].end(); ++it)
	  k_momenta.push_back( *it );
	for (auto it=k_momenta_left[id].begin(); it!=k_momenta_left[id].end(); ++it)
	  k_momenta.push_back( *it );
      }
      
      // checking if such combination of momenta has already appeared before?
      auto where = std::find( how_many.begin(), how_many.end(), k_momenta );
      int iwhere = how_many.size();
      if (where == how_many.end()){ // has not appeared before, hence is a new type diagram
	how_many.push_back(k_momenta); // remember this set of momenta
	bkgroups.push_back(id);        // first diagram of this type
      }else{
	iwhere = distance(how_many.begin(),where); // this is diagram with the same momenta, its position in how_many.
      }
      BKindex(id) = iwhere;
    }
    
    PQg.resize(bkgroups.size());
    PQg_new.resize(bkgroups.size());

    {// Now we sort BKindex, so that BKA==0 diagrams appear first (those with ladder at the right hand), followed by BKA==1 terms (ladder on the left side), followed by all other diagrams.
      vector<int> grp_index(bkgroups.size());
      for (int i=0; i<grp_index.size(); i++) grp_index[i]=i;  // setting up an index array
      Cmp_momenta cmp_momenta(how_many);// we will sort according to how_many[ig][0] in descending order
      std::stable_sort(grp_index.begin(), grp_index.end(), cmp_momenta); // actual sorting done
      vector<int> grp_index_1(bkgroups.size()); // will create an inverse index
      for (int i=0; i<grp_index.size(); i++) grp_index_1[grp_index[i]]=i;

      N_BKA0=0;
      if (how_many[grp_index[N_BKA0]][0]==1000) N_BKA0=1; // We have some diagrams with ladder on the right-hand side
      // BKA==0 needs diagrams in group 0
      N_BKA1=0; // BKA==1 needs diagrams from 1... N_BKA1
      // BKA==2 needs diagrams from N_BKA1 to bkgroups.size()
      for (; N_BKA1<bkgroups.size(); N_BKA1++)
	if (how_many[grp_index[N_BKA1]][0]<100) break;
      
      if (debug) log << "N_BKA0=" << N_BKA0 << " N_BKA1="<< N_BKA1 << endl; 
      bl::Array<unsigned short,1> BKindex_new(Ndiags);// with the help of an inverse index, we can set each diagram to the new sorted index.
      for (int id=0; id<Ndiags; id++) BKindex_new(id) = grp_index_1[BKindex(id)]; // setting new BKindex
      for (int id=0; id<Ndiags; id++) BKindex(id) = BKindex_new(id); // and now rewriting the old BKindex
    
      BKgroups.resize(bkgroups.size()); // correcting BKgroups
      for (int ig=0; ig<BKgroups.size(); ig++){
	for (int id=0; id<Ndiags; id++)
	  if (BKindex(id)==ig) {
	    BKgroups(ig) = id;
	    break;
	  }
      }
    }
    
    BK_right.resize(BKgroups.size());
    BK_left.resize(BKgroups.size());
    BK_right = -1; BK_left = -1;
    //  BKindexr and  BKindexl not essential, can be removed
    bl::Array<int,1> BKindexr(Ndiags), BKindexl(Ndiags);
    BKindexr = -1; BKindexl = -1;
    std::deque< std::deque<int> > right_side;
    std::deque< std::deque<int> > left_side;

    for (int id=0; id<Ndiags; id++){
      int ig = BKindex(id);
      if (ig>=N_BKA0){ // need momenta on the right
	// checking if such combination of momenta has already appeared before?
	auto where = std::find( right_side.begin(), right_side.end(), k_momenta_right[id] );
	int iwhere = right_side.size();
	if (where == right_side.end()){ // has not appeared before, hence is a new type diagram
	  right_side.push_back(k_momenta_right[id]); // remember this set of momenta
	  BKgroupr.push_back( id );
	}else{
	  iwhere = distance(right_side.begin(),where); // this is diagram with the same momenta, its position in how_many.
	}
	BK_right(ig) = iwhere;
	BKindexr(id) = iwhere;
      }
      if (ig>=N_BKA1){ // need momenta on the left as well
	// checking if such combination of momenta has already appeared before?
	auto where = std::find( left_side.begin(), left_side.end(), k_momenta_left[id] );
	int iwhere = left_side.size();
	if (where == left_side.end()){ // has not appeared before, hence is a new type diagram
	  left_side.push_back(k_momenta_left[id]); // remember this set of momenta
	  BKgroupl.push_back( id );
	}else{
	  iwhere = distance(left_side.begin(),where); // this is diagram with the same momenta, its position in how_many.
	}
	BK_left(ig) = iwhere;
	BKindexl(id) = iwhere;
      }
    }

    ith_left.resize(BKgroupl.size());
    ith_right.resize(BKgroupr.size());
    ith_left=0;
    ith_right=0;
    ik_right.resize(BKgroupr.size());
    ik_left.resize(BKgroupl.size());
    
    if (debug){
      log<<"BKgroups for attaching the vertex:"<<endl;
      log<<"  #"<<" "<<"rep"<< ","<<" r"<<","<<"l "<<"  (typ;#nm) : "<<endl;
      for (int ig=0; ig<BKgroups.size(); ig++){
	log<< setw(3)<<ig<<" " << setw(3)<< BKgroups(ig)<< ","<< setw(2) << BK_right(ig)<<","<< setw(2) << BK_left(ig)<<" ";
      	int n=0;
	for (int id=0; id<Ndiags; id++) if (BKindex(id)==ig) n+=1;
	int BKA_type = 2;
	if (ig==0) BKA_type=0;
	else if (ig<N_BKA1) BKA_type=1;
	//log << " ("<<setw(3)<<how_many[grp_index[ig]][0]<<";"<<n<<") : ";
	log << " ("<<setw(3)<< BKA_type <<";#"<<setw(2)<<n<<") : ";
	for (int id=0; id<Ndiags; id++){
	  if (BKindex(id)==ig){
	    log<<" "<<id<<"=(";
	    for (int j=0; j<2*Norder; j++) log << diagsG(id,j)<<",";
	    log<<") ";
	  }
	}
	log<<endl;
      }
      log<<"BKgroups for the right side attachement"<<endl;
      for (int igr=0; igr<BKgroupr.size(); igr++){
	int n=0;
	for (int id=0; id<Ndiags; id++) if (BKindexr(id)==igr) n+=1;
	log << setw(3) << igr <<" " << setw(3) << BKgroupr[igr] << "   (#"<<setw(3)<<n<<") ";
	for (int id=0; id<Ndiags; id++){
	  if (BKindexr(id)==igr){
	    log<<" "<<id<<"=(";
	    for (int j=0; j<2*Norder; j++) log << diagsG(id,j)<<",";
	    log<<") ";
	  }
	}
	log<<endl;
      }
      log<<"BKgroups for the left side attachement"<<endl;
      for (int igl=0; igl<BKgroupl.size(); igl++){
	int n=0;
	for (int id=0; id<Ndiags; id++) if (BKindexl(id)==igl) n+=1;
	log << setw(3) << igl << " " << setw(3) << BKgroupl[igl] << "   (#"<<setw(3)<<n<<") ";
	for (int id=0; id<Ndiags; id++){
	  if (BKindexl(id)==igl){
	    log<<" "<<id<<"=(";
	    for (int j=0; j<2*Norder; j++) log << diagsG(id,j)<<",";
	    log<<") ";
	  }
	}
	log<<endl;
      }
    }
  }
  
  template <typename real>
  void TestGroups(const bl::Array<bl::TinyVector<real,3>,2>& mom_G){
    bl::Array<double,1> k_nrm(BKgroups.extent(0)), k_cth(BKgroups.extent(0));
    bl::Array<double,1> kp_nrm(BKgroups.extent(0)), kp_cth(BKgroups.extent(0));
    for (int ig=0; ig<BKgroups.extent(0); ig++){
      int id_representative = BKgroups(ig);
      bl::TinyVector<double,3> k = mom_G(id_representative,0);
      int inv_1  = i_diagsG(id_representative, 1);
      bl::TinyVector<double,3> kp = mom_G(id_representative,inv_1);
      k_nrm(ig) = norm(k);
      kp_nrm(ig) = norm(kp);
      k_cth(ig) = k(2)/k_nrm(ig);
      kp_cth(ig) = kp(2)/kp_nrm(ig);
    }
    int Ndiags = mom_G.extent(0);
    for (int id=0; id<Ndiags; id++){
      int ig = BKindex(id);
      if (ig>=N_BKA1){ // only for those they should obey the property
	bl::TinyVector<double,3> k = mom_G(id,0);
	int inv_1  = i_diagsG(id, 1);
	bl::TinyVector<double,3> kp = mom_G(id,inv_1);
	double nk = norm(k);
	double cos_theta = k(2)/nk;
	double nkp = norm(kp);
	double cos_thetap = kp(2)/nkp;
	if (fabs(k_nrm(ig)-nk)>1e-6){std::cerr<<"ERROR : 1) It seems BKgroups or BKindex was not properly computed!"<<std::endl; exit(1);}
	if (fabs(k_cth(ig)-cos_theta)>1e-6){ std::cerr<<"ERROR : 2) It seems BKgroups or BKindex was not properly computed! "<<std::endl; exit(1);}
	if (fabs(kp_nrm(ig)-nkp)>1e-6){
	  std::cerr<<"ERROR : 3) It seems BKgroups or BKindex was not properly computed!"<<std::endl;
	  std::cerr<<"id="<<id<<" BKindex="<<ig<<" nkp="<<nkp <<" kp_nrm="<<kp_nrm(ig) << endl;
	  exit(1);
	}
	if (fabs(kp_cth(ig)-cos_thetap)>1e-6){ std::cerr<<"ERROR : 4) It seems BKgroups or BKindex was not properly computed! "<<std::endl; exit(1);}
      }
    }
    // Now checking only the right-hand side
    for (int ig=0; ig<BKgroupr.size(); ig++){
      int id_representative = BKgroupr[ig];
      bl::TinyVector<double,3> k = mom_G(id_representative,0);
      k_nrm(ig) = norm(k);
      k_cth(ig) = k(2)/k_nrm(ig);
    }
    for (int id=0; id<Ndiags; id++){
      int ig = BK_right(BKindex(id));
      if (ig>=N_BKA0){ // the first group are those with ladders on the right-hand side, and will be removed in BKA>0
	bl::TinyVector<double,3> k = mom_G(id,0);
	double nk = norm(k);
	double cos_theta = k(2)/nk;
	if (fabs(k_nrm(ig)-nk)>1e-6){std::cerr<<"ERROR : 1) It seems BKgroupr or BK_right was not properly computed!"<<std::endl; exit(1);}
	if (fabs(k_cth(ig)-cos_theta)>1e-6){ std::cerr<<"ERROR : 2) It seems BKgroupr or BK_right was not properly computed! "<<std::endl; exit(1);}
      }
    }
    // Now checking only the left-hand side
    for (int ig=0; ig<BKgroupl.size(); ig++){
      int id_representative = BKgroupl[ig];
      int inv_1  = i_diagsG(id_representative, 1);
      bl::TinyVector<double,3> kp = mom_G(id_representative,inv_1);
      kp_nrm(ig) = norm(kp);
      kp_cth(ig) = kp(2)/kp_nrm(ig);
    }
    for (int id=0; id<Ndiags; id++){
      int ig = BK_left(BKindex(id));
      if (ig>=N_BKA1){
	int inv_1  = i_diagsG(id, 1);
	bl::TinyVector<double,3> kp = mom_G(id,inv_1);
	double nkp = norm(kp);
	double cos_thetap = kp(2)/nkp;
	if (fabs(kp_nrm(ig)-nkp)>1e-6){std::cerr<<"ERROR : 3) It seems BKgroupl or BK_left was not properly computed!"<<std::endl; exit(1);}
	if (fabs(kp_cth(ig)-cos_thetap)>1e-6){ std::cerr<<"ERROR : 4) It seems BKgroupl or BK_left was not properly computed! "<<std::endl; exit(1);}
      }
    }
  }

  template <typename real>
  void Meassure(int itt, int tmeassure, double PQ, double sp, int iiQ, bl::Array<double,1>& pl_t,
		const bl::TinyVector<double,3> Q, double aQ, const bl::Array<bl::TinyVector<real,3>,1>& mom_g,
		const bl::Array<unsigned short,2>& Gindx)
  {
    if (itt%(2000*tmeassure) == 0){ // For efficiency, do not check every time.
      if ( fabs(sum(PQg)/PQ-1) > 1e-4 ){
	std::cerr<<"ERROR  PQg and PQ are not equal : PQ="<< PQ << " and PQg="<< sum(PQg) << std::endl;
      }
    }

    if (Nw==1)
      pl_w(0) = Ker(0,0).real()*pl_t(0);
    else
      dot_product(pl_w, Ker, pl_t); // pl_w = dot(Ker, pl_t), i.e., polarization in Matsubara frequency
    
    for (int iw=0; iw<Nw; iw++) C_Pln0(iiQ,iw) += pl_w(iw).real() * sp;

    for (int igr=0; igr<BKgroupr.size(); igr++){
      int id_representative = BKgroupr[igr];
      bl::TinyVector<double,3> k  = mom_g(Gindx(id_representative,0));
      double ak = norm(k);
      //if (!Q0w0)
      {
	double cos_theta=0;
	if (ak>0) cos_theta = k[2]/ak;
	int ith = static_cast<int>(Nthbin * 0.5*(cos_theta+1));
	if (ith>=Nthbin) ith=Nthbin-1;
	ith_right(igr) = ith;
      }
      {//     determines where is |k|=ak in the discrete kxb mesh
	int Nkbin = kxb.extent(0)-1;
	int klo = 0, khi=Nkbin-1;
	ik_right(igr) = bisection(ak, klo, khi, Nkbin, kxb);
      }
    }
    for (int igl=0; igl<BKgroupl.size(); igl++){
      int id_representative = BKgroupl[igl];
      int inv_1  = i_diagsG(id_representative, 1);
      bl::TinyVector<double,3> kp = mom_g(Gindx(id_representative,inv_1));
      double akp = norm(kp);
      //if (!Q0w0)
      {
	double cos_thetap=0;
	if (akp>0) cos_thetap = kp[2]/akp;
	int ithp = static_cast<int>(Nthbin * 0.5*(cos_thetap+1));
	if (ithp>=Nthbin) ithp=Nthbin-1;
	ith_left(igl) = ithp;
      }
      {//     determines where is |k|=ak in the discrete kxb mesh
	int Nkbin = kxb.extent(0)-1;
	int klo = 0, khi=Nkbin-1;
	ik_left(igl) = bisection(akp, klo, khi, Nkbin, kxb);
      }
    }
    for (int ig=N_BKA0; ig<N_BKA1; ig++){ // for BKA==1
      double group_weight = PQg(ig)/PQ;
      double wgh = sp * group_weight;
      int igr = BK_right(ig);
      int ik = ik_right(igr);
      int ith = ith_right(igr);
      for (int iw=0; iw<Nw; iw++)
	C_Pln1(iiQ,iw) += (Vertex(ith,ik,iiQ,iw) * pl_w(iw)).real() * wgh;
    }
    for (int ig=N_BKA1; ig<BKgroups.extent(0); ig++){ // for BKA==2
      double group_weight = PQg(ig)/PQ;
      double wgh = sp * group_weight;
      int igr = BK_right(ig);
      int ik   = ik_right (igr);
      int ith  = ith_right(igr);
      int igl = BK_left(ig);
      int ikp  = ik_left  (igl);
      int ithp = ith_left (igl);
      for (int iw=0; iw<Nw; iw++){
	complex<double> tmp = Vertex(ith,ik,iiQ,iw) * pl_w(iw) * wgh;
	C_Pln1(iiQ,iw) += tmp.real();
	C_Pln2(iiQ,iw) += (tmp * Vertex(ithp,ikp,iiQ,iw)).real();
      }
    }
  }
  void BeforeReduce(){
    // seeting up C^2, so that we can later compute error
    for (int iq=0; iq<C_Pln2.extent(0); iq++){ // setting up information to compute the error
      C_Pln2(iq,Nw)  = C_Pln2(iq,0)*C_Pln2(iq,0);
      C_Pln1(iq,Nw)  = C_Pln1(iq,0)*C_Pln1(iq,0);
      C_Pln0(iq,Nw)  = C_Pln0(iq,0)*C_Pln0(iq,0);
    }
  }
  void AfterReduce(int size){
    // now computing the error
    for (int iq=0; iq<C_Pln2.extent(0); iq++){ // setting up information to compute the error
      C_Pln2(iq,Nw)  = sqrt(fabs(C_Pln2(iq,Nw) - C_Pln2(iq,0)*C_Pln2(iq,0))/size);
      C_Pln1(iq,Nw)  = sqrt(fabs(C_Pln1(iq,Nw) - C_Pln1(iq,0)*C_Pln1(iq,0))/size);
      C_Pln0(iq,Nw)  = sqrt(fabs(C_Pln0(iq,Nw) - C_Pln0(iq,0)*C_Pln0(iq,0))/size);
    }
  }
  void Normalize(double beta){
    C_Pln2(bl::Range::all(),bl::Range::all()) *=  1./beta;
    C_Pln1(bl::Range::all(),bl::Range::all()) *=  1./beta;
    C_Pln0(bl::Range::all(),bl::Range::all()) *=  1./beta;
  }
};


#endif // BAYMK
