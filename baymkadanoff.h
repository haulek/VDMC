// @Copyright 2018 Kristjan Haule and Kun Chen    
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
      {
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



/*
class BaymKadanoff_Symmetric_Q0W0_Data{
public:
  const bool Q0w0 = true;
  double Q_external;
  bl::Array<double,4>& C_Pln;
  bl::Array<double,1>& kxb;
  int Nthbin, Nlt, Nlq;
  bl::Array<int,1> BKgroups;
  bl::Array<unsigned short,1> BKindex;
  bl::Array<double,1> PQg, PQg_new;
  bl::Array<unsigned short,2> i_diagsG;
public:
  BaymKadanoff_Symmetric_Q0W0_Data(bl::Array<double,4>& _C_Pln_, bl::Array<double,1>& _kxb_, double _Q_external_=0.0) : C_Pln(_C_Pln_), kxb(_kxb_), Nlt(0), Nlq(0), Q_external(_Q_external_){
    Nthbin = C_Pln.extent(0);
    C_Pln = 0.0;
  }
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
*/
/*
class BaymKadanoff_Q0W0_Data{
public:
  const bool Q0w0 = true;
  double Q_external;
  bl::Array<double,2>& C_Pln;
  bl::Array<double,1>& kxb;
  int Nthbin, Nlt, Nlq;
  bl::Array<int,1> BKgroups;
  bl::Array<unsigned short,1> BKindex;
  bl::Array<double,1> PQg, PQg_new;
public:
  BaymKadanoff_Q0W0_Data(bl::Array<double,2>& _C_Pln_, bl::Array<double,1>& _kxb_, double _Q_external_=0.0) : C_Pln(_C_Pln_), kxb(_kxb_), Nlt(0), Nlq(0), Q_external(_Q_external_){
    Nthbin = C_Pln.extent(0);
    C_Pln = 0.0;
  }
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
*/
/*
class StandardData{
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
  void PQg_Initialize()                { }
  void PQg_Add(int id, double PQd)     { }
  void PQg_new_Initialize()            { }
  void PQg_new_Add(int id, double PQd) { }
  void Set_PQg_new_to_PQg()            { }
  void FindGroups(const bl::Array<unsigned short,2>& diagsG, const std::vector<std::vector<std::vector<int> > >& Loop_index, const std::vector<std::vector<std::vector<int> > >& Loop_type, ostream& log){}
  template <typename real>
  void TestGroups(const bl::Array<bl::TinyVector<real,3>,2>& mom_G){}
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

class Standard_Q0W0_Data{
public:
  const bool Q0w0 = true;
  double Q_external;
  bl::Array<double,2>& C_Pln;
  int Nlt, Nlq;
public:
  Standard_Q0W0_Data(bl::Array<double,2>& _C_Pln_, double _Q_external_=0.0) : C_Pln(_C_Pln_), Nlt(0), Nlq(0), Q_external(_Q_external_)
  { }
  void PQg_Initialize()                { }
  void PQg_Add(int id, double PQd)     { }
  void PQg_new_Initialize()            { }
  void PQg_new_Add(int id, double PQd) { }
  void Set_PQg_new_to_PQg()            { }
  void FindGroups(const bl::Array<unsigned short,2>& diagsG, const std::vector<std::vector<std::vector<int> > >& Loop_index, const std::vector<std::vector<std::vector<int> > >& Loop_type, ostream& log){}
  template <typename real>
  void TestGroups(const bl::Array<bl::TinyVector<real,3>,2>& mom_G){}
  template <typename real>
  void Meassure(int itt, int tmeassure, double PQ, double sp, bl::Array<double,1>& , bl::Array<double,1>& ,
		const bl::TinyVector<double,3> , double , const bl::Array<bl::TinyVector<real,3>,1>& mom_g,
		const bl::Array<unsigned short,2>& Gindx)
  {
    C_Pln(0,0) += sp;
  }
  void Normalize(double beta, double cutoffq){C_Pln *=  1./(beta);};
};
*/

#endif // BAYMK
