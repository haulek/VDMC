// @Copyright 2018 Kristjan Haule 
std::vector<double> VNrm(double lmbda, double cutoffk, int n)
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
