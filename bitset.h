// @Copyright 2018 Kristjan Haule and Kun Chen    
#include <iostream>
#include <sstream>
#include <deque>
//#include <set>

class BitSet{
public:
  long f;
  BitSet(long fi=0) : f(fi)
  { }
  operator long() {return f;}
  void Add(int sign, int inp){
    if (inp>14 || inp<-14) {std::cerr<<"Can not store a value larger than 14. i="<<inp<<std::endl; exit(1);}
    int shf=0;
    long fi = f;
    for (int i=0; i<12; i++){
      int i0    = (fi & 15);
      if (i0==0) break;
      //int sign0 = (fi & 16);
      fi = fi>>5;
      shf += 5;
      //cout<<"i="<<i<<" i0="<<i0<<" sgn="<<sign0<<endl;
    }
    //cout<<"shf="<<shf<<endl;
    long v = (inp+1);
    if (sign>0) v = v^16;
    //cout<<"v="<<v<<endl;
    v = v<<shf;
    f = f^v;
    //cout<<"f is now "<<f<<endl;
  }
  std::string repr() const
  {
    std::stringstream ss;
    long fi = f;
    for (int i=0; i<12; i++){
      int i0    = (fi & 15);
      int sign0 = (fi & 16);
      if (i0==0) break;
      fi = fi>>5;
      if (sign0) ss<<"+"<< (i0-1) <<" ";
      else ss<<"-"<< (i0-1) <<" ";
    }
    return ss.str();
  }
  std::deque<std::pair<int,int> > Value()
  {
    std::deque<std::pair<int,int> > dat;
    long fi = f;
    for (int i=0; i<12; i++){
      int i0    = (fi & 15);
      int sign0 = (fi & 16);
      if (i0==0) break;
      fi = fi>>5;
      if (sign0) dat.push_back( std::make_pair(1,i0-1) );
      else dat.push_back( std::make_pair(-1,i0-1) );
    }
    return dat;
  }
  int how_many_values()
  {
    long fi = f;
    int i=0;
    for ( ; i<12; i++){
      int i0    = (fi & 15);
      //int sign0 = (fi & 16);
      if (i0==0) break;
      fi = fi>>5;
    }
    return i;
  }
};
