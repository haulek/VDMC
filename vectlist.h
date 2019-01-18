// @Copyright 2018 Kristjan Haule and Kun Chen    
#ifndef VECLIST
#define VECLIST

#include <iostream>
#include <ostream>
#include <deque>
#include <map>
//#include <blitz/array.h>
//namespace bl = blitz;
//using namespace std;

class vecList{
public:
  std::map<int,int> m;
  vecList() {};
  
  vecList(std::initializer_list<int> ms, std::initializer_list<int> sgns){
    if (ms.size()!=sgns.size()){
      std::cout<<"ERROR in vecList initialization. Sizes not correct!"<<std::endl;
      return;
    }
    auto lm = ms.begin();
    auto ls = sgns.begin();
    for (; lm!=ms.end(); ++lm,++ls){
      m[*lm] = *ls;
    }
  }

  void add(int which, int sign){
    std::map<int,int>::iterator iter = m.find(which);
    if (iter != m.end() ){
      m[which] += sign;
      if (m[which]==0) m.erase(iter);
    }else{
      m[which] = sign;
    }
  }
  vecList operator+(const vecList& a) const{
    vecList n(*this);
    for (auto i=a.m.begin(); i!=a.m.end(); ++i)
      n.add(i->first,i->second);
    return n;
  }
  vecList operator-(const vecList& a) const{
    vecList n(*this);
    for (auto i=a.m.begin(); i!=a.m.end(); ++i)
      n.add(i->first,-i->second);
    return n;
  }
  
};

std::ostream& operator<<(std::ostream& out, const vecList& v)
{
  for (auto i=v.m.begin(); i!=v.m.end(); ++i){
    if (i->second==1) out<<"+";
    else if (i->second==-1) out<<"-";
    else if (i->second>0) out<<"("<<i->second<<")+";
    else if (i->second<0) out<<"("<<i->second<<")-";
    else out<<"(0)";
    out << i->first << " ";
  }
  return out;
}
/*
int main()
{
  vecList v1, v2;
  v1.add(0,1);
  v1.add(2,1);
  v1.add(3,-1);
  v2.add(0,-1);
  v2.add(4,+1);
  v2.add(3,1);
  std::cout << v1 << std::endl;
  std::cout << v2 << std::endl;
  vecList v3 = v1 - v2;
  std::cout << v3 << std::endl;

  vecList v4({0,2,4,5},{1,-1,1,-1});
  std::cout << v4 << std::endl;
  
  return 0;
}
*/
#endif
