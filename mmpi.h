// @Copyright 2018 Kristjan Haule and Kun Chen    
#ifndef MY_MPI
#define MY_MPI

class my_mpi{
public:
  //MPI_Comm communicator;
  int size;
  int rank;
  int master;
  my_mpi() : size(1), rank(0), master(0){};
};

#define _MPI
#ifdef _MPI
#include <mpi.h>
#endif

#endif // MY_MPI
