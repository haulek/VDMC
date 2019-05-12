#C++ = g++-7
#CFLAGS = -fopenmp -Ofast -fPIC -fno-strict-aliasing -fno-common -dynamic -arch x86_64 -arch x86_64
#LIBS = -L/usr/local/lib -lgsl -lgslcblas -framework Accelerate
#PYLIBS = -L/Applications/Canopy.app/appdata/canopy-1.6.2.3262.macosx-x86_64/Canopy.app/Contents/lib/python2.7/config -lpython2.7 -ldl -framework CoreFoundation
#PYBIND = -I ../ -I/System/Library/Frameworks/Python.framework/Versions/2.7/Extras/lib/python/scipy/weave/blitz # -I/usr/lib/python2.7/site-packages/weave/blitz 
#PYTHON = -I/Applications/Canopy.app/appdata/canopy-1.6.2.3262.macosx-x86_64/Canopy.app/Contents/include/python2.7

C++ = mpicxx
CFLAGS = -Ofast -fPIC 
LIBS = -L/usr/local/lib -lgsl -lgslcblas -framework Accelerate
PYLIBS =  -L/Applications/Canopy.app/appdata/canopy-2.1.9.3717.macosx-x86_64/Canopy.app/Contents/lib/python2.7/config -lpython2.7 -ldl -framework CoreFoundation
PYBIND = -I ../pybind11 -I/usr/lib/python2.7/site-packages/weave/blitz -I/usr/include -I/usr/include/python2.7 
PYTHON = 

equivalent.so : equivalent.cc
	$(C++) $(CFLAGS) -shared -std=c++11 $(PYBIND) $(PYTHON) equivalent.cc -o equivalent.so $(PYLIBS) $(LIBS)

clean :
	rm -f equivalent.so
	rm -rf equivalent.so.dSYM
