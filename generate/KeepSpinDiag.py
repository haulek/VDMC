#!/usr/bin/env python
# @Copyright 2018 Kristjan Haule 
from scipy import *
import glob
import os, sys
import loops as lp

def SpinKeepDiagram(cloop):
    KeepDiagram=0
    for c in cloop:
        if 0 in c and 1 in c:
            KeepDiagram = 1
            break
    return KeepDiagram


def KeepSpinDiags(fname, fnew):
    fi = open(fname, 'r')
    fo = open(fnew, 'w')
    s = fi.next()
    print >> fo, s,

    debug = True
    
    i=0
    n_all_diags=0
    while (s):
        try:
            s = fi.next()
            ij, s2 = s.split(' ', 1)
            diag, vtyp = s2.split(';',1)
            
            Gc = eval(diag)
            cVtyp = eval(vtyp)
            
            nord = len(Gc)/2
            first_line = ij+' '+diag+'; '    # first line only contains diagram and Vtype. The latter might change

            if i==0:  # at the first time only
                with_bit=[]
                for j in range(nord):
                    ii = (1<<j) # j-th bit is set
                    with_bit.append( [r for r in range(1<<nord) if (r & ii)] )
                
            
            KeepDiagram = True
            
            lines = [fi.next(),fi.next(),fi.next(),fi.next()]  # these lines contain "loop vertex", "loop type", "loop index", "vertex type"
            last_line = fi.next()   # this line contains Hugenholtz diagram signs, which need to be changed

            _ii_, _n_, vind = last_line.split(' ',2)
            sgn = eval(vind.partition('#')[0])

            if debug: print 'examine:', Gc, cVtyp, sgn
            
            # BUG-Jan 2019 : This generates the order, which is incompatible with C++ sampling. We need the order corresponding to the proper binary tree   
            #Exchanges = [ j for j,v in enumerate(cVtyp) if v>=10]
            Exchanges = [ j for j,v in enumerate(cVtyp) if v>=10][::-1]
            if debug: print 'Exchanges=', Exchanges
            cloop = lp.to_cycles(Gc) # how many fermionic loops
            
            KeepDiagram = SpinKeepDiagram(cloop) # Lets check if 0 and 1 appear in the same loop -> we need to keep the diagram
            
            Gp_all=[Gc]
            loops = [len(cloop)]
            ss = [(-2)**len(cloop)*(-1)**nord]                # this is the sign for ordinary diagrams, which can be compared to the quantity being read from the file
            sc = [(-2)**len(cloop)*(-1)**nord*KeepDiagram]    # this is the sign for the spin diagram, which can vanish

            if not KeepDiagram:
                which_vanish=[0]
            else:
                which_vanish=[]
            nex = len(Exchanges)
            n_diags = [KeepDiagram]
            #_how_many_appear_=[i_which]  # this is used only if Hugenholtz_All
            for l in range(1,2**len(Exchanges)):
                Gp = copy(Gc)                              # the diagram for which we try to find all Hugenholtz equivalents
                
                Vexchanged=[ Exchanges[j] for j in range(len(Exchanges)) if l & (1<<j) ] # which interactions need to be exchanged for this case, which is one out of 2^(n-1)
                if debug:
                    print '   x: Vexchanged=', Vexchanged
                    print '   x:', i, ': ',
                for ii in Vexchanged:
                    j1, j2 = 2*ii, 2*ii+1  # these are the vertices that need to be exchanged
                    if debug: print 'Exchanging (', j1,',', j2, ') -> (', j2,',',j1,')  ',# 'with Vtyp=', cVtyp,
                    i1, i2 = Gc.index(j1), Gc.index(j2)        # vertices to exchange
                    Gp[i1], Gp[i2] = Gp[i2], Gp[i1]            # with Hugenholtz exchange becomes Gp
                Gp_all.append(Gp)
                cloop = lp.to_cycles(Gp)
                
                KeepDiagram = SpinKeepDiagram(cloop)
                
                loops.append( len(cloop) )
                if debug: print '\n   x:  and got G=', Gp, ' with loops=', len(cloop), 'and keep=', KeepDiagram 

                ss.append( (-2)**len(cloop)*(-1)**nord )
                sc.append( (-2)**len(cloop)*(-1)**nord*KeepDiagram )
                n_diags.append( KeepDiagram )
                if not KeepDiagram: which_vanish.append(l)

            print 'ss=', ss, 'sc=', sc
            ss = array(ss)
            sc = array(sc)
            how_many_appear = ss/array(sgn)
            print 'how_many_appear=', how_many_appear

            #hma = set(how_many_appear)
            #if len(hma)!=1:
            #    print 'ERROR: how_many_appear should not have multiple values how_many_appear=', how_many_appear
            #    sys.exit(1)
            #how_many_appear = hma.pop()
            #if how_many_appear != 1:
            #    if debug: print 'WARNING : We do have degeneracy here', how_many_appear, 'which is OK when Hugenholtz_All is used or n>=6'
            #if debug: print 'ss=', ss/how_many_appear, 'sc=', sc/how_many_appear

            s_sgn = tuple(sc/how_many_appear)
            n_all_diags += sum( array(n_diags)/how_many_appear )

            # Here trying to see if we can skip the entire diagram, or, reduce its Hugenholtz
            if len(which_vanish)>0: 
                if nex==0:
                    if not KeepDiagram:
                        print 'WARNING : YES All vanish but have only one!!!'
                else:
                    #print 'HERE BE CAREFUL'
                    nm  = 1<<(nex-1)
                    # print 'nex=', nex, 'nm=', nm, 'with_bit[0]=', with_bit[0][:nm]
                    check = sum(abs(array(s_sgn)))
                    must_remove=[]
                    for j in range(nex):
                        can_vanish = with_bit[j][:nm]
                        if debug: print j+1, 'all that should vanish=', can_vanish, 'which vanish=', which_vanish
                        if len(set(can_vanish+which_vanish))==len(which_vanish):
                            nVtyp = list(cVtyp[:])
                            nVtyp[Exchanges[j]]-=10
                            must_remove += can_vanish
                            if debug : print 'WARNING : YES All vanish! nVtyp=', nVtyp, ' cVtyp=', cVtyp, ' s_sgn=', s_sgn
                            cVtyp = tuple(nVtyp)

                    if must_remove:
                        n_sgn = list(s_sgn[:])
                        for d in sorted(set(must_remove),reverse=True):
                            del n_sgn[d]
                        s_sgn = tuple(n_sgn)
                        if debug : print  '  ....becomes ....... s_sgn=', s_sgn
                        check2 = sum(abs(array(s_sgn))) 
                        if check2 != check:
                            print 'ERROR : Before removing diagrams the sum of signs was ', check, 'while after is', check2, 'but the sign should be preserved'
                            sys.exit(1)
                            
            first_line += str(cVtyp) + '\n'
            last_line = str(_ii_)+' '+str(_n_)+' '+str(s_sgn)+ '  # Hugenholtz diagram signs for spin\n'
            lines = [first_line] + lines + [last_line]
            for line in lines:
                print >> fo, line,
            i+=1
        except StopIteration:
            break

    print '# Number of spin-diagrams=', n_all_diags, 'at', fnew
        
if __name__ == '__main__':
    dr = 'sinput'
    if len(sys.argv)<2:
        print 'Give input directory name, i.e.,  sinput'
        sys.exit(1)
    else:
        dr = sys.argv[1]
    
    
    fls = glob.glob(dr+'/loops_Pdiags.*_cworder_*')
    dr2 = dr + '_spin'
    
    if not os.path.exists(dr2): os.makedirs(dr2)
    
    for fl in fls:
        fl2 = dr2 + '/' + os.path.split(fl)[1]
        print 'changing', fl, ' -> ', fl2
        KeepSpinDiags(fl,fl2)
