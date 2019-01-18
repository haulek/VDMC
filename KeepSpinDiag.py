from scipy import *
import glob
import os
# @Copyright 2018 Kristjan Haule and Kun Chen

def to_cycles(perm):
    #pi = {i+1: perm[i] for i in range(len(perm))}
    pi = {i: perm[i] for i in range(len(perm))}
    cycles = []

    while pi:
        elem0 = next(iter(pi)) # arbitrary starting element
        this_elem = pi[elem0]
        next_item = pi[this_elem]

        cycle = []
        while True:
            cycle.append(this_elem)
            del pi[this_elem]
            this_elem = next_item
            if next_item in pi:
                next_item = pi[next_item]
            else:
                break

        cycles.append(cycle)

    return cycles

def KeepSpinDiags(fname, fnew):
    fi = open(fname, 'r')
    fo = open(fnew, 'w')
    s = fi.next()
    print >> fo, s,
    i=0
    while (s):
        try:
            s = fi.next()
            ii, s2 = s.split(' ', 1)
            if ';' in s2:
                diag, vtyp = s2.split(';',1)
                dG = eval(diag)
            else:
                dG = eval(s2)

            cycles = to_cycles( dG )   # fermionic loops
            KeepDiagram = False
            for i,c in enumerate(cycles):
                if 0 in c and 1 in c:
                    KeepDiagram = True
                    break
            #print i, dG, cycles, KeepDiagram
            lines = [s,fi.next(),fi.next(),fi.next(),fi.next(),fi.next()]
            if KeepDiagram:
                for line in lines:
                    print >> fo, line,
            i+=1
        except StopIteration:
            break

if __name__ == '__main__':
    dr = 'sinput'
    fls = glob.glob(dr+'/loops_Pdiags.*')
    dr2 = dr + '_spin'

    if not os.path.exists(dr2): os.makedirs(dr2)
    
    for fl in fls:
        fl2 = dr2 + '/' + os.path.split(fl)[1]
        print 'changing', fl, ' -> ', fl2
        KeepSpinDiags(fl,fl2)
