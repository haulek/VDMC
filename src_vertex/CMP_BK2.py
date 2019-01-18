#!/usr/bin/env python

# @Copyright 2018 Kristjan Haule and Kun Chen    
import re
import subprocess
import sys

if __name__ == '__main__':
    
    e1 = '~/projects/electronGas/VDMC12/rupc02/src_vertex/tpvertex.py'
    e2 = '~/projects/electronGas/VDMC12/rupc02/src_vertex/S2BKScanAttach.py'
    
    execfile('params.py')
    for k in p.keys():
        exec( k+' = '+str(p[k]) )
    kF = (9*pi/4.)**(1./3.) /rs
    n0 = 3/(4*pi*rs**3)

    for lmbda in p['_lmbdas_']:
    #lmbda = p['_lmbdas_'][0]
    #for ii in range(1):
        fo = open('params2.py', 'w')
        fi = open('params.py', 'r')
        for line in fi:
            m1 = re.search("\'lmbda\'", line)
            m2 = re.search("\'dmu\'", line)
            if m1 is not None:
                newline = "     'lmbda'    : "+str(lmbda)+", # inverse screening length"
                print >> fo, newline
                print 'Replacing line', line.strip(), 'with', newline.strip()
            elif m2 is not None:
                #print line
                fii = open('Density_order_5_lmbda_'+str(lmbda)+'.dat', 'r')
                fii.next()
                line2 = fii.next()
                m2 = re.search('dmu=\s*([-]?\d\.\d+)', line2)
                if m2 is not None:
                    #print line2
                    dmu2 = float(m2.group(1))
                    #print dmu2
                newline = "     'dmu'      : "+str(dmu2)+", # chemical potentila shift from mu=-kF^2"
                print >> fo, newline
                print 'Replacing line', line.strip(), 'with', newline.strip()
            else:
                print >> fo, line,
        print >> fo, "p['Short'] = True"
        fo.close()

        cmd = 'mpirun '+e1+' > nohup.vertex'
        subprocess.call(cmd,shell=True,stdout=sys.stdout,stderr=sys.stderr)
        
        cmd = e2 + ' Pcof_lmbda_'+str(lmbda)+'_3_corder_0.npy > lmbda_'+str(lmbda)+'.dat'
        subprocess.call(cmd,shell=True,stdout=sys.stdout,stderr=sys.stderr)
