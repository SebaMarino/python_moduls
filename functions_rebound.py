import numpy as np
import sys, os


def rand_x(alpha, Xmin, Xmax):
    # returns a random number distributed as x^-alpha
    u=np.random.uniform(0.0,1.0)

    if alpha==1.0:
        return Xmin * (Xmax/Xmin)**u

    else:
        q=1.0-alpha
        return ( Xmin**q - u*(Xmin**q-Xmax**q) )**(1.0/q)

def col_track(sim_pointer,col): #collision search routine, hands collision_resolve functon a pointer to all initialised simulations 
    #col is an array which contains array indices in sim.particles of two particles that are colliding
    #see https://github.com/hannorein/rebound/blob/70cdfa1144752a12cacb78e08794e0a97d4c2aef/src/collision.c for description of how collisions are found
    #THE FOLLOWING FUNCTION IS EXACTLY THE SAME AS THE 'MERGE' FUNCTION IN THE UNDERLYING C CODE, EXCEPT SIMPLY REMOVING PARTICLE RATHER THAN MERGING. see line 446 onward in linked doc
    global global_col_track #global array where details of colliding particles are put. Later written to file. 
    # Global variable only called in this function. Therefore do not need to worry about pitfalls associated with global variables.
    sim=sim_pointer[0] # only initialise single simulation, this is therefore the first in the pointer array
    ps=sim.particles
    #collision_resolve function run twice with p1 and p2 interchanged (see above docs line 449).
    #only run this function once for each pair of particles
    if (ps[col.p1].lastcollision == sim.t or ps[col.p2].lastcollision == sim.t):
        return 0 #function already run for this pair of particles return nothing
    
    #print "time of collision ", sim.t/(2.*np.pi)
    #print "array indices of the particles in the sim.particles array that are colliding ",col.p1, col.p2
    #print "hash values of colliding particles ", ps[col.p1].hash.value, ps[col.p2].hash.value
    
    swap = 0
    i = col.p1
    j = col.p2 # want particle j to be removed
    
    if (j<i):
        swap = 1
        i=col.p2
        j=col.p1
    
    pi = ps[i]
    pj = ps[j]
    pi.lastcollision = sim.t


    col_track_array_pos = len(global_col_track[~np.all(global_col_track==0, axis=1)]) #time of collision, hash values of colliding particles, positions and velocites to file
    global_col_track[col_track_array_pos] = [sim.t/(2.*np.pi),pi.hash.value,pj.hash.value,pi.x,pi.y,pi.z,pi.vx*(2.*np.pi),pi.vy*(2.*np.pi),pi.vz*(2.*np.pi),pj.x,pj.y,pj.z,pj.vx*(2.*np.pi),pj.vy*(2.*np.pi),pj.vz*(2.*np.pi)]
    
    if swap ==0:
        #print "array indice in sim.particles of particle being removed ",j
        #print "hash value of particle that will be ejected ",pj.hash.value
        #print ""
        return 2 #remove particle p2 from simulation
    if swap ==1:
        #print "array indice in sim.particles of particle being removed ",j
        #print "hash value of particle that will be ejected ",pj.hash.value
        #print ""
        return 1 #remove particle p1 from simulation (it has larger array index)
