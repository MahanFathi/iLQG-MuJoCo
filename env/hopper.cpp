// Run iterative LQG, execute half the trajectory, run iLQG again, and again, ... store the final trajectory

#include <iostream>
#include <lindy.h>
#include "mujoco.h"
#include "ilqr.h"


int main(int argc, char** argv) {

    // activate and load model
    mj_activate("mjkey.txt");

    /* HOPPER */
    #define DOFNUM 6
    #define ACTNUM 3
    #define EXTRAV 2


    mjModel* m = 0;
    m = mj_loadXML("../model/hopper.xml", NULL, NULL, 0);
    int t_0 = atoi(argv[1]);

    // extract some handy parameters
    int nv = m->nv;
    int nu = m->nu;

    // make mjData for main thread and converging thread
    mjData* dmain = mj_makeData(m);
    mjData* d = mj_makeData(m);

    int T = 50; // lqr optimization horizon
    auto* deriv = (mjtNum*) mju_malloc(3*sizeof(mjtNum)*nv*nv);

    for( int i = 0; i < t_0; i++ )
        mj_step(m, dmain);

    // make an instance of ilqr
    ilqr ILQR(m, dmain, deriv, T, argv[1]);

    for( int i = 0; i < 500; i++ ) {
        printf("#################################\n");
        printf("\t\tTIME:%d\n", i);
        printf("#################################\n");
        ILQR.RunMPC();
    }



}
