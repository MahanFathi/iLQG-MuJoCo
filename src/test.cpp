#include <iostream>
#include <cstring>
#include <lindy.h>
#include "mujoco.h"

int main(int argc, char** argv) {

    // activate and load model
    mj_activate("mjkey.txt");


    mjModel* m = 0;
    if( strlen(argv[1])>4 && !strcmp(argv[1]+strlen(argv[1])-4, ".mjb") )
        m = mj_loadModel(argv[1], NULL);
    else
        m = mj_loadXML(argv[1], NULL, NULL, 0);
    if( !m )
    {
        printf("Could not load modelfile '%s'\n", argv[1]);
        return 1;
    }

    // make mjData
    mjData* d = mj_makeData(m);

    // allocate derivatives
    auto* deriv = (mjtNum*) mju_malloc(3*sizeof(mjtNum)*m->nv*m->nv);

    stateMat_t Fx; Fx.setZero();
    state_action_Mat_t Fu; Fu.setZero();

    stateVec_t pred;
    stateVec_t x1;
    stateVec_t x2;
    actionVec_t u1;


    for( int i = 0 ; i < 1000 ; i++ )
    {

        // advance simulation for a few time steps
        for( int t = 0; t < 10; t++ )
            mj_step(m, d);

        // perturb qpos, qvel, ctrl randomly (and manually!!)
        d->qpos[1] += 5e-1;
        d->ctrl[0] = 0.0;
//        d->ctrl[1] += -5e-1;
//        d->ctrl[2] += 5e-1;
        d->qvel[0] += 5e-1;
        d->qvel[1] += 5e-1;

        // get F
        get_derivs(deriv, m, d);
        Fx.setZero();
        calc_F(Fx.data(), Fu.data(), deriv, m, d);
        Fx += stateMat_t::Identity();

        mju_copy(x1.data(), d->qpos, m->nv);
        mju_copy(x1.data()+m->nv, d->qvel, m->nv);
        mju_copy(u1.data(), d->ctrl, m->nu);

        pred = Fx * x1 + Fu * u1;

        mj_step(m,d);

        // construct new [q qdot u] vector
        mju_copy(x2.data(), d->qpos, m->nv);
        mju_copy(x2.data() + m->nv, d->qvel, m->nv);


        // print results
        printf("pred %d:\t", i);
        std::cout<< pred.transpose();
        printf("\n");

        printf("new state %d:\t", i);
        std::cout<< x2.transpose();
        printf("\n\n");


    }

    return 0;
}