//
// Created by mahan on 12/27/17.
//

#include <cstring>
#include <cmath>
#include <mujoco.h>
#include "cost.h"
#include <Eigen/Dense>

cost::cost(const char *env, const mjModel* m, int T): m(m), T(T)
{
    cost_to_go = 0.0;

    lx.resize(T);
    lu.resize(T);
    lxx.resize(T);
    luu.resize(T);

    Vx.setZero();
    Vxx.setZero();

    for( int t = 0; t < T; t++ ){
        lx[t].setZero();
        lu[t].setZero();
        lxx[t].setZero();
        luu[t].setZero();
    }

//    // set cost matrices to zero at first
//    for( int t = 0; t < T; t++ ){
//        mju_zero(c[t], 2*m->nv+m->nu);
//        mju_zero(C[t], (2*m->nv+m->nu)*(2*m->nv+m->nu));
//    }

}

void cost::calc_costmats(const mjData *d, int t)
{
    // TODO: only for cart-pole. extend.

    lx[t](1,0) = 2 * k * d->qpos[1];
    lx[t](3,0) = 2 * k * d->qvel[1];
    lxx[t](1,1) = 2 * k;
    lxx[t](3,3) = 2 * k;

    lu[t](0,0) = 2 * d->ctrl[0];
    luu[t](0,0) = 2;


}

void cost::add_cost(const mjData *d) {

    cost_to_go +=
            k * (d->qpos[1]) * (d->qpos[1]) +
            k * (d->qvel[1]) * (d->qvel[1]) +
            (d->ctrl[0]) * (d->ctrl[0]);

//    printf("\nCOST: \t%f",cost_to_go);

}

stateVec_t cost::get_lx(stateVec_t x){

    Vx(1,0) = 2 * k * x(1,0);
    Vx(3,0) = 2 * k * x(3,0);
    return Vx;

}

stateMat_t cost::get_lxx(stateVec_t x){

    Vxx(1,1) = 2 * k;
    Vxx(3,3) = 2 * k;
    return Vxx;

}

void cost::reset_cost() {

    cost_to_go = 0.0;

}




