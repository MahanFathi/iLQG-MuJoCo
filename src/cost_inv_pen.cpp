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


void cost::reset_cost() {

    cost_to_go = 0.0;

}

/*################################################*/
/* ASSUMING THIS CASE TO BE THE INVERTED PENDULUM */
/*################################################*/

#if ACTNUM == 1
#if DOFNUM == 2

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

#endif
#endif


/*################################################*/
/*      ASSUMING THIS CASE TO BE THE HOPPER       */
/*################################################*/

#if ACTNUM == 3
#if DOFNUM == 6

void cost::calc_costmats(const mjData *d, int t)
{

    // GRAD

    lx[t](3,0) = (k*(torso*cos(d->qpos[3]) + 2*thigh*cos(d->qpos[3] - d->qpos[4]) -
                    2*leg*cos(d->qpos[3] - d->qpos[4] + d->qpos[4]))*(torso*sin(d->qpos[3]) +
                                                                      2*thigh*sin(d->qpos[3] - d->qpos[4]) - 2*leg*sin(d->qpos[3] - d->qpos[4] + d->qpos[4])))/2;

    lx[t](4,0) = k*(thigh*cos(d->qpos[3] - d->qpos[4]) - leg*cos(d->qpos[3] - d->qpos[4] + d->qpos[4]))*
                (-(torso*sin(d->qpos[3])) - 2*thigh*sin(d->qpos[3] - d->qpos[4]) + 2*leg*sin(d->qpos[3] - d->qpos[4] + d->qpos[4]));

    lx[t](5,0) = k*leg*cos(d->qpos[3] - d->qpos[4] + d->qpos[4])*(-(torso*sin(d->qpos[3])) - 2*thigh*sin(d->qpos[3] - d->qpos[4]) +
                                                                 2*leg*sin(d->qpos[3] - d->qpos[4] + d->qpos[4]));

    lu[t](0,0) = 2 * d->ctrl[0];
    lu[t](1,0) = 2 * d->ctrl[1];
    lu[t](2,0) = 2 * d->ctrl[2];


    // HESS

    lxx[t](3,3) = (k*(pow(torso,2)*cos(2*d->qpos[3]) +
                    4*(pow(thigh,2)*cos(2*(d->qpos[3] - d->qpos[4])) +
                       thigh*torso*cos(2*d->qpos[3] - d->qpos[4]) -
                       2*leg*thigh*cos(2*d->qpos[3] - 2*d->qpos[4] + d->qpos[5]) +
                       pow(leg,2)*cos(2*(d->qpos[3] - d->qpos[4] + d->qpos[5])) -
                       leg*torso*cos(2*d->qpos[3] - d->qpos[4] + d->qpos[5]))))/2.;

    lxx[t](3,4) = -(k*(2*pow(thigh,2)*cos(2*(d->qpos[3] - d->qpos[4])) +
                     thigh*torso*cos(2*d->qpos[3] - d->qpos[4]) +
                     leg*(-4*thigh*cos(2*d->qpos[3] - 2*d->qpos[4] + d->qpos[5]) +
                          2*leg*cos(2*(d->qpos[3] - d->qpos[4] + d->qpos[5])) -
                          torso*cos(2*d->qpos[3] - d->qpos[4] + d->qpos[5]))));

    lxx[t](4,3) = lxx[t](3,4);

    lxx[t](3,5) = k*leg*(-2*thigh*cos(2*d->qpos[3] - 2*d->qpos[4] + d->qpos[5]) +
                       2*leg*cos(2*(d->qpos[3] - d->qpos[4] + d->qpos[5])) -
                       torso*cos(2*d->qpos[3] - d->qpos[4] + d->qpos[5]));

    lxx[t](5,3) = lxx[t](3,5);

    lxx[t](4,4) = -(k*leg*(-2*thigh*cos(2*d->qpos[3] - 2*d->qpos[4] + d->qpos[5]) +
                         2*leg*cos(2*(d->qpos[3] - d->qpos[4] + d->qpos[5])) +
                         torso*sin(d->qpos[3])*sin(d->qpos[3] - d->qpos[4] + d->qpos[5])));

    lxx[t](4,5) = 2*k*(pow(thigh*cos(d->qpos[3] - d->qpos[4]) - leg*cos(d->qpos[3] - d->qpos[4] + d->qpos[5]),2) +
                     (thigh*sin(d->qpos[3] - d->qpos[4]) - leg*sin(d->qpos[3] - d->qpos[4] + d->qpos[5]))*
                     (-(torso*sin(d->qpos[3]))/2. - thigh*sin(d->qpos[3] - d->qpos[4]) +
                      leg*sin(d->qpos[3] - d->qpos[4] + d->qpos[5])));

    lxx[t](5,4) = lxx[t](4,5);

    lxx[t](5,5) = k*leg*(2*leg*cos(2*(d->qpos[3] - d->qpos[4] + d->qpos[5])) +
                       (torso*sin(d->qpos[3]) + 2*thigh*sin(d->qpos[3] - d->qpos[4])) *
                       sin(d->qpos[3] - d->qpos[4] + d->qpos[5]));

    luu[t](0,0) = 2;
    luu[t](1,1) = 2;
    luu[t](2,2) = 2;

}


void cost::add_cost(const mjData *d) {

    cost_to_go += k * pow(( -torso/2.0 * sin(d->qpos[3]) +
                        thigh * sin(d->qpos[4] - d->qpos[3]) +
                        leg * sin(d->qpos[5] - d->qpos[4] - d->qpos[3])),2) +
                  (d->ctrl[0]) * (d->ctrl[0]) +
                  (d->ctrl[1]) * (d->ctrl[1]) +
                  (d->ctrl[2]) * (d->ctrl[2]);

}


stateVec_t cost::get_lx(stateVec_t x){

    // GRAD

    Vx(3,0) = (k*(torso*cos(x(3,0)) + 2*thigh*cos(x(3,0) - x(4,0)) -
                  2*leg*cos(x(3,0) - x(4,0) + x(4,0)))*(torso*sin(x(3,0)) +
                                                        2*thigh*sin(x(3,0) - x(4,0)) - 2*leg*sin(x(3,0) - x(4,0) + x(4,0))))/2;

    Vx(4,0) = k*(thigh*cos(x(3,0) - x(4,0)) - leg*cos(x(3,0) - x(4,0) + x(4,0)))*
              (-(torso*sin(x(3,0))) - 2*thigh*sin(x(3,0) - x(4,0)) + 2*leg*sin(x(3,0) - x(4,0) + x(4,0)));

    Vx(5,0) = k*leg*cos(x(3,0) - x(4,0) + x(4,0))*(-(torso*sin(x(3,0))) - 2*thigh*sin(x(3,0) - x(4,0)) +
                                                   2*leg*sin(x(3,0) - x(4,0) + x(4,0)));

    return Vx;

}


stateMat_t cost::get_lxx(stateVec_t x){

    // HESS

    Vxx(3,3) = (k*(pow(torso,2)*cos(2*x(3,0)) +
                   4*(pow(thigh,2)*cos(2*(x(3,0) - x(4,0))) +
                      thigh*torso*cos(2*x(3,0) - x(4,0)) -
                      2*leg*thigh*cos(2*x(3,0) - 2*x(4,0) + x(5,0)) +
                      pow(leg,2)*cos(2*(x(3,0) - x(4,0) + x(5,0))) -
                      leg*torso*cos(2*x(3,0) - x(4,0) + x(5,0)))))/2.;

    Vxx(3,4) = -(k*(2*pow(thigh,2)*cos(2*(x(3,0) - x(4,0))) +
                    thigh*torso*cos(2*x(3,0) - x(4,0)) +
                    leg*(-4*thigh*cos(2*x(3,0) - 2*x(4,0) + x(5,0)) +
                         2*leg*cos(2*(x(3,0) - x(4,0) + x(5,0))) -
                         torso*cos(2*x(3,0) - x(4,0) + x(5,0)))));

    Vxx(4,3) = Vxx(3,4);

    Vxx(3,5) = k*leg*(-2*thigh*cos(2*x(3,0) - 2*x(4,0) + x(5,0)) +
                      2*leg*cos(2*(x(3,0) - x(4,0) + x(5,0))) -
                      torso*cos(2*x(3,0) - x(4,0) + x(5,0)));

    Vxx(5,3) = Vxx(3,5);

    Vxx(4,4) = -(k*leg*(-2*thigh*cos(2*x(3,0) - 2*x(4,0) + x(5,0)) +
                        2*leg*cos(2*(x(3,0) - x(4,0) + x(5,0))) +
                        torso*sin(x(3,0))*sin(x(3,0) - x(4,0) + x(5,0))));

    Vxx(4,5) = 2*k*(pow(thigh*cos(x(3,0) - x(4,0)) - leg*cos(x(3,0) - x(4,0) + x(5,0)),2) +
                    (thigh*sin(x(3,0) - x(4,0)) - leg*sin(x(3,0) - x(4,0) + x(5,0)))*
                    (-(torso*sin(x(3,0)))/2. - thigh*sin(x(3,0) - x(4,0)) +
                     leg*sin(x(3,0) - x(4,0) + x(5,0))));

    Vxx(5,4) = Vxx(4,5);

    Vxx(5,5) = k*leg*(2*leg*cos(2*(x(3,0) - x(4,0) + x(5,0))) +
                      (torso*sin(x(3,0)) + 2*thigh*sin(x(3,0) - x(4,0))) *
                      sin(x(3,0) - x(4,0) + x(5,0)));

    return Vxx;

}

#endif
#endif
