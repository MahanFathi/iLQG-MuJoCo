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

    lx.resize(T+1);
    lu.resize(T);
    lxx.resize(T+1);
    luu.resize(T);

    Vx.setZero();
    Vxx.setZero();

    for( int t = 0; t < T; t++ ){
        lx[t].setZero();
        lu[t].setZero();
        lxx[t].setZero();
        luu[t].setZero();
    }
    lx[T].setZero();
    lxx[T].setZero();

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

#if ACTNUM == 1 && DOFNUM == 2

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

    cost_to_go += get_cost(d);

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

mjtNum cost::get_cost(const mjData* d) {

    mjtNum instant_cost =
            k * (d->qpos[1]) * (d->qpos[1]) +
            k * (d->qvel[1]) * (d->qvel[1]) +
            (d->ctrl[0]) * (d->ctrl[0]);

    return instant_cost;

}

extraVec_t cost::get_extra(const mjData* d) {

    extraVec_t extra = extraVec_t::Zero();
    extra(0, 0) = d->qpos[1];
    extra(1, 0) = d->qvel[1];

    return extra;

}

void cost::get_derivatives(const mjData* d, int t) {

    // extra_deriv must be calculated before

    lx[t] = 2.0 * k * d->qpos[1] * extra_deriv.row(0).transpose();
    lx[t] += 2.0 * k * d->qvel[1] * extra_deriv.row(1).transpose();
    lxx[t] = 2.0 * k * extra_deriv.row(0).transpose() * extra_deriv.row(0); // Approximate. See: https://math.stackexchange.com/questions/2349026/why-is-the-approximation-of-hessian-jtj-reasonable
    lxx[t] += 2.0 * k * extra_deriv.row(1).transpose() * extra_deriv.row(1); // Approximate. See: https://math.stackexchange.com/questions/2349026/why-is-the-approximation-of-hessian-jtj-reasonable
    lu[t](0,0) = 2 * d->ctrl[0];
    luu[t](0,0) = 2;

}

#endif


/*################################################*/
/*      ASSUMING THIS CASE TO BE THE HOPPER       */
/*################################################*/

#if ACTNUM == 3
#if DOFNUM == 6

void cost::add_cost(const mjData *d) {

    cost_to_go += get_cost(d);

}


stateVec_t cost::get_lx(stateVec_t x){

    // GRAD

    Vx(3,0) = (k_x*(torso*cos(x(3,0)) + 2*thigh*cos(x(3,0) - x(4,0)) -
                  2*leg*cos(x(3,0) - x(4,0) + x(4,0)))*(torso*sin(x(3,0)) +
                                                        2*thigh*sin(x(3,0) - x(4,0)) - 2*leg*sin(x(3,0) - x(4,0) + x(4,0))))/2;

    Vx(4,0) = k_x*(thigh*cos(x(3,0) - x(4,0)) - leg*cos(x(3,0) - x(4,0) + x(4,0)))*
              (-(torso*sin(x(3,0))) - 2*thigh*sin(x(3,0) - x(4,0)) + 2*leg*sin(x(3,0) - x(4,0) + x(4,0)));

    Vx(5,0) = k_x*leg*cos(x(3,0) - x(4,0) + x(4,0))*(-(torso*sin(x(3,0))) - 2*thigh*sin(x(3,0) - x(4,0)) +
                                                   2*leg*sin(x(3,0) - x(4,0) + x(4,0)));

    return Vx;

}


stateMat_t cost::get_lxx(stateVec_t x){

    // HESS

    Vxx(3,3) = (k_x*(pow(torso,2)*cos(2*x(3,0)) +
                   4*(pow(thigh,2)*cos(2*(x(3,0) - x(4,0))) +
                      thigh*torso*cos(2*x(3,0) - x(4,0)) -
                      2*leg*thigh*cos(2*x(3,0) - 2*x(4,0) + x(5,0)) +
                      pow(leg,2)*cos(2*(x(3,0) - x(4,0) + x(5,0))) -
                      leg*torso*cos(2*x(3,0) - x(4,0) + x(5,0)))))/2.;

    Vxx(3,4) = -(k_x*(2*pow(thigh,2)*cos(2*(x(3,0) - x(4,0))) +
                    thigh*torso*cos(2*x(3,0) - x(4,0)) +
                    leg*(-4*thigh*cos(2*x(3,0) - 2*x(4,0) + x(5,0)) +
                         2*leg*cos(2*(x(3,0) - x(4,0) + x(5,0))) -
                         torso*cos(2*x(3,0) - x(4,0) + x(5,0)))));

    Vxx(4,3) = Vxx(3,4);

    Vxx(3,5) = k_x*leg*(-2*thigh*cos(2*x(3,0) - 2*x(4,0) + x(5,0)) +
                      2*leg*cos(2*(x(3,0) - x(4,0) + x(5,0))) -
                      torso*cos(2*x(3,0) - x(4,0) + x(5,0)));

    Vxx(5,3) = Vxx(3,5);

    Vxx(4,4) = -(k_x*leg*(-2*thigh*cos(2*x(3,0) - 2*x(4,0) + x(5,0)) +
                        2*leg*cos(2*(x(3,0) - x(4,0) + x(5,0))) +
                        torso*sin(x(3,0))*sin(x(3,0) - x(4,0) + x(5,0))));

    Vxx(4,5) = 2*k_x*(pow(thigh*cos(x(3,0) - x(4,0)) - leg*cos(x(3,0) - x(4,0) + x(5,0)),2) +
                    (thigh*sin(x(3,0) - x(4,0)) - leg*sin(x(3,0) - x(4,0) + x(5,0)))*
                    (-(torso*sin(x(3,0)))/2. - thigh*sin(x(3,0) - x(4,0)) +
                     leg*sin(x(3,0) - x(4,0) + x(5,0))));

    Vxx(5,4) = Vxx(4,5);

    Vxx(5,5) = k_x*leg*(2*leg*cos(2*(x(3,0) - x(4,0) + x(5,0))) +
                      (torso*sin(x(3,0)) + 2*thigh*sin(x(3,0) - x(4,0))) *
                      sin(x(3,0) - x(4,0) + x(5,0)));

    return Vxx;

}

Eigen::Matrix<mjtNum, 3, 1> cost::get_com(const mjData* d) {

    Eigen::Matrix<mjtNum, 3, 1> com = Eigen::Matrix<mjtNum, 3, 1>::Zero();
    mjtNum TotalMass = 0;
    for (int i = 1; i < m->nbody; i++)
    {
        TotalMass += m->body_mass[i];
        mju_addToScl3(com.data(), d->xipos+i*3, m->body_mass[i]);
    }
    com /= TotalMass;
    return com;

}

Eigen::Matrix<mjtNum, 3, 1> cost::get_body_coor(const mjData* d, int body_id) {

    Eigen::Matrix<mjtNum, 3, 1> body_coor;
    mju_copy3(body_coor.data(), d->xipos+body_id*3);
    return body_coor;

}

extraVec_t cost::get_extra(const mjData *d) {

    extraVec_t extra = extraVec_t::Zero();
    Eigen::Matrix<mjtNum, 3, 1> com = get_com(d);
    Eigen::Matrix<mjtNum, 3, 1> foot = get_body_coor(d, 4); // id 4 corresponds to foot
    // Eigen::Matrix<mjtNum, 3, 1> torso = get_body_coor(d, 1); // id 1 corresponds to torso
    extra(0,0) = com(0, 0) - foot(0,0); // x offset
    extra(1,0) = com(2, 0) - foot(2,0); // z stance

}

mjtNum cost::get_cost(const mjData *d) {

    mjtNum instant_cost;
    Eigen::Matrix<mjtNum, 3, 1> com = get_com(d);
    Eigen::Matrix<mjtNum, 3, 1> foot = get_body_coor(d, 4); // id 4 corresponds to foot
    instant_cost = k_x * pow(com(0,0) - foot(0,0), 2.0) +
                   k_z * pow(0.55 - (com(2,0) - foot(2,0)), 2.0) +
                   k_u * pow(d->ctrl[0], 2.0) +
                   k_u * pow(d->ctrl[1], 2.0) +
                   k_u * pow(d->ctrl[2], 2.0);
    return instant_cost;

}

void cost::get_derivatives(const mjData *d, int t) {

    // extra_deriv must be calculated before

    extraVec_t extra = get_extra(d);

    lx[t] = 2.0 * k_x * extra(0,0) * extra_deriv.row(0).transpose();
    lx[t] += 2.0 * k_z * (extra(1,0) - 0.55) * extra_deriv.row(1).transpose();
    lxx[t] = 2.0 * k_x * extra_deriv.row(0).transpose() * extra_deriv.row(0); // Approximate. See: https://math.stackexchange.com/questions/2349026/why-is-the-approximation-of-hessian-jtj-reasonable
    lxx[t] += 2.0 * k_z * extra_deriv.row(1).transpose() * extra_deriv.row(1); // Approximate. See: https://math.stackexchange.com/questions/2349026/why-is-the-approximation-of-hessian-jtj-reasonable
    lu[t](0,0) = 2 * k_u * d->ctrl[0];
    lu[t](1,0) = 2 * k_u * d->ctrl[1];
    lu[t](2,0) = 2 * k_u * d->ctrl[2];
    luu[t](0,0) = 2 * k_u;
    luu[t](1,1) = 2 * k_u;
    luu[t](2,2) = 2 * k_u;

}




#endif
#endif
