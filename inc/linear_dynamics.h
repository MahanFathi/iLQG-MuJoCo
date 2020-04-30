#pragma once

#include <eigen3/Eigen/Core>
#include "mujoco.h"

#include "mjderivative.h"


template<int nv, int nu>
class LinearDynamics
{
public:

    // typedefs for hopper env matrices
    typedef Eigen::Matrix<mjtNum, 2*nv, 2*nv> A_t;
    typedef Eigen::Matrix<mjtNum, 2*nv, nu> B_t;
    typedef Eigen::Matrix<mjtNum, 2*nv, 1> x_t;
    typedef Eigen::Matrix<mjtNum, nu, 1> u_t;
    // typedefs for hopper env maps
    typedef Eigen::Map<Eigen::Matrix<mjtNum, nv, nv>> dqdq_t;
    typedef Eigen::Map<Eigen::Matrix<mjtNum, nv, nu>> dqdu_t;
    typedef Eigen::Map<Eigen::Matrix<mjtNum, nv, 1>> qpos_t;
    typedef Eigen::Map<Eigen::Matrix<mjtNum, nv, 1>> qvel_t;
    typedef Eigen::Map<Eigen::Matrix<mjtNum, nv, 1>> ctrl_t;

    /*      Data     */
    // MuJoCo model and data
    mjModel* m;
    mjData* d;
    // storage for derivatives
    mjtNum* deriv;
    // matrices mapped to deriv
    dqdq_t* dqaccdq;
    dqdq_t* dqaccdqvel;
    dqdu_t* dqaccdctrl;
    // linear dynamics
    A_t* A;
    B_t* B;

    /*      Funcs    */
    LinearDynamics(mjModel* m, mjData* d):
        m(m), d(d)
    {
        // allocate memory for deriv and bind to eigen matrices
        deriv = (mjtNum*) mju_malloc(m->nv*(2*m->nv+m->nu)*sizeof(mjtNum));
        dqaccdq = new dqdq_t(deriv);
        dqaccdqvel = new dqdq_t(deriv + nv*nv);
        dqaccdctrl = new dqdu_t(deriv + 2*nv*nv);

        // initialize invariant parts of linear dynamics matrices
        A = new A_t;
        B = new B_t;
        (*A).block(0, 0, nv, nv).setIdentity();
        (*A).block(0, nv, nv, nv).setIdentity();
        (*A).block(0, nv, nv, nv) *= m->opt.timestep;
        (*B).setZero();
    }

    void updateDerivatives()
    {

        calcMJDerivatives(m, d, deriv);

        (*A).block(nv, 0, nv, nv) = (*dqaccdq) * m->opt.timestep;
        (*A).block(nv, nv, nv, nv).setIdentity();
        (*A).block(nv, nv, nv, nv) += (*dqaccdqvel) * m->opt.timestep;
        (*B).block(nv, 0, nv, nu) = (*dqaccdctrl) * m->opt.timestep;
    }

};
