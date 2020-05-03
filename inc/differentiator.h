#pragma once

#include <eigen3/Eigen/Core>
#include "mujoco.h"

#include "mjderivative.h"


template<int nv, int nu>
class Differentiator
{
public:

    // typedefs for env matrices/vectors
    typedef Eigen::Matrix<mjtNum, 2*nv, 2*nv> A_t;
    typedef Eigen::Matrix<mjtNum, 2*nv, nu> B_t;
    typedef Eigen::Matrix<mjtNum, 2*nv, 1> x_t;
    typedef Eigen::Matrix<mjtNum, nu, 1> u_t;
    // typedefs for env maps
    typedef Eigen::Map<Eigen::Matrix<mjtNum, nv, nv>> dqdq_mt;
    typedef Eigen::Map<Eigen::Matrix<mjtNum, nv, nu>> dqdu_mt;
    typedef Eigen::Map<Eigen::Matrix<mjtNum, nv, 1>> qpos_mt;
    typedef Eigen::Map<Eigen::Matrix<mjtNum, nv, 1>> qvel_mt;
    typedef Eigen::Map<Eigen::Matrix<mjtNum, nv, 1>> ctrl_mt;
    typedef Eigen::Map<Eigen::Matrix<mjtNum, 2*nv, 1>> x_mt;
    typedef Eigen::Map<Eigen::Matrix<mjtNum, nu, 1>> u_mt;

    /*      Data     */
    // MuJoCo model and data
    mjModel* m;
    mjData* d;
    // storage for derivatives
    mjtNum* deriv;
    // matrices mapped to deriv
    dqdq_mt* dqaccdq;
    dqdq_mt* dqaccdqvel;
    dqdu_mt* dqaccdctrl;
    // step cost function and derivatives
    stepCostFn_t &stepCostFn;
    x_mt* dgdx;
    u_mt* dgdu;
    // vectors mapped to (qpos, qvel) and ctrl
    x_mt* x;
    u_mt* u;
    // linear dynamics
    A_t* A;
    B_t* B;

    /*      Funcs       */
    Differentiator(mjModel* m, mjData* d, stepCostFn_t &stepCostFn):
        m(m), d(d), stepCostFn(stepCostFn)
    {
        // allocate memory for deriv and bind to eigen matrices/vectors
        deriv = (mjtNum*) mju_malloc((nv*(2*nv+nu)+2*nv+nu)*sizeof(mjtNum));
        dqaccdq = new dqdq_mt(deriv);
        dqaccdqvel = new dqdq_mt(deriv + nv*nv);
        dqaccdctrl = new dqdu_mt(deriv + 2*nv*nv);
        dgdx = new x_mt(deriv + 2*nv*nv+nv*nu);
        dgdu = new u_mt(deriv + 2*nv*nv+nv*nu+2*nv);
        x = new x_mt(d->qpos);   // note that qpos and qvel are contiguous in memory
        u = new u_mt(d->ctrl);

        // initialize invariant parts of linear dynamics matrices
        A = new A_t;
        B = new B_t;
        (*A).block(0, 0, nv, nv).setIdentity();
        (*A).block(0, nv, nv, nv).setIdentity();
        (*A).block(0, nv, nv, nv) *= m->opt.timestep;
        (*B).setZero();
    }

    void setMJData(mjData* dStar)
    {
        d = dStar;

        // using 'placement new' syntax to only chage the
        // memory address for map instead of reallocation
        new (x) x_mt(d->qpos);  // note that qpos and qvel are contiguous in memory
        new (u) u_mt(d->ctrl);

    }

    void updateDerivatives() // derivatives are taken at d
    {
        calcMJDerivatives(m, d, deriv, stepCostFn);

        (*A).block(nv, 0, nv, nv) = (*dqaccdq) * m->opt.timestep;
        (*A).block(nv, nv, nv, nv).setIdentity();
        (*A).block(nv, nv, nv, nv) += (*dqaccdqvel) * m->opt.timestep;
        (*B).block(nv, 0, nv, nu) = (*dqaccdctrl) * m->opt.timestep;
    }

};
