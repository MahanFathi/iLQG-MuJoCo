#include <iostream>
#include <eigen3/Eigen/Core>
#include "mujoco.h"

#include "lqr.h"
#include "mjderivative.h"


LQR::LQR(mjModel* m, mjData* d):
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
    (*A).block<nv, nv>(0, 0).setIdentity();
    (*A).block<nv, nv>(0, nv).setIdentity();
    (*A).block<nv, nv>(0, nv) *= m->opt.timestep;
    (*B).setZero();
}

void LQR::updateDerivatives()
{
    calcMJDerivatives(m, d, deriv);
    (*A).block<nv, nv>(nv, 0) = (*dqaccdq) * m->opt.timestep;
    (*A).block<nv, nv>(nv, nv).setIdentity();
    (*A).block<nv, nv>(nv, nv) += (*dqaccdqvel) * m->opt.timestep;
    (*B).block<nv, nu>(nv, 0) = (*dqaccdctrl) * m->opt.timestep;
}
