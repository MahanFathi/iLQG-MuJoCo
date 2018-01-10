//
// Created by mahan on 12/26/17.
//

#ifndef ILGQ_ILQR_H
#define ILGQ_ILQR_H

#include "cost.h"
#include "types.h"
#include "mjmodel.h"
#include "mjdata.h"
#include <Eigen/Dense>

class ilqr {
public:

    // defined in main, have access to pointer
    mjModel* m;
    mjData* d; // optimizing data thread
    mjtNum* deriv;

    // states and control
    stateThread X;
    actionThread U;
    stateThread x;
    actionThread u;

    // Cost Class is included here
    cost Cost;

    // Improved line search vars
    mjtNum alpha{1.0};
    mjtNum decay = 0.8;
    mjtNum min_cost;
    mjtNum prev_cost;
    int decay_count = 0;
    int decay_limit = 3;

    int maxiter = 1;

    // Intermediate optimization vars
    stateMat_t Qxx;
    actionMat_t Quu;
    state_action_Mat_t Qxu;
    action_state_Mat_t Qux;
    stateVec_t Qx;
    actionVec_t Qu;
    stateMat_t Vxx;
    stateVec_t Vx;

    // Backward Output
    action_state_MatThread K;
    actionThread k;

    // Linear Dynamics
    stateMatThread Fx;
    state_action_MatThread Fu;

    int T; // optimization horizon
    int nv;
    int nu;

    // Auxiliary variables for fwd loop
    stateVec_t del_x;
    actionVec_t del_u;

    mjtNum mu;
    mjtNum mu_min;
    mjtNum delta;
    mjtNum delta_0;

    bool bwd_flag;
    bool use_regularization = true;

    int iter = 0;
    int min_iter = 2;
    bool converged = false;

    // constructor
    ilqr(mjModel *m, mjData *d,
         mjtNum *deriv, int T,
         const char *env);

    // runs the forward pass on ilqr, stores new states, ...
    // calculates linear dynamics and cost matrices
    void fwd_ctrl(bool init);

    // deconstructor: free stacks
    ~ilqr();

    // LQR backward pass
    void bwd_lqr();

    // Regularization
    void increase_mu();
    void decrease_mu();

    // One Iteration
    void iterate();

    // Till Convergence
    void manager();


};


#endif //ILGQ_ILQR_H
