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
    stateThread X_MinCost;
    actionThread U_MinCost;

    // Cost Class is included here
    cost Cost;

    // Improved line search vars
    mjtNum alpha{1.0};
    mjtNum decay = 0.9;
    mjtNum min_cost;
    mjtNum prev_cost;
    int decay_count = 0;
    int decay_limit = 10;
    bool bool_backtrack = true;

    // Intermediate optimization vars
    stateMat_t Qxx;
    actionMat_t Quu;
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
    mjtNum max_mu = 1e30;
    mjtNum mu_factor = 1.05;

    mjtNum lambda;
    mjtNum lamb_factor;

    int max_iter = 10;

    bool bwd_flag;
    bool pd_sanity = false;
    bool LevenbergMarquardt = false;

    int iter = 0;
    bool done = false;

    action_state_MatThread K_MinCost;
    actionThread k_MinCost;
    mjtNum alpha_MinCost = 1.0;
    bool bool_updateK_MinCost;
    action_state_Mat_t K_MPC;

    mjtNum torque_lower_bound = -1;
    mjtNum torque_upper_bound = +1;

#if ACTNUM == 1 && DOFNUM == 2
    int step_ratio = 1; // >= 1
#endif

#if ACTNUM == 3 && DOFNUM == 6
    int step_ratio = 5; // >= 1
#endif

    // constructor
    ilqr(mjModel *m, mjData *d,
         mjtNum *deriv, int T,
         const char *env);

    // runs the forward pass on ilqr, stores new states, ...
    // calculates linear dynamics and cost matrices
    void rollout(bool init, mjtNum alpha_given);
    void backtrack();

    // deconstructor: free stacks
    ~ilqr();

    // LQR backward pass
    void bwd_lqr();

    // Approximate Vx[T] with Vx[T-1] and Vxx[T-1]
    stateVec_t get_lx_T(void);

    // Deals with all derivative related parts
    void do_derivatives(mjData* d, int t);

    // Regularization
    void increase_mu();
    void decrease_mu();

    void big_step(mjData *d);
    void MPC_step(bool feedback=true);
    actionMat_t lm_inv(actionMat_t Quu, mjtNum lambda);
    void iterate();
    actionVec_t RunMPC();
};


#endif //ILGQ_ILQR_H
