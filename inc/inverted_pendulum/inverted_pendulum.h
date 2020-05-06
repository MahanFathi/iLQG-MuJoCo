#pragma once

#include "mujoco.h"

#include "ilqr.h"
#include "mjderivative.h"



class InvertedPendulum
{
public:

    /*      Data     */
    // MuJoCo model and data
    mjModel* m = NULL;
    mjData* d = NULL;

    // specific env settings
    static inline constexpr int nv = 2;
    static inline constexpr int nu = 1;
    static inline constexpr int N = 40;

    static inline constexpr int maxIterUtilConvergence = 20;

    // iterative LQR class
    ILQR<nv, nu, N>* iLQR;

    // cost function
    stepCostFn_t stepCostFn;

    /*      Funcs    */
    InvertedPendulum(mjModel* m, mjData* d);
    void forward();
};
