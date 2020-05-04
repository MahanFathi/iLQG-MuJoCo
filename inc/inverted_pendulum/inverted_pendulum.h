#pragma once

#include "mujoco.h"

#include "ilqr.h"



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
    static inline constexpr int N = 100;

    static inline constexpr int maxIterUtilConvergence = 100;

    // iterative LQR class
    ILQR<nv, nu, N>* iLQR;

    /*      Funcs    */
    InvertedPendulum(mjModel* m, mjData* d);
    void forward();
};
