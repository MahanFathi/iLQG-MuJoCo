#pragma once

#include <eigen3/Eigen/Core>
#include "mujoco.h"

// nv and nu for hopper env
inline constexpr int nv = 6;
inline constexpr int nu = 3;
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



class LQR
{
public:

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
    LQR(mjModel* m, mjData* d);

    void updateDerivatives();

};
