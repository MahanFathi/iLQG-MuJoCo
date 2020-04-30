#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>

#include <eigen3/Eigen/Core>
#include "mujoco.h"

#include "linear_dynamics.h"
#include "mjderivative.h"
#include "update.h"


int main(int argc, const char** argv)
{

    using x_t = LinearDynamics<6, 3>::x_t;
    using u_t = LinearDynamics<6, 3>::u_t;

    // activate mujoco
    mj_activate("mjkey.txt");

    // create model and data
    char error[1000] = "Could not load binary model";
    mjModel* m = mj_loadXML("./res/hopper.xml", 0, error, 1000);
    mjData* dStar = mj_makeData(m);

    // advance simulation
    for (auto i = 0; i < 500; i++)
        mj_step(m, dStar);

    // bind data to eigen
    Eigen::Map<Eigen::VectorXd> qStar(dStar->qpos, m->nv);
    Eigen::Map<Eigen::VectorXd> qvelStar(dStar->qvel, m->nv);
    Eigen::Map<Eigen::VectorXd> ctrlStar(dStar->ctrl, m->nu);

    // change controls
    ctrlStar = ctrlStar.array() - 0.1;

    // save first state and contorls
    x_t x_star; x_star << qStar, qvelStar;
    u_t u_star; u_star << ctrlStar;

    // linearize around set state and contorl
    LinearDynamics<6, 3>* ld = new LinearDynamics<6, 3>(m, dStar);
    ld->updateDerivatives();

    // copy data to d_prime
    mjData* d = mj_makeData(m);
    cpMjData(m, d, dStar);

    // advance simulation on dStar and save next state
    mj_step(m, dStar);
    x_t xStarNext; xStarNext << qStar, qvelStar;

    // bind data to eigen
    Eigen::Map<Eigen::VectorXd> q(d->qpos, m->nv);
    Eigen::Map<Eigen::VectorXd> qvel(d->qvel, m->nv);
    Eigen::Map<Eigen::VectorXd> ctrl(d->ctrl, m->nu);

    // perturb around x* and save x and u
    mjtNum eps = 1e-6;
    q = q.array() + eps;
    qvel = qvel.array() + eps;
    ctrl = ctrl.array() + eps;
    x_t x; x << q, qvel;
    u_t u; u << ctrl;

    // advance simulation on perturbed data
    mj_step(m, d);
    x_t xNext; xNext << q, qvel;
    u_t uNext; uNext << ctrl;

    // compare
    x_t xNextPrediction = *(ld->A) * (x - x_star) + *(ld->B) * (u - u_star) + xStarNext;
    std::cout << "xNextPrediction - xNext:" << '\n';
    std::cout << (xNextPrediction - xNext).transpose() << '\n';
    std::cout << "--------------------" << '\n';
    std::cout << "xNext - xStarNext:" << '\n';
    std::cout << (xNext - xStarNext).transpose() << '\n';
    std::cout << "--------------------" << '\n';
    std::cout << "xNext:" << '\n';
    std::cout << xNext.transpose() << '\n';

}
