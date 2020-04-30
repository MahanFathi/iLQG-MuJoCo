#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>

#include <eigen3/Eigen/Core>
#include "mujoco.h"

#include "lqr.h"
#include "mjderivative.h"
#include "update.h"


int main(int argc, const char** argv)
{

    // activate mujoco
    mj_activate("mjkey.txt");

    // create model and data
    char error[1000] = "Could not load binary model";
    mjModel* m = mj_loadXML(argv[1], 0, error, 1000);
    mjData* d_star = mj_makeData(m);

    // advance simulation
    for (auto i = 0; i < 500; i++)
        mj_step(m, d_star);

    // bind data to eigen
    Eigen::Map<Eigen::VectorXd> q_star(d_star->qpos, m->nv);
    Eigen::Map<Eigen::VectorXd> qvel_star(d_star->qvel, m->nv);
    Eigen::Map<Eigen::VectorXd> ctrl_star(d_star->ctrl, m->nu);

    // change controls
    ctrl_star = ctrl_star.array() - 0.1;

    // save first state and contorls
    x_t x_star; x_star << q_star, qvel_star;
    u_t u_star; u_star << ctrl_star;

    // linearize around set state and contorl
    LQR* lqr = new LQR(m, d_star);
    lqr->updateDerivatives();

    // copy data to d_prime
    mjData* d = mj_makeData(m);
    cpMjData(m, d, d_star);

    // advance simulation on d_star and save next state
    mj_step(m, d_star);
    x_t x_star_next; x_star_next << q_star, qvel_star;

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
    x_t x_next; x_next << q, qvel;
    u_t u_next; u_next << ctrl;

    // compare
    x_t xNext = *(lqr->A) * (x - x_star) + *(lqr->B) * (u - u_star) + x_star_next;
    std::cout << (xNext - x_next).transpose() << '\n';
    std::cout << "==================" << '\n';
    std::cout << (x_next - x_star_next).transpose() << '\n';
    std::cout << "==================" << '\n';
    std::cout << xNext.transpose() << '\n';

}
