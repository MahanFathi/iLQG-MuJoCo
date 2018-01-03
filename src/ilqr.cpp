//
// Created by mahan on 12/26/17.
//

#include "cost.h"
#include <mujoco.h>
#include <iostream>
#include <lindy.h>
#include "ilqr.h"

ilqr::ilqr(mjModel *m, mjData *d,
           mjtNum *deriv, int T,
           const char *env):
        m(m), d(d), T(T), Cost(env, m, T),
        deriv(deriv), nv(m->nv), nu(m->nu)
{

    // initialize value functions, etc.
    X.resize(T+1);
    U.resize(T);
    x.resize(T+1);
    u.resize(T);
    K.resize(T);
    k.resize(T);
    Fx.resize(T);
    Fu.resize(T);

    // set first policy to "indifference"
    for( int t = 0; t < T; t++ ){
        Fx[t].setZero();
        Fu[t].setZero();
        k[t].setZero();
        K[t].setZero();
        U[t].setZero();
        X[t].setZero();
    }


    // proceed for a few steps
    for( int i = 0; i < 80; i++ )
        mj_step(m, d);


    // run forward ilqr pass to initialize and more
    fwd_ctrl(true);
    printf("Cost at start: \t%f\n", Cost.cost_to_go);


}


void ilqr::fwd_ctrl(bool init=false) {

    // run forward simulation for control inputs, save q & qdot and ...
    // calculate linear dynamics

    // copy data for now
    mjData* d_cp = mj_copyData(nullptr, m, d);

    if (!init) {
        prev_cost = Cost.cost_to_go;
        Cost.reset_cost();
    }

    for( int t = 0; t < T; t++ )
    {

        // get controls
        mju_copy(del_x.data(), d_cp->qpos, nv);
        mju_copy(del_x.data() + nv, d_cp->qvel, nv);
        del_x -= X[t];
        del_u = K[t] * del_x + alpha * k[t];

        // store new controls
        U[t] += del_u;

        // update the state
        mju_copy(X[t].data(), d_cp->qpos, nv);
        mju_copy(X[t].data()+nv, d_cp->qvel, nv);

        // instruct new controls
        mju_copy(d_cp->ctrl, U[t].data(), nu);

//        printf("\n U[%d]: \t %f\n", t, U[t](0));
//        printf("\n U[%d]: \t %f\n", t, d_cp->ctrl[0]);

        // add cost
        Cost.add_cost(d_cp);

        // run with real dynamics
        mj_step(m, d_cp);

//        printf("State at time %d:\t\n", t);
//        std::cout<<X[t]<<"\n";
//        printf("K %d:\t\n", t);
//        std::cout<<K[t]<<"\n";
//        printf("k %d:\t\n", t);
//        std::cout<<k[t]<<"\n";
//        printf("Controls at time %d:\t\n", t);
//        std::cout<<U[t]<<"\n";

//        printf("\ntheta[%d]: \t%f\n", t, d_cp->qpos[1]);
//        printf("\nU[%d]: \t%f\n", t, d_cp->ctrl[0]);

    }

    mju_copy(X[T].data(), d_cp->qpos, nv);
    mju_copy(X[T].data() + nv, d_cp->qvel, nv);
    mju_zero(d_cp->ctrl, nu); // assuming an identical final state cost
    Cost.add_cost(d_cp);

    if (init) {
        min_cost = Cost.cost_to_go;
        for( int t = 0; t < T; t++ ) {

            mju_copy(d_cp->ctrl, X[t].data() + 2*nv, nu);
            mju_copy(d_cp->qpos, X[t].data(), nv);
            mju_copy(d_cp->qvel, X[t].data() + nv, nv);

            // get cost matrices
            Cost.calc_costmats(d_cp, t);
            Cost.add_cost(d_cp);

            // get linear dynamics TODO: probably not the fastest parallelization
            get_derivs(deriv, m, d_cp);
            Fx[t].setZero();
            calc_F(Fx[t].data(), Fu[t].data(), deriv, m, d_cp);
            Fx[t].block<2*DOFNUM, 2*DOFNUM>(0, 0) += stateMat_t::Identity();
        }
    }
    else if (decay_count == 0 || Cost.cost_to_go < prev_cost) {
        //  && decay_count < decay_limit
        // backup previous trajectory
        for( int t = 0; t < T; t++ ){
            x[t] = X[t];
            u[t] = U[t];
        }
        x[T] = X[T];
//        printf("ran with alpha = %.3f\n", alpha);
        alpha *= decay;
        decay_count++;
        fwd_ctrl(false);
    }
    else {
        // re-init line-search params
        decay_count = 0;
        alpha = 1.0;
        // reuse backups
        for( int t = 0; t < T; t++ ){
            X[t] = x[t];
            U[t] = u[t];
        }
        X[T] = x[T];
        for( int t = 0; t < T; t++ ) {
            // use data from the latest X
            mju_copy(d_cp->ctrl, X[t].data() + 2*nv, nu);
            mju_copy(d_cp->qpos, X[t].data(), nv);
            mju_copy(d_cp->qvel, X[t].data() + nv, nv);

            // get cost matrices
            Cost.calc_costmats(d_cp, t);

            // get linear dynamics TODO: probably not the fastest parallelization
            get_derivs(deriv, m, d_cp);
            Fx[t].setZero();
            calc_F(Fx[t].data(), Fu[t].data(), deriv, m, d_cp);
            Fx[t].block<2*DOFNUM, 2*DOFNUM>(0, 0) += stateMat_t::Identity();
//            std::cout << "Fx["
        }
    }

}

ilqr::~ilqr() {

}


void ilqr::bwd_lqr() {

    // start with V and v = 0
    Vxx = Cost.get_lxx(X[T]);
    Vx = Cost.get_lx(X[T]);

    for( int t = T-1; t >= 0; t-- ){

        Qx = Cost.lx[t] + Fx[t].transpose() * Vx;
        Qu = Cost.lu[t] + Fu[t].transpose() * Vx;

        Qxx = Cost.lxx[t] + Fx[t].transpose() * Vxx * Fx[t];
        Quu = Cost.luu[t] + Fu[t].transpose() * Vxx * Fu[t];
        Qux = Fu[t].transpose() * Vxx * Fx[t];

        // K = -Quu^-1 Qux
        K[t] = Quu.ldlt().solve(Qux) * (-1);
        k[t] = Quu.ldlt().solve(Qu) * (-1);

//        std::cout<< "\nK[" << t << "]: \n" << K[t] << std::endl;
//        std::cout<< "\nk[" << t << "]: \n" << k[t] << std::endl;

        // calc V and v
        Vxx = Qxx + K[t].transpose() * Quu * K[t] + K[t].transpose() * Qux + Qux.transpose() * K[t] + mu * stateMat_t::Identity();
        Vx = Qx + K[t].transpose() * Quu * k[t] + K[t].transpose() * Qu + Qux.transpose() * k[t] + 2 * mu * stateVec_t::Ones();

    }

}

void ilqr::iterate() {

    int iter = 0;
    do{
        bwd_lqr();
        fwd_ctrl();
        printf("\nCost at iter %d:\t%f\n", iter, Cost.cost_to_go);
        iter++;
    }
    while( iter < maxiter );
}






