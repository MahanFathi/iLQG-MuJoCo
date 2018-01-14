//
// Created by mahan on 12/26/17.
//

#include "cost.h"
#include <mujoco.h>
#include <iostream>
#include <lindy.h>
#include "ilqr.h"
#include <cmath>
#include <boost/algorithm/clamp.hpp>

ilqr::ilqr(mjModel *m, mjData *d,
           mjtNum *deriv, int T,
           const char *env):
        m(m), d(d), T(T), Cost(env, m, T),
        deriv(deriv), nv(m->nv), nu(m->nu)
{

    printf("nQ: %d\tnV: %d\tnbody: %d\n", m->nq, m->nv, m->nbody);

    // initialize regularization parameters
    mu = 1e-3;
    mu_min = 1e-6;
    delta = 1.0;
    delta_0 = 2.0;

    // initialize value functions, etc.
    X.resize(T+1);
    U.resize(T);
    x.resize(T+1);
    u.resize(T);
    X_MinCost.resize(T+1);
    U_MinCost.resize(T);
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
//    for( int i = 0; i < 200; i++ )
//        mj_step(m, d);


    // run forward ilqr pass to initialize and more
    fwd_ctrl(true);



}


void ilqr::fwd_ctrl(bool init=false) {

    // run forward simulation for control inputs, save q & qdot and ...
    // calculate linear dynamics

    // copy data for now
    mjData* d_cp = mj_copyData(nullptr, m, d); // TODO: I think redefining this on every fwd_ctrl() callback is going to consume some time. There should be a better way for implementing this.

    Cost.reset_cost();

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

        // clip control inputs
        for( int j = 0; j < nu; j++ )
            U[t](j,0) = boost::algorithm::clamp(U[t](j,0), torque_lower_bound, torque_upper_bound);

        // instruct new controls
        mju_copy(d_cp->ctrl, U[t].data(), nu);

//        std::cout << "\nU: \t" << U[t] << std::endl;

//        printf("\n U[%d]: \t %f\n", t, U[t](0));
//        printf("\n U[%d]: \t %f\n", t, d_cp->ctrl[0]);

        // add cost
        Cost.add_cost(d_cp);

        // run with real dynamics
//        for( int i = 0; i < 5; i++ ){
//            mj_step1(m, d_cp);
//            mj_step2(m, d_cp);
//        }
        big_step(d_cp);

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

    if (!init)
        printf("\nalpha = %.3f\t cost = %.2f", alpha, Cost.cost_to_go);

    if (init) {
        printf("Cost at start: \t%f\n", Cost.cost_to_go);
        min_cost = Cost.cost_to_go;
        prev_cost = Cost.cost_to_go;
        for( int t = 0; t < T; t++ ) {

            x[t] = X[t];
            u[t] = U[t];
            X_MinCost[t] = X[t];
            U_MinCost[t] = U[t];

            mju_copy(d_cp->ctrl, X[t].data() + 2*nv, nu);
            mju_copy(d_cp->qpos, X[t].data(), nv);
            mju_copy(d_cp->qvel, X[t].data() + nv, nv);

            // get linear dynamics TODO: probably not the fastest parallelization
            do_derivatives(d_cp, t);

        }
        x[T] = X[T];
        X_MinCost[T] = X[T];
    }
    else if (decay_count < decay_limit || Cost.cost_to_go + 1e-2 < prev_cost) {
        // Hint: prev_cost is the min cost of this forward pass among various alphas,
        // while min_cost is the all-time lowest. Also note:
        // x, u --> prev_cost
        // X_MinCost, U_MinCost --> min_cost

        if ( decay_count == 0 || Cost.cost_to_go + 1e-2 < prev_cost ) {
            prev_cost = Cost.cost_to_go;
            for (int t = 0; t < T; t++) {
                x[t] = X[t];
                u[t] = U[t];
            }
            x[T] = X[T];
            if ( decay_count != 0 )
                printf("\nUpdated local min at decay: #%d", decay_count);
        }

        if (Cost.cost_to_go < min_cost) {
            for (int t = 0; t < T; t++) {
                X_MinCost[t] = X[t];
                U_MinCost[t] = U[t];
            }
            X_MinCost[T] = X[T];
            min_cost = Cost.cost_to_go;
        }
        alpha *= decay;
        decay_count++;
        fwd_ctrl(false);
    }
    else if (iter >= min_iter) {
        done = true;
        // re-init line-search params
        decay_count = 0;
        alpha = 1.0;
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

            // get linear dynamics TODO: probably not the fastest parallelization
            do_derivatives(d_cp, t);
        }
    }

    mj_deleteData(d_cp);

}

ilqr::~ilqr() {

}


void ilqr::bwd_lqr() {

    // start with V and v = 0
    Vxx = Cost.get_lxx(X[T]);
    Vx = Cost.get_lx(X[T]);
    Vxx = 0.5 * (Vxx + Vxx.transpose());

    bwd_flag = true; // true when bwd complete, false else

//    Vxx.setZero();
//    Vx.setZero();

    for( int t = T-1; t >= 0; t-- ){

        Quu = Cost.luu[t] + Fu[t].transpose() * (Vxx + mu * stateMat_t::Identity()) * Fu[t]; // + mu * actionMat_t::Identity();

        // check if Quu is positive (semi)definite
        Eigen::LLT<actionMat_t> lltofQuu(Quu);
        if (use_regularization)
        if(lltofQuu.info() == Eigen::NumericalIssue) {
            printf("\niter: \t%d \ntime: \t%d\n mu: \t%f\n", iter, t, mu);
            increase_mu();
            bwd_flag = false;
            break;
        }

        Qxx = Cost.lxx[t] + Fx[t].transpose() * Vxx * Fx[t];
        Qux = Fu[t].transpose() * (Vxx + mu * stateMat_t::Identity()) * Fx[t];

        Qx = Cost.lx[t] + Fx[t].transpose() * Vx;
        Qu = Cost.lu[t] + Fu[t].transpose() * Vx;

        // K = -Quu^-1 Qux
//        K[t] = Quu.fullPivHouseholderQr().solve(Qux) * (-1);
//        k[t] = Quu.fullPivHouseholderQr().solve(Qu) * (-1);

//        K[t] = Quu.llt().solve(Qux) * (-1);
//        k[t] = Quu.llt().solve(Qu) * (-1);

        K[t] = - Quu.inverse() * Qux;
        k[t] = - Quu.inverse() * Qu;

//        std::cout<< "\nQuu[" << t << "]: \n" << Quu << std::endl;
//        std::cout<< "\nQxx[" << t << "]: \n" << Qxx << std::endl;
//        std::cout<< "\nQux[" << t << "]: \n" << Qux << std::endl;
//        std::cout<< "\nVxx[" << t << "]: \n" << Vxx << std::endl;////
//        std::cout<< "\nFx[" << t << "]: \n" << Fx[t] << std::endl;
//        std::cout<< "\nFu[" << t << "]: \n" << Fu[t] << std::endl;
//        std::cout<< "\nK[" << t << "]: \n" << K[t] << std::endl;
//        std::cout<< "\nk[" << t << "]: \n" << k[t] << std::endl;
//        std::cout<< "\nlx[" << t << "]: \n" << Cost.lx[t] << std::endl;
//        std::cout<< "\nlu[" << t << "]: \n" << Cost.lu[t] << std::endl;

        // calc V and v
        Vxx = Qxx + K[t].transpose() * Quu * K[t] + K[t].transpose() * Qux + Qux.transpose() * K[t];
        Vx = Qx + K[t].transpose() * Quu * k[t] + K[t].transpose() * Qu + Qux.transpose() * k[t];
        Vxx = 0.5 * (Vxx + Vxx.transpose());

    }

}


void ilqr::do_derivatives(mjData* d, int t) {

    Fx[t].setZero();
    calc_derivatives(Fx[t].data(), Fu[t].data(), deriv, m, d, &Cost, t);
    Fx[t] *= step_ratio * m->opt.timestep;
    Fu[t] *= step_ratio * m->opt.timestep;
    Fx[t] += stateMat_t::Identity();
    Fx[t].block<DOFNUM, DOFNUM>(0, m->nv) += step_ratio * m->opt.timestep * Eigen::Matrix<mjtNum, DOFNUM, DOFNUM>::Identity();
    Cost.get_derivatives(d, t);

}


void ilqr::increase_mu() {

    delta = fmax(delta_0, delta * delta_0);
    mu = fmax(mu_min, mu * delta);

}


void ilqr::decrease_mu() {

    delta = fmin(1 / delta_0, delta / delta_0);
    if ( mu * delta > mu_min )
        mu = mu * delta;
    else
        mu = 0.0;

}


void ilqr::iterate() {

    mjtNum PC = prev_cost;
    do {
        do {
            bwd_lqr();
        } while (!bwd_flag);
        fwd_ctrl();
        if (PC > prev_cost)
            increase_mu();
    } while(PC < prev_cost || mu > max_mu );

    decrease_mu();

    if (done)
        printf("\nDone at iter %d.", iter);
    else {
        printf("\nMin Cost So Far:\t%.2f", min_cost);
        printf("\nCost at iter %d:\t%.2f\n", iter, prev_cost);
    }

    iter++;

}


void ilqr::big_step(mjData* d) {

    for( int t = 0; t < step_ratio; t++ )
        mj_step(m, d);

}

//}

void ilqr::RunMPC() {



}



























//bool ilqr::PositiveDefinite(actionMat_t Quu) {
//
//    if( Quu(0,0) < 0 )
//        return false;
//    if (Quu.block<2,2>(0,0).determinant() < 0 )
//        return false;
//    if (Quu.determinant() < 0)
//        return false;
//
//    return true;
//
