#pragma once

#include <iostream>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include "mujoco.h"

#include "mjderivative.h"
#include "differentiator.h"
#include "util.h"


template<int nv, int nu, int N>
class ILQR
{
public:

    // typedefs for env matrices/vectors
    typedef Eigen::Matrix<mjtNum, 2*nv, 2*nv> A_t;
    typedef Eigen::Matrix<mjtNum, 2*nv, nu> B_t;
    typedef Eigen::Matrix<mjtNum, 2*nv, 1> x_t;
    typedef Eigen::Matrix<mjtNum, nu, 1> u_t;
    // typedefs for env maps
    typedef Eigen::Map<Eigen::Matrix<mjtNum, nv, nv>> dqdq_mt;
    typedef Eigen::Map<Eigen::Matrix<mjtNum, nv, nu>> dqdu_mt;
    typedef Eigen::Map<Eigen::Matrix<mjtNum, nv, 1>> qpos_mt;
    typedef Eigen::Map<Eigen::Matrix<mjtNum, nv, 1>> qvel_mt;
    typedef Eigen::Map<Eigen::Matrix<mjtNum, nv, 1>> ctrl_mt;
    typedef Eigen::Map<Eigen::Matrix<mjtNum, 2*nv, 1>> x_mt;
    typedef Eigen::Map<Eigen::Matrix<mjtNum, nu, 1>> u_mt;
    typedef Eigen::Map<Eigen::Matrix<mjtNum, 1, 2*nv>> q_mt;
    typedef Eigen::Map<Eigen::Matrix<mjtNum, 1, nu>> r_mt;
    // typedefs specific to iLQR (redundancies are only for better traceability)
    typedef Eigen::Matrix<mjtNum, nu, 2*nv> K_t;
    typedef Eigen::Matrix<mjtNum, nu, 1> k_t;
    typedef Eigen::Matrix<mjtNum, 2*nv, 2*nv> V_t;
    typedef Eigen::Matrix<mjtNum, 1, 2*nv> v_t;
    typedef Eigen::Matrix<mjtNum, 2*nv, 2*nv> Q_t;
    typedef Eigen::Matrix<mjtNum, nu, nu> R_t;


    /*      Data     */
    // MuJoCo model and data
    mjModel* m;
    mjData* d = NULL;

    // of course the differentiator
    Differentiator<nv, nu>* differentiator;

    // intermediate state/ctrl storage
    mjData* dArray[N+1];     // d[N]: init state,    d[0]: landing state
    mjData* dArrayBuffer[N+1];     // d[N]: init state,    d[0]: landing state

    // to store data when solving in backward pass
    V_t* V; v_t* v;
    K_t K[N+1]; k_t k[N+1]; // we don't need K/k[0]

    // vectors mapped to (qpos, qvel) and ctrl
    x_mt* x;
    u_mt* u;
    x_mt* xStar;
    u_mt* uStar;

    // store step cost function
    stepCostFn_t stepCostFn;

    // Levenberg-Marquardt parameters
    mjtNum muMin = 1e-6;
    mjtNum delta0 =  2.0;
    mjtNum mu = 1000.0;
    mjtNum delta;

    // line search
    mjtNum alpha = 1.0;
    mjtNum alphaDecay = 0.8;


    /*      Funcs    */
    ILQR(mjModel* m, mjData* dmain, stepCostFn_t &stepCostFn):
        m(m), stepCostFn(stepCostFn)
    {
        // set up d
        d = mj_makeData(m);
        setDInit(dmain);

        // set up differentiator
        differentiator = new Differentiator<nv, nu>(m, d, stepCostFn);

        // initialize trajectory:
        //      dmain is just the initial state
        //      d[*] are the optimization results
        for (int n = N; n >= 0; n--)
        {
            dArray[n] = mj_makeData(m);
            dArrayBuffer[n] = mj_makeData(m);
            cpMjData(m, dArray[n], d);
            mj_step(m, d);
        }   // note that d is shit now

        // bind x and u vectors to u
        x = new x_mt(d->qpos);   // note that qpos and qvel are contiguous in memory
        u = new u_mt(d->ctrl);
        xStar = new x_mt(d->qpos);   // note that qpos and qvel are contiguous in memory
        uStar = new u_mt(d->ctrl);

        V = new V_t;
        v = new v_t;
    }


    virtual void initV()
    {
        differentiator->setMJData(dArray[0]);
        differentiator->updateDerivatives();

        *v = *(differentiator->dgdx);
        (*V).noalias() = (*v).transpose()*(*v);
        (*v).setZero();
    }


    void setDInit(mjData* dInit)
    {
        cpMjData(m, d, dInit);
    }


    bool forwardPass()
    {   // apply control policies in K and k

        // NOTE: (important) mjData* d, should be set to target initial state before *each* forwardPass()

        // keep track of total cost
        initV();
        mjtNum cost(0.0);
        mjtNum costStar(0.0);

        for (int n = N; n >= 0; n--)
        {
            // bind xStar to reference point
            new (xStar) x_mt(dArray[n]->qpos);
            new (uStar) u_mt(dArray[n]->ctrl);
            (*u).noalias() = K[n] * (*x - *xStar) + (alpha * k[n].array()).matrix() + *uStar;
            costStar += stepCostFn(dArray[n]);
            cost += stepCostFn(d);
            if (n == 0)
            {
                costStar += ((*xStar).transpose() * (*V) * (*xStar))(0,0);
                costStar += ((*v) * (*xStar))(0,0);
                cost += ((*x).transpose() * (*V) * (*x))(0,0);
                cost += ((*v) * (*x))(0,0);
            }
            cpMjData(m, dArrayBuffer[n], d);
            mj_step(m, d);
        }   // since K/k[0] are shit, for dArray[0] only the state is valid

        bool success = (cost <= costStar);
        if (success)
            for (int n = N; n >= 0; n--)
                cpMjData(m, dArray[n], dArrayBuffer[n]);
        return !success;
    }


    bool backwardPass()
    {   // calculate K and k

        Q_t Q; R_t R; x_t c;
        static A_t &A = *(differentiator->A);
        static B_t &B = *(differentiator->B);
        static q_mt &q = *(differentiator->dgdx);
        static r_mt &r = *(differentiator->dgdu);

        initV();

        for (int n = 1; n <= N; n++)
        {
            std::cout << "----------" << '\n';
            std::cout << "n: \t" << n << '\n';

            // symmetric V
            (*V) = (*V + (*V).transpose().eval()).array() / 2;

            // get derivatives
            differentiator->setMJData(dArray[n]);
            differentiator->updateDerivatives();

            // approximate Hessians
            Q = q.transpose() * q;
            R = r.transpose() * r;

            // get c, i.e. x_{n-1}^* - x_n^*
            x_mt xn1(dArray[n-1]->qpos);
            x_mt xn(dArray[n]->qpos);
            c = xn1 - xn;

            // create inverse matrix
            (*V).diagonal().array() += mu;
            Eigen::LDLT<Eigen::Matrix<mjtNum, nu, nu>> ldlt = (2*B.transpose()*(*V)*B + 2*R).ldlt();
            if (!ldlt.isPositive() && nu != 1)
            {
                std::cout << mu << '\n';
                std::cout << (2*B.transpose()*(*V)*B + 2*R) << '\n';
                return true;
            }
            (*V).diagonal().array() -= mu;

            // claculate K & k
            K[n].noalias() = ldlt.solve(-2*B.transpose()*(*V)*A);
            k[n].noalias() = ldlt.solve(-B.transpose()*((*v).transpose()+2*(*V)*c)-r.transpose());

            // calculate V & v
            *V = (A+B*K[n]).transpose()*(*V)*(A+B*K[n])+Q+K[n].transpose()*R*K[n];
            *v = 2*(k[n].transpose()*B.transpose()+c.transpose())*(*V)*(A+B*K[n])+(*v)*(A+B*K[n])+q+2*k[n].transpose()*R*K[n];
        }
        return false;
    }


    void iterate()
    {
        // forward to get [(x*, u*), ...]
        while(forwardPass())
        {
            alpha *= alphaDecay;
            setDInit(dArray[N]);
        }
        alpha = 1.0;
        setDInit(dArray[N]);

        // backward to get [(K, k), ...]
        while (backwardPass())
            increaseMu();
        decreaseMu();
    }


    void increaseMu()
    {
        delta = mjMAX(delta0, delta*delta0);
        mu = mjMAX(muMin, mu*delta);
    }
    void decreaseMu()
    {
        delta = mjMIN(1.0/delta0, delta/delta0);
        if (mu*delta > muMin)
            mu = mjMAX(muMin, mu*delta);
        if (mu*delta <= muMin)
            mu = 0.0;
    }
};
