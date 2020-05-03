#pragma once


#include <eigen3/Eigen/Core>
#include "mujoco.h"

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
    mjData* d;

    // of course the differentiator
    Differentiator<nv, nu>* differentiator;

    // intermediate state/ctrl storage
    mjData* dArray[N+1];     // d[N]: init state,    d[0]: landing state

    // to store data when solving in backward pass
    V_t* V; v_t* v;
    K_t K[N+1]; k_t k[N+1]; // we don't need K/k[0]

    // vectors mapped to (qpos, qvel) and ctrl
    x_mt* x;
    u_mt* u;


    /*      Funcs    */
    ILQR(mjModel* m, mjData* d, stepCostFn_t &stepCostFn):
        m(m), d(d)
    {
        // set up differentiator
        differentiator = new Differentiator<nv, nu>(m, d);

        // initialize trajectory:
        //      dmain is just the initial state
        //      d[*] are the optimization results
        for (int n = N; n >= 0; n--)
        {
            dArray[n] = mj_makeData(m);
            cpMjData(m, dArray[n], d);
            mj_step(m, d);
        }   // note that d is shit now

        // bind x and u vectors to u
        x = new x_mt(d->qpos);   // note that qpos and qvel are contiguous in memory
        u = new u_mt(d->ctrl);
    }


    virtual void initV()
    {
        (*V).setZero();
        (*v).setZero();
    }


    void forwardPass()
    {   // apply control policies in K and k

        // set init state
        cpMjData(m, d, dArray[N]);

        x_mt xStar;
        u_mt uStar;

        for (int n = N; n >= 0; n--)
        {
            // bind xStar to reference point
            new (&xStar) x_mt(dArray[n]->qpos);
            new (&uStar) u_mt(dArray[n]->ctrl);
            (*u).noalias() = K[n] * (*x - xStar) + k[n] + uStar;
            cpMjData(m, dArray[n], d);
            mj_step(m, d);
        }   // since K/k[0] are shit, for dArray[0] only the state is valid
    }


    void backwardPass()
    {   // calculate K and k

        Q_t Q; R_t R; x_t c;
        static A_t &A = *(differentiator->A);
        static B_t &B = *(differentiator->B);
        static x_t &q = *(differentiator->dgdx);
        static u_t &r = *(differentiator->dgdu);

        initV();

        for (int n = 1; n <= N; n++)
        {
            // symmetric V
            (*V) = (*V + (*V).transpose().eval()).array() / 2;

            // get derivatives
            differentiator->setMJData(dArray[n]);
            differentiator->updateDerivatives();

            // approximate Hessians
            Q = q * q.transpose();
            R = r * r.transpose();

            // get c, i.e. x_{n-1}^* - x_n^*
            x_mt xn1(dArray[n-1]->qpos);
            x_mt xn(dArray[n]->qpos);
            c = xn1 - xn;

            // claculate K & k
            Eigen::ColPivHouseholderQR<R_t> temp = (-2*B.transpose()*(*V)*B+2*R).colPivHouseholderQr();
            K[n] = temp.solve(2*B.transpose()*(*V)*A);
            k[n] = temp.solve(B.transpose()*((*v).transpose()+2*(*V)*c));

            // calculate V & v
            *V = (A+B*K[n]).transpose()*(*V)*(A+B*K[n])+Q+K[n].transpose()*R*K[n];
            *v = 2*(k[n].transpose()*B.transpose()+c.transpose())*(*V)*(A+B*K[n])+(*v)*(A+B*K[n])+q+2*k[n]*R*K[n];
        }
    }

};
