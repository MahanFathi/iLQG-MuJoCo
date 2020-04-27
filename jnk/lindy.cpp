
#include "mujoco.h"
#include <iostream>
#include <types.h>
#include <cost.h>



// enable compilation with and without OpenMP support
#if defined(_OPENMP)
#include <omp.h>
#else


// omp functions used below
void omp_set_dynamic(int) {}
void omp_set_num_threads(int) {}
int omp_get_num_procs(void) {return 1;}
#endif


// global variables: user-defined, with defaults
const int MAXTHREAD = 64;   // maximum number of threads allowed
int nthread = 0;            // number of parallel threads (default set later)
int niter = 30;             // fixed number of solver iterations for finite-differencing
int nwarmup = 3;            // center point repetitions to improve warmstart
double eps = 1e-6;          // finite-difference epsilon


// worker function for parallel finite-difference computation of derivatives
void _worker(mjtNum* deriv, const mjModel* m, const mjData* dmain, mjData* d, int id, cost* Cost, int t)
{
    int nv = m->nv;

    // allocate stack space for result at center
    mjMARKSTACK
    mjtNum* center = mj_stackAlloc(d, nv);
    mjtNum* warmstart = mj_stackAlloc(d, nv);

    // prepare static schedule: range of derivative columns to be computed by this thread
    int chunk = (m->nv + nthread-1) / nthread;
    int istart = id * chunk;
    int iend = mjMIN(istart + chunk, m->nv);

    // copy state and control from dmain to thread-specific d
    d->time = dmain->time;
    mju_copy(d->qpos, dmain->qpos, m->nq);
    mju_copy(d->qvel, dmain->qvel, m->nv);
    mju_copy(d->qacc, dmain->qacc, m->nv);
    mju_copy(d->qacc_warmstart, dmain->qacc_warmstart, m->nv);
    mju_copy(d->qfrc_applied, dmain->qfrc_applied, m->nv);
    mju_copy(d->xfrc_applied, dmain->xfrc_applied, 6*m->nbody);
    mju_copy(d->ctrl, dmain->ctrl, m->nu);

    // run full computation at center point (usually faster than copying dmain)
    mj_forward(m, d);

    // extra solver iterations to improve warmstart (qacc) at center point
    for( int rep=1; rep<nwarmup; rep++ )
        mj_forwardSkip(m, d, mjSTAGE_VEL, 1);

    // select output from forward dynamics
    mjtNum* acc = d->qacc;

    // save output for center point and warmstart (needed in forward only)
    mju_copy(center, acc, nv);
    mju_copy(warmstart, d->qacc_warmstart, nv);
    extraVec_t center_extra = Cost->get_extra(d);
    extraVec_t prtrbd_extra;

    // select target vector and original vector for force or acceleration derivative
    mjtNum* target = d->qfrc_applied;
    const mjtNum* original = dmain->qfrc_applied;

    // finite-difference over force: skip = mjSTAGE_VEL
    for( int i=istart; i<iend; i++ )
    {
        // perturb selected target
        target[i] += eps;

        // evaluate dynamics, with center warmstart
            mju_copy(d->qacc_warmstart, warmstart, m->nv);
            mj_forwardSkip(m, d, mjSTAGE_VEL, 1);

        // undo perturbation
        target[i] = original[i];

        // compute column i of derivative 2
        for( int j=0; j<nv; j++ )
            deriv[2*nv*nv + i + j*nv] = (acc[j] - center[j])/eps;
    }

    // finite-difference over velocity: skip = mjSTAGE_POS
    for( int i=istart; i<iend; i++ )
    {
        // perturb velocity
        d->qvel[i] += eps;

        // get perturbed extra
        prtrbd_extra = Cost->get_extra(d);
        for( int j=0; j < Cost->extra_dim; j++ )
            Cost->extra_deriv(j, i+nv) = (prtrbd_extra(j,0)-center_extra(j,0))/eps;

        // evaluate dynamics, with center warmstart
            mju_copy(d->qacc_warmstart, warmstart, m->nv);
            mj_forwardSkip(m, d, mjSTAGE_POS, 1);

        // undo perturbation
        d->qvel[i] = dmain->qvel[i];

        // compute column i of derivative 1
        for( int j=0; j<nv; j++ )
            deriv[nv*nv + i + j*nv] = (acc[j] - center[j])/eps;
    }

    // finite-difference over position: skip = mjSTAGE_NONE
    for( int i=istart; i<iend; i++ )
    {
        // get joint id for this dof
        int jid = m->dof_jntid[i];

        // get quaternion address and dof position within quaternion (-1: not in quaternion)
        int quatadr = -1, dofpos = 0;
        if( m->jnt_type[jid]==mjJNT_BALL )
        {
            quatadr = m->jnt_qposadr[jid];
            dofpos = i - m->jnt_dofadr[jid];
        }
        else if( m->jnt_type[jid]==mjJNT_FREE && i>=m->jnt_dofadr[jid]+3 )
        {
            quatadr = m->jnt_qposadr[jid] + 3;
            dofpos = i - m->jnt_dofadr[jid] - 3;
        }

        // apply quaternion or simple perturbation
        if( quatadr>=0 )
        {
            mjtNum angvel[3] = {0,0,0};
            angvel[dofpos] = eps;
            mju_quatIntegrate(d->qpos+quatadr, angvel, 1);
        }
        else
            d->qpos[m->jnt_qposadr[jid] + i - m->jnt_dofadr[jid]] += eps;

        // get perturbed extra TODO: Not sure if works for quaternions
        prtrbd_extra = Cost->get_extra(d);
        for( int j=0; j < Cost->extra_dim; j++ )
            Cost->extra_deriv(j, m->jnt_qposadr[jid] + i - m->jnt_dofadr[jid]) =
                    (prtrbd_extra(j,0)-center_extra(j,0))/eps;

        // evaluate dynamics, with center warmstart
        mju_copy(d->qacc_warmstart, warmstart, m->nv);
        mj_forwardSkip(m, d, mjSTAGE_NONE, 1);

        // undo perturbation
        mju_copy(d->qpos, dmain->qpos, m->nq);

        // compute column i of derivative 0
        for( int j=0; j<nv; j++ )
            deriv[i + j*nv] = (acc[j] - center[j])/eps;
    }

    mjFREESTACK
}


void get_derivs(mjtNum* deriv, mjModel* m, const mjData* dmain, cost* Cost, int t)
{
    // default nthread = number of logical cores (usually optimal)
    nthread = omp_get_num_procs();

    mjData* d[MAXTHREAD];
    for( int n=0; n<nthread; n++ )
        d[n] = mj_makeData(m);

    // set up OpenMP (if not enabled, this does nothing)
    omp_set_dynamic(0);
    omp_set_num_threads(nthread);

    // save solver options
    int save_iterations = m->opt.iterations;
    mjtNum save_tolerance = m->opt.tolerance;


    // set solver options for finite differences
    m->opt.iterations = niter;
    m->opt.tolerance = 0;

    // run worker threads in parallel if OpenMP is enabled
    #pragma omp parallel for schedule(static)
    for( int n=0; n<nthread; n++ )
        _worker(deriv, m, dmain, d[n], n, Cost, t);

    // set solver options for main simulation
    m->opt.iterations = save_iterations;
    m->opt.tolerance = save_tolerance;

    // free memory
    for( int n=0; n<nthread; n++ )
        mj_deleteData(d[n]);

}



void calc_derivatives(mjtNum* Fxdata, mjtNum* Fudata, mjtNum* deriv, mjModel* m, mjData* dmain, cost* Cost, int t){

    get_derivs(deriv, m, dmain, Cost, t);

//    mju_scl(deriv, deriv, 5*m->opt.timestep, 3*m->nv*m->nv);

    mjtNum dqpqv[2*m->nv*m->nv];
    mju_copy(dqpqv, deriv, 2*m->nv*m->nv);
//    mju_transpose(dqpqv, dqpqv, 2*m->nv, m->nv);
    mju_copy(Fxdata + 2*m->nv*m->nv, dqpqv, 2*m->nv*m->nv);

    mjtNum dctrl[m->nv*m->nu]; // dctrl is nv x nu
    mju_mulMatMatT(dctrl, deriv + 2*m->nv*m->nv, dmain->actuator_moment, m->nv, m->nv, m->nu);
//    mju_transpose(dctrl,dctrl,m->nu,m->nv);
    mju_copy(Fudata + m->nv*m->nu, dctrl, m->nv*m->nu);


}
