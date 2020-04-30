#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "mujoco.h"


// enable compilation with and without OpenMP support
#if defined(_OPENMP)
    #include <omp.h>
#else
    // omp timer replacement
    #include <chrono>
    double omp_get_wtime(void)
    {
        static std::chrono::system_clock::time_point _start = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed = std::chrono::system_clock::now() - _start;
        return elapsed.count();
    }

    // omp functions used below
    void omp_set_dynamic(int) {}
    void omp_set_num_threads(int) {}
    int omp_get_num_procs(void) {return 1;}
#endif


// gloval variables: internal
const int MAXTHREAD = 16;   // maximum number of threads allowed


// global variables: user-defined, with defaults
static int nthread = 4;            // number of parallel threads (default set later)
static int niter = 30;             // fixed number of solver iterations for finite-differencing
static int nwarmup = 3;            // center point repetitions to improve warmstart
static double eps = 1e-6;          // finite-difference epsilon


void cpMjData(const mjModel* m, mjData* d_dest, const mjData* d_src)
{
    d_dest->time = d_src->time;
    mju_copy(d_dest->qpos, d_src->qpos, m->nq);
    mju_copy(d_dest->qvel, d_src->qvel, m->nv);
    mju_copy(d_dest->qacc, d_src->qacc, m->nv);
    mju_copy(d_dest->qacc_warmstart, d_src->qacc_warmstart, m->nv);
    mju_copy(d_dest->qfrc_applied, d_src->qfrc_applied, m->nv);
    mju_copy(d_dest->xfrc_applied, d_src->xfrc_applied, 6*m->nbody);
    mju_copy(d_dest->ctrl, d_src->ctrl, m->nu);
}


// worker function for parallel finite-difference computation of derivatives
void worker(const mjModel* m, const mjData* dmain, mjData* d, int id, mjtNum* deriv)
{
    int nv = m->nv;
    int nu = m->nu;

    // assumption: nv >= nu

    // allocate stack space for result at center
    mjMARKSTACK
    mjtNum* temp = mj_stackAlloc(d, nv);
    mjtNum* warmstart = mj_stackAlloc(d, nv);

    // prepare static schedule: range of derivative columns to be computed by this thread
    int chunk = (m->nv + nthread-1) / nthread;
    int istart = id * chunk;
    int iend = mjMIN(istart + chunk, nv);

    // copy state and control from dmain to thread-specific d
    cpMjData(m, d, dmain);

    // run full computation at center point (usually faster than copying dmain)
    mj_forward(m, d);

    // extra solver iterations to improve warmstart (qacc) at center point
    for( int rep=1; rep<nwarmup; rep++ )
        mj_forwardSkip(m, d, mjSTAGE_VEL, 1);

    // set output forward dynamics
    mjtNum* output = d->qacc;

    // save output for center point and warmstart (needed in forward only)
    mju_copy(warmstart, d->qacc_warmstart, nv);

    // finite-difference over force or acceleration: skip = mjSTAGE_VEL
    for( int i=istart; i<iend; i++ )
    {
        // check limit, due to the assumption nv > nu
        if (i >= nu)
            break;

        // perturb selected target +
        d->ctrl[i] += eps;

        // evaluate dynamics, with center warmstart
        mju_copy(d->qacc_warmstart, warmstart, m->nv);
        mj_forwardSkip(m, d, mjSTAGE_VEL, 1);

        // copy and store +perturbation
        mju_copy(temp, output, nv);

        // perturb selected target -
        d->ctrl[i] = dmain->ctrl[i] - eps;

        // evaluate dynamics, with center warmstart
        mju_copy(d->qacc_warmstart, warmstart, m->nv);
        mj_forwardSkip(m, d, mjSTAGE_VEL, 1);

        // undo perturbation
        d->ctrl[i] = dmain->ctrl[i];

        // compute column i of derivative 2
        for( int j=0; j<nv; j++ )
            deriv[2*nv*nv + i + j*nu] = (temp[j] - output[j])/(2*eps);
    }

    // finite-difference over velocity: skip = mjSTAGE_POS
    for( int i=istart; i<iend; i++ )
    {
        // perturb velocity +
        d->qvel[i] += eps;

        // evaluate dynamics, with center warmstart
        mju_copy(d->qacc_warmstart, warmstart, m->nv);
        mj_forwardSkip(m, d, mjSTAGE_POS, 1);

        // copy and store +perturbation
        mju_copy(temp, output, nv);

        // perturb velocity -
        d->qvel[i] = dmain->qvel[i] - eps;

        // evaluate dynamics, with center warmstart
        mju_copy(d->qacc_warmstart, warmstart, m->nv);
        mj_forwardSkip(m, d, mjSTAGE_POS, 1);

        // undo perturbation
        d->qvel[i] = dmain->qvel[i];

        // compute column i of derivative 1
        for( int j=0; j<nv; j++ )
            deriv[nv*nv + i + j*nv] = (temp[j] - output[j])/(2*eps);
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

        // apply quaternion or simple perturbation +
        if( quatadr>=0 )
        {
            mjtNum angvel[3] = {0,0,0};
            angvel[dofpos] = eps;
            mju_quatIntegrate(d->qpos+quatadr, angvel, 1);
        }
        else
            d->qpos[m->jnt_qposadr[jid] + i - m->jnt_dofadr[jid]] += eps;

        // evaluate dynamics, with center warmstart
        mju_copy(d->qacc_warmstart, warmstart, m->nv);
        mj_forwardSkip(m, d, mjSTAGE_NONE, 1);

        // copy and store +perturbation
        mju_copy(temp, output, nv);

        // undo perturbation
        mju_copy(d->qpos, dmain->qpos, m->nq);

        // apply quaternion or simple perturbation +
        if( quatadr>=0 )
        {
            mjtNum angvel[3] = {0,0,0};
            angvel[dofpos] = -eps;
            mju_quatIntegrate(d->qpos+quatadr, angvel, 1);
        }
        else
            d->qpos[m->jnt_qposadr[jid] + i - m->jnt_dofadr[jid]] -= eps;

        // evaluate dynamics, with center warmstart
        mju_copy(d->qacc_warmstart, warmstart, m->nv);
        mj_forwardSkip(m, d, mjSTAGE_NONE, 1);

        // undo perturbation
        mju_copy(d->qpos, dmain->qpos, m->nq);

        // compute column i of derivative 0
        for( int j=0; j<nv; j++ )
            deriv[i + j*nv] = (temp[j] - output[j])/(2*eps);
    }

    mjFREESTACK
}


void calcMJDerivatives(mjModel* m, mjData* dmain, mjtNum* deriv)
{

    // deriv: dacc/dpos, dacc/dvel, dacc/dctrl
    // default nthread = number of logical cores (usually optimal)
    nthread = omp_get_num_procs();

    // make mjData: per-thread
    mjData* d[MAXTHREAD];
    for( int n=0; n<nthread; n++ )
        d[n] = mj_makeData(m);

    // allocate derivatives
    // deriv = (mjtNum*) mju_malloc(m->nv*(2*m->nv+m->nu)*sizeof(mjtNum));

    // set up OpenMP (if not enabled, this does nothing)
    omp_set_dynamic(0);
    omp_set_num_threads(nthread);

    // save solver options
    int save_iterations = m->opt.iterations;
    mjtNum save_tolerance = m->opt.tolerance;


    // set solver options for main simulation
    m->opt.iterations = save_iterations;
    m->opt.tolerance = save_tolerance;

    // set solver options for finite differences
    m->opt.iterations = niter;
    m->opt.tolerance = 0;

    // run worker threads in parallel if OpenMP is enabled
    #pragma omp parallel for schedule(static)
    for( int n=0; n<nthread; n++ )
        worker(m, dmain, d[n], n, deriv);

    for( int n=0; n<nthread; n++ )
        mj_deleteData(d[n]);

    // reset solver options for simulation
    m->opt.iterations = save_iterations;
    m->opt.tolerance = save_tolerance;
}
