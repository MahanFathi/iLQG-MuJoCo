#include "inverted_pendulum/inverted_pendulum.h"
#include "inverted_pendulum/cost.h"



InvertedPendulum::InvertedPendulum(mjModel* m, mjData* d):
    m(m), d(d)
{
    // cost function
    stepCostFn = stepCost;

    for (auto i = 0; i < 10; i++)
        mj_step(m, d);

    // setup iLQR
    iLQR = new ILQR<nv, nu, N>(m, d, stepCostFn);
}

void InvertedPendulum::forward()
{
    iLQR->setDInit(d);
    for (int i = 0; i < maxIterUtilConvergence; i++)
        iLQR->iterate();

    // get first u
    mju_copy(d->ctrl, iLQR->dArray[N]->ctrl, nu);

    // proceed simulation
    mj_step(m, d);
}
