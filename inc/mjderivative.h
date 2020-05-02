#pragma once

#include "mujoco.h"

typedef mjtNum (*stepCostFn_t)(mjData*);

void cpMjData(const mjModel* m, mjData* d_dest, const mjData* d_src);
void calcMJDerivatives(mjModel* m, mjData* dmain, mjtNum* deriv, stepCostFn_t stepCostFn);
