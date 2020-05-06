#pragma once

#include "mujoco.h"

typedef mjtNum (*stepCostFn_t)(const mjData*);

void calcMJDerivatives(mjModel* m, mjData* dmain, mjtNum* deriv, stepCostFn_t stepCostFn);
