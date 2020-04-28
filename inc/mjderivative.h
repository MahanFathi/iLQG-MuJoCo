#pragma once

#include "mujoco.h"


void calcMJDerivatives(mjModel* m, mjData* dmain, mjtNum* deriv);
