#pragma once

#include "mujoco/mujoco.h"


void cpMjData(const mjModel* m, mjData* d_dest, const mjData* d_src);
