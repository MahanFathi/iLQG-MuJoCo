#pragma once

#include "mujoco/mujoco.h"


void forwardStep(mjModel* model, mjData* data);

void forwardFrame(mjModel* model, mjData* data);
