#pragma once


#include "mujoco/mujoco.h"


inline mjtNum stepCost(const mjData* d)
{
    mjtNum cost;
    cost =
        1.0  * d->qpos[0] * d->qpos[0] +
        10.0 * d->qpos[1] * d->qpos[1] +
        1.0  * d->qvel[0] * d->qvel[0] +
        10.0 * d->qvel[1] * d->qvel[1] +
        1.0  * d->ctrl[0] * d->ctrl[0];
    return cost;
}
