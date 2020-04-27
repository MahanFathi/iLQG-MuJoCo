#include "mujoco.h"
#include <iostream>


constexpr float fps = 60.0;


void forwardStep(mjModel* model, mjData* data)
{
    mj_step(model, data);
}


void forwardFrame(mjModel* model, mjData* data)
{
    mjtNum simstart = data->time;
    while( data->time - simstart < 1.0/fps ) {
        forwardStep(model, data);
    }
}
