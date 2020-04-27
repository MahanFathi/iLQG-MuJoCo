//
// Created by mahan on 12/25/17.
//

#ifndef ILGQ_LINDY_H
#define ILGQ_LINDY_H

#include "types.h"
#include "cost.h"
#include <mjmodel.h>
#include <mjdata.h>

//void _worker(mjtNum* deriv, const mjModel* m, const mjData* dmain, mjData* d, int id, cost* Cost);

// calculates raw derivatives
void get_derivs(mjtNum* deriv, mjModel* m, const mjData* dmain, cost* Cost, int t);

// turns raw derivs into Eigen matrices
void calc_derivatives(mjtNum* Fxdata, mjtNum* Fudata, mjtNum* deriv, mjModel* m, mjData* dmain, cost* Cost, int t);

#endif //ILGQ_LINDY_H
