//
// Created by mahan on 12/27/17.
//

#ifndef ILGQ_COST_H
#define ILGQ_COST_H

#include "types.h"
#include <iostream>
#include "mjmodel.h"
#include "mjdata.h"
#include <Eigen/Dense>

class cost {
public:

    const mjModel* m;
    int T;                  // horizon

    stateThread lx;
    actionThread lu;
    stateMatThread lxx;
    actionMatThread luu;
    // lux is assumed to be zero throughout this project

    stateMat_t Vxx;
    stateVec_t Vx;

    mjtNum cost_to_go;      // total accumulative cost


    /*      ASSUMING THIS CASE TO BE THE HOPPER       */
    #if ACTNUM == 1
    #if DOFNUM == 2
        float k = 1;          // relative weight in cost
    #endif
    #endif


    /*      ASSUMING THIS CASE TO BE THE HOPPER       */
    #if ACTNUM == 3
    #if DOFNUM == 6
        float k = 1;
        // float stance_z = 1.25;
        float torso = 0.4;
        float thigh = 0.45;
        float leg = 0.5;
    #endif
    #endif


    cost(const char *env, const mjModel* m, int T);

    void calc_costmats(const mjData* d, int t);

    void add_cost(const mjData* d);

    void reset_cost();

    stateVec_t get_lx(stateVec_t x);

    stateMat_t get_lxx(stateVec_t x);
};


#endif //ILGQ_COST_H
