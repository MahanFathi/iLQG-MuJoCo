//
// Created by mahan on 12/31/17.
//

#ifndef ILGQ_TYPES_H
#define ILGQ_TYPES_H

#include <Eigen/Dense>
#include <Eigen/StdVector>
#include "mjmodel.h"

#define DOFNUM 2
#define ACTNUM 1

typedef Eigen::Matrix<mjtNum, 2*DOFNUM, 1> stateVec_t;
typedef Eigen::Matrix<mjtNum, 2*DOFNUM, 2*DOFNUM, Eigen::RowMajor> stateMat_t;



typedef Eigen::Matrix<mjtNum, ACTNUM, 1> actionVec_t;
typedef Eigen::Matrix<mjtNum, ACTNUM, ACTNUM> actionMat_t;


typedef Eigen::Matrix<mjtNum, ACTNUM, 2*DOFNUM> action_state_Mat_t;
typedef Eigen::Matrix<mjtNum, 2*DOFNUM, ACTNUM> state_action_Mat_t;
//typedef Eigen::Matrix<mjtNum, 2*DOFNUM, 2*DOFNUM+ACTNUM, Eigen::ColMajor> state_stateaction_Mat_t; // note that it's column major; for the sake of mujoco deriv code.


typedef std::vector<stateVec_t, Eigen::aligned_allocator<stateVec_t>> stateThread; // Usage: X, lx
typedef std::vector<actionVec_t, Eigen::aligned_allocator<actionVec_t>> actionThread; // Usage: U, k, lu
typedef std::vector<stateMat_t, Eigen::aligned_allocator<stateMat_t>> stateMatThread; // Usage: Fx, lxx
typedef std::vector<actionMat_t, Eigen::aligned_allocator<actionMat_t>> actionMatThread; // Usage: luu
typedef std::vector<state_action_Mat_t, Eigen::aligned_allocator<state_action_Mat_t>> state_action_MatThread; // Usage: Fu
typedef std::vector<action_state_Mat_t, Eigen::aligned_allocator<action_state_Mat_t>> action_state_MatThread; // Usage: K


#endif //ILGQ_TYPES_H
