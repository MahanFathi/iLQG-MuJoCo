//
// Created by mahan on 12/31/17.
//

#ifndef ILGQ_ANIM_H
#define ILGQ_ANIM_H

#include "glfw3.h"
#include "mjrender.h"
#include "mjvisualize.h"
#include "mjdata.h"
#include "mjmodel.h"

// MuJoCo data structures
extern mjModel *m;
extern mjData *d;
extern mjvCamera cam;                      // abstract camera
extern mjvOption opt;                      // visualization options
extern mjvScene scn;                       // abstract scene
extern mjrContext con;                     // custom GPU context

// mouse interaction
extern bool button_left = false;
extern bool button_middle = false;
extern bool button_right =  false;
extern double lastx = 0;
extern double lasty = 0;

void keyboard(GLFWwindow* window, int key, int scancode, int act, int mods);

void mouse_button(GLFWwindow* window, int button, int act, int mods);

void mouse_move(GLFWwindow* window, double xpos, double ypos);

void scroll(GLFWwindow* window, double xoffset, double yoffset);

#endif //ILGQ_ANIM_H
