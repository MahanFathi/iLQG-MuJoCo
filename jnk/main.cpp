// Runs the Visual Environment for a few Iterations
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <lindy.h>
#include "mujoco.h"
#include "ilqr.h"
#include <glfw3.h>




// MuJoCo data structures
mjModel *m = NULL;
mjData *d = NULL;
mjvCamera cam;                      // abstract camera
mjvOption opt;                      // visualization options
mjvScene scn;                       // abstract scene
mjrContext con;                     // custom GPU context


// mouse interaction
bool button_left = false;
bool button_middle = false;
bool button_right =  false;
double lastx = 0;
double lasty = 0;


// keyboard callback
void keyboard(GLFWwindow* window, int key, int scancode, int act, int mods)
{
    // backspace: reset simulation
    if( act==GLFW_PRESS && key==GLFW_KEY_BACKSPACE )
    {
        mj_resetData(m, d);
        mj_forward(m, d);
    }
}


// mouse button callback
void mouse_button(GLFWwindow* window, int button, int act, int mods)
{
    // update button state
    (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT)==GLFW_PRESS);
    button_middle = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE)==GLFW_PRESS);
    button_right =  (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT)==GLFW_PRESS);

    // update mouse position
    glfwGetCursorPos(window, &lastx, &lasty);
}


// mouse move callback
void mouse_move(GLFWwindow* window, double xpos, double ypos)
{
    // no buttons down: nothing to do
    if( !button_left && !button_middle && !button_right )
        return;

    // compute mouse displacement, save
    double dx = xpos - lastx;
    double dy = ypos - lasty;
    lastx = xpos;
    lasty = ypos;

    // get current window size
    int width, height;
    glfwGetWindowSize(window, &width, &height);

    // get shift key state
    bool mod_shift = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT)==GLFW_PRESS ||
                      glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT)==GLFW_PRESS);

    // determine action based on mouse button
    mjtMouse action;
    if( button_right )
        action = mod_shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
    else if( button_left )
        action = mod_shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V;
    else
        action = mjMOUSE_ZOOM;

    // move camera
    mjv_moveCamera(m, action, dx/height, dy/height, &scn, &cam);
}


// scroll callback
void scroll(GLFWwindow* window, double xoffset, double yoffset)
{
    // emulate vertical mouse motion = 5% of window height
    mjv_moveCamera(m, mjMOUSE_ZOOM, 0, -0.05*yoffset, &scn, &cam);
}




int main(int argc, char** argv) {

    // activate and load model
    mj_activate("mjkey.txt");


    mjModel* m = 0;
#if ACTNUM == 1 && DOFNUM == 2
    m = mj_loadXML("../model/inverted_pendulum.xml", NULL, NULL, 0);
    int t_0 = 120;
#endif
#if ACTNUM == 3 && DOFNUM == 6
    m = mj_loadXML("../model/hopper.xml", NULL, NULL, 0);
    int t_0 = atoi(argv[1]);
#endif

    // extract some handy parameters
    int nv = m->nv;
    int nu = m->nu;

    // make mjData for main thread and converging thread
    mjData* dmain = mj_makeData(m);
    mjData* d = mj_makeData(m);

    int T = 50; // lqr optimization horizon
    auto* deriv = (mjtNum*) mju_malloc(3*sizeof(mjtNum)*nv*nv);

    for( int i = 0; i < t_0; i++ )
        mj_step(m, dmain);

   // make an instance of ilqr
    ilqr ILQR(m, dmain, deriv, T, argv[1]);

//    for( int i = 0; i < 500; i++ ) {
//        printf("#################################\n");
//        printf("\t\tTIME:%d\n", i);
//        printf("#################################\n");
//        ILQR.RunMPC();
//    }

//    mju_scl(m->cam_pos0, m->cam_pos0, 20, 3);

    // init GLFW
    if( !glfwInit() )
        mju_error("Could not initialize GLFW");

    // create window, make OpenGL context current, request v-sync
    GLFWwindow* window = glfwCreateWindow(1200, 900, "Demo", NULL, NULL);
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    // initialize visualization data structures
    mjv_defaultCamera(&cam);
    mjv_defaultOption(&opt);
    mjr_defaultContext(&con);
    mjv_makeScene(&scn, 1000);                   // space for 1000 objects
    mjr_makeContext(m, &con, mjFONTSCALE_100);   // model-specific context

    // install GLFW mouse and keyboard callbacks
    glfwSetKeyCallback(window, keyboard);
    glfwSetCursorPosCallback(window, mouse_move);
    glfwSetMouseButtonCallback(window, mouse_button);
    glfwSetScrollCallback(window, scroll);

    cam.distance *= 1.5;


    int t = 0;
    for( int iter = 0; iter < 50; iter++ ) {
        d = mj_copyData(nullptr, m, dmain);
        t = 0;
        while (!glfwWindowShouldClose(window) && t < T) {
            // advance interactive simulation for 1/60 sec
            //  Assuming MuJoCo can simulate faster than real-time, which it usually can,
            //  this loop will finish on time for the next frame to be rendered at 60 fps.
            //  Otherwise add a cpu timer and exit this loop when it is time to render.
            mjtNum simstart = d->time;


            while( d->time - simstart < 1.0/200.0 && t < T){ // I've messed with the timing: 1.0/60.0
                if ( t < T ){
                    mju_copy(d->ctrl, ILQR.u[t].data(), nu);
                    t++;
                    ILQR.big_step(d);
                }

            }

            if (t == T)
                break;


            // get framebuffer viewport
            mjrRect viewport = {0, 0, 0, 0};
            glfwGetFramebufferSize(window, &viewport.width, &viewport.height);

            // update scene and render
            mjv_updateScene(m, d, &opt, NULL, &cam, mjCAT_ALL, &scn);
            mjr_render(viewport, &scn, &con);

            // swap OpenGL buffers (blocking call due to v-sync)
            glfwSwapBuffers(window);

            // process pending GUI events, call GLFW callbacks
            glfwPollEvents();
        }

        ILQR.iterate();

    }

    // close GLFW, free visualization storage
    glfwTerminate();
    mjv_freeScene(&scn);
    mjr_freeContext(&con);
    // free MuJoCo model and data, deactivate
    mj_deleteData(d);
    mj_deleteModel(m);
    mj_deactivate();


    // free memory
    mju_free(deriv);

    return 0;
}


