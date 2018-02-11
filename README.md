# iLQG-MuJoCo
Iterative LQG for a couple of MuJoCo models

### Demo for _Inverted Pendulum_ and _Hopper_
A few iterations before convergence is shown here. Hopper in particular does some crazy stuff, but it will get there (no wonder why evolution wiped out these guys). This whole process below is considered as a signle MPC run. You can set contact solref[0] to a slightly higher number in the model, for more smooth dynamics and hence better derivative behaviors. 

<img src="https://i.imgur.com/kOUsrXA.gif" width="200"> <img src="https://i.imgur.com/8z8WXNd.gif" width="200"> 

### This Repo Contains:
- Iterative LQR algorithm, based on this [paper](https://homes.cs.washington.edu/~todorov/papers/TassaIROS12.pdf
): 
  - Improved Regularization 
  - Backtrack Linesearch 
- Finite difference calculation of derivatives, in **parallel**
- The Levenberg-Marquardt heuristic
- Hessian approximation from gradients 
- MPC mode

### Dependencies
- [MuJoCo](MuJoCo.org)
- [Eigen 3](http://eigen.tuxfamily.org)
- OpenGL 

### Usage 
`mjpro151` is also included in this repo. You just have to copy your `mjkey.txt` in the `bin` folder. Make excecutables with `make` at `./MakeFile` and run with `./bin/main (some number for init)`. As I'm using fixed-size eigen matrices in `./include/types.h`, you need to modify this according to your model, and build again.   

### TODOs
- Make it faster of course 
- Extend to walking robots 
- Use proposed cost function from the paper 

### Acknowledgement 
Big thanks to Taylor Apgar, from Dynamic Robotics Laboratory at Oregon State University. 

