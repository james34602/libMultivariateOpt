## Gradient free optimization
### Features

- Differential evolution

Bezier Search, almost parameter free, allow bound constraints

- Nelder-Mead Simplex search with bound constraints

- Flower pollination search

Standard approach, allow bound constraints

### Examples

- Simple function optimization
- Nth bands digital peaking filter curve fitting, 2N order cascade form IIR filter (Real world application)
- TO BE ADDED

#### Q/A

- Can I add nonlinear/linear constraints other than just bound constraints?

Yes, those are problem dependent, you have to modify your cost function in some ways.

- Why are the optimizer function argument could be that long?

We allow user to choose probability density functions(PDF), different PDF affects the random number distribution of the optmizer itself.
Last argument is callback function pointer, it allows programmer display the process of optimization to GUI, if there is no GUI, pass NULL, if there is a GUI, implement your own callback function.
