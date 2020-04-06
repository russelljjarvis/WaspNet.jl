- [X] `Network` constructor method to take FF layers and make FF network without specifying connections
- [X] Generalize `Network` constructor function so all methods call to one primary method
- [ ] Utility to generate networks of varying widths
- [ ] Utility functions should not assume inputs are connected *directly* to neurons

- [ ] HH Neuron Model (including tests)
- [ ] QIF Neuron Model (including tests)
- [ ] FNH Neuron Model (including tests)

- [ ] Tests for all utility functions in multiple use-cases
- [X] Test deepcopy utility function
- [ ] Tests for heterogeneous networks

- [X] Update `Layer` to handle the weight matrices at construction-time for both Arrays + Block Arrays
- [X] Update `Layer` to preallocate space for input vectors to avoid re-allocation at every `update!`
- [X] Organize the `Layer` constructors better; agree on typical use case and write for that
- [ ] Test all `Layer` method

