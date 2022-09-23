# neural_nets

> My study on neural networks during my Computer Science MSc.

I started this project to learn more of the rust's ecosystem, such as testing, and documentation features, and also to test my know-how on some neural networks topics.

## Tasks

- [ ] Add docs for [logistic regression](src/package/logistic_regression.rs) functions.
- [ ] Add docs for [linear regression](src/package/linear_regression.rs) functions.

## Considering

> Nothing here yet!

## Need help / future

- Save datasets for testing using `json` or something else, instead of declaring it on the test module, as in [here](src/package/linear_regression.rs#L88)
- Find a better way to test methods like linear regression. Currently I'm asserting the `mean squared error` is below a given threshold.
