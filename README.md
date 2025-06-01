# LinearRegressionJava
Just a simple Java program implementing linear regression, optimizers, and regularization.

## Example Linear Regression training with Stochastic Gradient Descent:
### Initialize loss function (mean squared error) and model:
```Mse mse = new Mse();
 LinearRegression model = new LinearRegression();
```
### Optional regularization for the loss:
```
// l1 regularization with lambda = 2
Regularization reg = new Regularization(1, 2);
Mse mse_reg = new Mse(reg);
```

### Initialize optimizer:
```
double learning_rate = 0.00001;
byte batch_size = (byte)4;
int n_epochs = 200;

StochasticGradientDescent sgd = new StochasticGradientDescent(
      mse_reg, learning_rate, batch_size, n_epochs, model
);
```

### Train with features x and targets y:
```
model.fit(x_mat, y_mat, sgd);
```


