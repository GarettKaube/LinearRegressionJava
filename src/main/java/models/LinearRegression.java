package models;

import lossfunctions.Mse;
import optimization.StochasticGradientDescent;
import optimization.Utils;
import org.ejml.simple.SimpleMatrix;
import optimization.Optimizer;

import java.util.Random;

public class LinearRegression extends Model {
    public LinearRegression() {
    }

    public void initParams() {
        int p = this.X.getNumCols();
        Random rand = new Random();
        this.params = SimpleMatrix.randomNormal(SimpleMatrix.identity(p).scale(0.01), rand);
    }

    public void initParams(SimpleMatrix X) {
        int p = X.getNumCols();
        Random rand = new Random();
        this.params = SimpleMatrix.randomNormal(SimpleMatrix.identity(p).scale(0.01), rand);
    }

    public SimpleMatrix predict(SimpleMatrix X) {
        return X.mult(this.params);
    }

    public SimpleMatrix predict(SimpleMatrix X, SimpleMatrix theta) {
        return X.mult(theta);
    }

    /**
     * Fits the model with the provided Optimizer
     * @param x Data
     * @param y Target
     * @param optimizer
     */
    public void fit(SimpleMatrix x, SimpleMatrix y, Optimizer optimizer) {
        this.X = x;
        this.y = y;

        // Initialize model parameters and pass them to optimizer
        this.initParams();
        optimizer.setParams(this.params);

        // Optimize and set the models params to the optimized ones
        optimizer.optimize(x, y);
        this.setParams(optimizer.params);
    }

    /**
     * Implements the closed form solution for the params of linear regression
    */
    public void fit(SimpleMatrix x, SimpleMatrix y) {
        int n = x.getNumCols();
        SimpleMatrix theta = x.transpose().mult(x).invert().mult(x.transpose()).mult(y);
        this.setParams(theta);
    }

    /**
    Computes the gradient w.r.t the parameters for the model y = XB grad(XB) = X
     */
    @Override
    public SimpleMatrix getGrad(SimpleMatrix x) {
        return x;
    }

    public static void main(String[] args) {
        models.LinearRegression model = new models.LinearRegression();
        Utils utils = new Utils();
        Random rand = new Random();

        // create sample data generated by linear process
        int n_samples = 5000;
        double[][] theta = {{0.2}, {0.5}};

        SimpleMatrix x_mat = SimpleMatrix.random(n_samples, 2);
        x_mat = utils.setColumn(x_mat, 0, 1);
        SimpleMatrix theta_mat = new SimpleMatrix(theta);
        SimpleMatrix random_noise = SimpleMatrix.randomNormal(SimpleMatrix.identity(x_mat.getNumRows()), rand);

        SimpleMatrix y_mat = x_mat.mult(theta_mat).plus(random_noise);

        // fit the model
        Mse mse = new Mse();

        double learning_rate = 0.00001;
        byte batch_size = (byte)4;
        byte n_epochs = (byte)100;

        StochasticGradientDescent sgd = new StochasticGradientDescent(
                mse, learning_rate, batch_size, n_epochs
        );

        model.fit(x_mat, y_mat, sgd);
        model.params.print();

        // closed form solution
        model.fit(x_mat, y_mat);
        model.params.print();
    }
}
