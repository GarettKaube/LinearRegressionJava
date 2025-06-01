package lossfunctions;

import org.ejml.simple.SimpleMatrix;
import models.Model;
import regularization.Regularization;

public class Mse extends LossFunctions {
    Regularization regularizer;
    /**
     Implements the sum of squares error
     Requires y and theta to be an n x 1, (p + 1) x 1 matrix respectively
     where p is the number of predictors
     */

    public Mse(Regularization regularizer) {
        this.regularizer = regularizer;
    }

    public Mse() {
    }

    public double sse(SimpleMatrix x, SimpleMatrix y, SimpleMatrix theta) {
        SimpleMatrix e = y.minus(x.mult(theta));
        SimpleMatrix residual = e.transpose().mult(e);

        return residual.get(0,0);
    }

    /**
     Evaluates sse then divides by the number of samples n and adds the regularization term
     */
    public double mse(SimpleMatrix x, SimpleMatrix y, SimpleMatrix theta) {
        try {
            return  (this.sse(x, y, theta) / x.getNumRows()) + this.regularizer.computeRegularization(theta);
        }

        catch (Exception e) {
            System.out.println(e);
            return  (this.sse(x, y, theta) / x.getNumRows());
        }
    }

    @Override
    public SimpleMatrix grad(SimpleMatrix x, SimpleMatrix y, SimpleMatrix theta, Model model) {
        SimpleMatrix gradTheta = model.getGrad(x, theta);
        SimpleMatrix error = model.predict(x, theta).minus(y);
        SimpleMatrix grad = gradTheta.transpose().mult(error).scale(2);

        if (this.regularizer != null) {
            return grad.plus(this.regularizer.dL(theta));
        }
        else {
            return grad;
        }
    }

    public static void main(String[] args) {
        double[][] x = {{1, 1}, {1, 2}, {1, 3}};
        double[][] y = {{1}, {5}, {3}};
        double[][] theta = {{0.2}, {0.5}};

        SimpleMatrix x_mat = new SimpleMatrix(x);
        SimpleMatrix y_mat = new SimpleMatrix(y);
        SimpleMatrix theta_mat = new SimpleMatrix(theta);

        System.out.println("x:");
        x_mat.print();

        System.out.println("Testing sse:");
        Mse loss = new Mse();
        double res = loss.sse(x_mat, y_mat, theta_mat);
        System.out.println("residual:");
        System.out.println(res);

        System.out.println("Testing mse:");
        double mse = loss.mse(x_mat, y_mat, theta_mat);
        System.out.println("mse:");
        System.out.println(mse);

        System.out.println("Testing mse with l1 regularization:");
        Regularization reg = new Regularization(1, 0.4);

        loss.regularizer = reg;
        double mse_reg = loss.mse(x_mat, y_mat, theta_mat);
        System.out.println("mse reg:");
        System.out.println(mse_reg);

        System.out.println("Testing Linear regression MSE grad");

        models.LinearRegression model = new models.LinearRegression();
        SimpleMatrix grad = loss.grad(x_mat, y_mat, theta_mat, model);
        grad.print();
    }
}
