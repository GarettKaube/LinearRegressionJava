import lossfunctions.Mse;
import models.LinearRegression;
import optimization.StochasticGradientDescent;
import optimization.Utils;
import org.ejml.simple.SimpleMatrix;
import regularization.Regularization;

import java.util.Random;

public class main {
    public static void main(String[] args) {
        Utils utils = new Utils();
        Random rand = new Random(42);

        int n_samples = 2000;
        double[][] theta = {{0.2}, {0.5}};

        double[][] randomArray = new double[n_samples][2];

        for (int i = 0; i < n_samples; i++) {
            for (int j = 0; j < 2; j++) {
                randomArray[i][j] = rand.nextDouble();  // uniform [0, 1)
            }
        }

        SimpleMatrix x_mat = new SimpleMatrix(randomArray);

        x_mat = utils.setColumn(x_mat, 0, 1);
        SimpleMatrix theta_mat = new SimpleMatrix(theta);
        SimpleMatrix random_noise = SimpleMatrix.randomNormal(SimpleMatrix.identity(x_mat.getNumRows()), rand);

        SimpleMatrix y_mat = x_mat.mult(theta_mat).plus(random_noise);

        // Initialize MSE loss
        Mse mse = new Mse();

        LinearRegression model = new LinearRegression();


        // Initialize SGD with MSE loss
        StochasticGradientDescent sgd = new StochasticGradientDescent(
                mse, 0.0001, (byte)4, 200, model
        );


        System.out.println("Before optimization");
        sgd.initParams(x_mat);
        SimpleMatrix params = sgd.model.getParams();
        params.print();

        sgd.optimize(x_mat, y_mat);
        params = sgd.model.getParams();
        System.out.println("After optimization");
        params.print();

        // Test regularization
        LinearRegression model2 = new LinearRegression();
        Regularization reg = new Regularization(1, 0);

        Mse mse_reg = new Mse(reg);

        StochasticGradientDescent sgd2 = new StochasticGradientDescent(
                mse_reg, 0.0001, (byte)4, 200, model2
        );

        LinearRegression model_reg = new LinearRegression();

        model_reg.fit(x_mat, y_mat, sgd2);
        System.out.println("After optimization with l2 regularization");
        model_reg.getParams().print();
    }
}
