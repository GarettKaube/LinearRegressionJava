package regularization;

import org.ejml.simple.SimpleMatrix;

public class Regularization {
    int l;
    double lambda;

    /**
     Implements Lp regularization with lambda as the tuning parameter
     */
    public Regularization(int l, double lambda) {
        this.l = l;
        this.lambda = lambda;
    }

    public double computeLpNorm(SimpleMatrix theta) {
        double sum = 0.0;

        for (int i = 0; i < theta.getNumElements(); i++) {
            double param = theta.get(i);
            sum += Math.pow(Math.abs(param), this.l);
        }

        return Math.pow(sum, 1.0 / this.l);
    }

    public double computeRegularization(SimpleMatrix theta) {
        double lp = this.computeLpNorm(theta);
        return this.lambda * lp;
    }

    /**
     * Implements the gradient of the Lp norm multiplied by regularization parameter
     * @param theta
     * @return
     */
    public SimpleMatrix dL(SimpleMatrix theta) {
        int num_params = theta.getNumElements();
        double[][] vec = new double[num_params][1];

        // Fill in the vector component of the gradient
        for (int i = 0; i < num_params; i++) {
            double param = theta.get(i);
            double abs_value = Math.abs(param);
            vec[i][0] = Math.pow(abs_value, this.l - 1) * Math.signum(param);
        }

        SimpleMatrix vec_mat = new SimpleMatrix(vec);

        // Compute the scaler component of the gradient
        double reg = this.computeLpNorm(theta);

        double reg_lp = Math.pow(reg, 1 - this.l);
        double scalar = this.lambda * reg_lp;


        SimpleMatrix dl = vec_mat.scale(scalar);

        return dl;
    }

    public static void main(String[] args) {
        double[][] theta = {{0.2}, {0.5}};
        SimpleMatrix theta_mat = new SimpleMatrix(theta);

        Regularization reg = new Regularization(3, 0.001);

        SimpleMatrix grad = reg.dL(theta_mat);
        grad.print();
    }
}
