package models;

import org.ejml.simple.SimpleMatrix;

import java.util.Random;

public class Model {
    SimpleMatrix params;
    SimpleMatrix X;
    SimpleMatrix y;

    public SimpleMatrix getGrad(SimpleMatrix x) {
        return x;
    }

    public SimpleMatrix getGrad(SimpleMatrix x, SimpleMatrix params) {
        return x;
    }

    /**
     * Initializes the models params.
     */
    public void initParams() {
        Random rand = new Random();
        int p = 2;
        this.params = SimpleMatrix.random(p, 1);
    }

    /**
     * Initializes the models params. Uses info from X to initialize parameters
     * @param X SimpleMatrix
     */
    public void initParams(SimpleMatrix X) {
        Random rand = new Random();
        int p = 2;
        this.params = SimpleMatrix.random(p, 1);
    }

    /**
     *
     * @param params SimpleMatrix containing model parameters
     */
    public void setParams(SimpleMatrix params) {
        this.params = params;
    }

    public SimpleMatrix getParams() {
        return this.params ;
    }

    public SimpleMatrix predict(SimpleMatrix X, SimpleMatrix params) {
        return X.mult(params);
    }

    public SimpleMatrix predict(SimpleMatrix X) {
        return X.mult(this.params);
    }

    public void fit(SimpleMatrix X, SimpleMatrix y) {
        this.X = X;
        this.y = y;
    }
}
