package optimization;

import models.Model;
import org.ejml.simple.SimpleMatrix;
import lossfunctions.LossFunctions;

public class Optimizer {
    public LossFunctions loss;
    public SimpleMatrix params;
    public Model model;

    public Optimizer(LossFunctions loss, SimpleMatrix params, Model model) {
        this.loss = loss;
        this.params = params;
        this.model = model;
    }

    public Optimizer(LossFunctions loss, SimpleMatrix params) {
        this.loss = loss;
        this.params = params;
    }

    public Optimizer(LossFunctions loss) {
        this.loss = loss;
    }

    public Optimizer(LossFunctions loss, Model model) {
        this.loss = loss;
        this.model = model;
    }

    public void setModel(Model model) {
        this.model = model;
    }

    public void setParams(SimpleMatrix params) {
        this.params = params;
    }

    public void optimize(SimpleMatrix x, SimpleMatrix y) {
        double[][] a = {{0.0}};
        SimpleMatrix A = new SimpleMatrix(a);
    }
}



