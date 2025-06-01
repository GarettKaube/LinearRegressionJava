package optimization;

import lossfunctions.LossFunctions;
import models.Model;
import utils.Pair;
import org.ejml.simple.SimpleMatrix;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.*;

public class StochasticGradientDescent extends Optimizer {
    public double learningRate;
    public byte batch_size;
    public int n_epochs;

    public StochasticGradientDescent(
            LossFunctions loss,
            double learningRate,
            SimpleMatrix params,
            byte batch_size,
            int n_epochs,
            Model model
    ) {
        super(loss, params, model);
        this.learningRate = learningRate;
        this.batch_size = batch_size;
        this.n_epochs = n_epochs;
    }

    public StochasticGradientDescent(
            LossFunctions loss,
            double learningRate,
            byte batch_size,
            int n_epochs,
            Model model
    ) {
        super(loss, model);
        this.learningRate = learningRate;
        this.batch_size = batch_size;
        this.n_epochs = n_epochs;
    }

    public StochasticGradientDescent(
            LossFunctions loss,
            double learningRate,
            byte batch_size,
            int n_epochs
    ) {
        super(loss);
        this.learningRate = learningRate;
        this.batch_size = batch_size;
        this.n_epochs = n_epochs;
    }

    public void initParams() {
        this.model.initParams();
        this.params = this.model.getParams();
    }

    public void initParams(SimpleMatrix X) {
        this.model.initParams(X);
        this.params = this.model.getParams();
    }

    /**
     * Takes the x, y SimpleMatrix data pair and shuffles them while maintaining the predictors-label pairs, i.e.
     * (x[i], y[i]) = (x[j], y[j]) where j is the new index after shuffling.
     * @param x SimpleMatrix
     * @param y SimpleMatrix
     * @return Pair<SimpleMatrix, SimpleMatrix>
     */
    private Pair<SimpleMatrix, SimpleMatrix> randomizeData(SimpleMatrix x, SimpleMatrix y) {
        List<double[]> x_list = new ArrayList<>();
        List<double[]> y_list = new ArrayList<>();

        for (int i = 0; i < x.getNumRows(); i++) {
            x_list.add(x.extractVector(true, i).getDDRM().getData().clone());
            y_list.add(y.extractVector(true, i).getDDRM().getData().clone());
        }

        List<Pair<double[], double[]>> pairs = new ArrayList<>();
        for (int i = 0; i < x_list.size(); i++) {
            pairs.add(new Pair<>(x_list.get(i), y_list.get(i)));
        }

        Collections.shuffle(pairs, new Random());

        x_list.clear();
        y_list.clear();
        for (Pair<double[], double[]> p : pairs) {
            x_list.add(p.first);
            y_list.add(p.second);
        }

        for (int i = 0; i < x.getNumRows(); i++) {
            x.setRow(i, 0, x_list.get(i));
            y.setRow(i, 0, y_list.get(i));
        }

        return new Pair<>(x, y);
    }

    private List<Pair<SimpleMatrix, SimpleMatrix>> splitIntoBatches(SimpleMatrix x, SimpleMatrix y) {
            int n = x.getNumRows();
            int n_c = x.getNumCols();
            int nBatches = (int) Math.ceil((double)n / (double)this.batch_size);

            List<Pair<SimpleMatrix, SimpleMatrix>> batches = new ArrayList<>();

            System.out.println("Creating batches...");
            int j = 0;
            for (int i = 0; i < nBatches; i++) {
                int slice = (i * this.batch_size) + this.batch_size - 1;

                // Make sure the last batch bounds are not greater than the size of the data
                if (slice + 1 > n) {
                    slice = n - 1;
                }

                SimpleMatrix x_batch = x.extractMatrix(j, slice+1, 0, n_c);
                SimpleMatrix y_batch = y.extractMatrix(j, slice+1, 0, 1);
                batches.add(new Pair<>(x_batch, y_batch));
                j = ((1+i)*this.batch_size);
            }
            return batches;
    }

    private void step(SimpleMatrix grad) {
        this.params = grad.scale(-this.learningRate).plus(this.params);
    }

    private SimpleMatrix StochasticGrad(SimpleMatrix x, SimpleMatrix y) {
        // Compute Stochastic gradient
        List<SimpleMatrix> grads = new ArrayList<>();
        for (int i = 0; i < x.getNumRows(); i++) {
            SimpleMatrix ithGradient = this.loss.grad(x.getRow(i), y.getRow(i), this.params, this.model);
            grads.add(ithGradient);
        }

        // Sum the gradients
        SimpleMatrix stochGrad = grads.get(0);
        for (int i = 1; i < grads.size(); i++) {
            stochGrad = stochGrad.plus(grads.get(i));
        }

        // Divide gradient by batch size
        stochGrad = stochGrad.scale(grads.size());
        return stochGrad;
    }

    @Override
    public void optimize(SimpleMatrix x, SimpleMatrix y) {
        Pair<SimpleMatrix, SimpleMatrix> a = this.randomizeData(x, y);
        SimpleMatrix x_ = a.first;
        SimpleMatrix y_ = a.second;

        List<Pair<SimpleMatrix, SimpleMatrix>> batches = this.splitIntoBatches(x_, y_);
        for (int epoch = 0; epoch < this.n_epochs; epoch++) {
            for (Pair<SimpleMatrix, SimpleMatrix> p: batches) {
                SimpleMatrix x_batch = p.first;
                SimpleMatrix y_batch = p.second;
                // Compute Stochastic gradient
                SimpleMatrix stochasticGrad = StochasticGrad(x_batch, y_batch);

                // Take step
                this.step(stochasticGrad);
                this.model.setParams(this.params);
            }
        }

        // set the optimized parameters
        this.model.setParams(this.params);
    }
}

