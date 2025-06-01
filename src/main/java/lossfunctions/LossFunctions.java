package lossfunctions;

import org.ejml.simple.SimpleMatrix;
import models.Model;

public class LossFunctions {
    public SimpleMatrix grad(SimpleMatrix x, SimpleMatrix y, SimpleMatrix theta, Model model) {
        return x;
    }

}



