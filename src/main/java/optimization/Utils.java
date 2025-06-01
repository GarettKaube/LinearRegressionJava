package optimization;

import org.ejml.simple.SimpleMatrix;

public class Utils {
    public SimpleMatrix setColumn(SimpleMatrix M, int colIndex, double value) {
        for (int i = 0; i < M.getNumRows(); i++) {
            M.set(i, colIndex, value);
        }
        return M;
    }
}
