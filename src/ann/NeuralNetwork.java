package ann;

import java.util.*;

/**
 * Represents a multi-layered neural network. The network can have an arbitrary
 * number of hidden layers with an arbitrary number of neurons in each layer.
 */
public class NeuralNetwork {
    /** Weights */
    private double[][][] w;

    /** Biases */
    private double[][] b;

    /** Number of neurons per layer */
    private final int[] NEURONS;

    /** Number of layers */
    private final int N_LAYERS;

    /** Size of input */
    public final int INPUT_SIZE;

    /** Size of output */
    public final int OUTPUT_SIZE;

    /**
     * Creates a new, multi-layered neural network with a given number of neurons in
     * each layer.
     * 
     * @param neurons - number of neurons in each layer. For example the
     *                array layers = [12, 6, 4] would have 12 input neurons, 6
     *                hidden-layer neurons and 4 output neurons.
     * 
     * @throws IllegalArgumentException if less than 2 layers are specified
     */
    public NeuralNetwork(int[] neurons) {
        /*
         * Dimension arrays for w and b. Note that even though both arrays have N_LAYERS
         * in the outermost dimension we will only actually be using layers 1 to
         * N_LAYERS - 1. This will help us maintain consistency with the notation in our
         * reference material:
         * 
         * http://neuralnetworksanddeeplearning.com/chap2.html
         */

        // Validate given network size
        if (neurons.length < 2) {
            throw new IllegalArgumentException("Invalid network size. Network must have at least 2 layers.");
        }

        NEURONS = neurons;
        N_LAYERS = neurons.length;
        INPUT_SIZE = NEURONS[0];
        OUTPUT_SIZE = NEURONS[1];
        w = new double[N_LAYERS][][];
        b = new double[N_LAYERS][];

        for (int l = 1; l < N_LAYERS; l++) {
            w[l] = new double[NEURONS[l]][NEURONS[l - 1]];
            b[l] = new double[NEURONS[l]];
        }

        randomizeWeightsAndBiases();
    }

    /**
     * Resets the network by assigning Gaussian random values to w and b
     */
    private void randomizeWeightsAndBiases() {
        Random random = new Random();
        for (int l = 1; l < N_LAYERS; l++) {
            for (int j = 0; j < NEURONS[l]; j++) {
                for (int k = 0; k < NEURONS[l - 1]; k++) {
                    w[l][j][k] = random.nextGaussian();
                }

                b[l][j] = random.nextGaussian();
            }
        }
    }

    /**
     * Trains the network using randomized minibatches from a set of training
     * examples.
     * 
     * @param x         - training inputs
     * @param y         - training outputs
     * @param eta       - learning rate, must be > 0
     * @param batchSize - number of examples per minibatch, must evenly
     *                  divide the total number of training inputs
     * 
     * @throws IllegalArgumentException if there is a mismatch in the size of
     *                                  training inputs and outputs, an invalid
     *                                  learning rate, or a batch size that does not
     *                                  evenly divide the training set.
     */
    public void train(double[][] x, double[][] y, double eta, int batchSize) {
        if (x.length != y.length) {
            String msg = "Mismatch in x.length and y.length: (" + x.length +
                    ", " + y.length + ")";
            throw new IllegalArgumentException(msg);
        }

        if (eta <= 0) {
            String msg = "Invalid training rate eta: " + eta + ". Must be >= 0.";
            throw new IllegalArgumentException(msg);
        }

        if (x.length % batchSize != 0) {
            String msg = "batchSize (" + batchSize + ") does not evenly " +
                    "divide x.length (" + x.length + ")";
            throw new IllegalArgumentException(msg);
        }

        // Generate batches
        // Get a list of the indices of x and shuffle it
        final int N_EXAMPLES = x.length;
        List<Integer> indices = new ArrayList<>(N_EXAMPLES);
        for (int i = 0; i < N_EXAMPLES; i++) {
            indices.add(i);
        }

        Collections.shuffle(indices);

        // Now generate the batches
        final int N_BATCHES = (N_EXAMPLES / batchSize);
        double[][][] xBatches = new double[N_BATCHES][batchSize][];
        double[][][] yBatches = new double[N_BATCHES][batchSize][];

        for (int n = 0; n < N_BATCHES; n++) {
            for (int i = 0; i < batchSize; i++) {
                int index = indices.get(n * batchSize + i);
                xBatches[n][i] = x[index];
                yBatches[n][i] = y[index];
            }
        }

        // Loop over batches and apply sgd to each batch
        for (int n = 0; n < N_BATCHES; n++) {
            sgd(xBatches[n], yBatches[n], eta);
        }

    }

    /**
     * Performs a steepest gradient descent step, updating the weights and
     * biases for the network using the average slope from all the examples.
     * 
     * @param x   - training inputs
     * @param y   - training outputs
     * @param eta - learning rate
     */
    private void sgd(double[][] x, double[][] y, double eta) {
        final int N_EXAMPLES = x.length;

        // Activations
        double[][] a = new double[N_LAYERS][];

        // Weighted input
        double[][] z = new double[N_LAYERS][];

        // Errors
        double[][] delta = new double[N_LAYERS][];

        // Estimated derivatives of cost function with respect to w and b
        double[][][] dCdw = new double[N_LAYERS][][];
        double[][] dCdb = new double[N_LAYERS][];

        // Initialize arrays
        for (int l = 1; l < N_LAYERS; l++) {
            a[l] = new double[NEURONS[l]];
            z[l] = new double[NEURONS[l]];
            delta[l] = new double[NEURONS[l]];
            dCdw[l] = new double[NEURONS[l]][NEURONS[l - 1]];
            dCdb[l] = new double[NEURONS[l]];
        }

        // Loop over examples
        for (int n = 0; n < N_EXAMPLES; n++) {
            // Feed forward
            a[0] = x[n];
            for (int l = 1; l < N_LAYERS; l++) {
                z[l] = sum(mult(w[l], a[l - 1]), b[l]);
                a[l] = sigmoid(z[l]);
            }

            // Get output error - first get grad_a C. The equation here
            // assumes C = (y - a)^2 / 2
            double[] gradC = diff(a[N_LAYERS - 1], y[n]);
            double[] sigPrime = sigmoidPrime(z[N_LAYERS - 1]);
            delta[N_LAYERS - 1] = hadamard(gradC, sigPrime);

            // Backpropagate the error
            for (int l = N_LAYERS - 2; l > 0; l--) {
                double[] wlp1Tdlp1 = mult(transpose(w[l + 1]), delta[l + 1]);
                delta[l] = hadamard(wlp1Tdlp1, sigmoidPrime(z[l]));
            }

            // Update gradients
            for (int l = 1; l < N_LAYERS; l++) {
                for (int j = 0; j < NEURONS[l]; j++) {
                    for (int k = 0; k < NEURONS[l - 1]; k++) {
                        dCdw[l][j][k] += a[l - 1][k] * delta[l][j];
                    }

                    dCdb[l][j] += delta[l][j];
                }
            }
        }

        // Update weights and biases
        for (int l = 1; l < N_LAYERS; l++) {
            for (int j = 0; j < NEURONS[l]; j++) {
                for (int k = 0; k < NEURONS[l - 1]; k++) {
                    w[l][j][k] -= eta * dCdw[l][j][k] / x.length;
                }

                b[l][j] -= eta * dCdb[l][j] / x.length;
            }
        }
    }

    /**
     * Evaluates the neural network for an input x.
     * 
     * @param x - input, must be of length {@link #INPUT_SIZE}
     * @return the network's output, of length {@link #OUTPUT_SIZE}
     * @throws IllegalArgumentException if x.length != {@link #INPUT_SIZE}
     */
    public double[] evaluate(double[] x) {
        if (x.length != NEURONS[0]) {
            String msg = "Invalid size for x: " + x.length + ", " +
                    INPUT_SIZE + " required";
            throw new IllegalArgumentException(msg);
        }

        double[] a = x;
        for (int l = 1; l < N_LAYERS; l++) {
            a = sigmoid(sum(mult(w[l], a), b[l]));
        }

        return a;
    }

    // MATRIX MATH / HELPER FUNCTIONS BELOW HERE //

    /**
     * Performs the matrix multiplication m*v where m is a rank-2 array m (a
     * matrix) and v is a rank-1 array. Note, that this assumes that the
     * elements of v represent a column vector, though it's stored as a flat
     * array for ease of use.
     * 
     * @param m - rank-2 array with dimensions nr x nc
     * @param v - array with length nc
     * @return - array with length nr representing m*v
     * @throws IllegalArgumentException if dimensions of m and v are incompatible
     */
    private static double[] mult(double[][] m, double[] v) {
        // Check dimensions for consistency
        for (int r = 0; r < m.length; r++) {
            if (m[r].length != v.length) {
                String msg = "Invalid dimension detected: m[" + r + "].length" +
                        " (" + m[r].length + ") does not equal v.length (" +
                        v.length + ")";
                throw new IllegalArgumentException(msg);
            }
        }

        double[] mv = new double[m.length];
        for (int i = 0; i < m.length; i++) {
            for (int j = 0; j < v.length; j++) {
                mv[i] += m[i][j] * v[j];
            }
        }

        return mv;
    }

    /**
     * Tranposes a rank-2 array (matrix) m, returning the transposed
     * matrix mT where mT[i][j] = m[j][i] (i.e. rows and columns flipped).
     * 
     * @param m - input matrix
     * @return the transposed matrix
     * @throws IllegalArgumentException if m has inconsistent dimensions
     */
    private static double[][] transpose(double[][] m) {
        // Validate consistency of dimensions in input matrix
        for (double[] r : m) {
            if (r.length != m[0].length) {
                String msg = "Inconsistent dimensions in m";
                throw new IllegalArgumentException(msg);
            }
        }

        final int NR_M = m.length;
        final int NC_M = m[0].length;
        double[][] mT = new double[NC_M][NR_M];
        for (int i = 0; i < NR_M; i++) {
            for (int j = 0; j < NC_M; j++) {
                mT[j][i] = m[i][j];
            }
        }

        return mT;
    }

    /**
     * Calculates the Hadamard (element-wise) product of two arrays.
     *
     * @param a
     * @param b
     * @return the array had such that had[i] = a[i] * b[i]
     * @throws IllegalArgumentException if a and b are different sizes
     */
    private static double[] hadamard(double[] a, double[] b) {
        if (a.length != b.length) {
            String msg = "Dimension mismatch: a.length = " + a.length +
                    ", b.length = " + b.length;
            throw new IllegalArgumentException(msg);
        }

        final int N = a.length;
        double[] had = new double[N];
        for (int i = 0; i < N; i++) {
            had[i] = a[i] * b[i];
        }

        return had;
    }

    /**
     * Returns the element-wise sum of two arrays.
     * 
     * @param a
     * @param b
     * @return array s such that s[i] = a[i] + b[i]
     * @throws IllegalArgumentException if a and b are different sizes
     */
    private static double[] sum(double[] a, double[] b) {
        // Validate
        if (a.length != b.length) {
            String msg = "Dimension mismatch: a.length = " + a.length +
                    ", b.length = " + b.length;
            throw new IllegalArgumentException(msg);
        }

        final int N = a.length;
        double[] aPlusB = new double[N];
        for (int i = 0; i < N; i++) {
            aPlusB[i] = a[i] + b[i];
        }

        return aPlusB;
    }

    /**
     * Returns the element-wise difference of two arrays.
     * 
     * @param a
     * @param b
     * @return array s such that s[i] = a[i] - b[i]
     * @throws IllegalArgumentException if a and b are different sizes
     */
    private static double[] diff(double[] a, double[] b) {
        if (a.length != b.length) {
            String msg = "Dimension mismatch: a.length = " + a.length +
                    ", b.length = " + b.length;
            throw new IllegalArgumentException(msg);
        }

        final int N = a.length;
        double[] aMinusB = new double[N];
        for (int i = 0; i < N; i++) {
            aMinusB[i] = a[i] - b[i];
        }

        return aMinusB;
    }

    /**
     * Returns an array consisting of the sigmoid function of each element in z.
     * 
     * @param z
     * @return the array sig where sig[i] = sigmoid(z[i])
     */
    private static double[] sigmoid(double[] z) {
        final int SIZE = z.length;
        double[] sig = new double[SIZE];

        for (int i = 0; i < SIZE; i++) {
            sig[i] = sigmoid(z[i]);
        }

        return sig;
    }

    /**
     * Returns an array consisting of the sigmoid function of each element in z.
     * 
     * @param z
     * @return the array sig where sig[i] = sigmoid(z[i])
     */
    private static double[] sigmoidPrime(double[] z) {
        final int SIZE = z.length;
        double[] sigPrime = new double[SIZE];

        for (int i = 0; i < SIZE; i++) {
            sigPrime[i] = sigmoid(z[i]) * (1 - sigmoid(z[i]));
        }

        return sigPrime;
    }

    /**
     * @param z
     * @return the sigmoid function 1 / (1 + exp(-z))
     */
    private static double sigmoid(double z) {
        return 1 / (1 + Math.exp(-z));
    }
}
