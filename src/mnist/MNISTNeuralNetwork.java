package mnist;

import ann.NeuralNetwork;

import java.util.List;

public class MNISTNeuralNetwork extends NeuralNetwork {
    private static final int MNIST_IMAGE_DIM = MNISTReader.MNIST_IMAGE_SIZE;

    /**
     * Creates a three-layer neural network for MNIST number recognition with
     * nHidden neurons in the hidden layer
     *
     * @param nHidden - number of neurons in hidden layer
     */
    public MNISTNeuralNetwork(int nHidden) {
        super(new int[] {MNIST_IMAGE_DIM * MNIST_IMAGE_DIM, nHidden, 10});
    }

    /**
     * Trains the network using randomized minibatches from the training
     * examples given
     * @param trainingImages - MNIST-formatted training images
     * @param trainingLabels - training labels
     * @param testImages - MNIST-formatted testing images, used to measure
     *                   network's accuracy
     * @param testLabels - testing labels, used to measure network's accuracy
     * @param eta - learning rate
     * @param batchSize - number of examples per minibatch, must evenly
     *                  divide the total number of training inputs
     */
    public void train(List<int[][]> trainingImages,
                      int[] trainingLabels,
                      List<int[][]> testImages,
                      int[] testLabels,
                      double eta, int batchSize) {

        if (trainingImages.size() != trainingLabels.length) {
            String msg = "Mismatch between number of training images and " +
                    "number of training labels: (" + trainingImages.size() +
                    "," + trainingLabels.length + ")";
            throw new IllegalArgumentException(msg);
        }

        if (testImages.size() != testLabels.length) {
            String msg = "Mismatch between number of test images and number " +
                    "of test labels: (" + trainingImages.size() + "," +
                    trainingLabels.length + ")";
            throw new IllegalArgumentException(msg);
        }

        // Adapt images and labels to be consistent with NeuralNetwork class
        final int N_IMAGES = trainingImages.size();
        double[][] x = new double[N_IMAGES][MNIST_IMAGE_DIM * MNIST_IMAGE_DIM];
        double[][] y = new double[N_IMAGES][10];

        for (int n = 0; n < N_IMAGES; n++) {
            x[n] = flatten(trainingImages.get(n));
            y[n][trainingLabels[n]] = 1.0;
        }

        // Call learn from super class
        super.train(x, y, eta, batchSize);

        // Print out accuracy
        int nCorrect = 0;
        final int N_TESTS = testImages.size();
        for (int i = 0; i < N_TESTS; i++) {
            if (mostLikelyDigit(testImages.get(i)) == testLabels[i])  {
                nCorrect++;
            }
        }

        double accuracy = (double) nCorrect / N_TESTS;
        System.out.printf("Accuracy: %.2f%%\n", 100 * accuracy);
    }

    /**
     * Evaluates an image using the network and returns the network's
     * estimate of the most likely digit
     * @param image - MNIST-formatted image
     * @return the network's most likely estimate of the digit shown in the
     * image
     */
    private double mostLikelyDigit(int[][] image) {
        double[] output = evaluate(image);

        double pMax = Double.MIN_VALUE;
        int digit = -1;
        for (int i = 0; i < output.length; i++) {
            if (output[i] > pMax) {
                pMax = output[i];
                digit = i;
            }
        }

        return digit;
    }

    /**
     * Returns the network's output for an image
     * @param image - MNIST-formatted image
     * @return the network's output for the image
     */
    public double[] evaluate(int[][] image) {
        return super.evaluate(flatten(image));
    }

    /**
     * Converts an MNIST-formatted image into a flat array that can be used
     * by the NeuralNetwork class
     * @param image - MNIST-formatted image
     * @return
     */
    private static double[] flatten(int[][] image) {
        double[] x = new double[MNIST_IMAGE_DIM * MNIST_IMAGE_DIM];
        for (int r = 0; r < MNIST_IMAGE_DIM; r++) {
            for (int c = 0; c < MNIST_IMAGE_DIM; c++) {
                x[r * MNIST_IMAGE_DIM + c] = (double) image[r][c] / 255;
            }
        }

        return x;
    }
}
