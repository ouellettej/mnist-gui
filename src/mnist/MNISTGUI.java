package mnist;

import javax.swing.*;
import java.awt.*;
import java.util.List;

public class MNISTGUI extends JFrame {
    private static final int WIDTH = 450;
    private static final int HEIGHT = 700;
    private static final String TITLE = "MNIST Neural Network";
    private static final String PATH = "res/mnist/";
    private static final String[] digits =
            {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"};

    // GUI COMPONENTS //
    /** Displays image */
    private JPanel viewer;

    /** Bar chart of neural net confidences */
    private JPanel bars;

    /** Button for testing random image */
    private JButton randomTestButton;

    /** Input for learning rate */
    private JTextField etaInput;

    /** Input for batch size */
    private JTextField batchSizeInput;

    /** Input for number of epochs */
    private JTextField nEpochsInput;

    /** Perform training */
    private JButton trainButton;

    // NEURAL NETWORK //
    /** Neural network */
    private MNISTNeuralNetwork network;

    // MNIST DATA //
    /** Training images used to train network */
    private List<int[][]> trainingImages;

    /** Training labels used to train network */
    private int[] trainingLabels;

    /** Test images used to verify network */
    private List<int[][]> testImages;

    /** Test labels used to verify network */
    private int[] testLabels;

    public MNISTGUI() {
        loadData();
        network = new MNISTNeuralNetwork(30);

        setSize(WIDTH, HEIGHT);
        setTitle(TITLE);
        //setResizable(false);
        setDefaultCloseOperation(EXIT_ON_CLOSE);
        addComponents();
        randomImage();
        setVisible(true);
    }

    /**
     * Loads the MNIST data
     */
    private void loadData() {
        trainingImages = MNISTReader.getImages(PATH + "training-images");
        trainingLabels = MNISTReader.getLabels(PATH + "training-labels");
        testImages = MNISTReader.getImages(PATH + "test-images");
        testLabels = MNISTReader.getLabels(PATH + "test-labels");
    }

    /**
     * Sets up the GUI
     */
    private void addComponents() {
        // Visualizations
        JPanel main = new JPanel();
        BoxLayout layout = new BoxLayout(main, BoxLayout.Y_AXIS);
        main.setLayout(layout);


        GridLayout pixelGrid = new GridLayout(MNISTReader.MNIST_IMAGE_SIZE,
                MNISTReader.MNIST_IMAGE_SIZE);
        viewer = new JPanel(pixelGrid);
        viewer.setPreferredSize(new Dimension(400,400));
        bars = new JPanel();
        bars.setPreferredSize(new Dimension(400, 100));
        main.add(viewer);
        main.add(bars);

        // Right side is for input
        JPanel inputPanel = new JPanel();
        inputPanel.add(new JLabel("Learning Rate:"));
        etaInput = new JTextField(3);
        etaInput.setText("3");
        inputPanel.add(etaInput);

        inputPanel.add(new JLabel("Minibatch Size:"));
        batchSizeInput = new JTextField(3);
        batchSizeInput.setText("10");
        inputPanel.add(batchSizeInput);

        inputPanel.add(new JLabel("Epochs:"));
        nEpochsInput = new JTextField(3);
        nEpochsInput.setText("3");
        inputPanel.add(nEpochsInput);
        main.add(inputPanel);

        JPanel buttonPanel = new JPanel();
        trainButton = new JButton("Train");
        trainButton.addActionListener(actionEvent -> train());
        buttonPanel.add(trainButton);

        randomTestButton = new JButton("Random Test Image");
        randomTestButton.addActionListener(actionEvent -> randomImage());
        buttonPanel.add(randomTestButton);
        main.add(buttonPanel);

        add(main);
    }

    /**
     * Train the network using the inputs from the gui
     */
    private void train() {
        double eta = Double.parseDouble(etaInput.getText());
        int batchSize = Integer.parseInt(batchSizeInput.getText());
        int nEpochs = Integer.parseInt(nEpochsInput.getText());

        for (int n = 0; n < nEpochs; n++) {
            network.train(trainingImages, trainingLabels,
                          testImages, testLabels, eta, batchSize);
        }
    }

    /**
     * Picks a random image and displays it along with the network's output
     */
    private void randomImage() {
        // Get the image
        int r = (int) (Math.random() * trainingImages.size());
        int[][] image = trainingImages.get(r);

//        for (r = 0; r < MNISTReader.MNIST_IMAGE_SIZE; r++) {
//            for (int c = 0; c < MNISTReader.MNIST_IMAGE_SIZE; c++) {
//                image[r][c] = (int) (256 * Math.random());
//            }
//        }

        // Put on the viewer
        setViewer(image);

        // Set bar graph
        setBars(network.evaluate(image));
    }

    /**
     * Sets the pixels in the viewer to match an MNIST formatted imgae
     * @param image - MNIST-formatted image
     */
    private void setViewer(int[][] image) {
        viewer.removeAll();
        viewer.revalidate();
        viewer.repaint();
        for (int i = 0; i < MNISTReader.MNIST_IMAGE_SIZE; i++) {
            for (int j = 0; j < MNISTReader.MNIST_IMAGE_SIZE; j++) {
                int rgb = image[i][j];
                JLabel label = new JLabel();
                label.setOpaque(true);
                label.setBackground(new Color(rgb, rgb, rgb));
                viewer.add(label);
            }
        }
    }

    /**
     * Sets the bars underneath the image using the network's output. Bigger
     * bars for a digit indicate the network is more confident of its answer
     * @param networkOutput
     */
    private void setBars(double[] networkOutput) {
        // Normalize output and convert to percentages
        double sum = 0;
        for (double aLi : networkOutput) {
            sum += aLi;
        }

        int[] barSizes = new int[10];
        for (int i = 0; i < 10; i++) {
            barSizes[i] = Math.round((float) (100 * networkOutput[i] / sum));
        }

        // Reset the bars and redraw
        bars.removeAll();
        bars.revalidate();
        bars.repaint();
        for (int i = 0; i < 10; i++) {
            JProgressBar bar = new JProgressBar(JProgressBar.VERTICAL, 0, 100);
            bar.setPreferredSize(new Dimension(35, 100));
            bar.setValue(barSizes[i]);
            bar.setStringPainted(true);
            bar.setString(Integer.toString(i));
            bars.add(bar);
        }
    }

    public static void main(String[] args) {
        new MNISTGUI();
    }

}
