package tdlearningsutton;

/**
 * Nonlinear TD/Backprop pseudo C-code
 * <p>
 * Written by Allen Bonde Jr. and Rich Sutton in April 1992. Updated in June and
 * August 1993. Copyright 1993 GTE Laboratories Incorporated. All rights
 * reserved. Permission is granted to make copies and changes, with attribution,
 * for research and educational purposes.
 * <p>
 * This pseudo-code implements a fully-connected two-adaptive-layer network
 * learning to predict discounted cumulative outcomes through Temporal
 * Difference learning, as described in Sutton (1988), Barto et al. (1983),
 * Tesauro (1992), Anderson (1986), Lin (1992), Dayan (1992), et alia. This is a
 * straightforward combination of discounted TD(lambda) with backpropagation
 * (Rumelhart, Hinton, and Williams, 1986). This is vanilla backprop; not even
 * momentum is used. See Sutton & Whitehead (1993) for an argument that backprop
 * is not the best structural credit assignment method to use in conjunction
 * with TD. Discounting can be eliminated for absorbing problems by setting
 * gamma=1. Eligibility traces can be eliminated by setting lambda=0. Setting
 * both of these parameters to 0 should give standard backprop except where the
 * input at time t has its target presented at time t+1.
 * <p>
 * This is pseudo code: before it can be run it needs I/O, a random number
 * generator, library links, and some declarations. We welcome extensions by
 * others converting this to immediately usable C code.
 * <p>
 * where x, h, and y are (arrays holding) the activity levels of the input,
 * hidden, and output units respectively, v and w are the first and second layer
 * weights, and ev and ew are the eligibility traces for the first and second
 * layers (see Sutton, 1989). Not explicitly shown in the figure are the biases
 * or threshold weights. The first layer bias is provided by a dummy nth input
 * unit, and the second layer bias is provided by a dummy (num-hidden)th hidden
 * unit. The activities of both of these dummy units are held at a constant
 * value (biasInputStrenght).
 * <p>
 * In addition to the main program, this file contains 4 routines:
 * <p>
 * initNetwork, which initializes the network data structures.
 * <p>
 * computeOutputs, which does the forward propagation, the computation of all
 * unit activities based on the current input and weights.
 * <p>
 * updateWeights, which does the backpropagation of the TD errors, and updates
 * the weights.
 * <p>
 * updateEligibilities, which updates the eligibility traces.
 * <p>
 * These routines do all their communication through the global variables shown
 * in the diagram above, plus old_y, an array holding a copy of the last time
 * step's output-layer activities.
 * <p>
 * REFERENCES
 * <p>
 * Anderson, C.W. (1986) Learning and Problem Solving with Multilayer
 * Connectionist Systems, UMass. PhD dissertation, dept. of Computer and
 * Information Science, Amherts, MA 01003.
 * <p>
 * Barto, A.G., Sutton, R.S., & Anderson, C.W. (1983) "Neuron-like adaptive
 * elements that can solve difficult learning control problems," IEEE
 * Transactions on Systems, Man, and Cybernetics SMC-13: 834-846.
 * <p>
 * Dayan, P. "The convergence of TD(lambda) for general lambda," Machine
 * Learning 8: 341-362.
 * <p>
 * Lin, L.-J. (1992) "Self-improving reactive agents based on reinforcement
 * learning, planning and teaching," Machine Learning 8: 293-322.
 * <p>
 * Rumelhart, D.E., Hinton, G.E., & Williams, R.J. (1986) "Learning internal
 * representations by error propagation," in D.E. Rumehart & J.L. McClelland
 * (Eds.), Parallel Distributed Processing: Explorations in the Microstructure
 * of Cognition, Volume 1: Foundations, 318-362. Cambridge, MA: MIT Press.
 * <p>
 * Sutton, R.S. (1988) "Learning to predict by the methods of temporal
 * differences," Machine Learning 3: 9-44.
 * <p>
 * Sutton, R.S. (1989) "Implementation details of the TD(lambda) procedure for
 * the case of vector predictions and backpropagation," GTE Laboratories
 * Technical Note TN87-509.1, May 1987, corrected August 1989. Available via ftp
 * from ftp.gte.com as /pub/reinforcement-learning/sutton-TD-backprop.ps
 * <p>
 * Sutton, R.S., Whitehead, S.W. (1993) "Online learning with random
 * representations," Proceedings of the Tenth National Conference on Machine
 * Learning, 314-321. Soon to be available via ftp from ftp.gte.com as
 * /pub/reinforcement-learning/sutton-whitehead-93.ps.Z
 * <p>
 * Tesauro, G. (1992) "Practical issues in temporal difference learning,"
 * Machine Learning 8: 257-278.
 * <p>
 */
public class TDLearningOriginal {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        TDLearningOriginal tdLearnind = new TDLearningOriginal(16, 16, 1, 20, 1, 0.1, 0.1, 0.9, 0.7);
        tdLearnind.learn();
    }
    double alpha; // 1st layer learning rate (typically 1/inputUnits)
    double beta; // 2nd layer learning rate (typically 1/hiddenUnits)
    double biasInputStrenght; // strength of the bias (constant input) contribution
    final double[] error; // TD error
    double gamma; // discount-rate parameter (typically 0.9)
    final double[] hiddenLayer; // hidden layer
    final double[][][] hiddenTrace; // hidden trace
    final int hiddenUnits; 
    // Network Data Structure:
    final double[][] input; // input data (units)
    // Experimental Parameters:
    final int inputUnits;
    final double[][] inputWeights; // weights
    double lambda; // trace decay parameter (should be <= gamma)

    // Learning Data Structure:
    final double[] old_outputLayer;
    final double outputDerivatedFnet; // for temporal calculations
    final double[] outputLayer; // output layer
    final double[][] outputTrace; // output trace
    final int outputUnits; // number of inputs, hidden, and output units
    final double[][] outputWeights; // weights
    final double[][] reward;// reward

    int t; // current time step
    final int time_steps; // number of time steps to simulate

    /**
     * Setup the training constants
     * <p>
     * @param inputUnits        number of input neurons
     * @param hiddenUnits       number of hidden neurons
     * @param outputUnits       number of output neurons
     * @param time_steps        number of time steps to simulate
     * @param biasInputStrenght strength of the bias (constant input)
     *                          contribution
     * @param alpha             1st layer learning rate (typically 1/n)
     * @param beta              2nd layer learning rate (typically 1/num_hidden)
     * @param gamma             discount-rate parameter (typically 0.9)
     * @param lambda            trace decay parameter (should be <= gamma)
     */
    public TDLearningOriginal(int inputUnits, int hiddenUnits, int outputUnits, int time_steps, double biasInputStrenght, double alpha, double beta, double gamma, double lambda) {
        this.inputUnits = inputUnits;
        this.hiddenUnits = hiddenUnits;
        this.outputUnits = outputUnits;
        this.time_steps = time_steps;
        this.biasInputStrenght = biasInputStrenght;
        this.alpha = alpha;
        this.beta = beta;
        this.gamma = gamma;
        this.lambda = lambda;

        // Network Data Structure:
        input = new double[time_steps][inputUnits + 1]; // input data (units). +1 for the bias
        hiddenLayer = new double[hiddenUnits + 1]; // hidden layer. +1 for the bias
        outputLayer = new double[outputUnits]; // output layer
        outputWeights = new double[hiddenUnits][outputUnits]; // weights
        inputWeights = new double[inputUnits][hiddenUnits]; // weights

        // Learning Data Structure:
        old_outputLayer = new double[outputUnits];
        hiddenTrace = new double[inputUnits][hiddenUnits][outputUnits]; // hidden trace
        outputTrace = new double[hiddenUnits][outputUnits]; // output trace
        reward = new double[time_steps][outputUnits]; // reward
        error = new double[outputUnits]; // TD error
        outputDerivatedFnet = new double[outputUnits]; //for temporal calculations
    }

    /**
     * Compute hidden layer and output predictions
     */
    public void computeOutputs() {
        
        hiddenLayer[hiddenUnits] = biasInputStrenght;
        input[t][inputUnits] = biasInputStrenght;
        
        for ( int hiddenUnit = 0; hiddenUnit < hiddenUnits; hiddenUnit++ ) {
            hiddenLayer[hiddenUnit] = 0.0;
            for ( int inputUnit = 0; inputUnit <= inputUnits; inputUnit++ ) {
                hiddenLayer[hiddenUnit] += input[t][inputUnit] * inputWeights[inputUnit][hiddenUnit];
            }
            hiddenLayer[hiddenUnit] = 1.0 / (1.0 + Math.exp(-hiddenLayer[hiddenUnit])); // asymmetric sigmoid
        }
        for ( int outputUnit = 0; outputUnit < outputUnits; outputUnit++ ) {
            outputLayer[outputUnit] = 0.0;
            for ( int hiddenUnit = 0; hiddenUnit <= hiddenUnits; hiddenUnit++ ) {
                outputLayer[outputUnit] += hiddenLayer[hiddenUnit] * outputWeights[hiddenUnit][outputUnit];
            }
            outputLayer[outputUnit] = 1.0 / (1.0 + Math.exp(-outputLayer[outputUnit])); // asymmetric sigmoid (OPTIONAL)
        }
    }// end computeOutputs


    /**
     * Initialize weights and biases
     */
    public void initNetwork() {
        for ( int s = 0; s < time_steps; s++ ) {
            input[s][inputUnits] = biasInputStrenght;
        }
        hiddenLayer[hiddenUnits] = biasInputStrenght;
        for ( int hiddenUnit = 0; hiddenUnit <= hiddenUnits; hiddenUnit++ ) {
            for ( int outputUnit = 0; outputUnit < outputUnits; outputUnit++ ) {
                outputWeights[hiddenUnit][outputUnit] = Math.random() / 10d; // some small random value
                outputTrace[hiddenUnit][outputUnit] = 0.0;
                old_outputLayer[outputUnit] = 0.0;
            }
            for ( int inputUnit = 0; inputUnit <= inputUnits; inputUnit++ ) {
                inputWeights[inputUnit][hiddenUnit] = Math.random() / 10d; //some small random value
                for ( int outputUnit = 0; outputUnit < outputUnits; outputUnit++ ) {
                    hiddenTrace[inputUnit][hiddenUnit][outputUnit] = 0.0;
                }
            }
        }
    }// end initNetwork


    /**
     * Train the neural network
     */
    public void learn() {
        int outputUnit;
        initNetwork();
        t = 0; // No learning on time step 0
        
        computeOutputs(); // Just compute old response (old_y)...
        
        for ( outputUnit = 0; outputUnit < outputUnits; outputUnit++ ) {
            old_outputLayer[outputUnit] = outputLayer[outputUnit];
        }
        updateEligibilities(); // ...and prepare the eligibilities
        
        for ( t = 1; t <= time_steps; t++ ) { // a single pass through time series data
            computeOutputs(); // forward pass - compute activities
            
            for ( outputUnit = 0; outputUnit < outputUnits; outputUnit++ ) {
                error[outputUnit] = reward[t][outputUnit] + gamma * outputLayer[outputUnit] - old_outputLayer[outputUnit]; // form errors
            }
            
            updateWeights(); // backward pass - learning
            computeOutputs(); // forward pass must be done twice to form TD errors
            for ( outputUnit = 0; outputUnit < outputUnits; outputUnit++ ) {
                old_outputLayer[outputUnit] = outputLayer[outputUnit]; // for use in next cycle's TD errors
            }
            updateEligibilities(); // update eligibility traces
            
        } // end t
    }
    /**
     * Calculate new weight eligibilities
     */
    public void updateEligibilities() {

        for ( int outputUnit = 0; outputUnit < outputUnits; outputUnit++ ) {
            outputDerivatedFnet[outputUnit] = outputLayer[outputUnit] * (1 - outputLayer[outputUnit]);
        }

        for ( int hiddenUnit = 0; hiddenUnit <= hiddenUnits; hiddenUnit++ ) {
            for ( int outputUnit = 0; outputUnit < outputUnits; outputUnit++ ) {
                outputTrace[hiddenUnit][outputUnit] = lambda * outputTrace[hiddenUnit][outputUnit]
                        + outputDerivatedFnet[outputUnit] * hiddenLayer[hiddenUnit];
                for ( int inputUnit = 0; inputUnit <= inputUnits; inputUnit++ ) {
                    hiddenTrace[inputUnit][hiddenUnit][outputUnit]
                            = lambda * hiddenTrace[inputUnit][hiddenUnit][outputUnit]
                            + outputDerivatedFnet[outputUnit] * outputWeights[hiddenUnit][outputUnit]
                            * (hiddenLayer[hiddenUnit] * (1 - hiddenLayer[hiddenUnit]))
                            * input[t][inputUnit];
                }
            }
        }
    }// end updateEligibilities

    /**
     * Update weight vectors
     */
    public void updateWeights() {
        for ( int outputUnit = 0; outputUnit < outputUnits; outputUnit++ ) {
            for ( int hiddenUnit = 0; hiddenUnit <= hiddenUnits; hiddenUnit++ ) {
                outputWeights[hiddenUnit][outputUnit] += beta * error[outputUnit] * outputTrace[hiddenUnit][outputUnit];
                for ( int inputUnit = 0; inputUnit <= inputUnits; inputUnit++ ) {
                    inputWeights[inputUnit][hiddenUnit] += alpha * error[outputUnit] * hiddenTrace[inputUnit][hiddenUnit][outputUnit];
                }
            }
        }
    }// end updateWeights

}
