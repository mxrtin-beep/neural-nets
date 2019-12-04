package NN;

import java.io.*;
import java.util.*;

/**
 * @author Martin Bourdev
 * @version September 24, 2019
 *
 * Perceptron2 allows the user to create a multi-layer network. The number of layers, the input values, the weights, and
 * the numbers of units come from the user in the form of instance variables or doubles in an input document.
 * The hidden layer values and output layer values come from the sum of
 * the products of previous layers with their respective weights.
 *
 */
public class Perceptron2
{
   
   private static final double MAX_LITTLE_ENDIAN_VALUE_ = 16777216.0;
   
   private final double BITMAP_SCALING_FACTOR_ = MAX_LITTLE_ENDIAN_VALUE_;
   
   private DibDump2 dibDump_;
   private Node[][] nodes_;
   private double[][] inputs_;
   private double[][] truths_;
   
   // The following constants define the end conditions for training and come from the input document.
   
   private final double MIN_ERROR_;    // The minimum error to reach before terminating training.
   
   // Initial and termination conditions for the learning factor.
   private final double INITIAL_LEARNING_FACTOR_;
   private final double LEARNING_FACTOR_CHANGE_;   // The factor by which to multiply or divide the learning factor each iteration.
   private final double MIN_LEARNING_FACTOR_;
   private final double MAX_LEARNING_FACTOR_;
   
   private final double MAX_TRAINING_ITERATIONS_;
   private double INTERMED_TRAINING_ITERATIONS_;         // The number of iterations before an intermediate image is produced
                                                         // during training
   
   private final int OUTPUT_FILE_NAME_ROW_ = 0;
   private final int OUTPUT_FILE_NAME_COL_ = 0;
   
   private final int WEIGHTS_FILE_NAME_ROW_ = 1;
   private final int WEIGHTS_FILE_NAME_COL_ = 0;
   
   private final int ACTIVATION_ROW_ = 2;    // The row in the document that holds the number of activations in each layer.
   private final int INPUT_DIM_COL_ = 0;     // The column in ACTIVATION_ROW_ that holds the number of input activations.
   
   private final int INITIAL_WEIGHT_ROW_ = 3;
   private final int INITIAL_WEIGHT_MIN_COL_ = 0;
   private final int INITIAL_WEIGHT_MAX_COL_ = 1;
   
   private final int MIN_ERROR_ROW_ = 4;     // The row in the document that holds the minimum error.
   private final int MIN_ERROR_COL_ = 0;     // The column in MIN_ERROR_ROW_ that holds the minimum error.
   
   /*
    * The row in the document that holds the initial learning factor, change in learning factor, min learning factor,
    * and max learning factor.
    */
   private final int LEARNING_FACTOR_ROW_ = 5;
   
   // The column in LEARNING_FACTOR_ROW_ that holds the initial learning factor.
   private final int INITIAL_LEARNING_FACTOR_COL_ = 0;
   
   // The column in LEARNING_FACTOR_ROW_ that holds the change in learning factor.
   private final int LEARNING_FACTOR_CHANGE_COL_ = 1;
   
   // The column in LEARNING_FACTOR_ROW_ that holds the min learning factor.
   private final int MIN_LEARNING_FACTOR_COL_ = 2;
   
   // The column in LEARNING_FACTOR_ROW_ that holds the max learning factor.
   private final int MAX_LEARNING_FACTOR_COL_ = 3;
   
   
   // The row in the document that holds the maximum number of training iterations.
   private final int TRAINING_ITERATIONS_ROW_ = 6;
   
   // The column in MAX_TRAINING_ITERATIONS_ROW_ that holds the maximum number of training iterations.
   private final int MAX_TRAINING_ITERATIONS_COL_ = 0;
   
   private final int INTERMED_TRAINING_ITERATIONS_COL_ = 1;
   
   
   
   private final int NUM_CASES_ROW_ = 7;                 // The row in the document that holds the number of cases.
   private final int NUM_CASES_COL_ = 0;                 // The column in NUM_CASES_ROW_ that holds the number of cases.
   
   private final int TRUTH_TABLE_ROW_ = 8;               // The row in the document where the truth table starts.
   
   private double INITIAL_WEIGHT_MIN_;
   private double INITIAL_WEIGHT_MAX_;
   
   private int NUM_CASES_;          // The number of cases.
   private int INPUT_DIM_;          // The number of input activations.
   private int NUM_LAYERS_;         // The number of layers.
   private int OUTPUT_DIM_;         // The number of output activations.
   private int OUTPUT_LAYER_INDEX_; // The index of the output layer.
   
   private String outputFileName_;
   private String weightsFileName_;
   
   // The following constants represent the state of the perceptron while training and are only used in the method train().
   
   private final int DEFAULT_TRAINING_STATE_ = 0;  // The perceptron is still training.
   private final int END_MIN_ERROR_ = 1;           // The perceptron has ended training due to reaching the minimum error.
   
   // The perceptron has ended training due to the error not changing across training iterations
   private final int END_ERROR_CONSTANT_ = 2;
   
   // The perceptron has ended training due to reaching the minimum learning factor.
   private final int END_MIN_LAMBDA_ = 3;
   
   // The perceptron has ended training due to reaching the maximum learning factor.
   private final int END_MAX_LAMBDA_ = 4;
   
   // The perceptron has ended training due to reaching the maximum number of iterations
   private final int END_MAX_ITERATIONS_ = 5;
   
   public final int MILLISEC_PER_SEC_ = 1000;
   public final int SECS_PER_MIN_ = 60;
   
   /**
    * Perceptron2 constructor.
    * Collects configurable data from a file given a filename. Initializes the inputs_ and truths_ array based on document
    * values. Initializes the nodes_ array.
    *
    * @param filename The name of the file from which to retrieve configurable data.
    */
   public Perceptron2 (String filename)
   {
      String[][] rawData = getRawData(filename); // Stores the data from the input document into a matrix.
      
      
      
      /*
       * Initializes the start and end conditions for training.
       * Initializes the minimum error, initial learning factor, change in learning factor, maximum
       * learning factor, minimum learning factor, and maximum training iterations.
       */
      MIN_ERROR_ = Double.parseDouble(rawData[MIN_ERROR_ROW_][MIN_ERROR_COL_]); // Minimum error is the first value of MIN_ERROR_ROW_.
      
      // Initializes the learning factor start and end conditions based on their respective columns.
      INITIAL_LEARNING_FACTOR_ = Double.parseDouble(rawData[LEARNING_FACTOR_ROW_][INITIAL_LEARNING_FACTOR_COL_]);
      LEARNING_FACTOR_CHANGE_ = Double.parseDouble(rawData[LEARNING_FACTOR_ROW_][LEARNING_FACTOR_CHANGE_COL_]);
      MIN_LEARNING_FACTOR_ = Double.parseDouble(rawData[LEARNING_FACTOR_ROW_][MIN_LEARNING_FACTOR_COL_]);
      MAX_LEARNING_FACTOR_ = Double.parseDouble(rawData[LEARNING_FACTOR_ROW_][MAX_LEARNING_FACTOR_COL_]);
      
      MAX_TRAINING_ITERATIONS_ = Double.parseDouble(rawData[TRAINING_ITERATIONS_ROW_][MAX_TRAINING_ITERATIONS_COL_]);
      
      
      NUM_CASES_ = (int) Double.parseDouble(rawData[NUM_CASES_ROW_][NUM_CASES_COL_]);
      INPUT_DIM_ = (int) Double.parseDouble(rawData[ACTIVATION_ROW_][INPUT_DIM_COL_]); // The number of input activations.
      NUM_LAYERS_ = rawData[ACTIVATION_ROW_].length;
      OUTPUT_LAYER_INDEX_ = NUM_LAYERS_ - 1;
      OUTPUT_DIM_ = (int) Double.parseDouble(rawData[ACTIVATION_ROW_][OUTPUT_LAYER_INDEX_]);
      
      
      inputs_ = new double[NUM_CASES_][INPUT_DIM_];    // Will store all input arrays.
      truths_ = new double[NUM_CASES_][OUTPUT_DIM_];   // Will store all truths.
      
      // Initializes the inputs array and truths array.
      for (int i = 0; i < NUM_CASES_; i++) // Iterates through all cases.
      {
         for (int j = 0; j < INPUT_DIM_; j++) // Iterates through all input Nodes.
         {
            inputs_[i][j] = Double.parseDouble(rawData[TRUTH_TABLE_ROW_ + i][j]); // Initializes input array.
         }
         for (int j = 0; j < OUTPUT_DIM_; j++) // Iterates through the number of outputs, which equals the number of truths.
         {
            // Initializes the truths, skipping the inputs in the truth table by adding INPUT_DIM_ to the second index.
            truths_[i][j] = Double.parseDouble(rawData[TRUTH_TABLE_ROW_ + i][INPUT_DIM_ + j]);
         }
      }
      
      nodes_ = new Node[NUM_LAYERS_][]; // Creates the 2D array of Nodes with a length of numLayers.
      
      /*
       * Stores the size of the previous layer of activations, starting with zero.
       * This number is required to create a Node, which takes in the size of the previous
       * layer to create the array of weights going into that Node. The inputs have no weights going into
       * them, so the previous size starts at zero.
       */
      int prevLayerSize = 0;
      int layerSize;
      
      // Initializes the array of Nodes.
      for (int i = 0; i < NUM_LAYERS_; i++)
      {
         /*
          * Defines the size of the current layer being initialized as the ith value in the activation
          * row of the document.
          */
         layerSize = (int) Double.parseDouble(rawData[ACTIVATION_ROW_][i]);
         
         nodes_[i] = new Node[layerSize];
         
         for (int j = 0; j < layerSize; j++) // Iterates through the number of Nodes in the current layer.
         {
            // Takes in a number of weights to feed into the Node equal to the size of the previous layer.
            nodes_[i][j] = new Node(prevLayerSize, -1, 1);
         }
         prevLayerSize = layerSize; // Updates the previous layer size.
         
      } // for (int i = 0; i < NUM_LAYERS_; i++)
   }
   
   /**
    * Perceptron2 constructor for image stuff.
    * @param config  The config file.
    * @param usingInputImage
    * @param usingOutputImage
    */
   public Perceptron2 (String config, boolean usingInputImage, boolean usingOutputImage, boolean usingWeightsFile)
   {
      String[][] rawData = getRawData(config); // Stores the data from the input document into a matrix.
   
      
      outputFileName_ = rawData[OUTPUT_FILE_NAME_ROW_][OUTPUT_FILE_NAME_COL_];
      weightsFileName_ = rawData[WEIGHTS_FILE_NAME_ROW_][WEIGHTS_FILE_NAME_COL_];
      
      /*
       * Initializes the start and end conditions for training.
       * Initializes the minimum error, initial learning factor, change in learning factor, maximum
       * learning factor, minimum learning factor, and maximum training iterations.
       */
      MIN_ERROR_ = Double.parseDouble(rawData[MIN_ERROR_ROW_][MIN_ERROR_COL_]); // Minimum error is the first value of MIN_ERROR_ROW_.
      
      // Initializes the learning factor start and end conditions based on their respective columns.
      INITIAL_LEARNING_FACTOR_ = Double.parseDouble(rawData[LEARNING_FACTOR_ROW_][INITIAL_LEARNING_FACTOR_COL_]);
      LEARNING_FACTOR_CHANGE_ = Double.parseDouble(rawData[LEARNING_FACTOR_ROW_][LEARNING_FACTOR_CHANGE_COL_]);
      MIN_LEARNING_FACTOR_ = Double.parseDouble(rawData[LEARNING_FACTOR_ROW_][MIN_LEARNING_FACTOR_COL_]);
      MAX_LEARNING_FACTOR_ = Double.parseDouble(rawData[LEARNING_FACTOR_ROW_][MAX_LEARNING_FACTOR_COL_]);
      
      MAX_TRAINING_ITERATIONS_ = Double.parseDouble(rawData[TRAINING_ITERATIONS_ROW_][MAX_TRAINING_ITERATIONS_COL_]);
      INTERMED_TRAINING_ITERATIONS_ = Double.parseDouble(rawData[TRAINING_ITERATIONS_ROW_][INTERMED_TRAINING_ITERATIONS_COL_]);
      
      INITIAL_WEIGHT_MIN_ = Double.parseDouble(rawData[INITIAL_WEIGHT_ROW_][INITIAL_WEIGHT_MIN_COL_]);
      INITIAL_WEIGHT_MAX_ = Double.parseDouble(rawData[INITIAL_WEIGHT_ROW_][INITIAL_WEIGHT_MAX_COL_]);
      
      NUM_CASES_ = Integer.parseInt(rawData[NUM_CASES_ROW_][NUM_CASES_COL_]);
   
   
      dibDump_ = new DibDump2();
      
      int truthStartCol;
      /*
      * Sets the value of INPUT_DIM_ to the number of pels in inputImgData if the perceptron is using
      * an input image, and the first value of the ACTIVATION_ROW_ of the config document if
      * the perceptron is not using an input image.
       */
      if(usingInputImage)
      {
         String sampleInputFile = rawData[TRUTH_TABLE_ROW_][0];
         createPelsFile(sampleInputFile, "inPels");
         String[][] inputImgData = getRawData("inPels");
         INPUT_DIM_ = inputImgData.length;
         assert(INPUT_DIM_ > 1);
         truthStartCol = 1;
      }
      else
      {
         INPUT_DIM_ = (int) Double.parseDouble(rawData[ACTIVATION_ROW_][INPUT_DIM_COL_]); // The number of input activations.
         truthStartCol = INPUT_DIM_;
      }
      
      NUM_LAYERS_ = rawData[ACTIVATION_ROW_].length;
      
      OUTPUT_LAYER_INDEX_ = NUM_LAYERS_ - 1;
      
      /*
       * Sets the value of OUPUT_DIM_ to the number of pels in outputImgData if the perceptron is using
       * an output image, and the last value of the ACTIVATION_ROW_ of the config document if
       * the perceptron is not using an output image.
       */
      if (usingOutputImage)
      {
         int lastTruthIndex = rawData[TRUTH_TABLE_ROW_].length - 1;
         String sampleOutputFile = rawData[TRUTH_TABLE_ROW_][lastTruthIndex];
         createPelsFile(sampleOutputFile, "outPels");
         String[][] outputImgData = getRawData("outPels");
         OUTPUT_DIM_ = outputImgData.length;
         assert(INPUT_DIM_ > 1);
         OUTPUT_DIM_ = outputImgData.length;
      }
      else
      {
         OUTPUT_DIM_ = (int) Double.parseDouble(rawData[ACTIVATION_ROW_][OUTPUT_LAYER_INDEX_]);
      }
      
      
      
      
      inputs_ = new double[NUM_CASES_][INPUT_DIM_];    // Will store all input arrays.
      truths_ = new double[NUM_CASES_][OUTPUT_DIM_];   // Will store all truths.
      
      // Initializes the inputs array and truths array.
      for (int i = 0; i < NUM_CASES_; i++) // Iterates through all cases.
      {
         if (!usingInputImage) // If not using input image, get inputs from truth table in config file.
         {
            for (int j = 0; j < INPUT_DIM_; j++) // Iterates through all input Nodes.
            {
               inputs_[i][j] = Double.parseDouble(rawData[TRUTH_TABLE_ROW_ + i][j]); // Initializes input array.
            }
         }
         else // If using input image, convert file into pels file, and use pels file as inputs.
         {
            String tableInput = rawData[TRUTH_TABLE_ROW_ + i][0];
            createPelsFile(tableInput, "inPels" + i);
            String[][] tableInputData = getRawData("SCALED " + "inPels");
            for (int j = 0; j < INPUT_DIM_; j++)
            {
               inputs_[i][j] = Double.parseDouble(tableInputData[j][0]);
            }
         }
         
         
         if (!usingOutputImage) // If not using input image, get inputs from truth table in config file.
         {
            for (int j = 0; j < OUTPUT_DIM_; j++) // Iterates through the number of outputs, which equals the number of truths.
            {
               // Initializes the truths, skipping the inputs in the truth table by adding INPUT_DIM_ to the second index.
               truths_[i][j] = Double.parseDouble(rawData[TRUTH_TABLE_ROW_ + i][truthStartCol + j]);
            }
         }
         else // If using input image, convert file into pels file, and use pels file as inputs.
         {
            String tableOutput = rawData[TRUTH_TABLE_ROW_ + i][rawData[TRUTH_TABLE_ROW_ + i].length - 1];
            createPelsFile(tableOutput, "outPels" + i);
            String[][] tableInputData = getRawData("SCALED " + "outPels");
            for (int j = 0; j < OUTPUT_DIM_; j++)
            {
               truths_[i][j] = Double.parseDouble(tableInputData[j][0]);
            }
         }
         
      }
      
      nodes_ = new Node[NUM_LAYERS_][]; // Creates the 2D array of Nodes with a length of numLayers.
      
      /*
       * Stores the size of the previous layer of activations, starting with zero.
       * This number is required to create a Node, which takes in the size of the previous
       * layer to create the array of weights going into that Node. The inputs have no weights going into
       * them, so the previous size starts at zero.
       */
      int prevLayerSize = 0;
      int layerSize;
      
      // Initializes the array of Nodes.
      for (int i = 0; i < NUM_LAYERS_; i++)
      {
         /*
          * Defines the size of the current layer being initialized as the ith value in the activation
          * row of the document.
          */
         if (i == 0)
         {
            layerSize = INPUT_DIM_;
         }
         else if (i == OUTPUT_LAYER_INDEX_)
         {
            layerSize = OUTPUT_DIM_;
         }
         else
         {
            layerSize = (int) Double.parseDouble(rawData[ACTIVATION_ROW_][i]);
         }
         
         
         nodes_[i] = new Node[layerSize];
         
         for (int j = 0; j < layerSize; j++) // Iterates through the number of Nodes in the current layer.
         {
            // Takes in a number of weights to feed into the Node equal to the size of the previous layer.
            nodes_[i][j] = new Node(prevLayerSize, INITIAL_WEIGHT_MIN_, INITIAL_WEIGHT_MAX_);
         }
         prevLayerSize = layerSize; // Updates the previous layer size.
         
      } // for (int i = 0; i < NUM_LAYERS_; i++)
      
      if(usingWeightsFile)
      {
         setWeightsFromFile(weightsFileName_);
      }
   }
   
   /**
    * Converts an integer array to a double array.
    * @param ints The given integer array.
    * @return  A double array with the same values as the integer array.
    */
   public double[][] intArrayToDoubleArray(int[][] ints)
   {
      double[][] dubs = new double[ints.length][];
      for(int i = 0; i < ints.length; i++)
      {
         dubs[i] = new double[ints[i].length];
         for(int j = 0; j < ints[i].length; j++)
         {
            dubs[i][j] = ints[i][j];
         }
      }
      return dubs;
   }
   /**
    * Creates a file of pels from a given image.
    * @param inFileName The name of the given image.
    * @param outFileName   The desired name of the pels file.
    */
   public void createPelsFile(String inFileName, String outFileName)
   {
      int[][] pels = dibDump_.getPels(inFileName);
   
      double[][] scaledPels = new double[pels.length][];
      
      for (int i = 0; i < pels.length; i++)
      {
         scaledPels[i] = new double[pels[i].length];
         for (int j = 0 ; j < pels[i].length; j++)
         {
            scaledPels[i][j] = scalePel(pels[i][j]);
         }
      }
      
      dibDump_.doubleArrayToFile("UNSCALED " + outFileName, intArrayToDoubleArray(pels));
      dibDump_.doubleArrayToFile("SCALED " + outFileName, scaledPels);
   }
   
   /**
    * Converts the given pel from a bitmap to a pel that can be interpreted by the perceptron.
    * @param i The pel from a bitmap.
    * @return  The same pel that can be interpreted by the perceptron.
    */
   public double scalePel(int i)
   {
      return (double) (dibDump_.swapInt(i)) / (BITMAP_SCALING_FACTOR_);
   }
   
   /**
    * Converts the given pel that the perceptron interprets to a pel that can be converted into a bitmap.
    * @param d The pel that the perceptron interprets.
    * @return  The same pel that can be converted into a bitmap.
    */
   public int unscalePel(double d)
   {
      return dibDump_.swapInt((int) (d * BITMAP_SCALING_FACTOR_));
   }
   /**
    * Given an array of inputs, initializes the activations of the first layer of Nodes.  Initializes the activations of
    * the hidden layer and output layer Nodes by calling the forward function of each Node after the input layer, passing
    * in the previous layer of Nodes.  Returns the activation of the first Node in the last layer.
    *
    * @param inputs  An array of the input values.
    * @return  The activation of the first Node in the last layer.
    */
   public double[] forward(double[] inputs)
   {
      assert(inputs.length == nodes_[0].length);
      
      // Sets the activations of the input Nodes of the perceptron to those of the given array of inputs.
      for (int i = 0; i < inputs.length; i++)
      {
         nodes_[0][i].activation_ = inputs[i];
      }
      
      /*
       * Calls the forward function of every Node after the input layer, passing in the
       * previous layer of Nodes. 
       * Iterates through layers, starting with 1 because the inputs (layer 0) are already initialized.
       */
      for (int layer = 1; layer < nodes_.length; layer++)
      {
         for (int i = 0; i < nodes_[layer].length; i++)
         {
            // Sets the activation of the Node to the sigmoid of the weighted sum of the activations of the previous layer.
            nodes_[layer][i].forward(nodes_[layer - 1]);
         }
      }
      
      // Create array of outputs to return.
      double[] outputActivations = new double[OUTPUT_DIM_];
      for (int i = 0; i < OUTPUT_DIM_; i++)
      {
         outputActivations[i] = nodes_[OUTPUT_LAYER_INDEX_][i].activation_;
      }
      return outputActivations;
   }
   
   /**
    * Given the truths and the outputs, calculates the weight gradient and adds it to the weight gradient sum for every Node,
    * except those of the first layer, which do not have weights feeding into them and therefore do not have
    * weight gradient sums.
    *
    * @param truths   The expected outputs of the net.
    * @param outputs  The actual outputs of the net.
    */
   public void backward(double[] truths, double[] outputs)
   {
      // Sets the incoming gradient of all Nodes in the output layer to -(truth - output).
      for (int i = 0; i < OUTPUT_DIM_; i++)
      {
         nodes_[OUTPUT_LAYER_INDEX_][i].incomingGradient_ = -(truths[i] - outputs[i]);
      }
      
      
      /*
       * First, sets the incoming gradients of all Nodes to zero, other than those of the last layer.
       * Calls the method backward on every Node  starting with the last one and going backward, passing in the previous
       * layer of Nodes. 
       *
       * Iterates through the layers, starting with the last layer and ending with the first hidden activation layer.
       */
      for (int layer = nodes_.length - 1; layer > 0; layer--)
      {
         
         // Sets the incoming gradient of the Nodes in the previous layer to 0.
         for (int i = 0; i < nodes_[layer - 1].length; i++)
         {
            nodes_[layer - 1][i].incomingGradient_ = 0;
         }
         
         /*
          * Calls backward on each Node in the current layer.
          * Backward adds to the weight gradient sum based on the incoming gradient, activation, and previous layer.
          * Backward also adds to the incoming gradient of the previous layer of Nodes. 
          */
         for (int i = 0; i < nodes_[layer].length; i++)
         {
            nodes_[layer][i].backward(nodes_[layer - 1]);
         }
      } // for (int layer = nodes_.length - 1; layer > 0; layer--)
   }
   
   /**
    * Returns the sum of the average weight gradients of every Node  except those of the input layer,
    * representing how much weights are changing.
    * The number is expected to be large if the error is large and small if the error is small.
    *
    * @return The sum of the average weight gradients of every Node  except those of the input layer.
    */
   private double getAvgGradientMagnitude()
   {
      double mag = 0;
      for (int layer = 1; layer < nodes_.length; layer++) // Iterates through the hidden and output layers.
      {
         for (int i = 0; i < nodes_[layer].length; i++) // Iterates through each Node in the current layer.
         {
            mag += nodes_[layer][i].getAvgGradientMagnitude(); // Returns the average gradient of all weights of the Node
         }
      }
      return mag;
   }
   
   /**
    * Returns the greatest weight gradient sum of all weights.
    * @return The greatest weight gradient sum.
    */
   public double getMaxGradientMagnitude()
   {
      double max = 0;
      for (int layer = 1; layer < nodes_.length; layer++) // Iterates through the hidden and output layers.
      {
         for (int i = 0; i < nodes_[layer].length; i++) // Iterates through each Node in the current layer.
         {
            for(int j = 0; j < nodes_[layer][i].weights_.length; j++)
            {
               double currentWeightGrad = nodes_[layer][i].weightsGradientSum_[j];
               if(currentWeightGrad > max)
               {
                  max = currentWeightGrad;
               }
            }
         }
      }
      return max;
   }
   /**
    * Updates the weights of all Nodes based on the learning factor.
    *
    * @param learningFactor   The factor to multiply the weight gradient of each Node 
    */
   public void updateWeights(double learningFactor)
   {
      for (int i = 0; i < nodes_.length; i++)
      {
         for (int j = 0; j < nodes_[i].length; j++)
         {
            /*
             * Calls updateWeights on every Node in the network,
             * which adds the weight gradient sums of each Node multiplied by the learning factor
             * to the respective weights of the Node 
             */
            nodes_[i][j].updateWeights(learningFactor);
         }
      }
   }
   
   /**
    * Returns the array of errors based on one case.
    *
    * @param inputs  The inputs to feed into the net.
    * @param truths   The expected output array.
    * @return  The sum of the squares of the differences between the outputs and truths, divided by two.
    */
   public double[] getErrorVector(double[] inputs, double[] truths)
   {
      double[] outputs = forward(inputs);
      double[] errors = new double[outputs.length];
      double difference;
      
      for (int i = 0; i < outputs.length; i++)
      {
         difference = outputs[i] - truths[i];
         errors[i] = 0.5 * difference * difference;
      }
      return errors;
   }
   
   /**
    * Updates the weightsGradientSum of every Node based on the given inputs and truth.
    *
    * @param inputs  The array of inputs to feed into the net.
    * @param truths   The expected outputs.
    */
   public void trainIteration(double[] inputs, double[] truths)
   {
      double[] outputs = forward(inputs);
      backward(truths, outputs);
   }
   
   /**
    * Runs trainIteration for all sets of inputs.
    */
   public void trainAllIterations()
   {
      for (int i = 0; i < inputs_.length; i++)
      {
         trainIteration(inputs_[i], truths_[i]);
      }
   }
   
   /**
    * Sets the weightGradientSum of all weights to zero.
    */
   public void resetWeightGradientSums()
   {
      for (int i = 0; i < nodes_.length; i++)
      {
         for (int j = 0; j < nodes_[i].length; j++)
         {
            nodes_[i][j].resetWeightGradientSums();
         }
      }
   }
   
   
   /**
    * Returns an array of average errors for all output activations over all cases.
    * The size of the array is equal to the number of outputs.
    *
    * @return an array that contains the average errors for all outputs over all cases.
    */
   public double[] getAverageErrorVector()
   {
      double[] sums = new double[OUTPUT_DIM_];
      for (int i = 0; i < NUM_CASES_; i++) // Iterates through number of cases.
      {
         double[] caseError = getErrorVector(inputs_[i], truths_[i]);
         for (int j = 0; j < OUTPUT_DIM_; j++) // Iterates through number of outputs.
         {
            sums[j] += caseError[j] / NUM_CASES_;
         }
      }
      return sums;
   }
   
   /**
    * Returns the error in a scalar form to be used in training.
    *
    * @return the average of all values in getAverageErrorVector().
    */
   public double getAverageErrorScalar()
   {
      double[] averageErrors = getAverageErrorVector();
      double sum = 0.0;
      for (int i = 0; i < averageErrors.length; i++)
      {
         sum += averageErrors[i];
      }
      return sum / OUTPUT_DIM_;
   }
   
   
   /**
    * Minimizes the error based on the principle of steepest descent. Until the average error is less than the
    * minimum error, the method computes the error before and after one iteration of trainIteration for one case.
    * The method updates the training factor, computes the average error across all cases, and updates the hidden
    * activations and output based on the new weights.
    * Pauses every 100 iterations and prints the index, change in error, learning factor, average magnitude of the
    * weight gradient, and average error every iteration.
    */
   public void train()
   {
      double averageError;
      int index = 0;
      double learningFactor = INITIAL_LEARNING_FACTOR_;
      double errorBefore;
      double mag;
      double error;
      double magMax;
      
      /*
       * Training ends when done is not zero.
       * Zero signifies that the perceptron is still training.
       * One signifies that the minimum average error has been reached.
       * Two signifies that the average error has not changed over a training iteration.
       * Three signifies that the minimum learning factor has been reached.
       * Four signifies that the maximum learning factor has been reached.
       * Five signifies that the maximum number of iterations has been reached.
       */
      int done = DEFAULT_TRAINING_STATE_;
      
      
      
      do 
      { //  while (done == DEFAULT_TRAINING_STATE_);
         
         backupWeights();              // Stores a backup of weights in case they need to be rolled back.
         resetWeightGradientSums();    // Sets the weight gradient sums to zero.
         
         errorBefore =  getAverageErrorScalar();   // Stores the error before weights are updated.
         trainAllIterations();                     // Creates new weight gradient sums.
         updateWeights(learningFactor);            // Updates the weights based on the new sums and lambda.
         
         error = getAverageErrorScalar();          // Stores the error after weights are updated.
         
         // Changes learning factor based on the change in error.
         if (errorBefore > error)
         {
            learningFactor *= LEARNING_FACTOR_CHANGE_;
         }
         else if (errorBefore < error)
         {
            restoreWeights(); // Rolls back all the weights.
            learningFactor /= LEARNING_FACTOR_CHANGE_;
         }
         
         
         averageError = getAverageErrorScalar(); // Computes the average error of all cases to update the end condition.
   
         /*
          * Average magnitude of all weight gradients of all weights in the network.
          * Expected to be large if error is large, small if error is small.
          */
         mag = getAvgGradientMagnitude();
         
         magMax = getMaxGradientMagnitude();
         
         System.out.println("DEBUG-- Index: " + index + " Error: " + errorBefore + "->" + error
                                    + " lambda:" + learningFactor + " MaxGradMag: " + magMax
                                    + " AvgGradSum: " + mag + " AvgErrorScalar: " + averageError);
         
         index++;
         
         if(index % INTERMED_TRAINING_ITERATIONS_ == 0)
         {
            convertOutputsToBMPs("intermed", index);
         }
         // Checking end conditions.
         if (averageError <= MIN_ERROR_) // The average error has reached the minimum.
         {
            done = END_MIN_ERROR_;
         }
         if (errorBefore == error) // The error before and after changing weights has not changed.
         {
            done = END_ERROR_CONSTANT_;
         }
         if (learningFactor <= MIN_LEARNING_FACTOR_) // The learning factor is less than or equal to the minimum.
         {
            done = END_MIN_LAMBDA_;
         }
         if (learningFactor >= MAX_LEARNING_FACTOR_) // The learning factor is greater than or equal to the maximum.
         {
            done = END_MAX_LAMBDA_;
         }
         if (index >= MAX_TRAINING_ITERATIONS_) // The index has reached the maximum number of training iterations.
         {
            done = END_MAX_ITERATIONS_;
         }
         
      } while (done == DEFAULT_TRAINING_STATE_); // Ends once done is changed.
      
      
      // Sets the error message based on the value of done.
      String errorMessage = "";
      if (done == END_MIN_ERROR_)
      {
         errorMessage = "The minimum average error has been reached.";
      }
      else if (done == END_ERROR_CONSTANT_)
      {
         errorMessage = "The initial and final average errors are equal.";
      }
      else if (done == END_MIN_LAMBDA_)
      {
         errorMessage = "The learning factor is less than or equal to the minimum.";
      }
      else if (done == END_MAX_LAMBDA_)
      {
         errorMessage = "The learning factor is greater than or equal to the maximum.";
      }
      else if (done == END_MAX_ITERATIONS_)
      {
         errorMessage = "The maximum number of iterations has been reached.";
      }
      System.out.println(errorMessage + "\n");
   }
   
   /**
    *  All Nodes store a backup of their current weights.
    */
   public void backupWeights()
   {
      for (int i = 0; i < nodes_.length; i++)
      {
         for (int j = 0; j < nodes_[i].length; j++)
         {
            nodes_[i][j].backupWeights();
         }
      }
   }
   
   /**
    *  All Nodes change their weights to their backups.
    */
   public void restoreWeights()
   {
      for (int i = 0; i < nodes_.length; i++)
      {
         for (int j = 0; j < nodes_[i].length; j++)
         {
            nodes_[i][j].restoreWeights();
         }
      }
   }
   
   /**
    * Returns a jagged 2D double array of the values in the input document.
    * The first index represents the line, and the second index represents the double in that line.
    *
    * @return a 2D double array of input document values.
    */
   public String[][] getRawData(String s)
   {
      //String s = "/Users/martinbourdev/IdeaProjects/Perceptron1/text/input";
      try 
      {
         BufferedReader counter1 = new BufferedReader(new FileReader(s));
         
         // Counts the number of lines in the document.
         String line = counter1.readLine();
         int lines = 0;
         while (line != null)
         {
            lines++;
            line = counter1.readLine();
         }
         counter1.close();
         String[][] vals = new String[lines][];
         
         
         // Assigns each row of the vals array to the same row in the document.
         BufferedReader counter2 = new BufferedReader(new FileReader(s));
         line = counter2.readLine();
         for (int i = 0; i < lines; i++)
         {
            vals[i] = parseStrings(line);
            line = counter2.readLine();
         }
         counter2.close();
         return vals;
      }
      catch (IOException e)
      {
         throw new IllegalArgumentException(e.toString());
      }
   }
   
   /**
    * Separates a given String by spaces and puts each piece into an array.
    * @param s
    * @return
    */
   public String[] parseStrings(String s)
   {
      Scanner sc = new Scanner(s);
   
      // Counts how many words in the String to initialize the length of the array.
      int length = 0;
      while (sc.hasNext())
      {
         sc.next();
         length++;
      }
      sc.close();
      String[] words = new String[length];
   
      // Assigns each double to the array.
      Scanner sc2 = new Scanner(s);
      for (int i = 0; i < length; i++)
      {
         words[i] = sc2.next();
      }
      return words;
   }
   
   /**
    * Parses a string and returns an array of doubles.
    * S must contain doubles separated by spaces.
    *
    * @param s the given string.
    * @return a double[] of doubles in the string.
    */
   public double[] parseDoubles(String s)
   {
      Scanner sc = new Scanner(s);
      
      // Counts how many doubles in the String to initialize the length of the array.
      int length = 0;
      while (sc.hasNextDouble())
      {
         length++;
         sc.nextDouble();
      }
      sc.close();
      double[] doubles = new double[length];
      
      // Assigns each double to the array.
      Scanner sc2 = new Scanner(s);
      for (int i = 0; i < length; i++)
      {
         doubles[i] = sc2.nextDouble();
      }
      return doubles;
   }
   
   /**
    * Prints the weights and activations of each layer and the input and expected values.
    */
   public void print()
   {
      // Indices for weights to be printed.
      int previousLayer;
      int startActivation;
      int endActivation;
      
      for (int i = 0; i < nodes_.length; i++) // Iterates through each layer of activations.
      {
         System.out.println("LAYER " + i);
         for (int j = 0; j < nodes_[i].length; j++)   // Iterates through each activation in layer i.
         {
            Node n = nodes_[i][j];
            for (int w = 0; w < n.weights_.length; w++)   // Iterates through each weight feeding into the current activation.
            {
               previousLayer = i - 1;
               startActivation = w;
               endActivation = j;
               
               System.out.println("w" + previousLayer + startActivation + endActivation +
                                          " : " + n.weights_[startActivation]);    // Prints the weight index and value.
            }
            System.out.println("a" + i + j + " : " + n.activation_ + "\n");   // Prints the activation of the current Node 
            
         } // for (int j = 0; j < nodes_[i].length; j++)   // Iterates through each activation in layer i.
      } // for (int i = 0; i < nodes_.length; i++) // Iterates through each layer of activations.
   }
   
   /**
    * Prints the Nodes, weights, and errors for each case.
    */
   public void printAll()
   {
      for (int i = 0; i < inputs_.length; i++) // Iterates through each case.
      {
         System.out.println("***** CASE " + i + " ******");
         
         forward(inputs_[i]);    // Updates all the activations.
         //print();                // Prints all weights and activations of the network.
         
         
         // Prints all the truths for case i.
         String truthsString = "";
         for (int j = 0; j < truths_[i].length; j++)
         {
            truthsString += truths_[i][j];
            truthsString += "\t ";
         }
         System.out.println("TRUTHS: \t" + truthsString);
         
         
         // Prints all the outputs for case i.
         String outputString = "";
         for (int j = 0; j < OUTPUT_DIM_; j++)
         {
            outputString += nodes_[OUTPUT_LAYER_INDEX_][j].activation_;
            outputString += "\t ";
         }
         System.out.println("OUTPUTS: \t" + outputString);
         
         
         // Prints all the errors for case i.
         String errorsString = "";
         double[] errors = getErrorVector(inputs_[i], truths_[i]);
         for (int j = 0; j < truths_[i].length; j++)
         {
            errorsString += errors[j];
            errorsString += "\t ";
         }
         System.out.println("ERRORS: \t" + errorsString + "\n");
         
      } // for (int i = 0; i < inputs_.length; i++) // Iterates through each case.
   }
   
   public void writeOutputsToFile()
   {
   
   }
   
   public double sigmoid(double x)
   {
      return 1.0 / (1 + Math.exp(-x));
   }
   
   public String[][] scale(String [][] ints)
   {
      String[][] dubs = new String[ints.length][];
      for (int i = 0; i < ints.length; i++)
      {
         dubs[i] = new String[ints[i].length];
         for (int j = 0; j < ints[i].length; j++)
         {
            dubs[i][j] = "" + (double) Integer.parseInt(ints[i][j]);
         }
      }
      return dubs;
   }
   
   /**
    * Prints all the activation values of the network.
    */
   public void printActivations()
   {
      for (int i = 0; i < nodes_.length; i++)
      {
         for (int j = 0; j < nodes_[i].length; j++)
         {
            System.out.println("a" + i + j + ": " + nodes_[i][j].activation_);
         }
      }
   }
   
   /**
    * Converts all the outputs to BMP files with an index.
    * @param filename The name of the ouptut file.
    */
   public void convertOutputsToBMPs(String filename, int index)
   {
      int[][] intOutputs = new int[0][0];
      for (int i = 0; i < NUM_CASES_; i++)
      {
         double[] doubleOutputs = forward(inputs_[0]);
         intOutputs = new int[doubleOutputs.length][1];
         try
         {
            PrintWriter out = new PrintWriter(new FileWriter("finalOutPels" + i));
            for (int j = 0; j < OUTPUT_DIM_; j++)
            {
               int unscaledPel = unscalePel(doubleOutputs[j]);
               out.println(unscaledPel);
               intOutputs[j][0] = unscaledPel;
            }
            out.close();
            int[][] pelsIntArray = dibDump_.pelsFileToArray("finalOutPels" + i);
            dibDump_.writeBMPFile(filename + i + "-" + index + ".bmp", pelsIntArray);
         }
         catch(IOException e)
         {
            throw new RuntimeException(e);
         }
      }
      
      assert(intOutputs.length != 0);
      
   }
   
   /**
    * Writes the weights of the network to a file.
    * @param filename   The desired name of the file.
    */
   public void writeWeightsToFile(String filename)
   {
      ArrayList<String> weights = new ArrayList<String>();
      for(int i = 0; i < nodes_.length; i++)
      {
         for(int j = 0; j < nodes_[i].length; j++)
         {
            Node n = nodes_[i][j];
            for(int k = 0; k < n.weights_.length; k++)
            {
               weights.add("" + n.weights_[k]);
            }
         }
      } // for(int i = 0; i < nodes_.length; i++)
      
      Iterator<String> weightItr = weights.iterator();
      
      FileUtil.saveFile(filename, weightItr);
   }
   
   /**
    * Sets the weights of the network to the doubles in the file.
    * @param filename   The name of the file.
    */
   public void setWeightsFromFile(String filename)
   {
      String[][] newWeights = getRawData(filename);
      int index = 0;
      
      for(int i = 0; i < nodes_.length; i++)
      {
         for(int j = 0; j < nodes_[i].length; j++)
         {
            Node n = nodes_[i][j];
            for(int k = 0; k < n.weights_.length; k++)
            {
               n.weights_[k] = Double.parseDouble(newWeights[index][0]);
               index++;
            }
         }
      } // for(int i = 0; i < nodes_.length; i++)
   }
   /**
    * Creates a new Perceptron and has it call the test() method.
    * @param args The arguments for the main method.
    */
   public static void main(String[] args)
   {
      double time1 = System.currentTimeMillis();
      Perceptron2 p = new Perceptron2("text/input", true, true, false);
      p.createPelsFile("smallA.bmp", "smallApels");
//      p.train();
//      p.writeWeightsToFile("weights");
//      p.convertOutputsToBMPs("finalImage", 0);
//
//
//      p.printAll();
//      double time2 = System.currentTimeMillis();
//      double timeElapsedMillis = time2 - time1;
//      double timeElapsedSec = timeElapsedMillis/p.MILLISEC_PER_SEC_;
//      double timeElapsedMin = timeElapsedSec/p.SECS_PER_MIN_;
//      System.out.println("Time Elapsed: " + timeElapsedMin + " minutes.");
      //p.print();
      //p.printActivations();
   }
   
}