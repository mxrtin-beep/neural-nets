package NN;

import java.lang.Math;
import java.io.*;
import java.util.*;

/**
 * Perceptron allows the user to create a multi-layer network. The number of layers, the input values, the weights, and
 * the numbers of units come from the user in the form of instance variables or doubles in an input document.
 * The hidden layer values and output layer values come from the sum of
 * the products of previous layers with their respective weights. The main function creates the arrays of weights
 * and units, assigns them values, calculates the error, and prints all of it.
 *
 *
 * createUnitArr()            Creates the array of units.
 * output(double)             Passes the given double through an output function.
 * createWeightArr()          Creates the array of weights.
 * assignWeights()            Assigns the weights values from either the input doc or the weight array.
 * getInputs()                Converts the values from the input doc into an ArrayList.
 * parseString(String)        Converts the given String of doubles separated by spaces into an array of doubles.
 * assignInputs()             Assigns the first layer of units values from either the input doc or the input array.
 * intRows()                  Returns an int[] of the values of docValues[rowsRow], which represents the
 *                            number of activation units in each layer.
 * assignHiddens()            Assigns the hidden layers values based on the sums of the products of previous layers
 *                            and their respective weights.
 * printNet()                 Prints all the units and weights of the net, with indices and values, as well as the error.
 * calculateError()           Calculates the error.
 * calculateConvenience()     Calculates the convenience.
 * runCurrentWeights()        Runs the net for all input matrices and returns the convenience.
 * outputDeriv(x)             Returns the derivative of the output function at x, given by f(x)(1-f(x)).
 * assignWeights(double[][][])
 *                            Changes the weights to the values of the given array.
 * optimizeWeights(double)    Optimizes the weights for the current input values and a given learning factor.
 * randomizeWeights(double, double)
 *                            Randomizes the weight values based on a minimum double, inclusive, and a maximum double, exclusive.
 * run(int)                   Creates the net, assigns values based on a given input row,
 *                            calculates the error, and prints the net.
 * runAll()                   Runs run() for every given input array, and runs and prints the convenience.
 * optimizeAll()              Creates a net with random weights and finds the optimal weights for the given input array.
 * main()                     Creates a Perceptron, which runs runAll().
 * @author Martin Bourdev
 * @version September 4, 2019
 */

public class Perceptron {
   
   private double[][][] weights;
   private double[][] units;
   
   private final double DEFAULT_WEIGHT = 0.5;   //The value of a weight if it is not explicitly defined.
   private final double DEFAULT_INPUT = 0.5;    //The value of an input if it is not explicitly defined.
   
   private double[][] docValues; //The 2D array of doubles in the input document.
   
   private double error; //Calculated by the calculateError() function.
   
   /*
    * The value of the double in this row signifies that that many rows under it will be read as
    * arrays of inputs for the net. The number also should match how many expected results should be
    * in the expectedResultRow row of the document. The indices of all other rows are initialized
    * based on this one.
    */
   private final int INPUT_NUM_ROW = 0;
   public int inputsRow;   //The row in the input document that stores the number of input arrays, starting from 1.
   
   /*
    * The row in the input document that stores the number of units in each column.
    * The number of doubles represents the number of layers in the network.
    */
   private int rowsRow;
   private final int ROWS_ROW_OFFSET = 1; //The number of rows below the input rows that the rowsRow row is.
   
   private int weightRow;                    //The row in the input document that stores the weights.
   private final int WEIGHT_ROW_OFFSET = 2;  //The number of rows below the input rows that the weightRow row is.
   
   private int expectedResultRow;         //The row in the input document that stores the expected result.
   private int EXPECTED_ROW_OFFSET = 3;   //The number of rows below the input rows that the expectedRow row is.
   
   /*
    * Stores the errors of all the times the net runs, based on the first value of the INPUT_NUM_ROW row
    * of the input document.
    */
   private double[] errors;
   
   public double convenience;    //Stores the convenience.
   
   public final int DEFAULT_INPUT_ROW = 1;   //Stores the input row that the net will read for inputs by default.
   
   public final double DEFAULT_LEARNING_FACTOR = 1.0; //Learning factor used by default.
   /*
    * Learning factor does not decrease below this number. After this number is reached,
    * the error is considered to be minimized.
    */
   public final double MINIMUM_LEARNING_FACTOR = 0.1;
   public final double MINIMUM_ERROR = 0.05; //After this error or below is reached, the error is considered to be minimized.
   
   /**
    * Perceptron constructor.
    * Retrieves the values of the input document, initializes the error array, and initializes the row indices
    * based on how many rows of inputs are given.
    */
   public Perceptron()
   {
      docValues = getInputs();  //Stores values from input document into 2D array.
      
      //Initializes the errors array to have as many spots as input rows.
      errors = new double[(int) (docValues[INPUT_NUM_ROW][0])];
      
      //Initializes row indices based on how many input rows there are, given by docValues[INPUT_NUM_ROW][0].
      inputsRow = INPUT_NUM_ROW + DEFAULT_INPUT_ROW;
      rowsRow = INPUT_NUM_ROW + ROWS_ROW_OFFSET + (int) (docValues[INPUT_NUM_ROW][0]);
      weightRow = INPUT_NUM_ROW + WEIGHT_ROW_OFFSET + (int) (docValues[INPUT_NUM_ROW][0]);
      expectedResultRow = INPUT_NUM_ROW + EXPECTED_ROW_OFFSET + (int) (docValues[INPUT_NUM_ROW][0]);
   }
   
   /**
    * Returns a 2D array of units.
    * The first dimension is the number of layers specified by the length of the rowsRow row of the doc.
    * The second dimension is the number of rows of units in the layer i specified by the ith
    * values of docValues[rowsRow], or the number
    * of values in the inputsRow row if initializes the array of inputs.
    *
    * @return a 2D array to store the units.
    */
   public double[][] createUnitArr()
   {
      double[][] units = new double[docValues[rowsRow].length][];
      for(int i = 0; i < docValues[rowsRow].length; i++)
      {
         if(i == 0)  //Initializes the input array based on the number of values in the inputRow row.
            units[i] = new double[docValues[inputsRow].length];
         else        //Initializes the hidden arrays based on the number of units in that layer
            units[i] = new double[(int) (docValues[rowsRow][i])];
      }
      return units;
   }
   
   /**
    * Passes the given double through an output function and returns it.
    * @param activation    the given activation
    * @return              the activation passed through an output function.
    */
   public double output(double activation)
   {
      return 1.0/(1.0 + Math.exp(-activation));
   }
   
   /**
    * Creates a 3D array of weights.
    * The first dimension is the number of layers minus one, specified by the length of the rowsRow row of the
    * doc, minus one.
    * The second dimension is the number of rows of units to the left of the weights, specified by
    * the ith value of the rowsRow row of the doc.
    * The third dimension is the number of rows of units to the right of the weights, specified by
    * the i+1th value of the rowsRow row of the doc.
    *
    * @return the array of weights.
    */
   public double[][][] createWeightArr()
   {
      double[][][] weights = new double[docValues[rowsRow].length - 1][][];
      
      for(int i = 0; i < docValues[rowsRow].length - 1; i++)
      {
         if(i == 0)
         {
            weights[i] = new double[docValues[inputsRow].length][(int) (docValues[rowsRow][i+1])];
         }
         else
            weights[i] = new double[(int) (docValues[rowsRow][i])][(int) (docValues[rowsRow][i+1])];
      }
      return weights;
      
   }
   
   /**
    * Assigns weights into the weights array from the weightRow row of the input file,
    * Iterates through each weight
    * first through each layer of weights, then on units on the left, then on units to the right.
    * If there are more weights to assign than given values in the weightRow row of the input file,
    * uses the default weight value, DEFAULT_WEIGHT.
    */
   public void assignWeights()
   {
      int index = 0;
      for(int i = 0; i < weights.length; i++) //Iterates over every layer of weights.
      {
         for(int j = 0; j < weights[i].length; j++) //Iterates over every unit on the left of the layer of weights.
         {
            for(int k = 0; k < weights[i][j].length; k++) //Iterates over every unit on the right of the layer of weights.
            {
               if(docValues[weightRow].length <= index) //If there are more weights than given weight values.
               {
                  weights[i][j][k] = DEFAULT_WEIGHT;
               }
               else //If reading weights from the weightRow row of the document.
               {
                  weights[i][j][k] = docValues[weightRow][index];
               }
               index++;
            } //for(int k = 0; k < weights[i][j].length; k++)
            
         } //for(int j = 0; j < weights[i].length; j++)
         
      } //for(int i = 0; i < weights.length; i++)
   }
   
   /**
    * Returns a jagged 2D double array of the values in the input document.
    * The first index represents the line, and the second index represents the number of doubles in that line.
    * @return a 2D double array of input document values.
    */
   public double[][] getInputs()
   {
      String s = "/Users/martinbourdev/IdeaProjects/Perceptron1/text/input";
      try {
         BufferedReader counter1 = new BufferedReader(new FileReader(s));
      
         //Counts the number of lines in the document for the first index of the array.
         String line = counter1.readLine();
         int lines = 0;
         while(line != null)
         {
            lines++;
            line = counter1.readLine();
         }
         counter1.close();
         double[][] vals = new double[lines][];
      
         //Counts the number of doubles in each line, assigning each row of the array to that number.
         BufferedReader counter2 = new BufferedReader(new FileReader(s));
         line = counter2.readLine();
         for(int i = 0; i < lines; i++)
         {
            vals[i] = parseString(line);
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
    * Parses a string and returns an array of doubles.
    * S must contain doubles separated by spaces.
    * @param s the given string.
    * @return a double[] of doubles in the string.
    */
   public double[] parseString(String s)
   {
      Scanner sc = new Scanner(s);
      
      //Counts how many doubles in the String to initialize the length of the array.
      int length = 0;
      while(sc.hasNextDouble())
      {
         length++;
         sc.nextDouble();
      }
      sc.close();
      double[] doubles = new double[length];
      
      //Assigns each double to the array.
      Scanner sc2 = new Scanner(s);
      for(int i = 0; i < length; i++)
      {
         doubles[i] = sc2.nextDouble();
      }
      return doubles;
   }
   
   /**
    * Assigns values to the input units based on the inputsRow row of the docs. If there are more
    * units in the first layer than values given, uses the default input value.
    */
   public void assignInputs()
   {
      for(int i = 0; i < units[0].length; i++)
      {
         if(docValues[inputsRow].length <= i) //If more spaces in the input array than given input values.
         {
            units[0][i] = DEFAULT_INPUT;
         }
         else
            units[0][i] = docValues[inputsRow][i];
      }
      
   }
   
   /**
    * Returns the docValues[rowsRow] array, but with ints instead of doubles to make it easier to use them as
    * discrete numbers.
    * @return an integer array of the values of docValues[rowsRow], which represents the number of activation units
    * in each layer.
    */
   public int[] intRows()
   {
      int[] rows = new int[docValues[rowsRow].length];
      for(int i = 0; i < rows.length; i++)
      {
         rows[i] = (int) (docValues[rowsRow][i]);
      }
      return rows;
   }
   
   /**
    * Assigns values to hidden units based on the values of input units and weights.
    * Assigns values as the sum of the products of weights and values.
    */
   public void assignHiddens()
   {
      
      int[] rows = intRows();
      
      
      for(int i = 1; i < rows.length; i++) //Iterates through each layer, starting with 1, skipping the input array.
      {
         
         for(int j = 0; j < rows[i]; j++) //Iterates through the units being added to.
         {
            //giving value to units[i][j]
            double total = 0.0;
            for(int k = 0; k < rows[i-1]; k++)  //Iterates through units, the values of which are used to add to unit[i][j].
            {
               total += units[i-1][k]*weights[i-1][k][j];
            }
            units[i][j] = output(total); //Passes the weighted sum through an output function.
            
         } //for(int i = 1; i < rows.length; i++) //Iterates through each layer, starting with 1, skipping the input array.
         
      } //for(int j = 0; j < rows[i]; j++) //Iterates through the units being added to.
   }
   
   
   /**
    * Prints the neural net layer-by-layer, providing the unit and weight indices and values.
    * Finishes by printing the error.
    * Prints "Units; Layer", followed by the layer and each unit index and value.
    * Prints "Weights; Layer", followed by the layer and each weight index and value.
    *
    */
   public void printNet()
   {
      //The number of layers in the net, minus 1 in order to point to the last value in the array
      int layersIndex = docValues[rowsRow].length - 1;
      //An int array of the values of the rowsRow row of the document.
      int[] unitIndex = intRows();
      
      //Iterates through all layers - 1, because last one doesn't have weights.
      for(int i = 0; i < layersIndex; i++)
      {
         System.out.print("Units; Layer " + i + "; \t"); //Prints which layer of units is being printed.
   
         /*
           For i = 0, which is the layer of inputs, reads from the inputsRow row instead of the
           rowsRow row in case there are more given input values than the first value of the rowsRow row.
          */
         if(i == 0)
         {
            for (int j = 0; j < docValues[inputsRow].length; j++) //Iterates through all units in the layer i.
            {
               System.out.print("Unit" + i + j + ": " + units[i][j] + ", ");
            }
         }
         else
         {
            for (int j = 0; j < unitIndex[i]; j++) //Iterates through all units in the layer i.
            {
               System.out.print("Unit" + i + j + ": " + units[i][j] + ", ");
            }
         }
         
         //Prints weights.
         System.out.println();
         System.out.print("Weights; Layer " + i + "; \t"); //Prints which layer of weights is being printed.
         
         for(int j = 0; j < weights[i].length; j++) //Iterates through all weights on the left.
         {
            for(int k = 0; k < weights[i][j].length; k++) //Iterates through all weights on the right.
            {
               System.out.print("Weight" + i + j + k + ": " + weights[i][j][k] + ", ");
            }
         }
         System.out.println();
      } //for(int i = 0; i < layersIndex; i++) Iterates through all layers - 1, because last one doesn't have weights.
      
      //Prints output layer.
      System.out.print("Units; Layer " + layersIndex + "; \t"); //Prints which unit layer is being printed.
      
      for(int i = 0; i < unitIndex[layersIndex]; i++) //Iterates through all units in the last unit array of units.
      {
         System.out.print("Unit" + layersIndex + i + ": " + units[layersIndex][i] + ", ");
      }
      System.out.println();
      
      //Prints error.
      System.out.print("Error: " + error + "\n");
      System.out.println();
   }
   
   /**
    * Calculates the error.
    * Takes the expected result from the (inputsRow - 1)th value in the expectedResultRow row of the doc,
    * which corresponds with the row of inputs being used.
    * Takes the difference between the actual result (value of first unit of last layer) and expected result,
    * squares it, and divides by two.
    * Assigns the respective index of the errors array to the error.
    */
   public void calculateError()
   {
      //Retrieves the test result from the first unit in the last layer.
      double actualResult = units[docValues[rowsRow].length - 1][0];
   
      /*
        Uses the expected result from the respective index in the expectedResultRow, which is inputsRow - 1.
        One is subtracted because inputsRow starts at 1, and its respective index is 0.
       */
      double expectedResult = docValues[expectedResultRow][inputsRow - 1];
      double difference = actualResult - expectedResult;
      error = difference * difference / 2;
      
      errors[inputsRow - 1] = error; //Assigns the error to the inputRow's respective index.
   }
   
   /**
    * Calculates the convenience. Takes the square root of the sum of the squares of all the values in
    * the error array.
    */
   public void calculateConvenience()
   {
      convenience = 0.0;
      for(int i = 0; i < (int) (docValues[INPUT_NUM_ROW][0]); i++)
      {
         convenience += errors[i]*errors[i];
      }
      convenience = Math.sqrt(convenience);
   }
   
   /**
    * Runs the Perceptron for all input matrices based on the current weights and returns the convenience.
    * @precondition All weights are assigned a value.
    * @return  The convenience.
    */
   public double runCurrentWeights()
   {
      docValues = getInputs();
      inputsRow = DEFAULT_INPUT_ROW;
      weights = createWeightArr();
      units = createUnitArr();
      for(int i = 1; i <= docValues[INPUT_NUM_ROW][0]; i++) //Iterates through all the input rows.
      {
         inputsRow = i;
         assignInputs();
         assignHiddens();
         calculateError();
      }
      calculateConvenience();
      return convenience;
   }
   
   /**
    * Takes the derivative of the output function at a given x.
    * @param x The given x value.
    * @return  The derivative, given by f(x)*(1-f(x)), where x is the output function.
    */
   public double outputDeriv(double x)
   {
      return output(x) * (1 - output(x));
   }
   
   /**
    * Assigns the weights to be those from a given 3D double array.
    * @precondition newWeights has the same dimension as weights.
    * @param newWeights The given 3D double array.
    */
   public void assignWeights(double[][][] newWeights)
   {
      for(int i = 0; i < weights.length; i++)
      {
         for(int j = 0; j < weights[i].length; j++)
         {
            for(int k = 0; k < weights[i][j].length; k++)
            {
               weights[i][j][k] = newWeights[i][j][k];
            }
         }
      }
   }
   
   /**
    * Optimizes the weights for the current set of inputs to fall below a minimum error, given a
    * starting learning factor.
    * Creates an array of temporary weights and finds the error. Then, until either the error
    * is smaller than the defined minimum or the learning factor reaches the defined minimum, for each weight, calculates
    * deltaW and adds it to the temporary weight array. After the temporary array has been fully updated,
    * finds the error for that weight array. If the new error is smaller than the initial, the learning
    * factor is multiplied by two and the weights are updated to the new ones. If the new error is larger than
    * the initial, the weights are rolled back to the original and the learning factor is divided by two.
    * When the end conditions are met, the reason is printed and the final learning factor
    * is returned.
    *
    * @precondition Arrays of units and weights have been created and initialized.
    * @param lFactor The starting learning factor.
    * @return  The final learning factor.
    */
   public double optimizeWeights(double lFactor)
   {
      double[][][] deltaWArr = createWeightArr();     //Array of delta weights in case of rollback.
      double[][][] tempWeights = createWeightArr();   //Array of temporary weights, set to the weight values.
      for(int i = 0; i < weights.length; i++)
      {
         for(int j = 0; j < weights[i].length; j++)
         {
            for(int k = 0; k < weights[i][j].length; k++)
            {
               tempWeights[i][j][k] = weights[i][j][k];
            }
         }
      }
      
      //Initializes a learning factor and initial error.
      double learningFactor = lFactor;
      double initialError = error;
      boolean allErrorsTooBig = true;
      boolean notEqualErrors = true;
      
      while(notEqualErrors && allErrorsTooBig && (learningFactor > MINIMUM_LEARNING_FACTOR))
      {
         System.out.println("Learning factor: " + learningFactor);
         //printNet();
         //Iterates through weight layers, starting from layer 2.
         for (int layer = weights.length - 1; layer >= 0; layer--)
         {
            //Iterates through length of unit layer to the left of weight layer.
            for (int left = 0; left < weights[layer].length; left++)
            {
               //Iterates through length of unit layer to the right of weight layer.
               for (int right = 0; right < weights[layer][left].length; right++)
               {
                  //Retrieves the actual result from the first unit in the last layer.
                  double actualResult = units[docValues[rowsRow].length - 1][0];
      
                  /*
                    Uses the expected result from the respective index in the expectedResultRow, which is inputsRow - 1.
                    One is subtracted because inputsRow starts at 1, and its respective index is 0.
                   */
                  double expectedResult = docValues[expectedResultRow][inputsRow - 1];
                  double deltaW = 0.0;
                  if (layer == 0) //If considering layer 1.
                  {
                     //System.out.println(layer + " " + left + " " + right);
                     
                     //Finds the weighted sum of the first layer based on weight 0jk.
                     double sum1 = 0.0;
                     for (int leftItr = 0; leftItr < left + 1; leftItr++) //Iterates through left nodes.
                     {
                        sum1 += units[layer][leftItr] * weights[layer][leftItr][right];
                     }
                     
                     //Finds the weighted sum of the second layer based on weight 0jk.
                     double sum2 = 0.0;
                     for (int rightItr = 0; rightItr < right + 1; rightItr++)
                     {
                        sum2 += units[layer + 1][rightItr] * weights[layer + 1][rightItr][0];
                     }
                     
                     //Calculates deltaW. i = 0, j = 0, k = 0.
                     deltaW = -units[layer][left] * outputDeriv(sum1) * (expectedResult - actualResult) *
                                      outputDeriv(sum2) * weights[layer][left][0];
                     //System.out.println("f" + weights[0][0][0]);
                  }
                  else if (layer == 1) //If considering layer 2.
                  {
                     //Calculates the sum of the right side.
                     double sum = 0.0;
                     for (int leftItr = 0; leftItr < units[layer+1].length + 1; leftItr++) //Iterates through units in the 2nd layer.
                     {
                        sum += units[layer][leftItr] * weights[layer][leftItr][right];
                     }
                     
                     //Calculates deltaW.
                     deltaW = -(expectedResult - actualResult) * outputDeriv(sum) * units[layer][left];
                  }
                  //Multiplies deltaW by the learning factor and adds it to the temporary weights array.
                  deltaW *= -learningFactor;
                  //System.out.println("DeltaW " + i + j + k + ": " + deltaW);
                  tempWeights[layer][left][right] += deltaW;
                  //System.out.println("q" + weights[0][0][0]);
                  deltaWArr[layer][left][right] = deltaW;
                  //System.out.println("z" + weights[0][0][0]);
               } //for(int k = 0; k < weights[i][j].length; k++)
               
            } //for(int j = 0; j < weights[i].length; j++)
            
         } //for(int i = 0; i < weights.length; i++)
         
         //TEMPORARY WEIGHT ARRAY FULLY DEFINED.
         
         //Calculates the error of the temporary weight array.
         assignWeights(tempWeights);
         assignHiddens();
         calculateError();
         printNet();
         double newError = error;
         System.out.println("InitialError " + initialError);
         System.out.println("newError for set " + inputsRow + " is " + newError);
         
         //Changes learning factor based on new error.
         if (newError < initialError) //If error becomes smaller, increases learning factor.
         {
            learningFactor *= 2;
         }
         else if (initialError < newError) //If error becomes larger, decreases the learning factor.
         {
            learningFactor /= 2;
            
            //Rolling the weights back.
            for(int a = 0; a < weights.length; a++)
            {
               for(int b = 0; b < weights[a].length; b++)
               {
                  for(int c = 0; c < weights[a][b].length; c++)
                  {
                     weights[a][b][c] -= deltaWArr[a][b][c];
                  }
               }
            } //Rolling the weights back.
         } //else if (initialError < newError) //If error becomes larger, decreases the learning factor.
         else
         {
            notEqualErrors = false;
         }
   
         allErrorsTooBig = !checkErrors(MINIMUM_ERROR);
         if(allErrorsTooBig)
         {
            //Increments input row.
            if (inputsRow == docValues[INPUT_NUM_ROW][0]) //If on last input row, moves to first.
               inputsRow = 1;
            else
               inputsRow++;   //Increments input row to train for new inputs.
   
            //Reassigns inputs, weights, hiddens, and error for next case.
            assignInputs();
            for (int i = 0; i < weights.length; i++)
            {
               for (int j = 0; j < weights[i].length; j++)
               {
                  for (int k = 0; k < weights[i][j].length; k++)
                  {
                     weights[i][j][k] = tempWeights[i][j][k];
                  }
               }
            }
            assignHiddens();
            calculateError();
            initialError = error;
         }
         
         
      } //while(notEqualErrors && allErrorsTooBig && (learningFactor > MINIMUM_LEARNING_FACTOR))
      
      //printNet();
      if(checkErrors(MINIMUM_ERROR))
         System.out.println("Optimization terminated because error passed minimum.");
      else if(learningFactor == 0)
         System.out.println("Optimization terminated because learning factor reached zero.");
      else if(!notEqualErrors)
         System.out.println("Initial and Final errors are equal.");
      return learningFactor;
   }
   
   /**
    * Returns true if the errors of all the cases are lower than the given minimum error.
    * @precondition Unit and weight arrays are created, and weight array is initialized.
    * @param minError   The minimum error for which to return true.
    * @return  True if every error for every case is lower than the minimum error.
    */
   public boolean checkErrors(double minError)
   {
      int initialInputRow = inputsRow;
      boolean bool = true;
      for(int i = 1; i <= docValues[INPUT_NUM_ROW][0]; i++) //Iterates through all input rows.
      {
         inputsRow = i;
         assignInputs();
         assignHiddens();
         calculateError();
         System.out.println("ERROR : " + i + " " + error);
         if(error > minError)
         {
            bool = false;
         }
         //printNet();
      }
      inputsRow = initialInputRow;
      return bool;
   }
   
   /**
    * Sets all the weights to random values from the range [left, right). Left and right are given doubles.
    * @param left    The given minimum, inclusive.
    * @param right   The given maximum, exclusive.
    */
   public void randomizeWeights(double left, double right)
   {
      for(int i = 0; i < weights.length; i++)
      {
         for(int j = 0; j < weights[i].length; j++)
         {
            for(int k = 0; k < weights[i][j].length; k++)
            {
               weights[i][j][k] = Math.random()*right + left;
            }
         }
      }
   }
   /**
    * Creates and prints a neural network based on the inputs from a given row.
    * Creates weight and unit arrays, assigns input values, weights, and hidden values, calculates the error,
    * and prints the net.
    * @param inputRow the index of the input row to read, starting from 1.
    */
   public void run(int inputRow)
   {
      docValues = getInputs();
      inputsRow = inputRow;
      weights = createWeightArr();
      units = createUnitArr();
      assignInputs();
      assignWeights();
      assignHiddens();
      calculateError();
      printNet();
   }
   
   /**
    * Runs and prints a neural net for every given array of inputs,
    * given by the INPUT_NUM_ROW row of the input document.
    * Calculates and prints the convenience.
    */
   public void runAll()
   {
      for(int i = 1; i <= docValues[INPUT_NUM_ROW][0]; i++) //Iterates through all the input rows.
      {
         run(i);
      }
      calculateConvenience();
      System.out.println("Convenience: " + convenience);
   }
   
   /**
    * Creates a net with random weights and every other configurable number from the input document,
    * starting with the first set of inputs. Then, optimizes the weights and stores the learning factor.
    *
    */
   public void optimizeAll()
   {
      docValues = getInputs();
      inputsRow = 1;
      weights = createWeightArr();
      units = createUnitArr();
      assignInputs();
      randomizeWeights(0.0, 5.0);
      assignHiddens();
      calculateError();
      optimizeWeights(DEFAULT_LEARNING_FACTOR);
      
      for(int i = 1; i <= docValues[INPUT_NUM_ROW][0]; i++)
      {
         inputsRow = i;
         assignInputs();
         assignHiddens();
         calculateError();
         printNet();
      }
      
      //printNet();
   }
   
   /**
    * Creates a new Perceptron and has it call the test() method.
    * @param args The arguments for the main method.
    */
   public static void main(String[] args)
   {
      Perceptron nn = new Perceptron();
      nn.printNet();
   }
   
}


