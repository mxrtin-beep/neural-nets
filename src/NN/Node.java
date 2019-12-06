package NN;

/**
 * @author Martin Bourdev
 * @version September 24, 2019
 *
 * The Node class represents an activation in a perceptron, and holds the weights feeding into the activation,
 * the incoming gradient, a weights gradient sum for each weight, and a backup weight for each weight.
 */
public class Node
{
   public double[] weights_;  // Modified by updateWeights.
   public double activation_; // Modified by forward.
   public double incomingGradient_; // Modified by backward.
   public double[] weightsGradientSum_; // Modified by backward.
   public double[] backupWeights_;
   
   private final double DEFAULT_ACTIVATION_ = 0.0;
   private final double DEFAULT_INCOMING_GRADIENT_ = 0.0;
   
   /**
    * Node constructor.
    * Stores numWeights weights in weights_ with a random value from minWeight to maxWeight.
    * Sets activation_ and incomingGradient_ to 0 by default.
    *
    * @param numWeights The number of weights contributing to the input of this Node.
    */
   Node(int numWeights, double min, double max)
   {
      weights_ = new double[numWeights];
      weightsGradientSum_ = new double[numWeights];
      backupWeights_ = new double[numWeights];
      
      for (int i = 0; i < numWeights; i++)
      {
         weights_[i] = min + (max - min) * Math.random();
      }
      
      activation_ = DEFAULT_ACTIVATION_;
      incomingGradient_ = DEFAULT_INCOMING_GRADIENT_;
   }
   
   /**
    * Sets the value of the activation to the sigmoid of the weighted sum of the activations of the previous layer.
    * @param prevLayer  The previous layer of Nodes.
    */
   public void forward(Node[] prevLayer)
   {
      double weightedSum = 0.0;
      for (int i = 0; i < weights_.length; i++)
      {
         weightedSum += prevLayer[i].activation_ * weights_[i];
      }
      activation_ = activationFunction(weightedSum);
   }
   
   /**
    * Calculates the weight gradient for each weight and adds it to arrays of weights gradient sums.
    * Adds to the incoming gradient of the Nodes of the previous layer.
    * Assumes that the incoming gradient of the Node calling this method is set.
    *
    * @param prevLayer  The Nodes of the previous layer.
    */
   public void backward(Node[] prevLayer)
   {
      double gradient = incomingGradient_ * activationDeriv(activation_);
      
      for (int i = 0; i < prevLayer.length; i++) // Iterates through the Nodes of the previous layer.
      {
         // Adds to incoming gradients of previous layer weights.
         prevLayer[i].incomingGradient_ += gradient * weights_[i];
      }
      for (int i = 0; i < weights_.length; i++)
      {
         // Adds to the weights gradient sum of this Node.
         weightsGradientSum_[i] += gradient * prevLayer[i].activation_;
      }
   }
   
   /**
    *  Sets the weight gradient sums of all the weights to zero.
    */
   public void resetWeightGradientSums()
   {
      for (int i = 0; i < weights_.length; i++)
      {
         weightsGradientSum_[i] = 0.0;
      }
   }
   
   /**
    * For each weight, subtracts the learning factor multiplied by the weights gradient sum of that weight.
    * @param learningFactor   The factor by which to multiply the weights gradient sum.
    */
   public void updateWeights(double learningFactor)
   {
      for (int i = 0; i < weights_.length; i++)
      {
         weights_[i] -= learningFactor * weightsGradientSum_[i];
      }
   }
   
   /**
    * Returns the average of the absolute value of all the weight gradient sums of every weight of this Node.
    * @return  The average magnitude of weight gradients.
    */
   public double getAvgGradientMagnitude()
   {
      double mag = 0.0;
      for (int i = 0; i < weights_.length; i++)
      {
         mag += Math.abs(weightsGradientSum_[i]);
      }
      return mag /  weights_.length;
   }
   
   /**
    * Returns the sigmoid of a given value.
    * @param x The given value to pass through the sigmoid function.
    * @return  1.0 / (1.0 + e^-x).
    */
   private double sigmoid(double x)
   {
      return 1.0 / (1.0 + Math.exp(-x));
   }
   
   /**
    * Returns the derivative of the sigmoid given an activation already passed through the sigmoid function.
    * @param y The given activation.
    * @return  The value y passed through the derivative of the sigmoid function.
    */
   private double sigmoidDeriv(double y)
   {
      return y * (1.0 - y);
   }
   
   /**
    * Returns the double x passed through an activation function.
    * @param x The given double.
    * @return  The value x scaled to the activation function.
    */
   private double activationFunction(double x)
   {
      return sigmoid(x);
   }
   
   /**
    * Returns a double y passed through the derivative of the activation function.
    * @param y The given double.
    * @return  The value y passed through the derivative of the activation function.
    */
   private double activationDeriv(double y)
   {
      return sigmoidDeriv(y);
   }
   
   /**
    * Puts the weights and activations of the Node in the form of a String.
    * @return All the weights and the activation in the form of a String.
    */
   public String toString()
   {
      String s = "";
      
      for (int i = 0; i < weights_.length; i++)
      {
         s += "Weight" + i + ": " + weights_[i] + " ";
      }
      
      s += "Activation: " + activation_;
      return s;
   }
   
   /**
    *  Stores the value of every weight in a respective backup.
    */
   public void backupWeights()
   {
      for (int i = 0; i < weights_.length; i++)
      {
         backupWeights_[i] = weights_[i];
      }
   }
   
   /**
    *  Sets the value of all the weights to the respective backups.
    */
   public void restoreWeights()
   {
      for (int i = 0; i < weights_.length; i++)
      {
         weights_[i] = backupWeights_[i];
      }
   }
}
