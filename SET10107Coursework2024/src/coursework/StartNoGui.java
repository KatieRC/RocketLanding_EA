package coursework;

import model.Fitness;
import model.LunarParameters.DataSet;
import model.NeuralNetwork;

/**
 * Example of how to to run the {@link ExampleEvolutionaryAlgorithm} without the need for the GUI
 * This allows you to conduct multiple runs programmatically 
 * The code runs faster when not required to update a user interface
 *
 */
public class StartNoGui {
	
	public static void main(String[] args) {
		
		/**
		 * Train the Neural Network using our Evolutionary Algorithm 
		 * 
		 */

		/*
		 * Set the parameters here or directly in the Parameters Class.
		 * Note you should use a maximum of 20,0000 evaluations for your experiments 
		 */
			

		    // Set the parameters
		    Parameters.maxEvaluations = 20000; // Used to terminate the EA after this many generations
		    Parameters.popSize = 50; //Population Size
		    Parameters.setHidden(5); //Hidden Nodes
		   

		    // Create and train the neural network
		    NeuralNetwork nn = new ExampleEvolutionaryAlgorithm();
		    nn.run();

		    // Print the best weights found during training
		    System.out.println("Best weights found: " + nn.best);

		    
		    // Print the fitness on the training set
		    Parameters.setDataSet(DataSet.Training);
		    double trainingFitness = Fitness.evaluate(nn);
		    System.out.println("Training set fitness: " + trainingFitness);
		 
		    // Test the trained network on the test set
		    Parameters.setDataSet(DataSet.Test);
		    double testFitness = Fitness.evaluate(nn);
		    System.out.println("Test set fitness: " + testFitness);


		    System.out.println(); // Add a blank line for readability
        
		
    }
		
}
