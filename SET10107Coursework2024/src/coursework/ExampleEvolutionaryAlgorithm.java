package coursework;

import java.util.ArrayList;
import java.util.Collections;

import model.Fitness;
import model.Individual;
import model.LunarParameters.DataSet;
import model.NeuralNetwork;

/**
 * Implements a basic Evolutionary Algorithm to train a Neural Network
 * 
 * You Can Use This Class to implement your EA or implement your own class that extends {@link NeuralNetwork} 
 * 
 */
public class ExampleEvolutionaryAlgorithm extends NeuralNetwork {
	

	/**
	 * The Main Evolutionary Loop
	 */
	@Override
	public void run() {		
		//Initialise a population of Individuals with random weights
		population = initialise();

		//Record a copy of the best Individual in the population
		best = getBest();
		System.out.println("Best From Initialisation " + best);

		/**
		 * main EA processing loop
		 */		
		
		while (evaluations < Parameters.maxEvaluations) {

			/**
			 * this is a skeleton EA - you need to add the methods.
			 * You can also change the EA if you want 
			 * You must set the best Individual at the end of a run
			 * 
			 */

			// Select 2 Individuals from the current population. Currently returns random Individual
			Individual parent1 = select(); 
			Individual parent2 = select();
	
			//Perform uniform crossover to generate children
			ArrayList<Individual> children = uniformCrossover(parent1, parent2);			
			
			//mutate the offspring
			mutate(children);
			
			// Evaluate the children
			evaluateIndividuals(children);			

			// Replace children in population
			elitistReplace(children);

			// check to see if the best has improved
			best = getBest();
			
			// Implemented in NN class. 
			outputStats();
			
			//Increment number of completed generations		
			//Increment evaluations, assuming one evaluation per generation
            evaluations++;
		}

		//save the trained network to disk
		saveNeuralNetwork();
	}

	

	/**
	 * Sets the fitness of the individuals passed as parameters (whole population)
	 * 
	 */
	private void evaluateIndividuals(ArrayList<Individual> individuals) {
		for (Individual individual : individuals) {
			individual.fitness = Fitness.evaluate(individual, this);
		}
	}

	/**
	 * Crossover / Reproduction
	 *  
	 */
	
	/**
     * Performs one-point crossover
     */
    private ArrayList<Individual> onePointCrossover(Individual parent1, Individual parent2) {
        int chromosomeLength = parent1.chromosome.length;
        int crossoverPoint = Parameters.random.nextInt(chromosomeLength);

        double[] child1Chromosome = new double[chromosomeLength];
        double[] child2Chromosome = new double[chromosomeLength];

        // Copy genetic material from parent1 up to crossover point
        System.arraycopy(parent1.chromosome, 0, child1Chromosome, 0, crossoverPoint);
        // Copy genetic material from parent2 up to crossover point
        System.arraycopy(parent2.chromosome, 0, child2Chromosome, 0, crossoverPoint);

        // Copy genetic material from parent2 after crossover point
        System.arraycopy(parent2.chromosome, crossoverPoint, child1Chromosome, crossoverPoint, chromosomeLength - crossoverPoint);
        // Copy genetic material from parent1 after crossover point
        System.arraycopy(parent1.chromosome, crossoverPoint, child2Chromosome, crossoverPoint, chromosomeLength - crossoverPoint);

        // Create child individuals
        Individual child1 = new Individual();
        child1.chromosome = child1Chromosome;
        Individual child2 = new Individual();
        child2.chromosome = child2Chromosome;

        // Return the two offspring individuals
        ArrayList<Individual> children = new ArrayList<>();
        children.add(child1);
        children.add(child2);
        return children;
    }
    
    /**
     * Performs two-point crossover
     */
    private ArrayList<Individual> twoPointCrossover(Individual parent1, Individual parent2) {
        int chromosomeLength = parent1.chromosome.length;
        
        // Select two random crossover points
        int crossoverPoint1 = Parameters.random.nextInt(chromosomeLength);
        int crossoverPoint2 = Parameters.random.nextInt(chromosomeLength);
        
        // Ensure crossoverPoint1 < crossoverPoint2
        if (crossoverPoint1 > crossoverPoint2) {
            int temp = crossoverPoint1;
            crossoverPoint1 = crossoverPoint2;
            crossoverPoint2 = temp;
        }

        double[] child1Chromosome = new double[chromosomeLength];
        double[] child2Chromosome = new double[chromosomeLength];

        // Copy genetic material from parent1 up to crossoverPoint1
        System.arraycopy(parent1.chromosome, 0, child1Chromosome, 0, crossoverPoint1);
        // Copy genetic material from parent2 between crossoverPoint1 and crossoverPoint2
        System.arraycopy(parent2.chromosome, crossoverPoint1, child1Chromosome, crossoverPoint1, crossoverPoint2 - crossoverPoint1);
        // Copy genetic material from parent1 after crossoverPoint2
        System.arraycopy(parent1.chromosome, crossoverPoint2, child1Chromosome, crossoverPoint2, chromosomeLength - crossoverPoint2);

        // Copy genetic material from parent2 up to crossoverPoint1
        System.arraycopy(parent2.chromosome, 0, child2Chromosome, 0, crossoverPoint1);
        // Copy genetic material from parent1 between crossoverPoint1 and crossoverPoint2
        System.arraycopy(parent1.chromosome, crossoverPoint1, child2Chromosome, crossoverPoint1, crossoverPoint2 - crossoverPoint1);
        // Copy genetic material from parent2 after crossoverPoint2
        System.arraycopy(parent2.chromosome, crossoverPoint2, child2Chromosome, crossoverPoint2, chromosomeLength - crossoverPoint2);

        // Create child individuals
        Individual child1 = new Individual();
        child1.chromosome = child1Chromosome;
        Individual child2 = new Individual();
        child2.chromosome = child2Chromosome;

        // Return the two offspring individuals
        ArrayList<Individual> children = new ArrayList<>();
        children.add(child1);
        children.add(child2);
        return children;
    }
    
    /**
     * Performs uniform crossover
     */
    private ArrayList<Individual> uniformCrossover(Individual parent1, Individual parent2) {
        int chromosomeLength = parent1.chromosome.length;
        double[] child1Chromosome = new double[chromosomeLength];
        double[] child2Chromosome = new double[chromosomeLength];
        
        // Define the probability of selecting genes from each parent
        double probability = 0.5; 

        for (int i = 0; i < chromosomeLength; i++) {
            // Randomly choose whether to select gene from parent1 or parent2
            if (Parameters.random.nextDouble() < probability) {
                child1Chromosome[i] = parent1.chromosome[i];
                child2Chromosome[i] = parent2.chromosome[i];
            } else {
                child1Chromosome[i] = parent2.chromosome[i];
                child2Chromosome[i] = parent1.chromosome[i];
            }
        }

        // Create child individuals
        Individual child1 = new Individual();
        child1.chromosome = child1Chromosome;
        Individual child2 = new Individual();
        child2.chromosome = child2Chromosome;

        // Return the two offspring individuals
        ArrayList<Individual> children = new ArrayList<>();
        children.add(child1);
        children.add(child2);
        return children;
    }
    
    /**
     * Performs arithmetic crossover
     */
    private ArrayList<Individual> arithmeticCrossover(Individual parent1, Individual parent2) {
        ArrayList<Individual> offspring = new ArrayList<>();
        double[] childChromosome = new double[parent1.chromosome.length];
        
        for (int i = 0; i < parent1.chromosome.length; i++) {
            // Compute the average of corresponding genes from the parents
            childChromosome[i] = (parent1.chromosome[i] + parent2.chromosome[i]) / 2.0;
        }
        
        // Create a new offspring individual with the averaged genes
        Individual child = new Individual();
        child.chromosome = childChromosome;
        offspring.add(child);
        
        return offspring;
    }



	/**
	 * Returns a copy of the best individual in the population
	 * 
	 */
	private Individual getBest() {
		best = null;;
		for (Individual individual : population) {
			if (best == null) {
				best = individual.copy();
			} else if (individual.fitness < best.fitness) {
				best = individual.copy();
			}
		}
		return best;
	}

	/**
	 * Generates a randomly initialised population
	 * 
	 */
	private ArrayList<Individual> initialise() {
		population = new ArrayList<>();
		for (int i = 0; i < Parameters.popSize; ++i) {
			//chromosome weights are initialised randomly in the constructor
			Individual individual = new Individual();
			population.add(individual);
		}
		evaluateIndividuals(population);
		return population;
	}


	/**
	 * Selection -- Method to call the selection operator 
	 * 
	 */
	private Individual select() {	
		
		//return randomSelection();
		return tournamentSelection(3);	
		//return rouletteWheelSelection();
	    //return rankBasedSelection();
	}
	
	/**
     * Performs random selection.
     * 
     */
	private Individual randomSelection() {	
		
		Individual parent = population.get(Parameters.random.nextInt(Parameters.popSize));
		return parent.copy();
	}
	
	 /**
     * Performs tournament selection.
     * 
     */
	private Individual tournamentSelection(int tournamentSize) {
        ArrayList<Individual> tournament = new ArrayList<>();
        for (int i = 0; i < tournamentSize; i++) {
            tournament.add(population.get(Parameters.random.nextInt(population.size())));
        }
        return Collections.min(tournament); // Select the individual with the lowest fitness
    }
	
	/**
     * Performs roulette wheel  selection.
     * 
     */
	private Individual rouletteWheelSelection() {
	    // Calculate total fitness of the population
	    double totalFitness = 0.0;
	    for (Individual individual : population) {
	        totalFitness += individual.fitness;
	    }
	    
	    // Generate a random number between 0 and the total fitness
	    double spin = Parameters.random.nextDouble() * totalFitness;
	    
	    // Iterate through the population and select the individual where the spin falls
	    double cumulativeFitness = 0.0;
	    for (Individual individual : population) {
	        cumulativeFitness += individual.fitness;
	        if (cumulativeFitness >= spin) {
	            return individual.copy(); // Select the individual
	        }
	    }
	    
	    // If no individual is selected, return null (shouldn't happen)
	    return null;
	}
	
	/**
     * Performs rank based selection.
     * 
     */
	private Individual rankBasedSelection() {
	    // Sort the population based on fitness ranks
	    Collections.sort(population);
	    
	    // Calculate selection probabilities based on rank
	    double totalProbability = 0.0;
	    for (int i = 0; i < population.size(); i++) {
	        totalProbability += (i + 1); // Rank-based probability
	    }
	    
	    // Generate a random number between 0 and the total probability
	    double spin = Parameters.random.nextDouble() * totalProbability;
	    
	    // Iterate through the sorted population and select the individual based on rank
	    double cumulativeProbability = 0.0;
	    for (int i = 0; i < population.size(); i++) {
	        cumulativeProbability += (i + 1); // Incremental rank-based probability
	        if (cumulativeProbability >= spin) {
	            return population.get(i).copy(); // Select the individual
	        }
	    }
	    
	    // If no individual is selected, return null (shouldn't happen)
	    return null;
	}
	
	/**
	 * Mutation
	 * 
	 * 
	 */
	private void mutate(ArrayList<Individual> individuals) {		
		for(Individual individual : individuals) {
			for (int i = 0; i < individual.chromosome.length; i++) {
				if (Parameters.random.nextDouble() < Parameters.mutateRate) {
					if (Parameters.random.nextBoolean()) {
						individual.chromosome[i] += (Parameters.mutateChange);
					} else {
						individual.chromosome[i] -= (Parameters.mutateChange);
					}
				}
			}
		}		
	}

	/**
	 * 
	 * Replacement 
	 * 
	 * Performs elitist replacement 
	 * 
	 */
	private void elitistReplace(ArrayList<Individual> individuals) {
		for(Individual individual : individuals) {
			int idx = getWorstIndex();		
			population.set(idx, individual);
		}		
	}
	
	/**
     * Performs steady state replacement 
     * 
     */
	private void replaceSteadyState(ArrayList<Individual> offspring) {
	    // Sort the population by fitness (assuming lower fitness is better)
	    Collections.sort(population);
	    
	    // Replace the least fit individuals with the offspring
	    for (int i = 0; i < offspring.size(); i++) {
	        // Find the index of the least fit individual
	        int worstIndex = population.size() - 1 - i;
	        
	        // Replace the least fit individual with the offspring
	        population.set(worstIndex, offspring.get(i));
	    }
	}
	
	/**
     * Performs Generational replacement 
     * 
     */
	private void replaceGenerational(ArrayList<Individual> offspring) {
	    // Clear the existing population
	    population.clear();
	    
	    // Add all the new offspring to the population
	    population.addAll(offspring);
	}

	/**
	 * Returns the index of the worst member of the population
	 * @return
	 */
	private int getWorstIndex() {
		Individual worst = null;
		int idx = -1;
		for (int i = 0; i < population.size(); i++) {
			Individual individual = population.get(i);
			if (worst == null) {
				worst = individual;
				idx = i;
			} else if (individual.fitness > worst.fitness) {
				worst = individual;
				idx = i; 
			}
		}
		return idx;
	}	

	@Override
	public double activationFunction(double x) {
		if (x < -20.0) {
			return -1.0;
		} else if (x > 20.0) {
			return 1.0;
		}
		return x;
		/**
	     * Performs tanh activation function
	     * 
	     */
		//return Math.tanh(x);*/
		/**
	     * Performs ReLU activation function
	     * 
	     */
		//return Math.max(0, x);
		/**
	     * Performs sigmoid activation function
	     * 
	     */
		//return 1 / (1 + Math.exp(-x));
	}


}
