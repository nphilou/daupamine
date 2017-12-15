package ann;



import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

public class Neuron {
	/** name of the neuron */
	String name;
	/** List of parent neurons, i.e. the list of neurons that are used as input */
	List<Neuron> parents;
	/** list of child neurons,i.e. the list of neurons that use the output of this neuron */
	List<Neuron> children;
	/** activation function */
	Activation h;
	/** weight of the input neurons: it maps an input neuron to a weight */
	Map<Neuron,Double> w;
	/** value of the learning rate */
	final static double eta = 0.01;
	/** current value of the output of the neuron */
	protected double out;
	/** current value of the error of the neuron */
	protected double error;
	/** random number generator */
	public static Random generator;

	/**
	 * returns the current value of the error for that neuron.
	 * @return
	 */
	public double getError(){return error;}

	/**
	 * Constructor
	 * @param h is an activation function (linear, sigmoid, tanh)
	 */
	public Neuron(Activation h){
		if (generator==null)
			generator = new Random();
		this.h=h;
		parents = new ArrayList<>();
		children = new ArrayList<>();
		w = new HashMap<Neuron,Double>();
	}

	public void addParent(Neuron parent){
		parents.add(parent);
	}

	public void addChild(Neuron child){
		children.add(child);
	}

	/**
	 * Initializes randomly the weights of the incoming edges
	 */
	public void initWeights(){
		for (Neuron neuron:parents) {
			w.put(neuron,generator.nextDouble()/parents.size());
			//w.put(neuron, -0.5+generator.nextDouble());///784));
		}
	}



	/**
	 * Computes the output of a neuron that is either in the hidden layer or in the output layer.
	 * (there are no arguments as the neuron is not an inputNeuron)
	 */
	public void feed(){
		out=0;
		for (Neuron neuron:parents) {
			out+=neuron.getCurrentOutput()*w.get(neuron);
		}
		out=h.activate(out);

	}



	/**
	 * back propagation for a neuron in the output layer
	 * @param target is the correct value.
	 */
	public void backPropagate(double target){
		error= this.out*(1-this.out)*(target-this.out);//((target-out);//
		if(children.isEmpty()){
			for (Neuron neuron : parents) {
				w.put(neuron, w.get(neuron) + eta * error * neuron.getCurrentOutput());
			}
		}
		else{
			error=0;
			for (Neuron child:children){
				error+=child.getError()*child.w.get(this);
			}
			error=error*out*(1-out);
			for (Neuron father:parents){
				w.put(father,w.get(father)+eta*error*father.getCurrentOutput());
			}
		}
	}


	/**
	 * returns the current ouput (it should be called once the output has been computed,
	 * i.e. after calling feed)
	 * @return the current value of the ouput
	 */
	public double getCurrentOutput(){
		return out;
	}

	/** returns the name of the neuron *
	 *
	 */
	public String toString(){
		return name + " out: " + out ;
	}


}