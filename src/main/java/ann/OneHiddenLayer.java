package ann;

import java.util.*;

public class OneHiddenLayer extends ANN{

	List<Neuron> hiddenLayer;

	int numHiddenNeurons = 10;


	public OneHiddenLayer(Map<Input, Output> trainingData, Map<Input, Output> testingData, int numHiddenNeurons) {
		generator = new Random();
		this.trainingData = trainingData;
		this.testingData = testingData;

		inLayer=new LinkedList<>();//initialisation List and Activation
		outLayer=new LinkedList<>();
		hiddenLayer=new LinkedList<>();
		Activation act=new Sigmoid();
		this.numHiddenNeurons=numHiddenNeurons;

		for (int i = 0; i < 10; i++) {       //initialize outlayer and inputlayer
			outLayer.add(new Neuron(act));
		}
		Iterator<Neuron> outiterator = outLayer.iterator();

		for (int i = 0; i < numHiddenNeurons; i++) {
			Neuron neuron=new Neuron(act);
			hiddenLayer.add(neuron);
			while(outiterator.hasNext()){
				neuron.addChild(outiterator.next());
			}
			outiterator=outLayer.iterator();
		}

		Iterator<Neuron> hiddeniterator = hiddenLayer.iterator();
		for (Neuron neuronout:outLayer) {
			while(hiddeniterator.hasNext()){
				neuronout.addParent(hiddeniterator.next());
			}
			hiddeniterator=hiddenLayer.iterator();
			neuronout.initWeights();
		}

		for (int i = 0; i < 784; i++) {
			InputNeuron inputNeuron=new InputNeuron(255);
			inLayer.add(inputNeuron);
			while(hiddeniterator.hasNext()){
				inputNeuron.addChild(hiddeniterator.next());
			}
			hiddeniterator=hiddenLayer.iterator();
		}
		Iterator<InputNeuron> initerator = inLayer.iterator();
		for (Neuron neuron: hiddenLayer) {
			while(initerator.hasNext()){
				neuron.addParent(initerator.next());
			}
			initerator=inLayer.iterator();
			neuron.initWeights();
		}

	}


	public Output feed(Input in){

		double[] ret=new double[10];

		Iterator<InputNeuron> inputNeuronIterator = inLayer.iterator();
		Iterator<Double> inIterator = in.iterator();
		while(inputNeuronIterator.hasNext()){
			inputNeuronIterator.next().feed(inIterator.next());
		}
		for (Neuron aHiddenLayer : hiddenLayer) {
			aHiddenLayer.feed();
		}
		for (int j=0;j<10;j++) {
			outLayer.get(j).feed();
			ret[j]=outLayer.get(j).getCurrentOutput();
		}
		return new Output(ret);
	}



	public Map<Integer,Double> train(int nbIterations) {

		Map<Integer,Double> ret=new HashMap<Integer, Double>();
		Iterator<Neuron> outiterator = outLayer.iterator();
		int j=0;
		double error=0;
		double[] value=new double[10];
		for (int i = 0; i < nbIterations; i++) {
			for (Map.Entry<Input, Output> d : trainingData.entrySet()) {
				Input in = d.getKey();
				value=d.getValue().getVal();
				feed(in);
				while (outiterator.hasNext()){
					outiterator.next().backPropagate(value[j]);
					j++;
				}
				for (Neuron hiddenneuron : hiddenLayer) {
					hiddenneuron.backPropagate(1);
				}

				j=0;
				outiterator=outLayer.iterator();
			}
			error=test(testingData,i+1);
			ret.put(i,error);
		}
		return ret;
	}

}



