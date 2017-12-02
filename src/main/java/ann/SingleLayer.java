package ann;

import java.util.*;

public class SingleLayer extends ANN{




	public SingleLayer(Map<Input, Output> trainingData, Map<Input, Output> testingData) {
		generator = new Random();
		this.trainingData = trainingData;
		this.testingData = testingData;
		inLayer=new LinkedList<>();//initialisation List and Activation
		outLayer=new LinkedList<>();
		Activation act=new Sigmoid();
		// to be completed
		for (int i = 0; i < 10; i++) {       //initialize outlayer and inputlayer
			outLayer.add(new Neuron(act));
		}
		Iterator<Neuron> outiterator = outLayer.iterator();
		for (int i = 0; i < 784; i++) {
			InputNeuron inputNeuron=new InputNeuron(255);
			inLayer.add(inputNeuron);
			while(outiterator.hasNext()){
				inputNeuron.addChild(outiterator.next());
			}
			outiterator=outLayer.iterator();
            /*for (int j = 0; j < 10; j++) {
                inputNeuron.addChild(outLayer.get(j));
            }*/
		}
		Iterator<InputNeuron> initerator = inLayer.iterator();
		for (Neuron neuron: outLayer) {
			while(initerator.hasNext()){
				neuron.addParent(initerator.next());
			}
            /*for (int j = 0; j < 784; j++) {
                neuron.addParent(inLayer.get(j));
            }*/
			initerator=inLayer.iterator();
			neuron.initWeights();
		}
	}

	public Output feed(Input in){
		// to be completed
		double[] ret=new double[10];
        /*for (int i = 0; i < 28; i++) {
            for (int j = 0; j < 28; j++) {
                inLayer.get(28*i+j).feed(in.getValue(i,j));
            }
        }*/
		Iterator<InputNeuron> inputNeuronIterator = inLayer.iterator();
		Iterator<Double> inIterator = in.iterator();
		while(inputNeuronIterator.hasNext()){
			inputNeuronIterator.next().feed(inIterator.next());
		}

		for (int j=0;j<10;j++) {
			outLayer.get(j).feed();
			ret[j]=outLayer.get(j).getCurrentOutput();
			//System.out.println("ret[j] = " + ret[j]);
		}

		return new Output(ret);
	}


	public Map<Integer,Double> train(int nbIterations) {
		// to be completed
		Map<Integer,Double> ret=new HashMap<Integer, Double>();
		Iterator<Neuron> outiterator = outLayer.iterator();
		int j=0;
		double error=0;
		double[] value=new double[10];
		for (int i = 0; i < nbIterations; i++) {
			error=test(testingData,i);
			for (Map.Entry<Input, Output> d : trainingData.entrySet()) {
				Input in = d.getKey();
				//Output out = d.getValue();
				value=d.getValue().getVal();
                /*for (int j = 0; j < 10; j++) {
                    //outLayer.get(j).backPropagate(value[j]);
                }*/
                /*double[] tebb= feed(in).getVal();
                for (int k = 0; k < 10; k++) {
                    System.out.println(value[k]);
                    System.out.println("tebb[k] = " + tebb[k]);
                }*/
				feed(in);
				j=0;
				outiterator=outLayer.iterator();
				while (outiterator.hasNext()){
					outiterator.next().backPropagate(value[j]);
					//outiterator.next();
					j++;
				}
				j=0;
				outiterator=outLayer.iterator();
			}

			ret.put(i,error);
		}
		return ret;
	}


}