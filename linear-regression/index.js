async function plotData(pointsArray, featureName, predictedPointsArray = null) {
    const series = ['orignial'];
    const data = { values: [pointsArray], series };

    if (Array.isArray(predictedPointsArray)) {
        data.values.push(predictedPointsArray);
        series.push("predicted");
      }

    const surface = { name: `${featureName} vs price`, tab: 'Charts' };
    const opt = { xLabel: featureName, yLabel: 'Price' };
    tfvis.render.scatterplot(surface, data, opt);
  }
  async function plotPredictionLine () {
    const [xs, ys] = tf.tidy(() => {
   
      const normalisedXs = tf.linspace(0, 1, 100);
      const normalisedYs = model.predict(normalisedXs.reshape([100, 1]));
   
      const xs = denormalise(normalisedXs, normalisedFeature.min, normalisedFeature.max);
      const ys = denormalise(normalisedYs, normalisedLabel.min, normalisedLabel.max);
   
      return [ xs.dataSync(), ys.dataSync() ];
    });
   
    const predictedPoints = Array.from(xs).map((val, i) => {
      return {x: val, y: ys[i]}
    });
   
    await plotData(points, "Square feet", predictedPoints);
  }
function normalise(tensor, previousMin = null, previousMax = null) {
    const min = previousMin || tensor.min();
    const max = previousMax || tensor.max();
    const normalisedTensor = tensor.sub(min).div(max.sub(min));
    return {
        tensor: normalisedTensor,
        min,
        max
    };
}
  

function denormalise(tensor, min, max) {
      return tensor.mul(max.sub(min)).add(min);
  }


let model = null;
function createModel() {
    model = tf.sequential();

    model.add(tf.layers.dense({
      units: 1, 
      useBias: false, 
      activation: 'linear',
      inputShape: [1]
    }));

    const optimizer = tf.train.sgd(0.1);
    model.compile({
      loss: 'meanSquaredError',
      optimizer
    });

    return model;
  }

async function trainModel(model, featureTraingTensor, labelTraingTensor) {

    const surface = { name: 'Training Performance', tab: 'Training' };

    return model.fit(featureTraingTensor, labelTraingTensor, {
        epochs: 20,
        batchSize: 512,
        validationSplit: 0.2,
        callbacks: tfvis.show.fitCallbacks(surface, ['loss', 'acc']),
        onEpochBegin: async function() {
            await plotPredictionLine();
            const layer = model.getLayer(undefined, 0);
            tfvis.show.layer({name: 'Layer 1'}, layer);
        }
    });
  }

let points;
let normalisedFeature, normalisedLabel;
let featureTraingTensor, featureTestingTensor, labelTraingTensor, labelTestingTensor;

async function run() {
    const housingDataset = tf.data.csv('/kc_house_data.csv');
    
    const pointsDataset = housingDataset.map(record => (
      {
        x: record.sqft_living,
        y: record.price
      }
    ));

    points = await pointsDataset.toArray();
    
    if (points.length % 2 !== 0) { //remove one element if number of records in odd
      points.pop();
    }

    tf.util.shuffle(points);
    plotData(points, 'Living Area');

    //feature tensor 
    const featureValues = await points.map(p => p.x);
    const featureTensor = tf.tensor2d(featureValues, [featureValues.length, 1]);

    //label tensor 
    const labelValues = await points.map(p => p.y);
    const labelTensor = tf.tensor2d(labelValues, [labelValues.length, 1]);

    //normalise tensors
    normalisedFeature = normalise(featureTensor);
    normalisedLabel = normalise(labelTensor);

    //denormalise tensors
    // const denormalisedFeature = denormalise(normalisedFeature.tensor, normalisedFeature.min, normalisedFeature.max);
    // const denormalisedLabel = denormalise(normalisedLabel.tensor, normalisedLabel.min, normalisedLabel.max);

    // denormalisedFeature.print();
    // denormalisedLabel.print();

    //split data into training and testing datasets

    [featureTraingTensor, featureTestingTensor] = tf.split(normalisedFeature.tensor, 2);
    [labelTraingTensor, labelTestingTensor] = tf.split(normalisedLabel.tensor, 2);

    //dispose used tensors to optimise memeory 
    featureTensor.dispose();
    labelTensor.dispose();

    // Update status and enable train button
    document.getElementById("model-status").innerHTML = "No model trained";
    document.getElementById("train-button").removeAttribute("disabled");
    document.getElementById("load-button").removeAttribute("disabled");
}

async function predict () {
   const predictionInput = parseInt(document.getElementById("prediction-input").value);

   if (isNaN(predictionInput)) {
    alert("Please enter a valid number");
  }
  else if (predictionInput < 200) {
    alert("Please enter a value equal or greater than 200");
  }
  else {
    tf.tidy(() => {
        const inputTensor = tf.tensor1d([predictionInput]);
        const normalisedInput = normalise(inputTensor, normalisedFeature.min, normalisedFeature.max);
        const normalisedOutputTensor = model.predict(normalisedInput.tensor);
        const outputTensor = denormalise(normalisedOutputTensor, normalisedLabel.min, normalisedLabel.max);
        const outputValue = outputTensor.dataSync()[0];
        document.getElementById("prediction-output").innerHTML = `The predicted house price is: <br />`
          + `<span style="font-size: 2em">\$${(outputValue/1000).toFixed(0)*1000}</span>`;
      });
  }
} 

let storageId = 'kc-house-price-prediction';
async function save() {
    const savedResults = await model.save(`localstorage://${storageId}`);
    document.getElementById("model-status").innerHTML = `Trained (saved model ${savedResults.modelArtifactsInfo.dateSaved})`;
}

async function load() {
    const storageKey = `localstorage://${storageId}`;
    const models = await tf.io.listModels();
    if (models !== null) {
        const modelInfo = models[storageKey];

        if (modelInfo) {
            model = await tf.loadLayersModel(storageKey);
            tfvis.show.modelSummary({name: 'Model Summary'}, model);
            const layer = model.getLayer(undefined, 0);
            tfvis.show.layer({name: 'Layer 1'}, layer);

            await plotPredictionLine();
    
            document.getElementById("model-status").innerHTML = `Trained (saved model ${modelInfo.dateSaved})`;
            document.getElementById("predict-button").removeAttribute("disabled");
        } else {
            alert('Model not found');
        }
    } else {
        alert('Model not found');
    }
}

async function test() {
    const lossTensor = model.evaluate(featureTestingTensor, labelTestingTensor);
    const loss = await lossTensor.dataSync();

    document.getElementById("testing-status").innerHTML = `Testing set loss: ${parseFloat(loss).toPrecision(5)}`;
}

async function train() {

    // Disable all buttons and update status
    ["train", "test", "load", "predict", "save"].forEach(id => {
        document.getElementById(`${id}-button`).setAttribute("disabled", "disabled");
    });
    document.getElementById("model-status").innerHTML = "Training...";

    //create model 

    model = createModel();
    //model.summary();

    tfvis.show.modelSummary({name: 'Model Summary'}, model);
    const layer = model.getLayer(undefined, 0);
    tfvis.show.layer({name: 'Layer 1'}, layer);

    //train model 
    const result = await trainModel(model, featureTraingTensor, labelTraingTensor);
    const trainingLoss = result.history.loss.pop();
    const validationLoss = result.history.val_loss.pop();

    await plotPredictionLine();

    document.getElementById("model-status").innerHTML = `Trained (unsaved)\nLoss: ${trainingLoss.toPrecision(5)}\nValidation loss: ${validationLoss.toPrecision(5)}`;
    document.getElementById("test-button").removeAttribute("disabled");
    document.getElementById("save-button").removeAttribute("disabled");
}

async function toggleVisor() {
    tfvis.visor().toggle()
}

run();