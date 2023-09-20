async function plotPredictionHeatmap (name = "Predicted class", size = 400) {
    const [ valuesPromise, xTicksPromise, yTicksPromise ] = tf.tidy(() => {
      const gridSize = 50;
      const predictionColumns = [];
      // Heatmap order is confusing: columns first (top to bottom) then rows (left to right)
      // We want to convert that to a standard cartesian plot so invert the y values
      for (let colIndex = 0; colIndex < gridSize; colIndex++) {
        // Loop for each column, starting from the left
        const colInputs = [];
        const x = colIndex / gridSize;
        for (let rowIndex = 0; rowIndex < gridSize; rowIndex++) {
          // Loop for each row, starting from the top
          const y = (gridSize - rowIndex) / gridSize;
          colInputs.push([x, y]);
        }
        
        const colPredictions = model.predict(tf.tensor2d(colInputs));
        predictionColumns.push(colPredictions);
      }
      const valuesTensor = tf.stack(predictionColumns);
 
      const normalisedLabelsTensor = tf.linspace(0, 1, gridSize);
      const xTicksTensor = denormalise(normalisedLabelsTensor,
        normalisedFeature.min[0], normalisedFeature.max[0]);
      const yTicksTensor = denormalise(normalisedLabelsTensor.reverse(),
        normalisedFeature.min[1], normalisedFeature.max[1]);
 
      return [ valuesTensor.array(), xTicksTensor.array(), yTicksTensor.array() ];
    });
 
    const values = await valuesPromise;
    const xTicks = await xTicksPromise;
    const xTickLabels = xTicks.map(l => (l/1000).toFixed(1)+"k sqft");
    const yTicks = await yTicksPromise;
    const yTickLabels = yTicks.map(l => "$"+(l/1000).toFixed(0)+"k");
    const data = {
      values,
      xTickLabels,
      yTickLabels,
    };
 
    tfvis.render.heatmap({
      name: `${name} (local)`,
      tab: "Predictions"
    }, data, { height: size });
    tfvis.render.heatmap({ 
      name: `${name} (full domain)`, 
      tab: "Predictions" 
    }, data, { height: size, domain: [0, 1] });
}
async function plotClasses (pointsArray, classKey, size = 400, equalizeClassSizes = false) {
  // Add each class as a series
  const allSeries = {};
  pointsArray.forEach(p => {
    // Add each point to the series for the class it is in
    const seriesName = `${classKey}: ${p.class}`;
    let series = allSeries[seriesName];
    if (!series) {
      series = [];
      allSeries[seriesName] = series;
    }
    series.push(p);
  });
  
  if (equalizeClassSizes) {
    // Find smallest class
    let maxLength = null;
    Object.values(allSeries).forEach(series => {
      if (maxLength === null || series.length < maxLength && series.length >= 100) {
        maxLength = series.length;
      }
    });
    // Limit each class to number of elements of smallest class
    Object.keys(allSeries).forEach(keyName => {
      allSeries[keyName] = allSeries[keyName].slice(0, maxLength);
      if (allSeries[keyName].length < 100) {
        delete allSeries[keyName];
      }
    });
  }
  
  tfvis.render.scatterplot(
    {
      name: `Square feet vs House Price`,
      styles: { width: "100%" }
    },
    {
      values: Object.values(allSeries),
      series: Object.keys(allSeries),
    },
    {
      xLabel: "Square feet",
      yLabel: "Price",
      height: size,
      width: size*1.5,
    }
  );
}
  
function normalise (tensor, previousMin = null, previousMax = null) {
  const featureDimensions = tensor.shape.length > 1 && tensor.shape[1];
  if (featureDimensions && featureDimensions > 1) {
    // More than one feature
    // Split into separate tensors
    const features = tf.split(tensor, featureDimensions, 1);

    // Normalise and find min/max values for each feature
    const normalisedFeatures = features.map((featureTensor, i) => 
      normalise(featureTensor, previousMin ? previousMin[i] : null, previousMax ? previousMax[i] : null));

    // Prepare return values
    // In this case the min and max properties will be arrays, with one
    // value for each feature
    const returnTensor = tf.concat(normalisedFeatures.map(f => f.tensor), 1);
    const min = normalisedFeatures.map(f => f.min);
    const max = normalisedFeatures.map(f => f.max);
    return { tensor: returnTensor, min, max};
  }
  else {
    // Just one feature
    const min = previousMin || tensor.min();
    const max = previousMax || tensor.max();
    const normalisedTensor = tensor.sub(min).div(max.sub(min));
    return {
      tensor: normalisedTensor,
      min,
      max
    };
  }
}
  

function denormalise(tensor, min, max) {
  const featureDimensions = tensor.shape.length > 1 && tensor.shape[1];
  if (featureDimensions && featureDimensions > 1) {
    // More than one feature
    // Split into separate tensors
    const features = tf.split(tensor, featureDimensions, 1);

    // Denormalise
    const denormalised = features.map((featureTensor, i) => denormalise(featureTensor, min[i], max[i]));

    const returnTensor = tf.concat(denormalised, 1);
    return returnTensor;
  }
  else {
    const denormalisedTensor = tensor.mul(max.sub(min)).add(min);
    return denormalisedTensor;
  }
}


let model = null;
function createModel() {
  model = tf.sequential();
  
  model.add(tf.layers.dense({
    units: 10,
    useBias: true,
    activation: 'sigmoid',
    inputDim: 2,
  }));
  model.add(tf.layers.dense({
    units: 10,
    activation: 'sigmoid',
    useBias: true,
  }));
  // Output layer:
  model.add(tf.layers.dense({
    units: 1,
    activation: 'sigmoid',
    useBias: true,
  }));
  
  const optimizer = tf.train.adam();
  model.compile({
    loss: 'binaryCrossentropy',
    optimizer,
    metrics: ['accuracy'],
  });
  
  return model;
}

async function trainModel(model, featureTraingTensor, labelTraingTensor) {

    const surface = { name: 'Training Performance', tab: 'Training' };

    return model.fit(featureTraingTensor, labelTraingTensor, {
        epochs: 500,
        batchSize: 32,
        validationSplit: 0.2,
        callbacks: tfvis.show.fitCallbacks(surface, ['loss', 'acc']),
        onEpochBegin: async function() {
          await plotPredictionHeatmap();
          const layer = model.getLayer(undefined, 0);
          tfvis.show.layer({name: 'Layer 1'}, layer);
        }
    });
  }

let points;
let normalisedFeature, normalisedLabel;
let featureTraingTensor, featureTestingTensor, labelTraingTensor, labelTestingTensor;

async function run() {
    const housingDataset = tf.data.csv('/kc_waterfronts_fake.csv');
    
    const pointsDataset = housingDataset.map(record => (
      {
        x: record.sqft_living,
        y: record.price,
        class: record.waterfront,
      }
    ));

    points = await pointsDataset.toArray();
    
    if (points.length % 2 !== 0) { //remove one element if number of records in odd
      points.pop();
    }

    tf.util.shuffle(points);
    plotClasses(points, "Waterfront");

    // Extract Features (inputs)
    const featureValues = points.map(p => [p.x, p.y]);
    const featureTensor = tf.tensor2d(featureValues);
    
    // Extract Labels (outputs)
    const labelValues = points.map(p => p.class);
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
  const predictionInputOne = parseInt(document.getElementById("prediction-input-1").value);
  const predictionInputTwo = parseInt(document.getElementById("prediction-input-2").value);
  if (isNaN(predictionInputOne)) {
    alert("Please enter a valid number");
  }
  else if (isNaN(predictionInputTwo)) {
    alert("Please enter a valid number");
  }
  else {
    tf.tidy(() => {
      const inputTensor = tf.tensor2d([[predictionInputOne, predictionInputTwo]]);
      const normalisedInput = normalise(inputTensor, normalisedFeature.min, normalisedFeature.max);
      const normalisedOutputTensor = model.predict(normalisedInput.tensor);
      const outputTensor = denormalise(normalisedOutputTensor, normalisedLabel.min, normalisedLabel.max);
      const outputValue = outputTensor.dataSync()[0];
      document.getElementById("prediction-output").innerHTML = `The likelihood of being a waterfront property is: ${(outputValue*100).toFixed(1)}%`;
    });
  }
}

let storageId = 'kc-house-price-binary';
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
    
            await plotPredictionHeatmap();
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
    await plotPredictionHeatmap();
    const trainingLoss = result.history.loss.pop();
    const validationLoss = result.history.val_loss.pop();

    document.getElementById("model-status").innerHTML = `Trained (unsaved)\nLoss: ${trainingLoss.toPrecision(5)}\nValidation loss: ${validationLoss.toPrecision(5)}`;
    document.getElementById("test-button").removeAttribute("disabled");
    document.getElementById("save-button").removeAttribute("disabled");
}

async function toggleVisor() {
    tfvis.visor().toggle()
}

run();