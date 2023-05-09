const tf = require('@tensorflow/tfjs');
const tfn = require('@tensorflow/tfjs-node');
const fs = require('fs');
const { v4_idxclass_map } = require("../models/models")

// loading the model
let model
async function load_graph_model() {
  const handler = tfn.io.fileSystem("tfjs_x_model_v4_tested30/model.json");
  model = await tf.loadGraphModel(handler);
  return model;
}

//defining the predict image function
async function predict_image(img_path) {
  const img = fs.readFileSync(img_path);
  const tensor = tfn.node.decodeImage(img);
  const resized = tf.image.resizeBilinear(tensor, [224, 224]);
  const casted = resized.cast("float32");
  const expanded = casted.expandDims(0);
  const normalized = expanded.div(255.0);
  if (!model) {
    // Check if the model has already been loaded
    await load_graph_model(); // Load the model if it hasn't been loaded yet
  }
  const prediction = model.predict(normalized);
  const probs = prediction.dataSync();
  const classIndex = probs.indexOf(Math.max(...probs));
  return [v4_idxclass_map[classIndex], probs[classIndex]];
}

module.exports = {
    load_graph_model,
    predict_image
};