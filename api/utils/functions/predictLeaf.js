const tf = require('@tensorflow/tfjs');
const tfn = require('@tensorflow/tfjs-node');
const fs = require('fs');
const {idx_map_leaf} = require("../models/models")

//loading the leaf model
let leaf_model
async function loadLeafGraphModel() {
  const handler = tfn.io.fileSystem("tfjs_leaf_model_dir/model.json");
  leaf_model = await tf.loadGraphModel(handler);
}

//defining the isleaf function
async function predict_leaf(img) {
  const img1 = fs.readFileSync(img);
  const tensor = tfn.node.decodeImage(img1);
  const resized = tf.image.resizeBilinear(tensor, [224, 224]);
  const casted = resized.cast("float32");
  const expanded = casted.expandDims(0);
  const normalized = expanded.div(255.0);
  if (!leaf_model) {
    // Check if the model has already been loaded
    await loadLeafGraphModel(); // Load the model if it hasn't been loaded yet
  }
  const prediction = leaf_model.predict(normalized);
  const probs = prediction.dataSync();
  const classIndex = probs.indexOf(Math.max(...probs));
  return [idx_map_leaf[classIndex], probs[classIndex]];
}

module.exports = {
    loadLeafGraphModel,
    predict_leaf
};