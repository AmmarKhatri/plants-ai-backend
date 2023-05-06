const express = require("express");
const router = express.Router();
const multer = require('multer');
const tf = require("@tensorflow/tfjs");
const fs = require("fs");
const tfHub = require("@tensorflow/tfjs-converter");
const tfn = require("@tensorflow/tfjs-node");
const sharp = require('sharp');
const { error } = require("console");
const upload = multer({ dest: 'uploads/' });

// MAP FOR OG MODEL, REVERSE MAP, IDX -> NAME, I USED THO

const class_idx_map = {
  Apple___Apple_scab: 0,
  Apple___Black_rot: 1,
  Apple___Cedar_apple_rust: 2,
  Apple___healthy: 3,
  Blueberry___healthy: 4,
  "Cherry_(including_sour)___Powdery_mildew": 5,
  "Cherry_(including_sour)___healthy": 6,
  "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": 7,
  "Corn_(maize)___Common_rust_": 8,
  "Corn_(maize)___Northern_Leaf_Blight": 9,
  "Corn_(maize)___healthy": 10,
  Grape___Black_rot: 11,
  "Grape___Esca_(Black_Measles)": 12,
  "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": 13,
  Grape___healthy: 14,
  "Orange___Haunglongbing_(Citrus_greening)": 15,
  Peach___Bacterial_spot: 16,
  Peach___healthy: 17,
  "Pepper,_bell___Bacterial_spot": 18,
  "Pepper,_bell___healthy": 19,
  Potato___Early_blight: 20,
  Potato___Late_blight: 21,
  Potato___healthy: 22,
  Raspberry___healthy: 23,
  Soybean___healthy: 24,
  Squash___Powdery_mildew: 25,
  Strawberry___Leaf_scorch: 26,
  Strawberry___healthy: 27,
  Tomato___Bacterial_spot: 28,
  Tomato___Early_blight: 29,
  Tomato___Late_blight: 30,
  Tomato___Leaf_Mold: 31,
  Tomato___Septoria_leaf_spot: 32,
  "Tomato___Spider_mites Two-spotted_spider_mite": 33,
  Tomato___Target_Spot: 34,
  Tomato___Tomato_Yellow_Leaf_Curl_Virus: 35,
  Tomato___Tomato_mosaic_virus: 36,
  Tomato___healthy: 37,
};

// MAP FOR DATASET V4
v4_idxclass_map = {
  0: 'Apple_scab',
  1: 'Bacterial_spot',
  2: 'Black_rot',
  3: 'Cedar_apple_rust',
  4: 'Cercospora_leaf_spot Gray_leaf_spot',
  5: 'Common_rust_',
  6: 'Early_blight',
  7: 'Esca_(Black_Measles)',
  8: 'Haunglongbing_(Citrus_greening)',
  9: 'Late_blight',
  10: 'Leaf_Mold',
  11: 'Leaf_blight_(Isariopsis_Leaf_Spot)',
  12: 'Leaf_scorch',
  13: 'Northern_Leaf_Blight',
  14: 'Powdery_mildew',
  15: 'Septoria_leaf_spot',
  16: 'Spider_mites Two-spotted_spider_mite',
  17: 'Target_Spot',
  18: 'Tomato_Yellow_Leaf_Curl_Virus',
  19: 'Tomato_mosaic_virus',
  20: 'healthy'
}

// MAP FOR DATASET V3

v3_idxclass_map = {
    0: 'Apple___healthy',
    1: 'Apple_diseased',
    2: 'Blueberry___healthy',
    3: 'Cherry_(including_sour)___healthy',
    4: 'Cherry_(including_sour)_diseased',
    5: 'Corn_(maize)___healthy',
    6: 'Corn_(maize)_diseased',
    7: 'Grape___healthy',
    8: 'Grape_diseased',
    9: 'Orange_diseased',
    10: 'Peach___healthy',
    11: 'Peach_diseased',
    12: 'Pepper,_bell___healthy',
    13: 'Pepper,_bell_diseased',
    14: 'Potato___healthy',
    15: 'Potato_diseased',
    16: 'Raspberry___healthy',
    17: 'Soybean___healthy',
    18: 'Squash_diseased',
    19: 'Strawberry___healthy',
    20: 'Strawberry_diseased',
    21: 'Tomato___healthy',
    22: 'Tomato_diseased'
}

// ORIGINAL MODEL ON DATASET V4, USING TRANSFER LEARNING

let modelv4

async function loadGraphModelV4() {
  const handler = tfn.io.fileSystem("tfjs_model_v4/model.json");

  modelv4 = await tf.loadGraphModel(handler);
  // outputTensor = model.outputs[0];
  // console.log(outputTensor.classNames); // ["class 0", "class 1", "class 2", ...]
  return modelv4;
}

async function predict_image_v4(img_path) {
  const img = fs.readFileSync(img_path);
  const tensor = tfn.node.decodeImage(img);

  const resized = tf.image.resizeBilinear(tensor, [224, 224]);
  const casted = resized.cast("float32");
  const expanded = casted.expandDims(0);
  const normalized = expanded.div(255.0);

  if (!modelv4) {
    // Check if the model has already been loaded
    await loadGraphModelV4(); // Load the model if it hasn't been loaded yet
  }
  const prediction = modelv4.predict(normalized);

  const probs = prediction.dataSync();
  const classIndex = probs.indexOf(Math.max(...probs));

  // console.log("Predicted class index:", classIndex);
  // console.log("Class: ", idx_class_map[classIndex])
  // console.log("Accuracy: ", probs[classIndex])

  return [v4_idxclass_map[classIndex], probs[classIndex]];
}

// MODEL V3 DATASET OF FORM PALNTX HEALTHY / NONHEALTHY


let modelv3

async function loadGraphModelV3() {
  const handler = tfn.io.fileSystem("tfjs_model_v3/model.json");

  modelv3 = await tf.loadGraphModel(handler);
  return modelv3;
}

async function predict_image_v3(img_path) {
  const img = fs.readFileSync(img_path);
  const tensor = tfn.node.decodeImage(img);

  const resized = tf.image.resizeBilinear(tensor, [224, 224]);
  const casted = resized.cast("float32");
  const expanded = casted.expandDims(0);
  const normalized = expanded.div(255.0);

  if (!modelv3) {
    // Check if the model has already been loaded
    await loadGraphModelV3(); // Load the model if it hasn't been loaded yet
  }
  const prediction = modelv3.predict(normalized);

  const probs = prediction.dataSync();
  const classIndex = probs.indexOf(Math.max(...probs));

  return [v3_idxclass_map[classIndex], probs[classIndex]];
}

const idx_class_map = {};
for (let key in class_idx_map) {
  let value = class_idx_map[key];
  idx_class_map[value] = key;
}

// ORIGNAL MODEL, FIRST DEPLOYED ON SERVER


let model;

async function loadGraphModel() {
  const handler = tfn.io.fileSystem("tfjs_model_dir/model.json");

  model = await tf.loadGraphModel(handler);
  return model;
}

async function predict_image(img) {
  // const img = fs.readFileSync("test_pics/images-3.jpeg");
  const tensor = tfn.node.decodeImage(img);

  const resized = tf.image.resizeBilinear(tensor, [224, 224]);
  const casted = resized.cast("float32");
  const expanded = casted.expandDims(0);
  const normalized = expanded.div(255.0);

  if (!model) {
    // Check if the model has already been loaded
    await loadGraphModel(); // Load the model if it hasn't been loaded yet
  }
  const prediction = model.predict(normalized);

  const probs = prediction.dataSync();
  const classIndex = probs.indexOf(Math.max(...probs));

  // console.log("Predicted class index:", classIndex);
  // console.log("Class: ", idx_class_map[classIndex])
  // console.log("Accuracy: ", probs[classIndex])

  return [idx_class_map[classIndex], probs[classIndex]];
}

loadGraphModel();

// predict_image("test_pics/apple-cedar-rust.jpeg").then(
//   result => {
//     console.log(result)
//   }
// )

// LEAF RECOGNITION MODEL, SAYS IS LEAF OR NOT LEAF


let leaf_model
async function loadLeafGraphModel() {
  const handler = tfn.io.fileSystem("tfjs_leaf_model_dir/model.json");

  leaf_model = await tf.loadGraphModel(handler);
  // outputTensor = model.outputs[0];
  // console.log(outputTensor.classNames); // ["class 0", "class 1", "class 2", ...]
  return leaf_model;
}

idx_map_leaf = {
  0 : "is_leaf",
  1 : "is_not_leaf"
}

async function predict_leaf_image(img) {
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

// AMMARS MODEL, TRAINED BY YAHYA ON OG DATASET, TESTED AS WELL


let ammar_model

async function loadGraphModel_Ammar() {
  const handler = tfn.io.fileSystem("tfjs_ammar_model_og/model.json");

  ammar_model = await tf.loadGraphModel(handler);
  return ammar_model;
}

async function predict_image_ammar(img_path) {
  const img = fs.readFileSync(img_path);
  const tensor = tfn.node.decodeImage(img);

  const resized = tf.image.resizeBilinear(tensor, [224, 224]);
  const casted = resized.cast("float32");
  const expanded = casted.expandDims(0);
  const normalized = expanded.div(255.0);

  if (!ammar_model) {
    // Check if the model has already been loaded
    await loadGraphModel_Ammar(); // Load the model if it hasn't been loaded yet
  }
  const prediction = ammar_model.predict(normalized);

  const probs = prediction.dataSync();
  const classIndex = probs.indexOf(Math.max(...probs));

  // console.log("Predicted class index:", classIndex);
  // console.log("Class: ", idx_class_map[classIndex])
  // console.log("Accuracy: ", probs[classIndex])

  return [idx_class_map[classIndex], probs[classIndex]];
}

// AMMARS MODEL, TRAINED BY HIM, NOT TESTED

let ammar_model_v4

async function loadGraphModel_Ammar_v4() {
  const handler = tfn.io.fileSystem("tfjs_ammar_model_v4/model.json");

  ammar_model_v4 = await tf.loadGraphModel(handler);
  return ammar_model_v4;
}

async function predict_image_ammar_v4(img_path) {
  const img = fs.readFileSync(img_path);
  const tensor = tfn.node.decodeImage(img);

  const resized = tf.image.resizeBilinear(tensor, [224, 224]);
  const casted = resized.cast("float32");
  const expanded = casted.expandDims(0);
  const normalized = expanded.div(255.0);

  if (!ammar_model_v4) {
    // Check if the model has already been loaded
    await loadGraphModel_Ammar_v4(); // Load the model if it hasn't been loaded yet
  }
  const prediction = ammar_model_v4.predict(normalized);

  const probs = prediction.dataSync();
  const classIndex = probs.indexOf(Math.max(...probs));

  // console.log("Predicted class index:", classIndex);
  // console.log("Class: ", idx_class_map[classIndex])
  // console.log("Accuracy: ", probs[classIndex])

  return [v4_idxclass_map[classIndex], probs[classIndex]];
}

// AMMARS MODEL, TRAINED BY YAHYA, TESTED WITH A TEST DATASET

let ammar_model_v4_tested

async function loadGraphModel_Ammar_v4_tested() {
  const handler = tfn.io.fileSystem("tfjs_ammar_model_v4_tested/model.json");

  ammar_model_v4_tested = await tf.loadGraphModel(handler);
  return ammar_model_v4_tested;
}

async function predict_image_ammar_v4_tested(img_path) {
  const img = fs.readFileSync(img_path);
  const tensor = tfn.node.decodeImage(img);

  const resized = tf.image.resizeBilinear(tensor, [224, 224]);
  const casted = resized.cast("float32");
  const expanded = casted.expandDims(0);
  const normalized = expanded.div(255.0);

  if (!ammar_model_v4_tested) {
    // Check if the model has already been loaded
    await loadGraphModel_Ammar_v4_tested(); // Load the model if it hasn't been loaded yet
  }
  const prediction = ammar_model_v4_tested.predict(normalized);

  const probs = prediction.dataSync();
  const classIndex = probs.indexOf(Math.max(...probs));

  return [v4_idxclass_map[classIndex], probs[classIndex]];
}



// for testing model
// predict_image_ammar_v4_tested("/Users/yahyaahmedkhan/Desktop/dev/UniProjects/plants-ai-backend/Unknown-4.jpg").then(
//   result => {
//     console.log(result)
//   }
// )






router.get("/", (req, res, next) => {
  // testing if taking requests
  res.status(200).json({
    res: "server is receiving requests",
  });
});
router.post("/", upload.single('file'), async (req, res, next) => {
  // put logic here
  const file = req.file;
  const filePath = file.path;
  // Read the uploaded file from disk
  const fileBuffer = fs.readFileSync(filePath);

  // Convert the file to JPEG format using sharp
  const converted = await sharp(fileBuffer).jpeg().toBuffer();
  predict_image(converted).then(result =>{
    res.status(201).json({
      type: result[0],
      accuracy: result[1]
    });
  }).catch(err=>{
    console.log(err);
    res.status(500).json({
      message: "Error processing file"
    });
  })
});

module.exports = router;
