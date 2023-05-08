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

let model

async function load_graph_model() {
  const handler = tfn.io.fileSystem("tfjs_x_model_v4_tested30/model.json");

  model = await tf.loadGraphModel(handler);
  return model;
}

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

let leaf_model
async function loadLeafGraphModel() {
  const handler = tfn.io.fileSystem("tfjs_leaf_model_dir/model.json");

  leaf_model = await tf.loadGraphModel(handler);
}

idx_map_leaf = {
  0 : "is_leaf",
  1 : "is_not_leaf"
}

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
// // for testing model
// predict_leaf("/Users/yahyaahmedkhan/Desktop/dev/UniProjects/plants-ai-backend/8b_strawberry_leaf_scorch_strang_uk_09hort002js309.jpg").then(
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

      // Delete the uploaded file
      fs.unlinkSync(filePath);

  }).catch(err=>{
    console.log(err);
    res.status(500).json({
      message: "Error processing file"
    });

      // Delete the uploaded file
      fs.unlinkSync(filePath);

  })
});

module.exports = router;
