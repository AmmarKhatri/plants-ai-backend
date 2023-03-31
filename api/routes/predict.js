const express = require("express");
const router = express.Router();

const tf = require("@tensorflow/tfjs");
const fs = require("fs");
const tfHub = require("@tensorflow/tfjs-converter");
const tfn = require("@tensorflow/tfjs-node");

tf.serialization.registerClass(tfHub.KerasLayer);

async function loadModel() {
  const hubUrl =
    "https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v2_050_224/classification/3/default/1";
  await tf.loadGraphModel(hubUrl, { fromTFHub: true });
  tf.engine().registerBackend(
    "tfjs",
    () => new tf.WebGLBackend(tf.getBackend())
  );
  tf.setBackend("tfjs");

  const handler = tfn.io.fileSystem(
    "/Users/yahyaahmedkhan/Desktop/dev/UniProjects/plants-ai-backend/tfjs_model/model.json"
  );

  const customLayers = { KerasLayer: tfHub.KerasLayer }; // uncomment and define customLayers

  const model = await tf.loadLayersModel(handler, {
    customObjects: customLayers,
  });

  const img = fs.readFileSync(
    "file:///Users/yahyaahmedkhan/Desktop/dev/UniProjects/plants-ai-backend/test_pics/images-3.jpeg"
  );

  const tensor = tf.node.decodeImage(img);

  const resized = tf.image.resizeBilinear(tensor, [224, 224]);
  const casted = resized.cast("float32");
  const expanded = casted.expandDims(0);
  const normalized = expanded.div(255.0);

  const prediction = model.predict(normalized);

  const probs = prediction.dataSync();
  const classIndex = probs.indexOf(Math.max(...probs));
  console.log("Predicted class index:", classIndex);
}


loadModel();


router.get("/", (req, res, next) => {
  // testing if taking requests
  res.status(200).json({
    res: "server is receiving requests",
  });
});
router.post("/", (req, res, next) => {
  // put logic here
  res.status(200).json({
    res: "(anything)",
  });
});

module.exports = router;
