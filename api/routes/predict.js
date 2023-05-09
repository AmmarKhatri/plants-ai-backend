const express = require("express");
const router = express.Router();
const multer = require('multer');
const fs = require("fs");
const upload = multer({ dest: 'uploads/' });
const {predict_leaf} = require("../utils/functions/predictLeaf")
const {predict_image} = require("../utils/functions/predictImage")

router.get("/", (req, res, next) => {
  // testing if taking requests
  res.status(200).json({
    res: "server is receiving requests",
  });
});

router.post("/", upload.single('file'), async (req, res, next) => {
  
  // loading the file path
  const file = req.file;
  const filePath = file.path;

  // calling predict leaf function to confirm if it is a leaf
  predict_leaf(filePath).then(isleaf => {
    if (isleaf[0] === "is_leaf"){

      // it is a leaf, we proceed to classify it
      predict_image(filePath).then(result =>{
        res.status(201).json({
          type: result[0],
          accuracy: result[1]
        });
          // Delete the uploaded file
          fs.unlinkSync(filePath);
    
      })
    } else {

      // if not a leaf, we proceed to prompt that it is not a leaf
      res.status(201).json({
        type: isleaf[0],
        accuracy: isleaf[1]
      });
      // Delete the uploaded file
      fs.unlinkSync(filePath);
    }
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
