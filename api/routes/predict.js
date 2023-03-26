const express = require('express');
const router = express.Router();
// load model here then use it in request to:
//1. get images from req
//2. input in model
//3. send response in json format about the status of the leaf

router.get('/', (req, res, next) => {
    // testing if taking requests
    res.status(200).json({
        res: "server is receiving requests"
    })
})
router.post('/', (req, res, next) => {
    // put logic here
    res.status(200).json({
        res: "(anything)"
    })
})

module.exports = router;