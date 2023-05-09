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


// MAP FOR ISLEAF MODEL
idx_map_leaf = {
    0 : "is_leaf",
    1 : "is_not_leaf"
}

module.exports = {
    idx_map_leaf,
    v4_idxclass_map
}
