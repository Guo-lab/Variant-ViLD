# Description

- ## data: 
  - checkpoint, frozen, model.ckpt*3, pipeline.config from Original downloaded model
  - ssd_inception_v2_pets: changed config file
    - https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/configuring_jobs.md
  - sw_label_map: mapping file


  - sw_train.record, sw_val.record: TFRecords from create_sw_tf_record.py  


-  ### protos, utils (Later, be deleted as API has been installed)
    - from [Tensorflow](https://github.com/tensorflow/models/tree/master/research/object_detection/utils) 
    - API has object_detection
 
