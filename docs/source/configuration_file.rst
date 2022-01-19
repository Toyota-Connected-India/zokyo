Zokyo Configuration file
=========================

The Builder class instance requires a config file to be passed to it based on which augmentation operations are performed on the given data.

The following is the JSON config file's parameters:

.. code-block:: python

    {
        "input_dir" : "input", # input directory
        "output_dir" : "output", # output directory
        "annotation_dir" : "annotation", # ground truth annotations directory (only Pascal VOC format for now)
        "annotation_format" : "pascal_voc", # annotation format 
        "mask_dir" : "masks", # segmentation masks directory 
        "sample" : 1000, # number of output samples required
        "debug": true, # set to true to enable logging
        "multi_threaded" : false, # Multi threading (TODO)
        "shuffle": false, # set to true to shuffle the data
        "batch_ingestion": false, # set to true to turn on batch ingestion to have internal batch size
        "internal_batch": 4, # internal batch size
        "save_annotation_mask" : false, # set to true to save output anotation masks
        "operations":[
            {
                "operation": OperationName,# operation name
                "operation_module" : "zokyo.augmentation", # module of that operation (use "zokyo.augmentation". You can write your own module which takes ZokyoData instance as input.)
                "args": { # arguments required by that operation
                    "probability": 0.5, # probability of applying that operation
                    "is_mask" : true, # set to true to apply augmentation operation to specific mask class label
                    "mask_label" : 2, # that specific mask label
                    "is_annotation" : true, # set to true to apply augmentation operation to specific annotation class label
                    "annotation_label" : 1 # that specific annotation label
                    # other args specific to that operation
                }
            },

            # other operations
        ]
    }
