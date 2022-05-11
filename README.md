# zokyo


*CV library for image data augmentation*

![Zokyo](https://github.com/toyotaconnected-India/zokyo/workflows/Zokyo/badge.svg?branch=master)

```

                                                                                             
                                                                                             
                                  kkkkkkkk                                                   
                                  k::::::k                                                   
                                  k::::::k                                                   
                                  k::::::k                                                   
zzzzzzzzzzzzzzzzz   ooooooooooo    k:::::k    kkkkkkkyyyyyyy           yyyyyyy ooooooooooo   
z:::::::::::::::z oo:::::::::::oo  k:::::k   k:::::k  y:::::y         y:::::yoo:::::::::::oo 
z::::::::::::::z o:::::::::::::::o k:::::k  k:::::k    y:::::y       y:::::yo:::::::::::::::o
zzzzzzzz::::::z  o:::::ooooo:::::o k:::::k k:::::k      y:::::y     y:::::y o:::::ooooo:::::o
      z::::::z   o::::o     o::::o k::::::k:::::k        y:::::y   y:::::y  o::::o     o::::o
     z::::::z    o::::o     o::::o k:::::::::::k          y:::::y y:::::y   o::::o     o::::o
    z::::::z     o::::o     o::::o k:::::::::::k           y:::::y:::::y    o::::o     o::::o
   z::::::z      o::::o     o::::o k::::::k:::::k           y:::::::::y     o::::o     o::::o
  z::::::zzzzzzzzo:::::ooooo:::::ok::::::k k:::::k           y:::::::y      o:::::ooooo:::::o
 z::::::::::::::zo:::::::::::::::ok::::::k  k:::::k           y:::::y       o:::::::::::::::o
z:::::::::::::::z oo:::::::::::oo k::::::k   k:::::k         y:::::y         oo:::::::::::oo 
zzzzzzzzzzzzzzzzz   ooooooooooo   kkkkkkkk    kkkkkkk       y:::::y            ooooooooooo   
                                                           y:::::y                           
                                                          y:::::y                            
                                                         y:::::y                             
                                                        y:::::y                              
                                                       yyyyyyy                               
                                                                                                                                                                 

```

Zokyo is a no-nonsense low-code computer vision augmentation library specifically built for automotive deep learning development which is
easy to integrate with your MLOps pipelines. 

## Installing from source

With your `venv` activated:

```bash
$ make install
```

### Running tests

From your activated `venv` run:

```bash
$ make test
```

## Generate documentation

After done adding a new module with necessary docstrings, make sure to run the following command to generate sphinx documentation.

```bash
$ make docs
```

## Usage

A Computer Vision or ML engineer can try out different operations to finalize a configuration file for augmenting their images. A sample Configuration json file for Zokyo looks like this

```
{
        "input_dir" : "images", # input directory
        "output_dir" : "output", # output directory
        "annotation_dir" : "annotations", # ground truth annotations directory (Pascal VOC format)
        "annotation_format" : "pascal_voc", # annotation format 
        "mask_dir" : "mask" # segmentation masks directory 
        "sample" : 5000, # number of output samples required
        "multi_threaded" : true, # TODO
        "batch_ingestion": true, # set to true to turn on batch ingestion to have internal batch size
        "internal_batch": 20, # internal batch size
        "save_annotation_mask" : false, # set to true to save output anotation masks
        "operations":[
            {
                "operation": "DarkenScene", # operation name
                "operation_module" : "zokyo.augmentation", # module of that operation, refer examples on how to write custom ops
                "args": { # arguments required by that operation
                    "probability": 0.7, # probability of applying that operation
                    "darkness" : 0.5, # argument specific to the augmentation operation
                    "is_mask" : true, # apply augmentation operation to specific mask class label
                    "mask_label" : 2, # that specific mask label
                    "is_annotation" : true, # apply augmentation operation to specific annotation class label
                    "annotation_label : 1 # that specific annotation label
                }
            }
            ...
        ]
}
```


The above created config can then be used by a DevOps engineer to load and augment the data with the following 3 lines of code. 

```
from zokyo.augmentation import Builder
pipeline = Builder(config_json="config.json")
pipeline.process_and_save()
```

For more usage tutorials, take a look at the notebooks in the [examples folder](/examples).

**Note:** Zokyo currently supports only Pascal VOC format. To convert other annotation formats to Pascal VOC see [this](/zokyo/utils/data_format_conversions.py).
