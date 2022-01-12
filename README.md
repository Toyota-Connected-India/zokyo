# zokyo


*CV library for image data augmentation*

![Zokyo](https://github.com/toyotaconnected-India/zokyo/workflows/Zokyo/badge.svg?branch=master)

```

                                                                                                
                                                                                                
                                  hhhhhhh              iiii                                     
                                  h:::::h             i::::i                                    
                                  h:::::h              iiii                                     
                                  h:::::h                                                       
    ssssssssss  ppppp   ppppppppp  h::::h hhhhh      iiiiiinnnn  nnnnnnnn   xxxxxxx      xxxxxxx
  ss::::::::::s p::::ppp:::::::::p h::::hh:::::hhh   i:::::n:::nn::::::::nn  x:::::x    x:::::x 
ss:::::::::::::sp:::::::::::::::::ph::::::::::::::hh  i::::n::::::::::::::nn  x:::::x  x:::::x  
s::::::ssss:::::pp::::::ppppp::::::h:::::::hhh::::::h i::::nn:::::::::::::::n  x:::::xx:::::x   
 s:::::s  ssssss p:::::p     p:::::h::::::h   h::::::hi::::i n:::::nnnn:::::n   x::::::::::x    
   s::::::s      p:::::p     p:::::h:::::h     h:::::hi::::i n::::n    n::::n    x::::::::x     
      s::::::s   p:::::p     p:::::h:::::h     h:::::hi::::i n::::n    n::::n    x::::::::x     
ssssss   s:::::s p:::::p    p::::::h:::::h     h:::::hi::::i n::::n    n::::n   x::::::::::x    
s:::::ssss::::::sp:::::ppppp:::::::h:::::h     h:::::i::::::in::::n    n::::n  x:::::xx:::::x   
s::::::::::::::s p::::::::::::::::ph:::::h     h:::::i::::::in::::n    n::::n x:::::x  x:::::x  
 s:::::::::::ss  p::::::::::::::pp h:::::h     h:::::i::::::in::::n    n::::nx:::::x    x:::::x 
  sssssssssss    p::::::pppppppp   hhhhhhh     hhhhhhiiiiiiiinnnnnn    nnnnnxxxxxxx      xxxxxxx
                 p:::::p                                                                        
                 p:::::p                                                                        
                p:::::::p                                                                       
                p:::::::p                                                                       
                p:::::::p                                                                       
                ppppppppp                                                                       
                                                                                                

```

A zokyo (/ˈsfɪŋks/ SFINGKS, Ancient Greek: σφίγξ [spʰíŋks], Boeotian: φίξ [pʰíːks], plural zokyoes or sphinges) is a mythical creature with the head of a human, a falcon, a cat, or a sheep and the body of a lion with the wings of an eagle.

Zokyo is a CV library for image data augmentation specifically built
for Toyota and Lexus in unit camera streams on top of Augmentor.

## Installing locally

With your `venv` activated:

```bash
$ python setup.py install
```

### Running tests

From your activated `.venv` run:

```bash
$ make test
```

## usage

Sample Configuration json for zokyo

```
{
            "input_dir" : "images",
            "output_dir" : "output",
            "annotation_dir" : "annotations",
            "annotation_format" : "pascal_voc",
            "mask_dir" : "mask"
            "sample" : 5000,
            "multi_threaded" : true,
            "run_all" : false,
            "batch_ingestion": true,
            "internal_batch": 20,
            "save_annotation_mask" : false,
            "operations":[
                {
                    "operation": "DarkenScene",
                    "operation_module" : "zokyo.augmentation",
                    "args": {
                        "probability": 0.7,
                        "darkness" : 0.5,
                        "is_mask" : true,
                        "mask_label" : 2,
                        "is_annotation" : true,
                        "annotation_label : 1
                    }
                },
                {
                    "operation": "Equalize",
                    "operation_module" : "zokyo.augmentation",
                    "args": {
                        "probability": 0.5,
                        "is_mask" : true,
                        "label" : 2
                    }
                },
                {
                    "operation": "RadialLensDistortion",
                    "operation_module" : "zokyo.augmentation",
                    "args": {
                        "probability": 0.5,
                        "is_annotation" : true,
                        "distortiontype" : "NegativeBarrel",
                        "is_mask" : true,
                    }
                }
            ]
        }
```

```
from zokyo.augmentation import Builder
pipeline = Builder(config_json="config.json")
pipeline.process_and_save()
```

