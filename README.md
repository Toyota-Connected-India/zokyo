# sphinx


*CV library for image data augmentation*

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

A sphinx (/ˈsfɪŋks/ SFINGKS, Ancient Greek: σφίγξ [spʰíŋks], Boeotian: φίξ [pʰíːks], plural sphinxes or sphinges) is a mythical creature with the head of a human, a falcon, a cat, or a sheep and the body of a lion with the wings of an eagle.

Sphinx is a CV library for image data augmentation specifically built
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
```
from sphinx.augmentation import do_augmentation
do_augmentation(dir_path=dir_path, no_of_sample=10)
```