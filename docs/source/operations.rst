Zokyo Augmentation Operations
===============================

The following are the operations that are currently supported inside Zokyo. Any extra operations needed can be raised a feature request or can be added as a separate module in "operation_module" in config.json.

#. **EqualizeScene:** Operation to equalize an image or specific annotation classes in that image. 
#. **DarkenScene:** Operation to darken an image or specific annotation classes in that image. Requires a darkness parameter [0, 1] in config.
#. **BrightenScene:** Operation to brighten an image or specific annotation classes in that image. Requires a brightness parameter [0, 1] in config.
#. **RandomBrightness:** Operation to randomly brighten an image or specific annotation classes in that image. Requires a distribution parameter (normal or uniform) in config.
#. **SnowScene:** Operation to add snow effect to an image or specific annotation classes in that image.
#. **RadialLensDistortion:** Operation to add radial distortion effect to an image or specific annotation classes in that image. Requires a distortion parameter (NegativeBarrel, PinCushion).
#. **TangentialLensDistortion:** Operation to add tangential distortion effect to an image or specific annotation classes in that image.
#. **RainScene:** Operation to add rain effect to an image or specific annotation classes in that image. Requires -
   * rain_type (drizzle, heavy, torrential)
   * drop_width [1, 5]
   * drop_length [0, 100]
   * brightness_coefficient [0, 1]
   * slant_lower, slant_upper (-20 <= slant_lower <= slant_upper <= 20)
   * drop_color (list of 3 pixel values)
#. **SunFlare:** Operation to add sun flare effect to an image. 
#. **MotionBlur:** TODO
#. **FogScene:** TODO

