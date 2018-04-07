# TextBoxes: A Fast Text Detector with a Single Deep Neural Network

Recommend: [TextBoxes++](https://github.com/MhLiao/TextBoxes_plusplus) is an extended work of TextBoxes, which supports oriented scene text detection. The recognition part is also included in [TextBoxes++](https://github.com/MhLiao/TextBoxes_plusplus).


### Contents
1. [Installation](#installation)
2. [Test](#test)
3. [Train](#train)

### Installation
1. Get the code. We will call the directory that you cloned Caffe into `$CAFFE_ROOT`
  ```Shell
  git clone https://github.com/MhLiao/TextBoxes.git
  
  cd TextBoxes
  
  make -j8
  
  make py

  pip install -r requirements.txt
  ```

### Test
1. run "python examples/TextBoxes/demo_server.py".
2. You can modify the "use_multi_scale" in the "examples/demo.py" script to control whether to use multi-scale or not.
3. The results are saved in the "examples/results/".


### Train
1. Train about 50k iterions on Synthetic data which refered in the paper.
2. Train about 2k iterions on corresponding training data such as ICDAR 2013 and SVT.
3. For more information, such as learning rate setting, please refer to the paper.

### Data preparation for training
The reference xml file is as following:
  
        <?xml version="1.0" encoding="utf-8"?>
        <annotation>
            <object>
                <name>text</name>
                <bndbox>
                    <xmin>158</xmin>
                    <ymin>128</ymin>
                    <xmax>411</xmax>
                    <ymax>181</ymax>
                </bndbox>
            </object>
            <object>
                <name>text</name>
                <bndbox>
                    <xmin>443</xmin>
                    <ymin>128</ymin>
                    <xmax>501</xmax>
                    <ymax>169</ymax>
                </bndbox>
            </object>
            <folder></folder>
            <filename>100.jpg</filename>
            <size>
                <width>640</width>
                <height>480</height>
                <depth>3</depth>
            </size>
        </annotation>

Please let me know if you encounter any issues.
