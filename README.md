# DA-MM
This is the transfer learning library for the following paper:
Deep transfer learning method based on automatic Domain Alignment and Moment Matching
# Deep Transfer Learning on Caffe

This is a caffe library for deep transfer learning. We fork the repository with version ID `29cdee7` from [Caffe](https://github.com/BVLC/caffe), [Xlearn](https://github.com/thuml/Xlearn),[Autodial](https://github.com/ducksoup/autodial), [B-JMMD](https://github.com/mengchuangji/balanced-joint-maximum-mean-discrepancy)and make our modifications. The main modifications is listed as follow:
- Add `mmd layer` described in paper "Learning Transferable Features with Deep Adaptation Networks" (ICML '15).
- Add `jmmd` layer` described in paper "Deep Transfer Learning with Joint Adaptation Networks" (ICML '17).
- Add `entropy layer` and outerproduct layer described in paper "Unsupervised Domain Adaptation with Residual Transfer Networks" (NIPS '16).
- Add `DialLayer`: implements the AutoDIAL layer described in paper "AutoDIAL: Automatic DomaIn Alignment Layers" (ICCV '17).
- Add ` EntropyLossLayer`: a simple entropy loss implementation with integrated softmax computation described in paper "AutoDIAL: Automatic DomaIn Alignment Layers" (ICCV '17)..
- Add `bjmmd layer` described in paper "Balanced joint maximum mean discrepancy for deep transfer learning" (AA '2020).


Data Preparation
---------------
In `data/office/*.txt`, we give the lists of three domains in [Office](https://cs.stanford.edu/~jhoffman/domainadapt/#datasets_code) dataset.

We have published the Image-Clef dataset we use [here](https://drive.google.com/file/d/0B9kJH0-rJ2uRS3JILThaQXJhQlk/view?usp=sharing).

Training Model
---------------

In `\models\autodial`, we give an example model based on different networks to show how to transfer from `amazon` to `webcam`. 
The [bvlc\_reference\_caffenet](http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel) is used as the pre-trained model for Alexnet. The [deep-residual-networks](https://github.com/KaimingHe/deep-residual-networks) is used as the pre-trained model for Resnet. We use Resnet-50. The[bvlc_googlenet.caffemodel](https://github.com/AleDel/deepdreamer-touchdesigner/blob/master/models/bvlc_googlenet.caffemodel) is used as the pre-trained model for Inception.
If the Office dataset and pre-trained caffemodel are prepared, the example can be run with the following command:

```
Auto+MMD： Examples of different network implementations

For Alexnet:
"TOOLS=./build/tools
LOG=models/autodial/alexnet/mmd-auto/office-caltech1/AC/logs-auto-AC-0.1-626-64.log
$TOOLS/caffe train \
--solver=models/autodial/alexnet/mmd-auto/office-caltech1/AC/solver.prototxt -weights models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel -gpu all 2>&1 | tee $LOG"

For Inception:
"TOOLS=./build/tools
LOG=models/autodial/inception/mmd-auto/office31/logs-aw-autoMMD-google-test-60000-703-1.0-0.2.log
$TOOLS/caffe train \
--solver=models/autodial/inception/mmd-auto/office31/solver.prototxt -weights models/bvlc_googlenet/bvlc_googlenet.caffemodel -gpu all 2>&1 | tee $LOG" 
```

```
The commands of Auto+JMMD and Auto+BJMMD are similar to those of Auto+MMD
```



Changing Transfer Task
---------------
If you want to change to other transfer tasks (e.g. `webcam` to `amazon`), you may need to:

- In `train_val.prototxt` please change the source and target datasets;
- In `solver.prototxt` please change `test_iter` to the size of the target dataset: `2817` for `amazon`, `795` for `webcam` and `498` for `dslr`;


## Citation
If you use this code for your research, please consider citing:
```
    @article{Zhang2022Deep,
        title={Deep transfer learning method based on automatic domain alignment and moment matching},
        author={Jingui Zhang and Chuangji Meng and Cunlu Xua* and Jingyong Ma and Wei Su},
        journal={},
        number={},
        year={},
    }
        
```

```
    @article{Chuangji2020Balanced,
        title={Balanced Joint Maximum Mean Discrepancy for Deep Transfer Learning},
        author={Chuangji Meng and Cunlu Xu and Qin Lei and Wei Su and Jinzhao Wu},
        journal={Analysis and Applications},
        number={2},
        year={2020},
    }
        
```

## Contact
If you have any problem about our code, feel free to contact 
- zhangjingui@lzu.edu.cn
- 2279767412@qq.com
- clxu@lzu.edu.cn

or describe your problem in Issues.
