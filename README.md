## Some bullshit

![sample](https://github.com/wusongchao/fast-rcnn-pytorch-pdf-table/blob/master/sample.png)

This is Fast-RCNN, not Faster-RCNN, and doesn't adopt some advanced methods like ROI Align(Mask-RCNN) or PSROI(R-FCN). 
Based on PyTorch 0.4.0(When i finish this, the 1.0 version had not come up).

So why I make this wheel, use the out-of-style model/method?

This repo comes from a task: to detect table region(if possible, parse the table structure) in PDF. To adopt RCNN base(two step) method
in this specific field, there's a idea comes to my mind: How to use the prior knowledge of this specific task, that is, the tables are often
highly structured(compared with "normal" paragraph), and usually contain many lines.

So, to merge the prior knowledge, i choose to re-design the region proposal phase, thus Fast-RCNN but not Faster-RCNN became my based model.

I develop a heuristic algorithm to replace the Selective-Search method in Fast-RCNN's origin paper. Generally, this algorithm generate
region proposals based on lines by a series of hand-design rules.

However, the performance is so low. I had adjusted many parameters for many times, but hardly had some improvement. The performance is 
worse than directly fine-tume Faster-RCNN. So i guess that the method itself has some inner problem, the reasons(i guess) are given as follow:

1. The bounding box regression. This phase important in RCNN-based model, to fix the predict targets coordinates. However, in my model's experiments, the existence of b-box hardly have had some influence on the final performance(for different IOU, 0.6/0.8 in the final metrics). The table have a variety of styles, but in the dataset, they are all classified as only one class, "table". So it's hard to learn the mapping from region-proposal to the gt-box. And since the region proposals came from a line-based heuristic algorithm, it's results are also hard to predict(compared with the fix size "anchor" of Faster-RCNN).

2. The region proposal method. We know that one of the most significant metrics to measure the performance of region proposal is the recall rate. It means that the region proposals produced by this phase should contain as much gt-box as possible. However, as i mentioned before, the heuristic algorithm i developed is "line-based", but there are many samples of tables that contains no line, while others are not wrapped with lines. So the recall rate of region proposal method is lower than the method that sliding on the image/feature maps. 

3. Hand designed prior? Take Canny for example, it is a classic method to label the edge in an image. This method relies on a series of hand designed features, like many typical CV methods. But this method have had been beaten by many new methods, like HED network and more further works. As RCNN, it has been widely proven that in many CV tasks, the implicit features/rules learned by the nerual network are better than the carefully hand designed features/rules. The datasets used for training contains the datas' prior.


## Usage

I did not implement the command arg parse phase, the entries of the train/validate are both in train.py, 
so to change the start strategy of this demo, one have to directly change the code. I know this is stupid.

I did not post the whole dataset, since this is too huge. But all the dataset are available on Internet(and that's how i collect them).

There'are three kinds of dataset, the ICDAR 2013, the ICDAR 2017 and the marmot dataset.

The directory to be mounted can be found at path_constant.py
