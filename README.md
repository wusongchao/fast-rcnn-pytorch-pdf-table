## Some bullshit

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
worse than directly fine-tume Faster-RCNN.

## Usage

I did not implement the command arg parse phase, the entries of the train/validate are both in train.py, 
so to change the start strategy of this demo, one have to directly change the code. I know this is stupid.

I did not post the whole dataset, since this is too huge. But all the dataset are available on Internet(and that's how i collect them).

There'are three kinds of dataset, the ICDAR 2013, the ICDAR 2017 and the marmot dataset.
