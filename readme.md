readme.md

Predict a integer image

Test data is stored in testdata directory

Dataset is the mnist standard train and test set

##Normal way

### Step1
Use train.py to train and store the two layer net model in two_layer_net.pkl
	
	$python3 train.py
	train acc, test acc | 0.10441666666666667, 0.1028
	train acc, test acc | 0.8023833333333333, 0.8073
	train acc, test acc | 0.8756666666666667, 0.8785
	train acc, test acc | 0.8978666666666667, 0.8999
	train acc, test acc | 0.9068666666666667, 0.9102
	train acc, test acc | 0.9144333333333333, 0.9148
	train acc, test acc | 0.9203166666666667, 0.9218
	train acc, test acc | 0.9235666666666666, 0.9244
	train acc, test acc | 0.92845, 0.9296
	train acc, test acc | 0.9311166666666667, 0.9317
	train acc, test acc | 0.9345833333333333, 0.9351
	train acc, test acc | 0.93695, 0.9374
	train acc, test acc | 0.94025, 0.9396
	train acc, test acc | 0.9419333333333333, 0.9396
	train acc, test acc | 0.9441833333333334, 0.9425
	train acc, test acc | 0.9457833333333333, 0.9445
	train acc, test acc | 0.9484833333333333, 0.9477

### Step2
Use two_layer_test.py to test the accuracy of the model

	$python3 two_layer_test.py
	Accuracy:0.9466

with the get_image function you can also test your own image
## CNN

###Step1
Use train_conv.py to train and store CNN net model in params.pkl

	$python3 train_conv.py
	train loss:2.300001846087919
	=== epoch:1, train acc:0.212, test acc:0.191 ===
	train loss:2.29748375539359
	train loss:2.294312634680256
	train loss:2.286364878660862
	train loss:2.2797246011860945
	train loss:2.2627870681081244
	train loss:2.250901554926544
	train loss:2.2377228652672003
	train loss:2.1955270072011173
	train loss:2.1901397325996363
	train loss:2.161590794937312
	train loss:2.0857797169219743
	train loss:2.0658089204986436
	train loss:2.0255680630766495
	train loss:1.9911901577369833
	...
finally loss can come to less than 0.001

###Step2
Use test.py to test your own image in testdata
	
	$ python3 test.py
	5
	[[ -3669.03271876  -2723.0830266   -1336.10245562  -4087.55747946
	-1673.66868792   1515.17049045    675.42958532 -11025.10787864
	609.69700127 -10413.63506436]]
