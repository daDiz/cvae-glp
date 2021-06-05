This is the code accompanied with the paper:

Sha, H., Al Hasan, M., & Mohler, G. (2021). Group Link Prediction Using Conditional Variational Autoencoder. Proceedings of the International AAAI Conference on Web and Social Media, 15(1), 656-667. Retrieved from https://ojs.aaai.org/index.php/ICWSM/article/view/18092


Please cite the paper if you use the code. 

## prerequisites

1. move the datasets folder to this folder

2. python-2.7 and Tensorflow < 2.0, see requirements.txt (e.g. pip install -r requirements.txt)

Note: This code is tested in python 2.7 on a linux machine.

## cvae

### member-recommendation

1. To prepare data:

	python preprocess.py

2. To train the model:

	python train.py -zd 32 -hd 128 -lr 0.0001 > train.log

3. To perform prediction on the test set:

	python test.py -zd 32 -hd 128 -lr 0.0001 > test.log

4. To do a grid search for hyper-parameters:

	4.1 train on a series of hyper-parameters:

		./gs.sh > gs.log

	4.2 determine the best set of hyper-parameters (using hit@10):
		
		python find_best.py 10

### group-recommendation

1. To prepare data:

	python preprocess.py

2. To train:

	python train.py -zd 32 -hd 128 -lr 0.0001 > train.log

3. To do validation:

	python valid.py -zd 32 -hd 128 -lr 0.0001 > valid.log

4. To test:

	python test.py -zd 32 -hd 128 -lr 0.0001 > test.log

5. To do grid search:

	5.1 train: ./gs.sh > gs.log
	5.2 valid: ./gs_valid.sh > gs_valid.sh
	5.3 determine the best hyper-parameters (w/ hit@10): python find_best.py 10

6. case study:

	python cs.py -zd 32 -hd 128 -lr 0.0001 > cs.log
	cd case_study
	count_pb.py --> determine a threshold
	who_are_they.py --> count identities
	count_tiles.py --> count job titles
 


## cvaeh

### member-recommendation

1. To prepare data:

	python preprocess.py 2

	Note: 2 is the historical window size. 

2. To train the model

	python train.py -ts 2 -zd 32 -hd 128 -lr 0.0001 > train.log

3. To perform prediction on the test set:

	python test.py -ts 2 -zd 32 -hd 128 -lr 0.0001 > test.log

4. To do a grid search for hyper-parameters:

	4.1 train on a series of hyper-parameters:

		./gs.sh > gs.log

	4.2 determine the best set of hyper-parameters (using hit@10):

		python find_best.py 10

### group-recommendation

1. To prepare data:

	python preprocess.py 2
	
	Note: 2 is time window size for hitorical event counts

2. To train:

	python train.py -ts 2 -zd 32 -hd 128 -lr 0.0001 > train.log

3. To do validation:

	python valid.py -ts 2 -zd 32 -hd 128 -lr 0.0001 > valid.log

4. To test:

	python test.py -ts 2 -zd 32 -hd 128 -lr 0.0001 > test.log

5. To do grid search:

	5.1 train: ./gs.sh > gs.log
	5.2 valid: ./gs_valid.sh > gs_valid.sh
	5.3 determine the best hyper-parameters (w/ hit@10): python find_best.py 10



## baselines

### graph based

#### common neighbors

	##### member-recommendation
	train-valid:	python cn_faster.py -f valid -m sum > cn_faster_valid.log
	test:		python cn_faster.py -f test -m sum > cn_faster_test.log
	grid search:	python find_best.py 10

	Note: -m = sum, max, min

	##### group-recommendation
	train-valid:	python cn_group.py -f valid -m mean > cn_group_valid.log
	test:		python cn_group.py -f test -m mean > cn_group_test.log
	grid search:	python find_best.py valid

	Note: -m = mean, sum, max, min


#### jaccard index

	##### member-recommendation
	train-valid:	python jc_faster.py -f valid -m sum > jc_faster_valid.log
	test:		python jc_faster.py -f test -m sum > jc_faster_test.log
	grid search:	python find_best.py 10

	Node: -m = sum, max, min

	##### group-recommendation
	train-valid:	python jc_group.py -f valid -m mean > jc_group_valid.log
	test:		python jc_group.py -f test -m mean > jc_group_test.log
	grid search:	python find_best.py valid

	Note: -m = mean, sum, max, min



#### preferential attachment

	##### member-recommendation
	train-valid:	python pa_faster.py -f valid -m sum > pa_faster_valid.log
	test:		python pa_faster.py -f test -m sum > pa_faster_test.log
	grid search:	python find_best.py 10

	Node: -m = sum, max, min

	##### group-recommendation
	train-valid:	python pa_group.py -f valid -m mean > pa_group_valid.log
	test:		python pa_group.py -f test -m mean > pa_group_test.log
	grid search:	python find_best.py valid

	Note: -m = mean, sum, max, min


#### adar

	##### member-recommendation
	train-valid:	python adar_faster.py -f valid -m sum > adar_faster_valid.log
	test:		python adar_faster.py -f test -m sum > adar_faster_test.log
	grid search:	python find_best.py 10

	Node: -m = sum, max, min

	##### group-recommendation
	train-valid:	python adar_group.py -f valid -m mean > adar_group_valid.log
	test:		python adar_group.py -f test -m mean > adar_group_test.log
	grid search:	python find_best.py valid

	Note: -m = mean, sum, max, min


#### katz_beta

	##### member-recommendation
	train-valid:	python katz.py -f valid -l 1 -b 0.1 > katz_valid.log
	test:		python katz.py -f test -l 1 -b 0.1 > katz_test.log
	grid search:	./gs.sh > gs.log
			python find_best.py 10

	##### group-recommendation
	train-valid:	python katz_group.py -f valid -l 1 -b 0.1 -m sum > katz_group_valid.log
	test:		python katz_group.py -f test -l 1 -b 0.1 -m sum > katz_group_test.log
	grid search:	./gs.sh > gs.log
			python find_best.py 10

	Note: -m = mean, sum, max, min



### embedding based

#### node2vec

	##### member-recommendation
	prepare data:	use node2vec to generate the embedding from the graph (as an example, we provide the embedding for enron in ./emb/enron)
	train-valid:	python n2v.py -f valid -d 32 -p 0.5 -q 0.5 -m sum > n2v_valid.log
	test:		python n2v.py -f test -d 32 -p 0.5 -q 0.5 -m sum > n2v_test.log
	grid search:	./gs.sh > gs.log
			python find_best.py 10

	##### group-recommendation
	prepare data:	use node2vec to generate the embedding from the graph (as an example, we provide the embedding for enron in ./emb/enron)
	train-valid:	python n2v.py -f valid -d 32 -p 0.5 -q 0.5 -m sum > n2v_valid.log
	test:		python n2v.py -f test -d 32 -p 0.5 -q 0.5 -m sum > n2v_test.log
	grid search:	./gs.sh > gs.log
			python find_best.py valid


### matrix factorization based

#### nmf
	
	##### member-recommendation
	prepare data:	python preprocess.py
	train:		python do_nmf.py -ncp 64 -alpha 0.1 -l1 0.1 > do_nmf.log
	validation:	python predict_H.py -s valid -m cos > hit_valid.txt
	test:		python predict_H.py -s test -m cos > hit_test.txt	
	grid search:	./gs.sh > gs.log
			python find_best.py 10

	##### group-recommendation
	prepare data:	python preprocess.py
	train:		python do_nmf.py -ncp 16 -alpha 0.1 -l1 0.1 > do_nmf.log
	validation:	python predict_hw.py -s valid -ncp 16 -alpha 0.1 -l1 0.1 > predict_valid.log
	test:		python predict_hw.py -s test -ncp 16 -alpha 0.1 -l1 0.1 > predict_test.log	
	grid search:	./gs.sh > gs.log
			python find_best.py 10

#### svd

	##### member-recommendation
	prepare data:	python preprocess.py
	train:		python do_svd.py -ncp 32 > do_svd.log
	validation:	python predict_H.py -s valid -m cos > hit_valid.txt
	test:		python predict_H.py -s test -m cos > hit_test.txt	
	grid search:	./gs.sh > gs.log
			python find_best.py 10

	##### group-recommendation
	prepare data:	python preprocess.py
	train:		python do_svd.py -ncp 16 > do_svd.log
	validation:	python predict_hw.py -s valid -ncp 16 > predict_valid.log
	test:		python predict_hw.py -s test -ncp 16 > predict_test.log	
	grid search:	./gs.sh > gs.log
			python find_best.py 10


### neural network methods

#### one-hot mlp

	##### member-recommendation
	prepare data:	python preprocess.py
	train:		python train.py -hd 64 -lr 0.0001 > train.log
	test:		python test.py -hd 64 -lr 0.0001 > test.log
	grid search:	./gs.sh > gs.log
			python find_best.py 10

	##### group-recommendation
	prepare data:	python preprocess.py
	train:		python train.py -hd 64 -lr 0.0001 > train.log
	test:		python test.py -hd 64 -lr 0.0001 > test.log
	grid search:	./gs.sh > gs.log
			python find_best.py 10


#### set2vec mlp

	##### member-recommendation
	prepare data:	python preprocess.py
	train:		python train.py -ed 32 -hd 128 -lr 0.001 -ts 3 > train.log
	test:		python test.py -ed 32 -hd 128 -lr 0.001 -ts 3 > test.log
	grid search:	./gs.sh > gs.log
			python find_best.py 10

	##### group-recommendation
	prepare data:	python preprocess.py
	train:		python train.py -ed 32 -hd 128 -lr 0.001 -ts 1 > train.log
	validation:	python valid.py -ed 32 -hd 128 -lr 0.001 -ts 1 > valid.log
	test:		python test.py -ed 32 -hd 128 -lr 0.001 -ts 1 > test.log
	grid search:	./gs.sh > gs.log
			./gs_valid.sh > gs_valid.log
			python find_best.py 10


#### set2vec bpr

	##### member-recommendation
	prepare data:	python preprocess.py
	train:		python train.py -ts 5 -hd 128 -lr 0.0001 > train.log
	test:		python test.py -ts 5 -hd 128 -lr 0.0001 > test.log
	grid search:	./gs.sh > gs.log
			python find_best.py 10

	##### group-recommendation
	prepare data:	python preprocess.py
	train:		python train.py -ts 5 -hd 128 -lr 0.0001 > train.log
	validation:	python pred_group.py -ts 5 -hd 128 -lr 0.0001 > pred_group.log
	test:		python pred_group_test.py -ts 5 -hd 128 -lr 0.0001 > pred_group_test.log
	grid search:	./gs.sh > gs.log
			./gs_valid.sh > gs_valid.log
			python find_best.py 10


#### lstm

	##### member-recommendation
	prepare data:	python preprocess.py -ts 2
	train:		python train.py -ts 2 -hd 128 -lr 1.0 > train.log
	validation:	python predict_valid.py -ts 2 -hd 128 -lr 1.0 > predict_valid.log
	test:		python predict_test.py -ts 2 -hd 128 -lr 1.0 > predict_test.log
	grid search:	./gs.sh > gs.log
			python find_best.py 10
