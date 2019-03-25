# Weekly meetings

Copy/paste and fill in the template below each week, BEFORE coming to the weekly meeting with your supervisor. Put the newest weekly meeting first. 

# Weekly meetings

Copy/paste and fill in the template below each week, BEFORE coming to the weekly meeting with your supervisor. Put the newest weekly meeting first. 

### Date: 25-03-2019

#### Who helped you this week?

* Britt helped me analyzing some of my results

#### Who did you help this week?

* I helped with organizing parts of the community day

#### What did you achieve?

* restructuring of code so it is easier to use and adapt to specific problems
* further analysis on results from meta learning. Add new meta-features, add cross-validation
* some work on the poster for ISBI (and the research day)

#### What did you struggle with?

* finding good data visualisations

#### What would you like to work on next week?

* Deciding whether I need to improve my meta features or my regression model, and then doing that :) 

#### Where do you need help from Veronika?

* Can you think of reasons why the regression is not working (even the training), except lack of quality and consistency of the meta features


### Date: 05-02-2019

#### Who helped you this week?

* Helped Britt and Colin with the usage of Saruman
* Helped Britt with setting up the online scrum board

#### Who did you help this week?

* Ishaan helped me with some issues I had with GitHub

#### What did you achieve?

* Found a reliable set of meta-features to use. it consists of 13 features, Table 1:( https://arxiv.org/ftp/arxiv/papers/1402/1402.0570.pdf ). 
* Trained (an approximation of) the winning decathlon model on hte chaos CT data

#### What did you struggle with?

* figuring out how to extract the meta features from the decathlon data. (bv. what is the difference between class entropy and feature entropy and attribute entropy? the source paper didnt probide a lot of information)
* not having enough time during the week to reach my goals. 

#### What would you like to work on next week?

* Finalize extracting the meta features and do the actual meta learning
* Some new information about the chaos challenge was released. I want to plan the next coming weeks to ensure that our submission can be finished in time without our individual projects suffering near the submission date.  

#### Where do you need help from Veronika?

* do you think the current set of meta-features I want to use is good enough (both in size and types of features?)

#### What are the agreements after this weekly meeting?

* 

### Date: 05-02-2019

#### What did you achieve?

* analyzed (lack of) results of meta learning approach. results were a bit dissapointing, but it could be expected because I doubted the quality of the meta labels, which  originated from models that I trained myself. (only 3 models and trained for short time)
* Started changing the method metnioned above to fit the decathlon results as meta_labels. 
* discussed participation in Chaos challenge and implemented one algorithm (Unet-ConvLSTM) 
* Background reading on meta learning and data analysis

#### What did you struggle with?

* My previous meta learning method (in which I trained models to create the meta-labels) didnt work. This problem could be smaller when using the decathlon results. However I wanted to know for sure that this idea could indeed work. I might have spent too much time analyzing whether using the decathlon results could solve the problem than needed. 

#### What would you like to work on next week?

* prepare meta labels from decathlon results, and train the meta learner
* continue on chaos challenge algorithm
* discuss next steps for the chaos challenge with Ishaan/Britt/Colin. (i.e. implementation of ensemble, use of bagging, etc)

#### Where do you need help from Veronika?

* is using the decathlon results as meta labels a good approach?

### Date: 05-02-2019

#### What did you achieve?

* implement meta learning framework we discussed last week (results probably next week)
* Defined the meta learning model more precise
* Background reading on data visualisation 

#### What did you struggle with?

* In my meta learning approach I want to make predictions on datasets, not single images. 

#### What would you like to work on next week?

* Obtaining and analyzing results of meta learning method
* If results are meaningful: outline next step in approach

#### Where do you need help from Veronika?

* Some feedback on the meta learing part of the method (which I now defined in more detail) 
* Get feedback on the project so far; how can I improve myself? 


### Date: 28-01-2019

#### What did you achieve?

* Did a more directed literature review into meta feature selection 
*	Background research in visualisation and statistical analysis of results

#### What did you struggle with?

* Making the distinction between relevant and less relevant literature

#### What would you like to work on next week?

* Finalizing literature review
* obtaining the needed training data for the meta learning model

#### Where do you need help from Veronika?

* Check whether proposed method for meta learning is viable


### Date: 22-01-2019

#### What did you achieve?

* Trained all of the 10 subsets of the decathlon dataset on VGG16, ResNet50 and MobileNet based encoder-decoder networks. 
*	Built framework to insert new pre trained networks easy
* Wrote research proposal: UPDATE https://www.overleaf.com/1856385788wchfdrhrjdbq


#### What did you struggle with?

* I wasnt able to set a goal for the end of my internship. The current research proposal is probably one or two months of work, but I think I can only know what comes next once I am a bit further in my research 

#### What would you like to work on next week?

* Identifying the meta-features that are useful to me 

#### Where do you need help from Veronika?

* Feedback on research proposal 
* How can I set long term goals for this project?

### Date: 15-01-2019

#### What did you achieve?

*	Identified suitable pre-trained feature extraction models
* Built decoder network on top of pre-trained VGG16 model

#### What did you struggle with?

* It is not entirely clear if it is smart to fine-tune a pretrained network with the new data (especially for this project)
* I found a lot of good feature extractors, but I am not sure of a slection criteria

#### What would you like to work on next week?

* write a clear research proposal
 

#### Where do you need help from Veronika?

* To what extent do I need to finetune the pretrained networks and the new decoder part of the network?
* how many feature extractors to comnpare? 

### Date: 8-01-2019

#### What did you achieve?

*	analyzed the first try on meta learning with the decathlon datatset
* made a research proposal based on this first result

#### What did you struggle with?

* how to take the right conclusions from the work with the decathlon dataset
* how to asses whether a long term goal is attainable for my master project

#### What would you like to work on next week?

* adjust the research proposal based on the feedback from our meeting
* start with the first steps of the research proposal
 

#### Where do you need help from Veronika?

* Feedback on the research prososal. I dont have it finished now unfortunately. I will post it on the main page of this github tomorrow before 11AM. If more convenient I can finish it a bit earlier. 



### Date: 18-12-2018

#### What did you achieve?

*	implemented Unet and started training all datasets on it 
* interpreted first results ('decathlon/december/results_december.csv'
* uploaded literature list and todo list to github (will be done before the meeting) 

#### What did you struggle with?

* running time
* what if the unet with a set of hyperparameters works on 8/10 datasets? is that acceptable?

#### What would you like to work on next week?

* work on meta learning approach
* monitoring unet training

#### Where do you need help from Veronika?

* no.2 of "what did you struggle with"

### Date: 11-12-2018

#### What did you achieve?

*	Read the given papers from last weeks meeting 
*	implemented Unet and started training all datasets on it 
* first outline of the meta learning method

#### What did you struggle with?

* whether I should do a 3D image segmentation or view the image as a collection of seperate 2d slices
* was ill for 4 days this week so wasnt able to achieve all my goals

#### What would you like to work on next week?

* Obtaining Unet results on all datasets
* further design of meta learning approach. 

#### Where do you need help from Veronika?

* I chose to use a 2D unet instead of a 3D unet. Do you think this is a good decision?


### Date: 4-12-2018

#### What did you achieve?

*	Expanded literature review
*	Plan of approach
 

#### What did you struggle with?

* Determining how much time the first step/first prototype will take

#### What would you like to work on next week?

* Get started with first step/prototype
*	Set up the lab computer (hopefully)


#### Where do you need help from Veronika?

* How do I choose the semantic segmentation models? And what if one segmentation model just outperforms all other models for every dataset? 

### Date: 27-11-2018

#### What did you achieve?

* Better understanding of the concept of meta learning
* Initial iterature review
* Start of research proposal 

#### What did you struggle with?

* It is is hard to find a specific challenge/topic for my project
* Had to finalize work at CytoSMART, had my internship presentation and had a symposium this week which limited my time a bit

#### What would you like to work on next week?

* More extensive/directed literature review
* Write research proposal

#### Where do you need help from Veronika?

* Ideas for interesting challenges in meta learning that fit with the current research in the imaging group
* format of the research propsoal 

#### Any other topics

*

## Template
### Date: 24-09-2018


#### What did you achieve?

* Replace this text with a bullet point list of what you achieved this week.
* It's ok if your list is only one bullet point long!

#### What did you struggle with?

* Replace this text with a bullet point list of where you struggled this week.
* It's ok if your list is only one bullet point long!

#### What would you like to work on next week?

* Replace this text with a bullet point list of what you would like to work on next week.
* It's ok if your list is only one bullet point long!
* Try to estimate how long each task will take.

#### Where do you need help from Veronika?

* Replace this text with a bullet point list of what you need help with.
* It's ok if your list is only one bullet point long!
* Try to estimate how long each task will take.

#### Any other topics

This space is yours to add to as needed.


### Credit
This template is partially derived from "Whitaker Lab Project Management" by Dr. Kirstie Whitaker and the Whitaker Lab team, used under CC BY 4.0. 
