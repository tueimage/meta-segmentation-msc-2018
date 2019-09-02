# First week of your thesis project

*	description of problem

When a new medical imaging problem is given the whole process of designing a suitable neural network is done from the ground up (choosing the correct algorithm, training the model, etc.). However with previous knowledge of already solved medical imaging problems it should be possible to predict the most suitable solution, and potentially also utilizing pre-trained versions of this solution. 

* research question / approach to a solution

Don’t know the exact format that is wanted for the research question. Is it supposed to be something like this?

“How can meta learning determine which deep learning approach on medical image datasets is optimal?”

The Decathlon dataset has binary segmention maps. Therefore the first part of this project will be to create a meta learning model to determine the best semantic segmentation method for a given problem, using the data from the Decathlon Challenge. 

*	what a first prototype might be

Meta-learning model that outputs a binary prediction. I will select two top performing encoder-decoder networks for semantic segmentation. Both networks should share the encoder and differ with the decoder, to limit the amount a variables in this stage of the project. Performance of both modes on the decathlon data will be measured. Then a meta learning model will be trained to determine which decoder network works best on which dataset. Cross-validation will be used. 

*	rough schedule

Depending on when computer is available. Hope to have first prototype ready in two weeks. Depending on the challenges I find in that process I will shape the further planning. 


## Problem definition

Create an initial problem definition that captures your understanding of the thesis’ goals.

* description of problem
* research question / approach to a solution
* what a first prototype might be
* rough schedule 

Write this down in README.MD

## Literature 

Decide on where you are going to track of literature. This can be Zotero, Jabref + Evernote, Jabref + Excel, etc. 

When you read any paper, add it to your list, and write down a few notes with a summary of the paper, and how it is relevant (or not) for your thesis.
This will also help you with writing your thesis later.

Find 2-4 papers for your topic and summarize them in this system.  

## Lab notebook

Decide on where you are going to keep your lab notebook. This can be an .MD file on Github, Word, Evernote, etc. 

Consider it like a captain's log on a ship. Every day you work on the project, you write what you did, what didn't work, why you decided 
to try something else, a screenshot of the results, etc. 

The notebook is for yourself, but you can bring it to the meeting to discuss problems and results. Also the notebook will help your future
self to figure out what you did when you want to write your thesis. 


## Todo list

Decide on where you are going to track of all your tasks. This can be on Github (via issues or projects), Todoist, Trello, or a paper planner. Make a habit
of creating tasks for yourself, such that each task can be completed in a few hours. This means the task has to have a verb, and have a definition of "complete".
For example, "Summarize 5 papers on topic X". 

If a todo comes up for you during the meeting, capture it straight away - either in your system, or write it on paper and add it to your system later. 
