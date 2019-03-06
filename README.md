## Code Test for Software Engineers interested in NLP

This is a code test for Software Engineers at Quantworks interested in NLP. Consider this an assessment rather than a graded exam. This test helps us gauge a candidate's level of familiarity and comfort with technologies and techniques.

### Project Description

This exercise provides data from the [National Transportation Safety Board](http://www.ntsb.gov/Pages/default.aspx)'s [database of aviation accidents](http://www.ntsb.gov/_layouts/ntsb.aviation/index.aspx). We'll ask you to perform some routine high-level analytic tasks with the data.

----

### Some guidance

1. Use open source tools, such as Python, R, or Java. Do not use proprietary tools, such as SAS, SPSS, JMP, Tableau, or Stata.
2. Fork this repository to your personal GitHub account and clone the fork to your computer.
3. Use NLP processing techniques you have familiarity with to perform analysis. We will discuss your approach and understanding of NLP concepts together.
4. Save and commit your answers to your fork of the repository, and push them back to your personal GitHub account. You can then provide a link to that fork of the repository if you need to show a code example.
5. Use the Internet as a resource to help you complete your work. We do it all the time.
6. Comment your code so that when you look back at it in a year, you'll remember what you were doing.
7. There are many ways to approach and solve the problems presented in this exercise.
8. Have fun!

----

### The Data

There are 145 files in this repository:

- `AviationData.xml`: This is a straight export of the database provided by the NTSB. The data has not been altered. It was retrieved by clicking "Download All (XML)" from [this page on the NTSB site.](http://www.ntsb.gov/_layouts/ntsb.aviation/index.aspx)

There are 144 files in the following format:

- `NarrativeData_xxx.json`: These files were created by taking the `EventId`s from `AviationData.xml` and collecting two additional pieces of data for each event:
  - `narrative`: This is the written narrative of an incident.
  - `probable_cause`: If the full narrative includes a "probable cause" statement, it is in this field.

----

### The Task

**Explore the data and prepare a 1/2 to 1 page written overview of your findings.**
**Submit all your work (preprocessing/processing/analysis) in well documented code.**
**Submit at least one visualization that supports your analysis and findings.**

_Additional Context:_

* Assume the audience for your write-up is a non-technical stakeholder.
* Assume the audience for your code is a colleague who may need to read or modify it in the future.
* Please document your code as appropriate

#### Possible approaches

This data is ripe for exploration. Working with the files provided, here are a few structured tasks that you can attempt. Note: these are suggested data exploration tasks. Feel free to complete none, some, or all of these steps as you explore the data.

This is a sample approach. Please feel free to come up with your own process.
1. Consume the data from files into datastructure/database you wish to work with
2. Determine what fields you have available for each incident record in the structured `AviationData.xml`
3. Perform initial exploratory analysis of the narrative text. What, if anything, can you learn from analyzing the text that you cannot learn from the structured data fields?
4. Perform initial exploratory analysis of the narrative text and determine an analytical question you would like to answer with this data.
5. Perform common preprocessing tasks ( removal of common stopwords, parsing, etc) as required by the analytic question you would like to explore. (i.e. what are the most frequent terms in the narrative text? Do the word frequencies change over time?)
6. Choose a text representation method for your analysis
7. Use topic modeling to cluster the incidents based on the narrative text and/or probable cause descriptions. How do the clusters differ from those created with the structured data fields?
8. Imagine you have a client with a fear of flying. That client wants to know what types of flights present the most risk of an incident. Use your findings from the earlier tasks to answer the client's question in a report no longer than 1 page.
