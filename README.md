# IntegratedDualAnalysisAproach_MDA

This project includes an integrated dual analysis approach for high-dimensional quantitative and qualitative data. 
The original idea is based on the dual analysis approach developed by Turkay et al. [1]. By treating both, dimensions and items, 
as first order objects, users can obtain an overview about the data and identify possible correlations by investigating changes 
among descriptive statistics after subset selection.  

[1] &nbsp; C. Turkay, P. Filzmoser, and H. Hauser, “Brushing Dimensions - A Dual Visual Analysis Model for High-Dimensional Data, ”IEEE Transactionson Visualization and Computer Graphics, vol. 17, no. 12, pp. 2591–2599,2011.

## Abstract
The Dual Analysis framework is a powerful enabling technology for the exploration of high dimensional quantitative data by treating data dimensions as first-class objects that can be explored in tandem with data values. In this work, we extend the Dual Analysis framework through the ***joint*** treatment of quantitative (numerical) and qualitative (categorical) dimensions. 
Computing common measures for all dimensions allows us to visualize both quantitative and qualitative dimensions in the same view. \revision{This enables} a natural ***joint*** treatment of mixed data during interactive visual exploration and analysis. Several measures of variation for nominal qualitative data can also be applied to ordinal qualitative and quantitative data. For example, instead of measuring variability from a mean or median, \revision{other} measures assess inter-data variation or average variation from a mode. In this work, we demonstrate how these measures can be integrated into the Dual Analysis framework to explore and generate hypotheses about high-dimensional mixed data. A medical case study using clinical routine data of patients suffering from Cerebral Small Vessel Disease~(CSVD), conducted with a senior neurologist and a medical student, shows that a joint Dual Analysis approach for quantitative and qualitative data can rapidly lead to new insights based on which new hypotheses may be generated. 

## How to run it
- download repository from github
- open project in development environment, such as IntelliJ idea
- right click on **app_MDA.py** and run it
- navigate to */templates/index.html* and select run

Now, the project is running.


## Modification
- new data table: 
    - add csv to */resources*
    - in **app_MDA.py** line 43 specify path to your csv file
