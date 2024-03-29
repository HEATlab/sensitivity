# Improvisational Teamwork Summer 2022 

# Sensitivity
This repository hosts a collection of programs written by team *Improvisational-Teamwork* (part of the Summer 2022 HEATlab group). This work resulted in a publication at the 33rd ICAPS conference on Automatic Planning and Scheduling titled *"Sensitivity Analysis for Dynamic Control of PSTNs with Skewed Distributions"*.  The paper is now [available](https://ojs.aaai.org/index.php/ICAPS/article/view/27183/26956).

<!--- Note that the files included in this repository need to be executed within the code environment created by team *Prob-In-Ctrl* of HEATlab, which can be accessed [here](https://github.com/HEATlab/Prob-in-Ctrl). Additionally, -->

The results for SREA and DREA can be obtained from the DREAM algorithm [source code](https://github.com/HEATlab/DREAM). The results for Min-Loss and Max-Gain can be obtained from the DCforPSTN [source code](https://github.com/HEATlab/DCforPSTN).
<!-- 
## Data Processing
The `dataset` folder hosts JSON representations of STNUs used in our research.
These STNUs were constructed from the *ROVERS* and *CAR-SHARING* PSTNs, made by [(Santana et. al. 2016)](https://www.aaai.org/ocs/index.php/ICAPS/ICAPS16/paper/view/13138) and available [here](http://groups.csail.mit.edu/mers/datasets/scheduling/), by replacing distributions with finite intervals over all contingent edges.  

Each STNU is encoded by a list of nodes and constraints between those nodes.
All 452 STNUs in the `dataset/dynamically_controllable` directory are dynamically controllable while the 169 STNUs in the `dataset/uncontrollable` directory are consistent, but not dynamically controllable.

This dataset of networks is referenced in [the paper](https://www.cs.hmc.edu/HEAT/papers/Akmal_et_al_ICAPS_2019.pdf) our team submitted to [ICAPS 2019](https://icaps19.icaps-conference.org/).

## Documentation
Files throughout the project are commented in doxygen style. -->

## Project Structure
The `stn` folder contains files describing an STN class (which is really a class for STNUs, and more generally could be easily extended to represetn PSTNs) and converting between the class and JSON representations of networks. 
The remaining programs are not organized in any particular way. Because of this, in the comments that follow we often describe as a function taking in an "STN" when it really also works with STNUs as input.

### Benchmark Variations
Two existing PSTN benchmarks are changed by replacing theirnormally-distributed  uncertainty  models  with  distributionsthat capture more natural sources of uncertainty. As a basis for our augmented benchmark, we use the DREAM bench-mark of 540 PSTNs (Abrahams et al. 2019) and the CAR-SHARING dataset of 169 PSTNs which was created by San-tana et al. (2016) and then edited by Akmal et al. (2019). Varied benchmarks can be found in the dataset folder.

### Algorithms Variations
We modify Min-Loss and Max-Gain from [Gao, Popowski, and Boerkoel](https://dblp.org/rec/conf/aaai/GaoPB20.html) to build probability-aware dispatching algorithm under dynamic controllability. Details of modified algorithms can be found on branch minloss+ and maxgain+.

### Primary Programs

#### algorithm.py
Implements conflict generating algorithms for checking if a STNU is dynamically controllable (DC).
If the network is not DC, there are functions for reporting the "conflicts" that prevent the network from achieving controllability.
##### Details
Implements the `DCDijkstra` algorithm described in [(Williams 2017)](https://www.ijcai.org/proceedings/2017/598).

#### dispatch.py
Implements a dispatch strategy and associated simulation for STNUs, based off Algorithm 2 from [(Nilsson 2014)](https://pdfs.semanticscholar.org/0313/af826f45d090a63fd5d787c92321666115c8.pdf).
##### Details
This strategy should always succeed for dynamically controllable STNUs.
It works by first leveraging a conversion to the `DC_STN` class to infer wait constraints, and then following early execution.

<!-- #### LP.py
Uses `PuLP` module to set up and solve several different LPs.
These LPs are primarily attempts at linearizing the calculation for degree of strong controllability.
The main structure of the constraints remains the same accross most of the programs, but there are many different types of objective functions used.
##### Details
This files contains 4 different LPs (Original LP has two different kinds objectives).
* *Original LP (with naive objective function)*: In this LP, we are minimizing the sum of contingent intervals shrinked.
* *Original LP*: In this LP, we are minimizing the sum of contingent intervals shrinked divided by original length of the contingent intervals.
* *Proportion LP*: In this LP, we are shrinking all contingent edges by the
same proportion, and we are minimizing the proportion shrinked.
* *Maximin LP*: In this LP, we are maximizing the minimum amount we can shrink from a contingent edge.
* *Minimax LP*: In this LP, we are minimizing the maximum amount we can shrink from a contingent edge.

Through experiments, we discovered that the *Original LP* was the best approximation for the degree of strong controllability.
This method is referred to in our paper as the DSC-LP. 


#### probability.py
Stores functions that, given conflicts from non-DC network, return the predicted probability of successful dispatch on those networks.
##### Details
The approximation here is an application of CLT to sums of uniformly distributed random variables. -->

<!-- #### relax.py
Using the routines from `algorithm.py` and `LP.py`, defines various different ways of selecting "maximal" subintervals.
##### Description
The approaches for computing subintervals in the strong controllability case involve using solutions from some LPs.
In the dynamic controllability case, the main approach implemented is the `optimalRelax` function, which is a very straightforward algorithm (with `O(k*log k)` complexity, if `k` is the number of edges in the conflict) that finds the true maximum subintervals (in the sense of maximizing resulting volume while ensuring the conflict is resolved). -->

#### stn/stn.py
Defines STN, Edge, and Vertex classes.
##### Details
A Vertex represents an event in an STN.
It is encoded as a single node with a unique integer ID.

An Edge represents a constraint in an STN.
It is encoded as a pair of vertex IDs labeled with an interval and type.
The type designates an edge as a requirement (`stc`) or contingent (`stcu`) edge.

An STN consists of events with constraints in between events.
An STN object is encoded as set of Vertices with Edges between some pairs of vertices.

In general, a Vertex with ID zero is treated as the zero-timepoint.

#### stn/stnjsontools.py
Provides functions to create STN objects from input JSON files.


### Secondary Programs

<!-- #### dc_stn.py
An older file.
Builds a `DC_STN` class that represents STNUs with an aim of manipulating the associated labeled distance graph.
Implements the DC checking algorithm described in [(Morris 2003)](https://pdfs.semanticscholar.org/6fb1/b64231b23924bd9e2a1c49f3b282a99e2f15.pdf).
Also has a function for checking if STNs are consistent or not. -->

#### empirical.py
Computes and plots empirical results, mainly related to strong controllability.
##### Details
Leverages functions from `LP.py`, `dispatch.py`, and `relax.py` to measure approximate and exact degrees of strong controllability, approximate degrees of dynamic controllability, and true success rates with different dispatch strategies.
Comtains some methods for plotting these results as well.

<!-- #### plot.py
A file that makes use of the `plotly` module to make some nice graphs of the results outputted by `empirical.py`.  

#### result_stats.py
A short file for computing correlations between some of the sets of data stored in the `result` folder.

#### simulation.py
Attempts at writing early and late strategies for dispatch on STNUs.
This program did not really end up getting used, since the attempts at implementing early and late strategies here are incorrect.
Instead, we only used the simulation presented in `dispatch.py`.

#### util.py
Holds a few helpful functions, which are called by other programs.
##### Details
Provides
- `STNtoDCSTN`: a function for converting from `STN` objects to `DC_STN` objects
- `normal`:     a function for getting STNUs in normal form (that is, all contingent edges have lower bound zero)
- `PriorityQueue`: a class implementing a standard min-heap

### Interfacing with NEOS

#### build_xml.py
A python program to take in AMPL model files and convert them to an XML format that can be submitted to the NEOS server.

#### model.py
Given an STNU, builds the corresponding optimization problem for computing the degree of strong controllability for the network.
##### Details
The output is an AMPL file.
The objective function is the logarithm of the product of the lengths of contingent subintervals.

#### NeosClient.py
A NEOS Python client (made available by the [NEOS server](https://neos-server.org/neos/downloads.html)) that has been mildly modified. This is used to submit optimization problems to NEOS, and then extract the values achieved by the objective function once the jobs are finished. -->



<!-- ### Other Files

#### result
This folder contains several JSON files storing results we found related to degree of controllability, empirical success rates for dispatch, and solutions to nonlinear optimization problems. -->

## Credits
The Summer 2022 HEATlab consisted of teams *Improvisational-Teamwork* and *MAD-TN*, which our team being the spiritual extention of 2021 team *Controlled-Choices*/ and 2019 team *Ctrl-Alt-Repeat*/
All teams worked at Harvey Mudd College under the supervision of [Professor Jim Boerkoel](https://www.cs.hmc.edu/~boerkoel/) as part of the HMC CS department's "[summer of CS](https://www.cs.hmc.edu/research/)".

- *Improvisational-Teamwork* ~ Team {Rosy Chen, Eryn Ma, Ingrid Wu}
  - Evaluate how well existing scheduling approaches for PSTN with normal, symmetric distributions extend to skewed, non-ordinary distributions
  - The paper can be accessed at https://ojs.aaai.org/index.php/ICAPS/article/view/27183.
  
  ## License
  This project is licensed under the terms of the MIT license.
