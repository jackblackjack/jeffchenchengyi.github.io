---
interact_link: content/machine-learning/08-genetic-algorithms/saga-fpga.ipynb
kernel_name: python3
has_widgets: false
title: 'Evolutionary Algorithms on FPGAs'
prev_page:
  url: /machine-learning/08-genetic-algorithms/README.html
  title: 'Species Adaptation Genetic Algorithms'
next_page:
  url: /machine-learning/08-genetic-algorithms/overview-of-es.html
  title: 'Overview of Evolutionary Strategies'
comment: "***PROGRAMMATICALLY GENERATED, DO NOT EDIT. SEE ORIGINAL FILES IN /content***"
---


# Evolutionary Algorithms for FPGA Configuration

By: Chengyi (Jeff) Chen



---
## What is an FPGA?

- FPGA stands for *Field Programmable Gate Arrays*
- FPGA replaces devices like CPU, I/O blocks, and Digital Signal Processors by using programmable logic (reconfiguration of the digital circuit design) to execute specific functions



---
## Experiment

Details:

- The Xilinx XC6216 FPGA was used which contains a 64 x 64 grid of configurable logic blocks (CLBs), but only a 10 x 10 corner was used

- It also contains I/O blocks in the parameter of the FPGA

- Genetic Algorithms can be used without any constraints because internal connections in the FPGA are **uni-directional**, meaning there is no possible way to configure the FPGA such that it breaks

Goal:

- Want to search for the best circuit design on those 10 x 10 grid of CLBs so that we can 
    
    1. Input (Tone Generator): 1kHz square wave --> Output (Analogue Integrator): +5V
    
    2. Input (Tone Generator): 10kHz square wave --> Output (Analogue Integrator): 0V


- This will be done by trying to maximize the difference between the sum of my integrator readings from the 1kHz input and the sum of my integrator readings from the 10kHz input
  
Algorithm:

1. Hardware-reset signal on FPGA to "clean" FPGA from previous configs

2. Initialize 50 random 1800-bit strings

3. For each 1800-bit string, configure 10 x 10 corner using the 1800-bit string (50 loops)

4. Send each input every 500ms:

    1. 5 1kHz tones --> $i_t$ integrator reading before output
    
    2. 5 10kHz tones --> $i_t$ integrator reading before output
    
4. Evaluate fitness for each configuration using

$$
\text{fitness} = \frac{1}{10} \vert (k_1 \sum_{t \in S_1} i_t) - (k_2 \sum_{t \in S_{10}} i_t) \vert\,\text{where}\,
\begin{cases}
    k_1 &= 1/30730.746 \\
    k_2 &= 1/30527.973
\end{cases}
$$

4. Choose the next generation

    1. Single fittest individual is twice as likely on average to be chosen to be used in next generation as compared to median-ranked 1800-bit string, but both follow 2 mutation patterns
        
        1. Probability of this (single-point cross-over) happening is 0.7: <img src='https://upload.wikimedia.org/wikipedia/commons/thumb/5/56/OnePointCrossover.svg/231px-OnePointCrossover.svg.png'/>
        
        2. This (per-bit mutation) happened **on average** 2.7 times for each 1800-bit string: <img src='https://localwire.pl/wp-content/uploads/2016/03/Mutation.svg'/>



---
## Results

- It takes 3,500 generations to get the perfect desired outcome and the results converged (did not get better with more generations)

- Mean fitness of the population jumped after the 2,600th generation

- Mean Hamming Distance (How different each of the 1800-bit strings were from each other) decreased sharply at the 2,600th generation and converged thereafter



---
## Analysis

Are there useless CLBs that we included in the configuration?

- We are also interested in finding which of the CLBs are actually indeed helping the configuration to get the desired output:
    
    - Some CLBs were clamped (set to a constant value of 0 / 1) and the evaluation is re-run to see if performance is affected to determine whether the CLB is actually helping
    
    - It showed that there were some CLBs that were extremly important for performance if they are a functional unit, some that might cause performance to degrade a little, and some that are completely useless

Does temperature affect the performance?

- Yes, but the experiment was carried out ~ $10^{\circ} C$ and it had no issues



---
## Conclusion

- Every FPGA chip is unique in that it will have different propagation delays / capacitances (meaning temperature is also another variable that will affect different chips in different ways) and therefore 2 chips might result in very different configurations for the same function. Furthermore, different regions on the same chip will also have very different configurations that achieve the same desired results.

Future works:

- Having ~ 10 nominally identical FPGA chips and held at differing temperatures to evaluate whether evolutionary algorithms can configure an FPGA optimally even at a wide range of temperatures. 



---
### Resources:
- [Adrian Thompson on "An evolved circuit, intrinsic in silicon, entwined with physics."](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.50.9691&rep=rep1&type=pdf)
- [Intel FPGA's YouTube video on Basics of Programmable Logic: FPGA Architecture](https://www.youtube.com/watch?v=jbOjWp4C3V4)
- [SAGA Research paper](https://pdfs.semanticscholar.org/09f1/150debcf83b9245ac6eb30dcfefad99be953.pdf)

