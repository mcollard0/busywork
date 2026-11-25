I wish to introduce heat to my office by way of my cpu and gpu. 

Program: 

Busy --cputhreads # by computational work. Busy --gputhread # gpu computation work. Attempt to busy cpucount-1 threads. 

--cpugoal #.## (default: cpucount-1) --gpugoal #.## (default: similar, processing units - 1)


Revise execution loop on a regular basis say every 5 seconds, to add or remove work to suit goal utilization.

Language: c++

