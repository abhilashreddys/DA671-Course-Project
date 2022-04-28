# DA671-Course-Project

This repository contains codes for the Optimistic Q - Learning Algorithm from paper titled _Model-free Reinforcement Learning in Infinite-horizon Average-reward Markov Decision Processes_ [here](https://arxiv.org/pdf/1910.07072.pdf).

This project is done by [Abhilash](https://github.com/abhilashreddys), [Rahul](https://github.com/chindimaga), [Bhavnick](https://github.com/Bhavnicksm)

## How to run
 - Uncomment the desired agent in main.py (line 88,89) (Default - OptimisticDiscountedAgent)
 - Set the desired horizon T and nb_runs. Default T=50000, nb_runs=5. 
 - To save the data in a new folder change the storage_counter (The data will be saved in log/environment/agent_'storage_counter').
 - To plot previously stored data you can set the corresponding storage_counter in alg_storage
Plots will be saved in plots directory by default.