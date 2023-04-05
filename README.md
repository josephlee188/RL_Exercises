# RL_Exercises
Reinforcement learning exercises from R.S. Sutton &amp; A. Barto's "Reinforcement Learning: An Introduction" (1992)

### Jack's car rental problem

Finding optimal strategy for Jack which gives optimal reward (please refer to the book for details of the problem).

<img src="https://user-images.githubusercontent.com/73336039/230065489-4202e420-6a8a-4531-84e2-8466b94e3562.png" alt="Day 0" width="30%"><img src="https://user-images.githubusercontent.com/73336039/230066912-6ffe83d1-7e2e-4b49-9930-c2adfc160801.png" alt="Day 1" width="30%"><img src="https://user-images.githubusercontent.com/73336039/230066924-674e069d-f745-4b57-812c-6aae70d1e658.png" alt="Day 2" width="30%">
<img src="https://user-images.githubusercontent.com/73336039/230066930-43cf3005-ce13-4523-9567-9fc074e5471e.png" alt="Day 3" width="30%">
<img src="https://user-images.githubusercontent.com/73336039/230066941-51fcadbf-8f2d-4db0-928a-bd11c2970d25.png" alt="Day 4" width="30%">
<img src="https://user-images.githubusercontent.com/73336039/230066953-f75d6f45-9cd8-486f-8781-eecece9fa663.png" alt="Day 5" width="30%">

where the heatmaps are through Day 0 ~ 5. <br>

$n_1$: Number of cars at parking lot 1.

$n_2$: Number of cars at parking lot 2. 

The colors represent the number of cars to be moved from lot 1 to 2.

&nbsp;

### Windy gridworld

Uses Sarsa on-policy TD algorithm to find the quickest route to the goal when wind is blowing upwards.

<img src="https://user-images.githubusercontent.com/73336039/230068152-fad9b8a3-68f9-499e-b2c2-2daf566a950a.png" alt="windygw" width="50%">

The color represents steps.

&nbsp;


### Mountain car problem

Using TD($\lambda$) with continuous state (discrete action) to find optimal policy for car to reach the goal.

After 10 episodes 

<img src="https://user-images.githubusercontent.com/73336039/230069460-15b01070-ad7c-4b9e-be41-01682ce1fe45.gif" alt="10" width="40%">

After 100 episodes

<img src="https://user-images.githubusercontent.com/73336039/230069807-438d779c-c0e5-486f-bef3-e5beece32320.gif" alt="100" width="40%">

After 1000 episodes

<img src="https://user-images.githubusercontent.com/73336039/230069956-43cc3b29-b635-45e8-b595-b8efb362fad5.gif" alt="1000" width="40%">

Value function (after 100 episodes)

<img src="https://user-images.githubusercontent.com/73336039/230070118-ac43d32a-b2d6-43f1-b2ce-a66c61342d22.png" alt="vf_100" width="40%">



