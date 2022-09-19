* validation 不會有reward紀錄，所以如果validatiion時reward出現奇怪的值是正常的

### run-Aug1-0
* 現在光跑大一點的instance拉大epoch_per_instance數試試看
* 刪掉optimizer的內容重跑

### train-Aug-30 - Overfitting single instance
* 跑10000個epoch以後可以把rfinwzwvtl的node數從12..->1078
* 畫個圖
### Tracking the training process
* ./track -bt 10000 ../saved_model/rfin ../case/case0/rfinwzwvtl.in

### train Sep-1 - Overfitting single instance
* overfit kdy 的狀況：跑10000個epoch以後可以把kdy的node數從7968 -> 7824
* 小結：即使要overfit一個instance，也要跑很久，而且可能是需要tune parameter的，效果也沒有很好
* 小結：這次沒有出現週期性跳動，所以或許可以相信實作沒有問題
* 小結：現在的reward設計依然會出現一直選擇移動來小賺分數的情況
    ? 為什麼可以全部都選1 ?
    try: 改reward
* 待辦：根據現在的結果，我們可以把instance的數量拉大，估計相應訓練需要的epoch數去train train看
* step_per_epoch爆掉之後壞掉有可能原因是step沒有歸零->先去看training時如何歸零的

* 感覺還是有大問題在裡面，讓我的overfit fit不起來
* 回去檢查這幾個部分：
    reward discounting, especially around episode resets
    advantage calculations, again especially around resets
    buffering and batching, especially pairing the wrong rewards with the wrong observations
    重新將計算過程寫過一次


### train Sep-9
* network wored on probe instance, by setting reward only on "place", we get higher prob for that action
* found the bug of inferencing that for the step function, the second parameter should be set as "true"
* still have to test for the inference processes of instances, eval.py is not working

### train Sep-12
* still observes periodic jumps in the number of node search, printing out the state encoding to see 
* 找到bug了，worst upper bound 更新得怪怪的，影響state encoding的計算
* probably still have bug:
    * first compute the reward in inference mode
    * then see if the reward really goes down using the model in /anf
    * if not, maybe there's bug for computing the discounted reward
    * 比較不會有bug的方法是讓inference也會call submit, 在該處計算real_reward

 ### Sep-13
* real_reward tracker for inference implemented
* model seems to be not learning, probably need another prob env to debug
* first debug the value network and see if we can get the value network to learn right
    * limit obs to only one state (000...000)
    * we can still have 4 actions
    * the reward be constant 1
    * the value network should learn the value of the state to be 1
    -> The value network can learn the value to be pretty close to 1, seems to be working

* try debug the agent network then
* adjust the reward magnitude, we might need to see KL divergence computation 

* see if the network can learn a predictable (not constant) reward
* limit the state to randomly two state (01 and 10)
* 01 provide reward 1, 10 provide reward -1
* see if the value network is able to learn the value of the state and see if the q value is the same for all action at one specific state

### One action, random +1/-1 observation, one timestep long, obs-dependent +1/-1 reward every time
* {0, 1} -> 1 , {1, 0} -> -1, {1, 1} -> 0
* need **one node** update to be set up for testing.
* value learned, but the Q(s, a) has slightly weird value distribution, in which the first action has zero prob 
    -> solved by tuning up the entropy ratio
* done, seems to be working, value network can learn the observation dependent reward and the action distribution is correct

## 4 action, random +1/-1 observation, one timestep long, reward&obs-dependent +1/-1 reward every time
* {1, -1} + [1 0 0 0] -> 1 , {-1, 1} + [0 1 0 0] -> -1
* with entropy_ratio = 0.1, the value get for same state can be different, which is quite weird.
    -> the reason for this is the value network hopes to get as close as the reward every time, but the reward highly depends on the action, so the value network can always have bad predect for the value of the state
* the value network should be updated using the mean of serveral trajectories return-to-go instead of the return-to-go of a single trajectory
* The update on value network becomes unstable when 
    1. the trajectory is very small
    2. the visited state space is too large and in one update only one sample is retrieved for each state
    These reasons provides tited value estimate therefore oscillates the update.
    The loss function of value network is designed to be 
    * loss = (return-to-go - value)**2.mean()
    however this only performs good when the same state is visited multiple times in a single trajectory, which is not the case for our problem.

    * solution: 
        * update after multiple trajectories
        * 將單一次TRPO走的量調小，用多epoch來彌補
        * 要注意start_idx的correctness
        * 如果buffer滿的話要怎麼控制？

### 待辦
* 先回去試著overfit單一instance (可以做到！) (done)
* 加入lstm試試看
* 或許需要把contour的行為print出來去做檢查
* 再跑一次沒有validation的檢查能不能overfit
* 觀察其他train RL agent的方式看看有沒有能借用的地方
