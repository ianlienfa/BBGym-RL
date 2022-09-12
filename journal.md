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

 
### 待辦
* 先回去試著overfit單一instance (可以做到！) (done)
* 加入lstm試試看
* 或許需要把contour的行為print出來去做檢查
* 再跑一次沒有validation的檢查能不能overfit
* 觀察其他train RL agent的方式看看有沒有能借用的地方
