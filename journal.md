* validation 不會有reward紀錄，所以如果validatiion時reward出現奇怪的值是正常的

### run-Aug1-0
* 現在光跑大一點的instance拉大epoch_per_instance數試試看
* 刪掉optimizer的內容重跑

### train-Aug-30 - Overfitting single instance
* 跑10000個epoch以後可以把rfinwzwvtl的node數從12..->1078
* 畫個圖
### Tracking the training process
* ./track -bt 10000 ../saved_model/rfin ../case/case0/rfinwzwvtl.in
* 檢查step_per_epoch 超過會壞掉的問題
* 為什麼會壞掉勒？ -> 不是step沒歸零，目前猜測是沒有進行last update
* 研究在early_stopping update那邊把update打開會不會壞掉
* optimal found -> 填補r
* 現在在修理「於epoch_end時不能call buffer->get的問題」，感覺idx出問題了
* 找到問題，要做一個大改動：將idx的上限調成max_size, 原本的上限是max_size-1

### 待辦
* 先回去試著overfit單一instance (可以做到！)
* 加入lstm試試看
* 或許需要把contour的行為print出來去做檢查
* 再跑一次沒有validation的檢查能不能overfit
* 觀察其他train RL agent的方式看看有沒有能借用的地方
