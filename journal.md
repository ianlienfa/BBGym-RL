* validation 不會有reward紀錄，所以如果validatiion時reward出現奇怪的值是正常的

### run-Aug1-0
* 現在光跑大一點的instance拉大epoch_per_instance數試試看
* 刪掉optimizer的內容重跑

### 待辦
* 先回去試著overfit單一instance
* 加入lstm試試看
* 或許需要把contour的行為print出來去做檢查
* 再跑一次沒有validation的檢查能不能overfit
* 觀察其他train RL agent的方式看看有沒有能借用的地方
