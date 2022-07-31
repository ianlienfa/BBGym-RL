* references
  * https://www.rapidtables.com/code/linux/gcc.html
  * https://www.mauriciopoppe.com/notes/computer-science/operating-systems/bin/make/

* Important!!
  * The job number should be manually defined in util/config.h !
  * This is due to the usage of <bitset> for performance enhancement
  * The bad news is that bitset for STL do not support dynamically change on its size
  * For the vanilla Lu algorithm, prune__OneRjSumCj__LU_AND_SAL__Theorem1 must be on

* Usage
  * train the network first: 
    make -j5 net 
    ./net -d **directory name**
  * prepare binary for inference
    change to "tester" branch, and compile binary for different labeler 
    move the binaries to current "build" directory and rename them with 
  * do inference:
    make -j5 main
    move the resulting .pt files to /inf directory
    ./main -d **test directory name**
  * draw:
    change the filename for evaluation in eval.ipynb
    run it

* Grid Search
  * make sure the corresponding variable is defined in config.h
  * make sure the corresponding variable is defined in cmakelist.txt
  * rerun the cmake command and assign the value to the variable, ex:
    * cmake -D V_HIDDEN_DIM='128' -D V_MAX_NUM_CNTR='9' -D V_LR_PI='4e-6' -D V_LR_Q='4e-5'
  * recompile the code

* use "-f" to provide the training set path
* by default the program requires a **/validation** subdirectory in the training set directory for validation purpose, user can provide a different directory by "-v"

