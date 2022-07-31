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
    ./net -d **test directory name**
  * draw:
    change the filename for evaluation in eval.ipynb
    run it

* use "-f" to provide the training set path
* by default the program requires a **/validation** subdirectory in the training set directory for validation purpose, user can provide a different directory by "-v"
