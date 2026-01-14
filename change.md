I want to have less dependence on `torchcodec` as I only use this for inference.
Let's create a version that's torchcodec free. 
create a plan to:
* Evaluate alternatives that are much easier lighter to install.
* It can do what torch codec do, probably slower is okay
* certing out faceing processor parts and function names stays the same
* implement the such code change