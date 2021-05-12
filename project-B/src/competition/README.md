## This folder contains the outputs of repeated competitions between our players.
Every filename starts with __out__. Then the name of the upper player is given. _eg._ `HA1-8-4-1` is the original heuristic agent with the heuristic parameters
```
_WEIGHTS = {
    'throw': 1,
    'kills': 8,
    'distance': 4,
    'diversity': 1
}
```
Then the lower agent is given in the same format after an underscore.

`HAL` is the Heuristic Agent with Lookahead.

To calculate the win rates for the player you can search for the words `upper`, `lower`, and `draw` in the output file.

## Setup
Call the program with `.\test-competition.ps1` from the `src` folder.

Initially you might get an error running PowerShell scripts, you have to give permission since the file could potentially be malicious. Check the file to make sure it isn't though you have my word. Then run PowerShell as admin, and you can give permission by running `set-executionpolicy remotesigned`.

Now it should work.