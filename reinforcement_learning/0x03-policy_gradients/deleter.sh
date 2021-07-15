#!/bin/bash
# deleted all lines starting with the string pecified
# useful if using the full verbose output with inlcuded print statements active
# in the train file. (save output to file called results.txt
sed -i '/^into/d' results.txt
sed -i '/^ \[/d' results.txt
sed -i '/^\[/d' results.txt
sed -i '/^made/d' results.txt
sed -i '/^selection/d' results.txt
sed -i '/^created/d' results.txt
sed -i '/^past/d' results.txt
sed -i '/^selected/d' results.txt
sed -i '/^state/d' results.txt
