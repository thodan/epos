#!/usr/bin/env python

# Shows stdin and stderr in terminal and saves it to a file at the same time.

# Example usage:
# python train.py |& ./tee.py log.txt

import sys
import os

if os.path.exists(sys.argv[1]):
  os.remove(sys.argv[1])

while True:
  line = sys.stdin.readline()
  if not line:
    break

  sys.stdout.write(line)

  with open(sys.argv[1], 'a') as handler:
    handler.write(line)
