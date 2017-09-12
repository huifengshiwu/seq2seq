#!/usr/bin/env bash

root_dir=`pwd`

/usr/bin/env pip3 install tensorflow-gpu python-dateutil pyyaml matplotlib --user --upgrade

cat >>~/.bashrc << EOL
alias get-best-score=${root_dir}/scripts/get-best-score.py
alias plot-loss=${root_dir}/scripts/plot-loss.py
alias txt-plot="${root_dir}/scripts/plot-loss.py --txt"
alias copy-model=${root_dir}/scripts/copy-model.py
alias ssh-plot=${root_dir}/scripts/ssh-plot.sh
shopt -s expand_aliases
EOL
