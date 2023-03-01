#!/bin/bash  

dir=$( dirname -- "$0"; ) #directory where the file exist

commands_file=$dir"/commands.sh"

source $commands_file >  text_from_terminal.txt 