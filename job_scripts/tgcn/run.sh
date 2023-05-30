#!/bin/sh

# array of censor levels
declare -a forecast_leads=("1" "24" "48")

# array of directories
declare -a dirs=("job_scripts/tgcn/aware" "job_scripts/tgcn/unaware")

# loop over directories
for dir in "${dirs[@]}"
do
  # check if directory exists
  if [ -d "$dir" ]
  then
    # loop over files in directory
    for file in "$dir"/*
    do
      # check if it is a file not a directory
      if [ -f "$file" ]
      then
        # loop over censor levels for static mode
        for i in "${forecast_leads[@]}"
        do
          # copy original script to a new file
          cp "$file" job_scripts/tgcn/tmp.sh

          # replace forecast lead in the new script
          sed -i "s/--forecast_lead=1/--forecast_lead=${i}/g" job_scripts/tgcn/tmp.sh
          # replace name
          sed -i "s/lead_1/lead_${i}/g"  job_scripts/tgcn/tmp.sh

          # submit the new job script
          bsub < job_scripts/tgcn/tmp.sh
          rm job_scripts/tgcn/tmp.sh
        done
      fi
    done
  else
    echo "$dir does not exist."
  fi
done
