#!/bin/sh

# array of censor levels
declare -a censor_levels=("2" "3")

# loop over censor levels for static mode
for i in "${censor_levels[@]}"
do
   # copy original script to a new file
   cp job_scripts/tgcn/job_tgcn.sh job_scripts/tgcn/job_${i}.sh

   # replace censor_level in the new script
   sed -i "s/--censor_level=<x>/--censor_level=${i}/g" job_scripts/tgcn/job_${i}.sh
   # replace name
   sed -i "s/#BSUB -J TGCN_cpnll_static_3/#BSUB -J TGCN_cpnll_static_${i}/g" job_scripts/tgcn/job_${i}.sh

   # submit the new job script
   bsub < job_scripts/tgcn/job_${i}.sh
   rm job_scripts/tgcn/job_${i}.sh
done

# array of censor levels for dynamic mode
declare -a censor_levels_dynamic=("1" "2")

# loop over censor levels for dynamic mode
for i in "${censor_levels_dynamic[@]}"
do
   # copy original script to a new file
   cp job_scripts/tgcn/job_tgcn.sh job_scripts/tgcn/job_dynamic_${i}.sh

   # replace censor_level in the new script and add --censor_dynamic
   sed -i "s/--censor_level=<x>/--censor_dynamic --censor_level=${i}/g" job_scripts/tgcn/job_dynamic_${i}.sh
   sed -i "s/#BSUB -J TGCN_cpnll_static_3/#BSUB -J TGCN_cpnll_dynamic_${i}/g" job_scripts/tgcn/job_dynamic_${i}.sh

   # submit the new job script
   bsub < job_scripts/tgcn/job_dynamic_${i}.sh
   rm job_scripts/tgcn/job_dynamic_${i}.sh
done
