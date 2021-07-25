# clone conda env; uncomment the following line to create the env first
# conda env create -f ./MitoMMP.yml
# activate the conda env
conda activate MitoMMP;
# Fig 3B
python model_simulation.py --compound "Antimycin A";
# Fig 3C
python model_simulation.py --compound "Rotenone";
# Fig 4AB
python model_simulation_3d.py --compound 'Deguelin' 'Azoxystrobin';
# Fig 5
python model_simulation.py --compound "FCCP";
# Fig 6
python model_simulation.py --compound "Oligomycin";